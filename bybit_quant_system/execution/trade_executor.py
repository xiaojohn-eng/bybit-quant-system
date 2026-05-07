"""
Trade Executor Module
=====================
Handles order execution, position management, stop-loss / take-profit,
trailing-stop updates and emergency close for Bybit linear perpetuals.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bybit_quant_system.risk.kelly_calculator import KellyCalculator
from bybit_quant_system.risk.dynamic_leverage import DynamicLeverageManager

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an active position."""
    symbol: str
    side: str                          # "Buy" or "Sell"
    qty: float
    entry_price: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_dist: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of a single signal execution."""
    success: bool
    symbol: str
    side: str
    qty: float
    price: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None
    message: str = ""
    closed_opposite: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TradeExecutor:
    """
    Executes trading signals with full position lifecycle management.

    Parameters
    ----------
    client : Any
        BybitClient (live) or PaperTradingEngine (paper).
    risk_manager : Any
        RiskManager instance for pre-trade checks.
    config : dict, optional
        Execution-level configuration overrides.
    """

    def __init__(
        self,
        client: Any,
        risk_manager: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.config = config or {}

        # Runtime state
        self.active_positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

        # Sub-components
        self.kelly = KellyCalculator()
        self.leverage_mgr = DynamicLeverageManager()

        # Configurable defaults
        self.default_sl_pct: float = self.config.get("default_sl_pct", 0.02)
        self.default_tp_pct: float = self.config.get("default_tp_pct", 0.04)
        self.trailing_enabled: bool = self.config.get("trailing_enabled", True)
        self.trailing_dist_pct: float = self.config.get("trailing_dist_pct", 0.015)
        self.max_positions: int = self.config.get("max_positions", 10)

        logger.info("TradeExecutor initialised")

    # ------------------------------------------------------------------ #
    # Signal execution
    # ------------------------------------------------------------------ #

    async def execute_signal(
        self,
        signal: Dict[str, Any],
        equity: float,
        current_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a trading signal end-to-end.

        Steps
        -----
        1. Risk manager pre-trade check.
        2. Compute dynamic leverage.
        3. Compute Kelly-optimal position size.
        4. Derive SL / TP prices.
        5. Close opposite position if present.
        6. Place the new order.
        7. Record in ``active_positions``.
        """
        symbol: str = signal["symbol"]
        side: str = signal["side"]          # "Buy" or "Sell"
        confidence: float = signal.get("confidence", 0.5)

        async with self._lock:
            # ---- 1. Risk check ---------------------------------------
            allowed, reason = await self.risk_manager.check_trade_allowed(
                symbol=symbol,
                side=side,
                equity=equity,
                active_positions=list(self.active_positions.values()),
            )
            if not allowed:
                logger.warning("Risk blocked %s %s: %s", side, symbol, reason)
                return ExecutionResult(
                    success=False, symbol=symbol, side=side,
                    qty=0.0, price=0.0, leverage=0.0,
                    message=f"Risk blocked: {reason}",
                )

            # ---- Fetch price if not provided --------------------------
            if current_price is None:
                try:
                    current_price = await self.client.get_latest_price(symbol)
                except Exception as exc:
                    logger.error("Price fetch failed for %s: %s", symbol, exc)
                    return ExecutionResult(
                        success=False, symbol=symbol, side=side,
                        qty=0.0, price=0.0, leverage=0.0,
                        message=f"Price fetch failed: {exc}",
                    )

            # ---- 2. Dynamic leverage ---------------------------------
            volatility: float = signal.get("volatility", 0.02)
            leverage: float = self.leverage_mgr.calculate(
                volatility=volatility,
                confidence=confidence,
            )

            # ---- 3. Kelly position size ------------------------------
            win_rate: float = signal.get("win_rate", 0.55)
            avg_win: float = signal.get("avg_win", self.default_tp_pct)
            avg_loss: float = signal.get("avg_loss", self.default_sl_pct)
            kelly_fraction: float = self.kelly.calculate_fraction(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            # Scale Kelly down for conservatism (half-Kelly)
            position_size: float = equity * kelly_fraction * 0.5 * leverage
            qty: float = position_size / current_price

            # Enforce minimum notional
            min_notional: float = self.config.get("min_notional", 5.0)
            if qty * current_price < min_notional:
                qty = min_notional / current_price

            # Round quantity to configured precision
            qty = round(qty, self.config.get("qty_precision", 3))

            # ---- 4. SL / TP ------------------------------------------
            stop_loss, take_profit = self._calculate_sl_tp(
                side=side,
                entry=current_price,
                volatility=volatility,
                signal=signal,
            )

            # ---- 5. Close opposite position --------------------------
            closed_opposite: bool = False
            if symbol in self.active_positions:
                existing: Position = self.active_positions[symbol]
                if existing.side != side:
                    logger.info("Closing opposite %s position on %s", existing.side, symbol)
                    await self.close_position(symbol, reason="reverse_signal")
                    closed_opposite = True
                else:
                    # Same direction – optional position add logic could go here
                    logger.info("Same-side position exists on %s; skipping", symbol)
                    return ExecutionResult(
                        success=False, symbol=symbol, side=side,
                        qty=0.0, price=current_price, leverage=leverage,
                        message="Same-side position already exists",
                    )

            # ---- 6. Place order --------------------------------------
            try:
                order_result = await self.client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type="Market",
                    price=None,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )
                order_id: Optional[str] = order_result.get("order_id") or order_result.get("id")
            except Exception as exc:
                logger.error("Order placement failed for %s: %s", symbol, exc)
                return ExecutionResult(
                    success=False, symbol=symbol, side=side,
                    qty=qty, price=current_price, leverage=leverage,
                    stop_loss=stop_loss, take_profit=take_profit,
                    message=f"Order failed: {exc}",
                )

            # ---- 7. Record position ----------------------------------
            position = Position(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=current_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=self._init_trailing_stop(side, current_price) if self.trailing_enabled else None,
                trailing_dist=current_price * self.trailing_dist_pct,
                order_id=order_id,
            )
            self.active_positions[symbol] = position
            self.pending_orders.pop(symbol, None)

            logger.info(
                "Executed %s %s @ %s qty=%s lev=%s SL=%s TP=%s",
                side, symbol, current_price, qty, leverage, stop_loss, take_profit,
            )

            return ExecutionResult(
                success=True,
                symbol=symbol,
                side=side,
                qty=qty,
                price=current_price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                order_id=order_id,
                message="Executed successfully",
                closed_opposite=closed_opposite,
            )

    # ------------------------------------------------------------------ #
    # Position management
    # ------------------------------------------------------------------ #

    async def manage_positions(self, current_data: Dict[str, Any]) -> None:
        """
        Iterate over active positions, check SL/TP, update trailing stops.

        Parameters
        ----------
        current_data : dict
            Mapping ``symbol -> {"price": float, ...}``.
        """
        async with self._lock:
            for symbol, pos in list(self.active_positions.items()):
                price_info = current_data.get(symbol)
                if price_info is None:
                    continue

                current_price: float = (
                    price_info["price"]
                    if isinstance(price_info, dict)
                    else float(price_info)
                )

                # Update unrealised PnL
                pos.unrealized_pnl = self._unrealized_pnl(pos, current_price)

                # Check stop-loss
                if pos.stop_loss is not None and self._hit_sl(pos, current_price):
                    logger.info("SL hit for %s @ %s (SL=%s)", symbol, current_price, pos.stop_loss)
                    await self._close_position_unlocked(symbol, reason="stop_loss")
                    continue

                # Check take-profit
                if pos.take_profit is not None and self._hit_tp(pos, current_price):
                    logger.info("TP hit for %s @ %s (TP=%s)", symbol, current_price, pos.take_profit)
                    await self._close_position_unlocked(symbol, reason="take_profit")
                    continue

                # Update trailing stop
                if self.trailing_enabled and pos.trailing_stop is not None:
                    await self._update_trailing_stop(pos, current_price)

    async def close_position(
        self,
        symbol: str,
        reason: str = "signal",
    ) -> Dict[str, Any]:
        """Public method to close a position by symbol."""
        async with self._lock:
            return await self._close_position_unlocked(symbol, reason)

    async def _close_position_unlocked(
        self,
        symbol: str,
        reason: str = "signal",
    ) -> Dict[str, Any]:
        """Internal close – must hold ``self._lock``."""
        if symbol not in self.active_positions:
            return {"success": False, "message": "No active position", "symbol": symbol}

        pos: Position = self.active_positions[symbol]
        try:
            close_side: str = "Sell" if pos.side == "Buy" else "Buy"
            result = await self.client.place_order(
                symbol=symbol,
                side=close_side,
                qty=pos.qty,
                order_type="Market",
                price=None,
            )
            pos.realized_pnl = pos.unrealized_pnl
            del self.active_positions[symbol]
            logger.info("Closed %s position on %s (reason=%s pnl=%.4f)",
                        pos.side, symbol, reason, pos.realized_pnl)
            return {
                "success": True,
                "symbol": symbol,
                "side": pos.side,
                "realized_pnl": pos.realized_pnl,
                "reason": reason,
                "order_result": result,
            }
        except Exception as exc:
            logger.error("Failed to close %s: %s", symbol, exc)
            return {"success": False, "symbol": symbol, "message": str(exc), "reason": reason}

    async def emergency_close_all(self, reason: str) -> Dict[str, Any]:
        """
        Close every active position immediately (market orders).

        Returns
        -------
        dict
            Summary with ``closed``, ``failed``, and ``total_pnl``.
        """
        async with self._lock:
            symbols: list = list(self.active_positions.keys())

        closed_count: int = 0
        failed_count: int = 0
        total_pnl: float = 0.0
        tasks = [self._close_position_unlocked(sym, reason=reason) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                failed_count += 1
                logger.error("Emergency close error: %s", res)
            elif res.get("success"):
                closed_count += 1
                total_pnl += res.get("realized_pnl", 0.0)
            else:
                failed_count += 1

        logger.critical(
            "EMERGENCY CLOSE ALL finished: closed=%d failed=%d total_pnl=%.4f (%s)",
            closed_count, failed_count, total_pnl, reason,
        )
        return {
            "success": True,
            "closed": closed_count,
            "failed": failed_count,
            "total_pnl": total_pnl,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_position_summary(self) -> Dict[str, Any]:
        """Return a snapshot of current positions and aggregate stats."""
        total_unrealized: float = sum(p.unrealized_pnl for p in self.active_positions.values())
        return {
            "position_count": len(self.active_positions),
            "positions": {
                sym: {
                    "side": p.side,
                    "qty": p.qty,
                    "entry": p.entry_price,
                    "leverage": p.leverage,
                    "unrealized_pnl": round(p.unrealized_pnl, 4),
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                }
                for sym, p in self.active_positions.items()
            },
            "total_unrealized_pnl": round(total_unrealized, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ---- SL / TP calculations -----------------------------------------

    def _calculate_sl_tp(
        self,
        side: str,
        entry: float,
        volatility: float,
        signal: Dict[str, Any],
    ):
        """Derive stop-loss and take-profit prices."""
        atr: float = signal.get("atr", entry * volatility)
        sl_pct: float = signal.get("sl_pct", self.default_sl_pct)
        tp_pct: float = signal.get("tp_pct", self.default_tp_pct)

        # Use ATR-based SL/TP when available for tighter risk control
        sl_dist: float = max(atr * 1.5, entry * sl_pct)
        tp_dist: float = max(atr * 3.0, entry * tp_pct)

        if side == "Buy":
            stop_loss = entry - sl_dist
            take_profit = entry + tp_dist
        else:
            stop_loss = entry + sl_dist
            take_profit = entry - tp_dist

        return round(stop_loss, 4), round(take_profit, 4)

    def _init_trailing_stop(self, side: str, entry: float) -> float:
        """Initial trailing stop price."""
        dist: float = entry * self.trailing_dist_pct
        return entry - dist if side == "Buy" else entry + dist

    async def _update_trailing_stop(self, pos: Position, current_price: float) -> None:
        """Advance trailing stop when price moves favourably."""
        if pos.side == "Buy":
            new_stop = current_price - pos.trailing_dist
            if new_stop > pos.trailing_stop:
                pos.trailing_stop = new_stop
                logger.debug("Trailing stop raised for %s -> %.4f", pos.symbol, new_stop)
        else:
            new_stop = current_price + pos.trailing_dist
            if new_stop < pos.trailing_stop:
                pos.trailing_stop = new_stop
                logger.debug("Trailing stop lowered for %s -> %.4f", pos.symbol, new_stop)

    # ---- PnL & hit detection ------------------------------------------

    @staticmethod
    def _unrealized_pnl(pos: Position, current_price: float) -> float:
        if pos.side == "Buy":
            return (current_price - pos.entry_price) * pos.qty
        return (pos.entry_price - current_price) * pos.qty

    @staticmethod
    def _hit_sl(pos: Position, price: float) -> bool:
        if pos.stop_loss is None:
            return False
        return price <= pos.stop_loss if pos.side == "Buy" else price >= pos.stop_loss

    @staticmethod
    def _hit_tp(pos: Position, price: float) -> bool:
        if pos.take_profit is None:
            return False
        return price >= pos.take_profit if pos.side == "Buy" else price <= pos.take_profit
