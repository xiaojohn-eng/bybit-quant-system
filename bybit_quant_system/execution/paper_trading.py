"""
Paper Trading Engine
====================
Simulates live trading on Bybit linear perpetuals with realistic margin,
fee deduction, funding-rate settlement, and liquidation logic.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEE_TAKER: float = 0.00055          # 0.055 % taker fee
FEE_MAKER: float = 0.0001           # 0.01 % maker fee
MAINTENANCE_MARGIN: float = 0.005   # 0.5 % mm rate
LIQUIDATION_BUFFER: float = 0.005   # extra 0.5 % buffer
FUNDING_INTERVAL_HOURS: int = 8


@dataclass
class _PaperPosition:
    """Internal position representation."""
    symbol: str
    side: str
    qty: float
    entry_price: float
    leverage: float
    margin: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    funding_paid: float = 0.0
    fees_paid: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_funding_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _TradeRecord:
    symbol: str
    side: str
    qty: float
    entry: float
    exit: float
    pnl: float
    fees: float
    opened_at: datetime
    closed_at: datetime
    close_reason: str


class PaperTradingEngine:
    """
    Paper-trading environment that mirrors Bybit inverse/linear mechanics.

    Parameters
    ----------
    client : Any, optional
        Optional real client for price feeds (falls back to supplied prices).
    initial_balance : float
        Starting wallet balance in USDT.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        initial_balance: float = 10000.0,
    ):
        self.client = client
        self.initial_balance: float = initial_balance

        # Balances
        self.balance: float = initial_balance
        self._used_margin: float = 0.0

        # Positions & history
        self.positions: Dict[str, _PaperPosition] = {}
        self.orders: List[Dict[str, Any]] = []
        self.trades: List[_TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []

        # Fee / funding tracking
        self.fees_paid: float = 0.0
        self.funding_paid: float = 0.0

        # Liquidation tracking
        self._liquidation_count: int = 0

        # Concurrency
        self._lock = asyncio.Lock()

        # Price cache (for _get_price when client unavailable)
        self._last_prices: Dict[str, float] = {}

        logger.info("PaperTradingEngine initialised balance=%.2f", initial_balance)

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a market (or limit) order.

        Returns
        -------
        dict
            Order result with ``order_id``, ``filled_price``, ``margin``, etc.
        """
        side = side.lower()
        async with self._lock:
            fill_price: float = price or await self._get_price(symbol)
            notional: float = qty * fill_price
            margin: float = notional / leverage
            fee: float = notional * (FEE_MAKER if order_type == "Limit" else FEE_TAKER)

            if margin + fee > self.balance:
                return {
                    "success": False,
                    "error": "Insufficient balance",
                    "required": margin + fee,
                    "available": self.balance,
                }

            self.balance -= margin + fee
            self._used_margin += margin
            self.fees_paid += fee

            order_id: str = f"paper_{symbol}_{side}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            order_record: Dict[str, Any] = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "order_type": order_type,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "margin": margin,
                "fee": fee,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.orders.append(order_record)

            # If adding to existing position, average the entry
            if symbol in self.positions:
                existing: _PaperPosition = self.positions[symbol]
                if existing.side == side:
                    total_qty: float = existing.qty + qty
                    existing.entry_price = (
                        existing.entry_price * existing.qty + fill_price * qty
                    ) / total_qty
                    existing.qty = total_qty
                    existing.margin += margin
                    existing.leverage = notional / existing.margin
                    existing.fees_paid += fee
                    if stop_loss:
                        existing.stop_loss = stop_loss
                    if take_profit:
                        existing.take_profit = take_profit
                    logger.info("Added to %s position %s qty=%s", side, symbol, qty)
                else:
                    # Opposite side – partial or full close then open new
                    close_qty: float = min(existing.qty, qty)
                    pnl: float = self._calc_pnl(existing, fill_price, close_qty)
                    self.balance += existing.margin * (close_qty / existing.qty) + pnl
                    existing.realized_pnl += pnl
                    existing.qty -= close_qty
                    existing.margin = existing.margin * (existing.qty / (existing.qty + close_qty)) if existing.qty > 0 else 0
                    existing.fees_paid += fee

                    if existing.qty <= 0:
                        self._record_trade(existing, fill_price, "reverse", fee)
                        del self.positions[symbol]

                    if qty > close_qty:
                        new_margin: float = (qty - close_qty) * fill_price / leverage
                        self.balance -= new_margin
                        self._used_margin += new_margin
                        self.positions[symbol] = _PaperPosition(
                            symbol=symbol,
                            side=side,
                            qty=qty - close_qty,
                            entry_price=fill_price,
                            leverage=leverage,
                            margin=new_margin,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            fees_paid=fee,
                        )
            else:
                self.positions[symbol] = _PaperPosition(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    entry_price=fill_price,
                    leverage=leverage,
                    margin=margin,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    fees_paid=fee,
                )

            logger.info(
                "Paper order %s %s %s @ %.4f qty=%.6s lev=%.1f margin=%.2f fee=%.4f",
                side, symbol, order_type, fill_price, qty, leverage, margin, fee,
            )
            return {
                "success": True,
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "filled_price": fill_price,
                "margin": margin,
                "fee": fee,
                "balance": self.balance,
            }

    # ------------------------------------------------------------------ #
    # Position close
    # ------------------------------------------------------------------ #

    async def close_position(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """Close the entire position for ``symbol``."""
        async with self._lock:
            if symbol not in self.positions:
                return {"success": False, "error": "No position", "symbol": symbol}

            pos: _PaperPosition = self.positions[symbol]
            exit_price: float = await self._get_price(symbol)
            pnl: float = self._calc_pnl(pos, exit_price, pos.qty)
            notional: float = pos.qty * exit_price
            fee: float = notional * FEE_TAKER

            self.balance += pos.margin + pnl - fee
            self._used_margin -= pos.margin
            self.fees_paid += fee

            result = {
                "success": True,
                "symbol": symbol,
                "side": pos.side,
                "qty": pos.qty,
                "entry": pos.entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "fee": fee,
                "balance": self.balance,
            }

            self._record_trade(pos, exit_price, "manual_close", fee)
            del self.positions[symbol]
            logger.info("Closed %s position PnL=%.4f balance=%.2f", symbol, pnl, self.balance)
            return result

    # ------------------------------------------------------------------ #
    # Tick simulation
    # ------------------------------------------------------------------ #

    async def tick(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a price tick: update PnL, check SL/TP/liquidation, settle funding.

        Parameters
        ----------
        price_data : dict
            ``{symbol: {"price": float, "funding_rate": float}, ...}``

        Returns
        -------
        dict
            Events that occurred (SL, TP, liquidation, funding).
        """
        async with self._lock:
            events: Dict[str, List[str]] = defaultdict(list)
            now: datetime = datetime.now(timezone.utc)

            for symbol, data in price_data.items():
                current_price: float = float(data["price"]) if isinstance(data, dict) else float(data)
                self._last_prices[symbol] = current_price  # Always cache latest price

                if symbol not in self.positions:
                    continue

                pos: _PaperPosition = self.positions[symbol]
                pos.unrealized_pnl = self._calc_pnl(pos, current_price, pos.qty)

                # ---- Check liquidation --------------------------------
                if self._check_liquidation(pos, current_price):
                    events[symbol].append("liquidation")
                    self._liquidation_count += 1
                    pnl: float = -pos.margin  # lose all margin
                    self.balance += pnl
                    self._used_margin -= pos.margin
                    self.fees_paid += pos.qty * current_price * FEE_TAKER
                    self._record_trade(pos, current_price, "liquidation", pos.qty * current_price * FEE_TAKER)
                    del self.positions[symbol]
                    logger.critical("LIQUIDATION %s @ %.4f margin=%.2f lost", symbol, current_price, pos.margin)
                    continue

                # ---- Check SL -----------------------------------------
                if pos.stop_loss is not None:
                    hit_sl: bool = (
                        current_price <= pos.stop_loss
                        if pos.side == "buy"
                        else current_price >= pos.stop_loss
                    )
                    if hit_sl:
                        events[symbol].append("stop_loss")
                        pnl = self._calc_pnl(pos, pos.stop_loss, pos.qty)
                        notional = pos.qty * current_price
                        fee = notional * FEE_TAKER
                        self.balance += pos.margin + pnl - fee
                        self._used_margin -= pos.margin
                        self.fees_paid += fee
                        self._record_trade(pos, pos.stop_loss, "stop_loss", fee)
                        del self.positions[symbol]
                        logger.info("SL hit %s @ %.4f pnl=%.4f", symbol, pos.stop_loss, pnl)
                        continue

                # ---- Check TP -----------------------------------------
                if pos.take_profit is not None:
                    hit_tp: bool = (
                        current_price >= pos.take_profit
                        if pos.side == "buy"
                        else current_price <= pos.take_profit
                    )
                    if hit_tp:
                        events[symbol].append("take_profit")
                        pnl = self._calc_pnl(pos, pos.take_profit, pos.qty)
                        notional = pos.qty * current_price
                        fee = notional * FEE_TAKER
                        self.balance += pos.margin + pnl - fee
                        self._used_margin -= pos.margin
                        self.fees_paid += fee
                        self._record_trade(pos, pos.take_profit, "take_profit", fee)
                        del self.positions[symbol]
                        logger.info("TP hit %s @ %.4f pnl=%.4f", symbol, pos.take_profit, pnl)
                        continue

                # ---- Funding settlement -------------------------------
                funding_rate: Optional[float] = data.get("funding_rate") if isinstance(data, dict) else None
                if funding_rate is not None:
                    hours_since: float = (now - pos.last_funding_at).total_seconds() / 3600
                    if hours_since >= FUNDING_INTERVAL_HOURS:
                        self._settle_funding(symbol, pos, funding_rate)
                        pos.last_funding_at = now
                        events[symbol].append(f"funding_{funding_rate:.6f}")

            # ---- Record equity curve --------------------------------
            equity: float = self._get_equity_unlocked()
            self.equity_curve.append({
                "timestamp": now.isoformat(),
                "equity": equity,
                "balance": self.balance,
                "used_margin": self._used_margin,
                "unrealized_pnl": sum(
                    p.unrealized_pnl for p in self.positions.values()
                ),
            })

            return dict(events)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_equity(self) -> float:
        """Return current equity (balance + unrealized PnL)."""
        return self._get_equity_unlocked()

    def _get_equity_unlocked(self) -> float:
        return self.balance + sum(
            p.margin + p.unrealized_pnl for p in self.positions.values()
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Compute full performance statistics.

        Returns
        -------
        dict
            total_return, win_rate, sharpe_ratio, max_drawdown,
            total_fees, total_funding, liquidation_count, trade_count.
        """
        if not self.trades:
            return {
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_fees": self.fees_paid,
                "total_funding": self.funding_paid,
                "liquidation_count": self._liquidation_count,
                "trade_count": 0,
                "avg_trade_pnl": 0.0,
                "profit_factor": 0.0,
            }

        pnls: np.ndarray = np.array([t.pnl for t in self.trades])
        wins: np.ndarray = pnls[pnls > 0]
        losses: np.ndarray = pnls[pnls < 0]

        total_return: float = (self.get_equity() - self.initial_balance) / self.initial_balance
        win_rate: float = len(wins) / len(pnls) if len(pnls) > 0 else 0.0

        # Sharpe (daily assumption: ~1 tick per step, annualise roughly)
        if len(pnls) > 1 and pnls.std() > 0:
            sharpe: float = (pnls.mean() / pnls.std()) * math.sqrt(365 * 24)
        else:
            sharpe = 0.0

        # Max drawdown from equity curve
        max_dd: float = self._max_drawdown()

        # Profit factor
        gross_profit: float = wins.sum() if len(wins) > 0 else 0.0
        gross_loss: float = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor: float = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_return_pct": round(total_return * 100, 2),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_fees": round(self.fees_paid, 4),
            "total_funding": round(self.funding_paid, 4),
            "liquidation_count": self._liquidation_count,
            "trade_count": len(self.trades),
            "avg_trade_pnl": round(pnls.mean(), 4),
            "profit_factor": round(profit_factor, 4),
        }

    # ------------------------------------------------------------------ #
    # Liquidation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_liquidation(pos: _PaperPosition, current_price: float) -> bool:
        """
        Check if position hits liquidation price.

        Long:  price <= entry * (1 - 1/L + mm + buffer)
        Short: price >= entry * (1 + 1/L - mm - buffer)
        """
        if pos.side == "buy":
            liq_price: float = pos.entry_price * (1 - 1 / pos.leverage + MAINTENANCE_MARGIN + LIQUIDATION_BUFFER)
            return current_price <= liq_price
        else:
            liq_price = pos.entry_price * (1 + 1 / pos.leverage - MAINTENANCE_MARGIN - LIQUIDATION_BUFFER)
            return current_price >= liq_price

    # ------------------------------------------------------------------ #
    # Funding settlement
    # ------------------------------------------------------------------ #

    def _settle_funding(
        self,
        symbol: str,
        position: _PaperPosition,
        funding_rate: float,
    ) -> None:
        """
        Apply funding payment.

        Long pays  funding_rate > 0; receives funding_rate < 0.
        Short receives funding_rate > 0; pays   funding_rate < 0.
        """
        notional: float = position.qty * position.entry_price
        payment: float = notional * funding_rate * (-1 if position.side == "buy" else 1)
        position.funding_paid += payment
        self.funding_paid += payment
        self.balance += payment
        logger.debug("Funding %s rate=%.6f payment=%.4f", symbol, funding_rate, payment)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.balance = self.initial_balance
        self._used_margin = 0.0
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.fees_paid = 0.0
        self.funding_paid = 0.0
        self._liquidation_count = 0
        logger.info("PaperTradingEngine reset")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _get_price(self, symbol: str) -> float:
        """Fetch latest price from cache, client, or raise."""
        # 1. Check cached price from tick()
        if symbol in self._last_prices:
            return float(self._last_prices[symbol])
        # 2. Try client
        if self.client is not None:
            try:
                return await self.client.get_latest_price(symbol)
            except Exception:
                pass
        raise RuntimeError(f"No price available for {symbol}; pass price explicitly")

    @staticmethod
    def _calc_pnl(pos: _PaperPosition, price: float, qty: float) -> float:
        """Calculate PnL for a given exit price and quantity."""
        if pos.side == "buy":
            return (price - pos.entry_price) * qty
        return (pos.entry_price - price) * qty

    def _record_trade(
        self,
        pos: _PaperPosition,
        exit_price: float,
        reason: str,
        fee: float,
    ) -> None:
        """Append a closed trade record."""
        self.trades.append(_TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            qty=pos.qty,
            entry=pos.entry_price,
            exit=exit_price,
            pnl=self._calc_pnl(pos, exit_price, pos.qty),
            fees=pos.fees_paid + fee,
            opened_at=pos.opened_at,
            closed_at=datetime.now(timezone.utc),
            close_reason=reason,
        ))

    def _max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        if not self.equity_curve:
            return 0.0
        equities: np.ndarray = np.array([e["equity"] for e in self.equity_curve])
        if len(equities) < 2:
            return 0.0
        peak: float = equities[0]
        max_dd: float = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd: float = (peak - eq) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd
