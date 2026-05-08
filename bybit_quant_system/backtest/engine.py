"""
Event-Driven Backtest Engine for Bybit Quantitative Trading System

Provides a bar-by-bar event-driven backtester that simulates order execution
with realistic fee models, slippage, stop-loss / take-profit checks, and
periodic funding-fee deductions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from bybit_quant_system.strategies.base_strategy import Signal as StrategySignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents a single open position."""

    side: str                      # "long" or "short"
    qty: float                     # absolute quantity (positive)
    entry_price: float
    leverage: float = 1.0
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    entry_time: Any = None
    entry_idx: int = 0
    cum_funding: float = 0.0       # accumulated funding fees

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1


@dataclass
class Trade:
    """A completed trade record."""

    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fee_paid: float
    funding_paid: float
    exit_reason: str               # "signal", "stop_loss", "take_profit"


@dataclass
class BacktestResult:
    """Container for backtest results."""

    strategy_name: str
    symbol: str
    metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    trades: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    parameters: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy Protocol
# ---------------------------------------------------------------------------

class StrategyProtocol(Protocol):
    """Protocol that all strategies must satisfy."""

    name: str

    def generate_signal(self, df: pd.DataFrame) -> StrategySignal:
        ...

    def get_parameters(self) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtest engine.

    Simulates bar-by-bar execution with configurable fees, slippage,
    leverage, stop-loss / take-profit, and funding-rate costs.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.00055,
        slippage_pct: float = 0.0001,
    ) -> None:
        """
        Parameters
        ----------
        initial_capital : float
            Starting equity in quote currency.
        maker_fee : float
            Maker fee rate (e.g. 0.0002 = 0.02%).
        taker_fee : float
            Taker fee rate (e.g. 0.00055 = 0.055%).
        slippage_pct : float
            Slippage applied per execution (e.g. 0.0001 = 0.01%).
        """
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct

        # Runtime state (reset per ``run``)
        self._capital: float = initial_capital
        self._position: Optional[Position] = None
        self._equity_curve: List[Dict[str, Any]] = []
        self._trades: List[Trade] = []
        self._trade_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: StrategyProtocol,
        df: pd.DataFrame,
        symbol: str,
        leverage: float = 1.0,
        funding_rate: float = 0.0,
        funding_interval_hours: float = 8.0,
    ) -> BacktestResult:
        """
        Run a full bar-by-bar backtest.

        Parameters
        ----------
        strategy : StrategyProtocol
            Strategy instance with ``generate_signal(df_slice) -> Signal``.
        df : pd.DataFrame
            OHLCV data. Must contain columns: open, high, low, close, volume
            and optionally ``funding_rate``.
        symbol : str
            Symbol being backtested (e.g. "BTCUSDT").
        leverage : float
            Position leverage (e.g. 2.0 = 2x).
        funding_rate : float
            Per-period funding rate used if df has no ``funding_rate`` col.
        funding_interval_hours : float
            How often funding fees are charged.

        Returns
        -------
        BacktestResult
        """
        if df.empty:
            raise ValueError("DataFrame is empty.")

        # Reset state
        self._reset_state()

        # Estimate bar interval in minutes for funding simulation
        bar_interval_min = self._estimate_bar_interval(df)
        bars_per_funding = max(
            1, int((funding_interval_hours * 60) / max(bar_interval_min, 1))
        )

        # Main loop
        for i in range(len(df)):
            bar = df.iloc[i]
            slice_df = df.iloc[: i + 1]

            # Current market prices
            open_p = float(bar["open"])
            high_p = float(bar["high"])
            low_p = float(bar["low"])
            close_p = float(bar["close"])
            timestamp = bar.name if hasattr(bar, "name") else i

            # 1. Check SL/TP for existing position
            if self._position is not None:
                exit_price, exit_reason = self._check_sl_tp(
                    self._position, high_p, low_p, close_p
                )
                if exit_price is not None:
                    self._close_position(
                        exit_price=exit_price,
                        exit_idx=i,
                        exit_reason=exit_reason,
                    )

            # 2. Generate signal
            signal = strategy.generate_signal(slice_df)

            # 3. Execute signal
            if signal.action in ("buy", "sell"):
                self._process_signal(
                    signal=signal,
                    bar=bar,
                    idx=i,
                    leverage=leverage,
                )

            # 4. Funding-fee simulation (every N bars)
            if self._position is not None and i % bars_per_funding == 0:
                fr = (
                    float(bar.get("funding_rate", funding_rate))
                    if "funding_rate" in bar
                    else funding_rate
                )
                self._apply_funding_fee(self._position, close_p, fr)

            # 5. Record equity
            equity = self._calculate_equity(close_p)
            self._equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": equity,
                    "close": close_p,
                    "position_side": (
                        self._position.side if self._position else "flat"
                    ),
                }
            )

        # Close any remaining position at last close
        if self._position is not None:
            last_close = float(df.iloc[-1]["close"])
            self._close_position(
                exit_price=last_close,
                exit_idx=len(df) - 1,
                exit_reason="end_of_data",
            )

        # Build result
        equity_df = pd.DataFrame(self._equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)

        trades_df = pd.DataFrame([self._trade_to_dict(t) for t in self._trades])

        parameters = {
            "initial_capital": self.initial_capital,
            "leverage": leverage,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "slippage_pct": self.slippage_pct,
            **strategy.get_parameters(),
        }

        metrics = self._calculate_comprehensive_metrics(equity_df)

        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades_df,
            parameters=parameters,
        )
        logger.info(
            "Backtest %s on %s finished: return=%.2f%%, sharpe=%.3f, trades=%d",
            strategy.name,
            symbol,
            metrics.get("total_return_pct", 0.0),
            metrics.get("sharpe_ratio", 0.0),
            metrics.get("total_trades", 0),
        )
        return result

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal: StrategySignal,
        bar: pd.Series,
        idx: int,
        leverage: float,
    ) -> None:
        """Process a buy/sell signal."""
        close_p = float(bar["close"])

        # Map strategy signal action to position side
        action_map = {"buy": "long", "sell": "short"}
        desired_side = action_map.get(signal.action, signal.action)

        # If we have an opposite position, close it first
        if self._position is not None and self._position.side != desired_side:
            self._close_position(
                exit_price=close_p,
                exit_idx=idx,
                exit_reason="signal_reverse",
            )

        # If flat (or just closed opposite), open new position
        if self._position is None:
            # Determine position size from confidence (default 0.95 if not set)
            size_pct = signal.confidence * 0.95 if signal.confidence is not None else 0.95
            position_value = self._capital * size_pct * leverage
            qty = position_value / (close_p * (1 + self.slippage_pct))

            # Apply slippage on entry
            fill_price = close_p * (1 + self.slippage_pct)

            # Calculate SL/TP prices from metadata or defaults
            sl_price = None
            tp_price = None
            sl_pct = signal.metadata.get("sl_pct", 0.02) if signal.metadata else 0.02
            tp_pct = signal.metadata.get("tp_pct", 0.04) if signal.metadata else 0.04
            if sl_pct is not None:
                if desired_side == "long":
                    sl_price = fill_price * (1 - sl_pct)
                else:
                    sl_price = fill_price * (1 + sl_pct)
            if tp_pct is not None:
                if desired_side == "long":
                    tp_price = fill_price * (1 + tp_pct)
                else:
                    tp_price = fill_price * (1 - tp_pct)

            # Deduct taker fee on entry
            fee = position_value * self.taker_fee
            self._capital -= fee

            self._position = Position(
                side=desired_side,
                qty=abs(qty),
                entry_price=fill_price,
                leverage=leverage,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_time=bar.name if hasattr(bar, "name") else idx,
                entry_idx=idx,
            )
            self._trade_count += 1

    def _close_position(
        self,
        exit_price: float,
        exit_idx: int,
        exit_reason: str,
    ) -> None:
        """Close the current position and record the trade."""
        if self._position is None:
            return

        pos = self._position

        # Apply slippage on exit
        if pos.side == "long":
            fill_price = exit_price * (1 - self.slippage_pct)
        else:
            fill_price = exit_price * (1 + self.slippage_pct)

        # PnL calculation
        if pos.side == "long":
            gross_pnl = (fill_price - pos.entry_price) * pos.qty * pos.leverage
        else:
            gross_pnl = (pos.entry_price - fill_price) * pos.qty * pos.leverage

        # Exit fee
        notional_exit = fill_price * pos.qty
        exit_fee = notional_exit * self.taker_fee
        self._capital -= exit_fee

        # Update capital with gross PnL
        self._capital += gross_pnl

        net_pnl = gross_pnl - exit_fee - pos.cum_funding

        trade = Trade(
            entry_idx=pos.entry_idx,
            exit_idx=exit_idx,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            qty=pos.qty,
            pnl=net_pnl,
            fee_paid=exit_fee,
            funding_paid=pos.cum_funding,
            exit_reason=exit_reason,
        )
        self._trades.append(trade)
        self._position = None

    def _check_sl_tp(
        self,
        position: Position,
        high: float,
        low: float,
        close: float,
    ) -> tuple[Optional[float], Optional[str]]:
        """
        Check if SL or TP was hit during this bar.

        Returns (exit_price, exit_reason) or (None, None).
        """
        if position.side == "long":
            # Stop-loss: low hit SL price
            if position.sl_price is not None and low <= position.sl_price:
                return position.sl_price, "stop_loss"
            # Take-profit: high hit TP price
            if position.tp_price is not None and high >= position.tp_price:
                return position.tp_price, "take_profit"
        else:  # short
            # Stop-loss: high hit SL price
            if position.sl_price is not None and high >= position.sl_price:
                return position.sl_price, "stop_loss"
            # Take-profit: low hit TP price
            if position.tp_price is not None and low <= position.tp_price:
                return position.tp_price, "take_profit"
        return None, None

    def _apply_funding_fee(
        self,
        position: Position,
        mark_price: float,
        funding_rate: float,
    ) -> None:
        """Apply periodic funding fee to open position."""
        position_value = position.qty * mark_price
        # Longs pay funding when rate > 0; shorts receive (and vice versa)
        fee = position_value * funding_rate * (-1.0 if position.side == "long" else 1.0)
        position.cum_funding += fee
        self._capital -= fee

    def _calculate_equity(self, current_price: float) -> float:
        """Total equity including unrealized PnL of open position."""
        equity = self._capital
        if self._position is not None:
            pos = self._position
            if pos.side == "long":
                unrealized = (current_price - pos.entry_price) * pos.qty * pos.leverage
            else:
                unrealized = (pos.entry_price - current_price) * pos.qty * pos.leverage
            equity += unrealized - pos.cum_funding
        return equity

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_dsr(self, sharpe: float, n_trials: int = 1) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2025)

        校正选择偏差后的统计显著夏普比率。
        DSR > 0.95 表示策略统计显著（非运气）。

        当只测试1个策略时，DSR = 标准正态CDF(Sharpe * sqrt(T))

        Returns:
            float: DSR值，范围[0,1]
        """
        if self._trade_count == 0:
            return 0.0

        T = self._trade_count  # 交易次数
        if T < 10:
            return 0.0

        # 收集交易收益用于偏度/峰度校正
        returns_list = []
        for trade in self._trades:
            if trade.entry_price and trade.exit_price:
                ret = (trade.exit_price - trade.entry_price) / trade.entry_price
                if trade.side == "short":
                    ret = -ret
                returns_list.append(ret)

        if len(returns_list) < 10:
            # 简化版本：Sharpe * sqrt(T/252)
            z = sharpe * np.sqrt(T / 252)
        else:
            returns_arr = np.array(returns_list)
            r_mean = np.mean(returns_arr)
            r_std = np.std(returns_arr) + 1e-10
            skew = np.mean(((returns_arr - r_mean) / r_std) ** 3)
            kurt = np.mean(((returns_arr - r_mean) / r_std) ** 4)
            z = sharpe * np.sqrt(T)
            # 偏度和峰度校正 (Edgeworth展开)
            z = z * (1 + skew * (z ** 2 - 1) / 6 + (kurt - 3) * (z ** 3 - 3 * z) / 24)

        # 标准正态CDF近似 (Abramowitz & Stegun)
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if z >= 0 else -1
        x = abs(z) / np.sqrt(2)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        dsr = 0.5 * (1.0 + sign * y)
        return max(0.0, min(1.0, dsr))

    def _calculate_comprehensive_metrics(
        self, equity_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate a comprehensive set of performance metrics."""
        if equity_df.empty:
            return {}

        equity = equity_df["equity"].values
        if len(equity) < 2:
            return {}

        # Returns series
        returns = np.diff(equity) / equity[:-1]
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Trading days per year estimate (crypto ~ 365)
        periods_per_year = 365 * 24  # hourly default
        n = len(returns)

        # Annualized return
        ann_return = (1 + total_return) ** (periods_per_year / n) - 1 if n > 0 else 0.0

        # Sharpe ratio (annualized, assumes risk-free rate = 0)
        ret_mean = np.mean(returns)
        ret_std = np.std(returns, ddof=1)
        sharpe = (
            (ret_mean / ret_std) * math.sqrt(periods_per_year) if ret_std > 1e-12 else 0.0
        )

        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-10
        sortino = (
            (ret_mean / downside_std) * math.sqrt(periods_per_year)
            if downside_std > 1e-12
            else 0.0
        )

        # Maximum drawdown & duration
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = float(np.max(drawdowns))

        # Max drawdown duration (in bars)
        peak_idx = 0
        max_dd_duration = 0
        current_dd_duration = 0
        for i in range(len(equity)):
            if equity[i] >= cummax[i]:
                peak_idx = i
                current_dd_duration = 0
            else:
                current_dd_duration = i - peak_idx
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration

        # Win rate & profit factor from trades
        trades_df = pd.DataFrame([self._trade_to_dict(t) for t in self._trades])
        total_trades = len(trades_df)

        if total_trades > 0:
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            win_rate = len(wins) / total_trades
            gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
            gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 1e-10
            profit_factor = gross_profit / gross_loss
            avg_trade_return = trades_df["pnl"].mean()
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0

        # Calmar ratio
        calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

        # Kelly fraction: W - (1-W)/(PF)  where PF = avg_win/avg_loss
        if total_trades > 0 and not trades_df.empty:
            wins_pnl = trades_df[trades_df["pnl"] > 0]["pnl"]
            losses_pnl = trades_df[trades_df["pnl"] <= 0]["pnl"]
            avg_win = wins_pnl.mean() if not wins_pnl.empty else 0.0
            avg_loss = abs(losses_pnl.mean()) if not losses_pnl.empty else 1e-10
            win_loss_ratio = avg_win / avg_loss
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0.0
            kelly = max(-1.0, min(1.0, kelly))  # Clamp
        else:
            kelly = 0.0

        # Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2025)
        dsr = self._calculate_dsr(sharpe)

        metrics = {
            "total_return_pct": round(total_return * 100, 4),
            "annualized_return_pct": round(ann_return * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "max_drawdown_duration_bars": int(max_dd_duration),
            "win_rate_pct": round(win_rate * 100, 4),
            "profit_factor": round(profit_factor, 4),
            "calmar_ratio": round(calmar, 4),
            "total_trades": total_trades,
            "avg_trade_return": round(avg_trade_return, 4),
            "kelly_fraction": round(kelly, 4),
            "final_equity": round(equity[-1], 4),
            "deflated_sharpe_ratio": round(dsr, 4),
        }
        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all runtime state for a fresh backtest."""
        self._capital = self.initial_capital
        self._position = None
        self._equity_curve = []
        self._trades = []
        self._trade_count = 0

    @staticmethod
    def _estimate_bar_interval(df: pd.DataFrame) -> float:
        """Estimate bar interval in minutes from datetime index."""
        if len(df) < 2:
            return 60.0  # Default 1h
        if hasattr(df.index, "freq") and df.index.freq is not None:
            # Try to infer from freq
            freq = df.index.freq
            if "H" in str(freq):
                return 60.0
            elif "T" in str(freq) or "min" in str(freq):
                return 1.0
            elif "D" in str(freq):
                return 1440.0
        # Infer from actual timestamps
        deltas = pd.Series(df.index).diff().dropna()
        if not deltas.empty:
            median_delta = deltas.median()
            return median_delta.total_seconds() / 60.0
        return 60.0

    @staticmethod
    def _trade_to_dict(trade: Trade) -> Dict[str, Any]:
        return {
            "entry_idx": trade.entry_idx,
            "exit_idx": trade.exit_idx,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "qty": trade.qty,
            "pnl": trade.pnl,
            "fee_paid": trade.fee_paid,
            "funding_paid": trade.funding_paid,
            "exit_reason": trade.exit_reason,
        }
