"""
Multi-Symbol Portfolio Backtest Module for Bybit Quantitative Trading System

Provides portfolio-level backtesting across multiple trading pairs,
correlation analysis, and weight optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """Container for multi-symbol portfolio backtest results."""

    individual_results: Dict[str, BacktestResult] = field(default_factory=dict)
    portfolio_equity: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    portfolio_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    weights: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Multi-Symbol Backtest
# ---------------------------------------------------------------------------

class MultiSymbolBacktest:
    """
    Run portfolio-level backtests across multiple trading symbols.

    Combines individual backtest equity curves using configurable weights,
    computes cross-symbol correlation, and calculates portfolio metrics.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        engine : BacktestEngine
            Shared backtest engine instance.
        config : dict, optional
            Global configuration overrides.
        """
        self.engine = engine
        self.config = config or {}

    # ------------------------------------------------------------------
    # Portfolio backtest
    # ------------------------------------------------------------------

    def run_portfolio_backtest(
        self,
        strategies: Dict[str, Any],
        data_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioResult:
        """
        Run backtests for multiple symbols and combine into portfolio.

        Parameters
        ----------
        strategies : dict
            Mapping ``symbol -> strategy_instance`` (or ``symbol ->
            (strategy_class, kwargs)`` if a tuple is provided).
        data_dict : dict
            Mapping ``symbol -> OHLCV DataFrame``.
        weights : dict, optional
            Allocation weight per symbol. If None, equal weights are used.
            Weights are normalised to sum to 1.0.

        Returns
        -------
        PortfolioResult
        """
        if not strategies or not data_dict:
            raise ValueError("strategies and data_dict must be non-empty.")

        symbols = list(strategies.keys())

        # Normalise weights
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}
        else:
            total = sum(weights.get(s, 0.0) for s in symbols)
            if total <= 0:
                total = 1.0
            weights = {s: weights.get(s, 0.0) / total for s in symbols}

        # Run individual backtests
        individual: Dict[str, BacktestResult] = {}
        equity_series: Dict[str, pd.Series] = {}

        for symbol in symbols:
            strategy_spec = strategies[symbol]
            df = data_dict.get(symbol)
            if df is None or df.empty:
                logger.warning("No data for %s, skipping.", symbol)
                continue

            # Resolve strategy instance
            if isinstance(strategy_spec, tuple):
                strategy_cls, kwargs = strategy_spec
                strategy = strategy_cls(**kwargs)
            else:
                strategy = strategy_spec

            logger.info("Running backtest for %s ...", symbol)
            result = self.engine.run(strategy, df, symbol)
            individual[symbol] = result

            # Extract equity series (normalised to 1.0 at start)
            eq = result.equity_curve["equity"].copy()
            if not eq.empty and eq.iloc[0] != 0:
                eq = eq / eq.iloc[0]
            equity_series[symbol] = eq

        if not individual:
            return PortfolioResult(weights=weights)

        # Align equity series to a common index (union of all timestamps)
        combined_index = self._union_index(equity_series)
        aligned: Dict[str, pd.Series] = {}
        for sym, eq in equity_series.items():
            aligned[sym] = eq.reindex(combined_index).ffill().bfill()

        # Build portfolio equity: weighted sum of normalised curves
        portfolio_eq = pd.Series(0.0, index=combined_index)
        for sym in aligned:
            portfolio_eq += aligned[sym] * weights.get(sym, 0.0)

        portfolio_equity_df = pd.DataFrame(
            {"portfolio_equity": portfolio_eq},
            index=combined_index,
        )

        # Correlation matrix from daily returns
        returns_dict = self._compute_returns(aligned)
        corr_matrix = self.calculate_correlation_matrix(returns_dict)

        # Portfolio metrics
        port_metrics = self._calculate_portfolio_metrics(
            portfolio_equity_df, individual, weights
        )

        result = PortfolioResult(
            individual_results=individual,
            portfolio_equity=portfolio_equity_df,
            portfolio_metrics=port_metrics,
            correlation_matrix=corr_matrix,
            weights=weights,
        )
        logger.info(
            "Portfolio backtest complete. Symbols=%d, final_equity=%.4f",
            len(individual),
            portfolio_equity_df["portfolio_equity"].iloc[-1]
            if not portfolio_equity_df.empty
            else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_correlation_matrix(
        returns_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate pairwise return correlation matrix.

        Parameters
        ----------
        returns_dict : dict
            symbol -> returns Series.

        Returns
        -------
        pd.DataFrame
            Symmetric correlation matrix.
        """
        if not returns_dict:
            return pd.DataFrame()

        df_rets = pd.DataFrame(returns_dict)
        # Drop all-NaN rows/cols
        df_rets = df_rets.dropna(how="all", axis=0).dropna(how="all", axis=1)
        corr = df_rets.corr(method="pearson")
        return corr.fillna(0.0)

    # ------------------------------------------------------------------
    # Weight optimization
    # ------------------------------------------------------------------

    @staticmethod
    def optimize_weights(
        results: Dict[str, BacktestResult],
        method: str = "equal",
        risk_aversion: float = 1.0,
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation weights.

        Parameters
        ----------
        results : dict
            symbol -> BacktestResult.
        method : str
            One of: ``"equal"``, ``"sharpe"``, ``"inverse_vol"``,
            ``"kelly"``.
        risk_aversion : float
            Risk aversion parameter (used with certain methods).

        Returns
        -------
        dict
            symbol -> weight mapping (weights sum to 1.0).
        """
        symbols = list(results.keys())
        if not symbols:
            return {}

        if method == "equal":
            w = 1.0 / len(symbols)
            return {s: w for s in symbols}

        elif method == "sharpe":
            # Weight proportional to Sharpe ratio (clip at 0)
            sharpe_vals = {
                s: max(0.0, r.metrics.get("sharpe_ratio", 0.0))
                for s, r in results.items()
            }
            total = sum(sharpe_vals.values())
            if total <= 0:
                return {s: 1.0 / len(symbols) for s in symbols}
            return {s: v / total for s, v in sharpe_vals.items()}

        elif method == "inverse_vol":
            # Weight proportional to inverse volatility
            vols = {}
            for s, r in results.items():
                eq = r.equity_curve["equity"]
                if len(eq) > 1:
                    rets = eq.pct_change().dropna()
                    vols[s] = 1.0 / (rets.std() + 1e-10)
                else:
                    vols[s] = 1.0
            total = sum(vols.values())
            return {s: vols[s] / total for s in symbols}

        elif method == "kelly":
            # Simplified Kelly: weight by kelly_fraction (clamped positive)
            kelly_vals = {
                s: max(0.0, r.metrics.get("kelly_fraction", 0.0))
                for s, r in results.items()
            }
            total = sum(kelly_vals.values())
            if total <= 0:
                return {s: 1.0 / len(symbols) for s in symbols}
            # Normalise and apply risk aversion
            raw = {s: v / total for s, v in kelly_vals.items()}
            # Risk aversion: shrink toward equal weight
            eq_w = 1.0 / len(symbols)
            final = {
                s: raw[s] / risk_aversion + eq_w * (1 - 1 / risk_aversion)
                for s in symbols
            }
            # Renormalise
            t = sum(final.values())
            return {s: v / t for s, v in final.items()}

        else:
            raise ValueError(f"Unknown weight optimization method: {method}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _union_index(series_dict: Dict[str, pd.Series]) -> pd.Index:
        """Build union index from multiple series indices."""
        idx: Optional[pd.Index] = None
        for s in series_dict.values():
            if idx is None:
                idx = s.index
            else:
                idx = idx.union(s.index)
        return idx if idx is not None else pd.Index([])

    @staticmethod
    def _compute_returns(
        aligned_equity: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Compute returns from aligned equity series."""
        returns = {}
        for sym, eq in aligned_equity.items():
            returns[sym] = eq.pct_change().dropna()
        return returns

    def _calculate_portfolio_metrics(
        self,
        portfolio_equity: pd.DataFrame,
        individual: Dict[str, BacktestResult],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate aggregate portfolio-level metrics."""
        eq = portfolio_equity["portfolio_equity"].values
        if len(eq) < 2:
            return {}

        initial = eq[0]
        final = eq[-1]
        total_return = (final - initial) / initial if initial != 0 else 0.0

        rets = np.diff(eq) / eq[:-1]
        n = len(rets)

        # Annualized (crypto: ~ 365 * 24 hourly bars)
        periods_per_year = 365 * 24
        ann_return = (1 + total_return) ** (periods_per_year / n) - 1 if n > 0 else 0.0

        ret_std = np.std(rets, ddof=1)
        sharpe = (
            (np.mean(rets) / ret_std) * np.sqrt(periods_per_year)
            if ret_std > 1e-12
            else 0.0
        )

        # Downside deviation
        downside = rets[rets < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 0 else 1e-10
        sortino = (
            (np.mean(rets) / downside_std) * np.sqrt(periods_per_year)
            if downside_std > 1e-12
            else 0.0
        )

        # Max drawdown
        cummax = np.maximum.accumulate(eq)
        drawdowns = (cummax - eq) / cummax
        max_dd = float(np.max(drawdowns))

        # Max drawdown duration
        peak_idx = 0
        max_dd_duration = 0
        for i in range(len(eq)):
            if eq[i] >= cummax[i]:
                peak_idx = i
            else:
                dd_dur = i - peak_idx
                if dd_dur > max_dd_duration:
                    max_dd_duration = dd_dur

        # Aggregate trade stats
        total_trades = sum(
            r.metrics.get("total_trades", 0) for r in individual.values()
        )

        # Weighted average win rate
        total_wr = sum(
            r.metrics.get("win_rate_pct", 0) * weights.get(s, 0.0)
            for s, r in individual.items()
        )

        # Portfolio volatility (std of portfolio returns annualized)
        port_vol = ret_std * np.sqrt(periods_per_year)

        # Calmar
        calmar = ann_return / max_dd if max_dd > 1e-10 else 0.0

        metrics = {
            "total_return_pct": round(total_return * 100, 4),
            "annualized_return_pct": round(ann_return * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "max_drawdown_duration_bars": int(max_dd_duration),
            "portfolio_volatility_annual": round(port_vol * 100, 4),
            "weighted_win_rate_pct": round(total_wr, 4),
            "total_trades": total_trades,
            "calmar_ratio": round(calmar, 4),
            "num_assets": len(individual),
            "final_equity": round(final, 4),
        }
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("MultiSymbolBacktest module loaded successfully.")
