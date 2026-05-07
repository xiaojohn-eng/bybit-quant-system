#!/usr/bin/env python3
"""
Standalone Backtest Runner
==========================
Supports single-symbol and multi-symbol backtests, parameter optimisation,
walk-forward analysis, and Monte-Carlo simulation.

Usage
-----
    python run_backtest.py --symbols BTCUSDT,ETHUSDT --days 90
    python run_backtest.py --symbols BTCUSDT --optimize --trials 200
    python run_backtest.py --symbols BTCUSDT --walk-forward --folds 5
    python run_backtest.py --symbols BTCUSDT --monte-carlo --mc-runs 1000
    python run_backtest.py --symbols BTCUSDT --output ./results --report-format html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from bybit_quant_system.backtest.engine import BacktestEngine
from bybit_quant_system.backtest.optimizer import StrategyOptimizer
from bybit_quant_system.backtest.multi_symbol_backtest import MultiSymbolBacktest
from bybit_quant_system.strategies.momentum_strategy import MomentumStrategy
from bybit_quant_system.strategies.mean_reversion_strategy import MeanReversionStrategy
from bybit_quant_system.strategies.breakout_strategy import BreakoutStrategy
from bybit_quant_system.strategies.funding_arbitrage import FundingArbitrage
from bybit_quant_system.data.bybit_client import BybitClient
from bybit_quant_system.config.settings import Config

logger = logging.getLogger("run_backtest")

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Type] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "funding_arbitrage": FundingArbitrage,
}


# ===========================================================================
# CLI arguments
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bybit Quant System – Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="Comma-separated symbols (default: BTCUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical data (default: 90)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="60",
        help="K-line interval in minutes (default: 60)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="momentum,mean_reversion,breakout",
        help="Comma-separated strategy names",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="Leverage for backtest (default: 3.0)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Initial balance (default: 10000)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna parameter optimisation",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run Walk-Forward Analysis (WFA)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="WFA fold count (default: 5)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="WFA training ratio (default: 0.7)",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte-Carlo simulation",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=1000,
        help="MC simulation runs (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backtest_results",
        help="Output directory (default: backtest_results)",
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["text", "json", "html"],
        default="json",
        help="Report format (default: json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


# ===========================================================================
# Single backtest execution
# ===========================================================================

def run_single_backtest(
    engine: BacktestEngine,
    strategy_class: Type,
    symbol: str,
    df: Any,
    leverage: float,
) -> Dict[str, Any]:
    """
    Run a single backtest and return the result dictionary.

    Parameters
    ----------
    engine : BacktestEngine
        The backtest engine instance.
    strategy_class : Type
        Strategy class to instantiate.
    symbol : str
        Trading symbol.
    df : pd.DataFrame
        OHLCV data.
    leverage : float
        Leverage multiplier.

    Returns
    -------
    dict
        Backtest metrics: total_return, sharpe, max_drawdown, trades, etc.
    """
    logger.info(
        "Running backtest: strategy=%s symbol=%s rows=%d lev=%.1f",
        strategy_class.__name__, symbol, len(df), leverage,
    )

    result: Dict[str, Any] = engine.run(
        strategy_class=strategy_class,
        symbol=symbol,
        df=df,
        leverage=leverage,
    )

    logger.info(
        "Backtest done: return=%.2f%% sharpe=%.3f maxdd=%.2f%% trades=%d",
        result.get("total_return_pct", 0),
        result.get("sharpe_ratio", 0),
        result.get("max_drawdown_pct", 0),
        result.get("trade_count", 0),
    )
    return result


# ===========================================================================
# Walk-Forward Analysis
# ===========================================================================

def run_walk_forward(
    engine: BacktestEngine,
    strategy_class: Type,
    symbol: str,
    df: Any,
    leverage: float,
    folds: int = 5,
    train_ratio: float = 0.7,
) -> Dict[str, Any]:
    """
    Walk-Forward Analysis: split data into folds,
    optimise on train, validate on test.

    Returns
    -------
    dict
        Aggregate WFA results.
    """
    import numpy as np
    import pandas as pd

    n: int = len(df)
    fold_size: int = n // folds
    wfa_results: List[Dict[str, Any]] = []

    for fold in range(folds):
        test_start: int = fold * fold_size
        test_end: int = test_start + fold_size if fold < folds - 1 else n
        train_end: int = test_start
        train_start: int = max(0, train_end - int(fold_size * train_ratio / (1 - train_ratio)))

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < 50 or len(test_df) < 20:
            logger.warning("Fold %d: insufficient data, skipping", fold)
            continue

        logger.info(
            "WFA fold %d/%d: train=[%d:%d] test=[%d:%d]",
            fold + 1, folds, train_start, train_end, test_start, test_end,
        )

        # Simple grid search on train
        best_sharpe: float = -float("inf")
        best_params: Dict[str, Any] = {}
        for rsi_p in [10, 14, 20]:
            for ema_f in [8, 12]:
                for ema_s in [26, 50]:
                    params: Dict[str, Any] = {
                        "rsi_period": rsi_p,
                        "ema_fast": ema_f,
                        "ema_slow": ema_s,
                    }
                    train_result = engine.run(
                        strategy_class=strategy_class,
                        symbol=symbol,
                        df=train_df,
                        leverage=leverage,
                        strategy_params=params,
                    )
                    sharpe: float = train_result.get("sharpe_ratio", -999)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

        # Validate on test
        test_result = engine.run(
            strategy_class=strategy_class,
            symbol=symbol,
            df=test_df,
            leverage=leverage,
            strategy_params=best_params,
        )
        test_result["fold"] = fold + 1
        test_result["best_params"] = best_params
        wfa_results.append(test_result)

    # Aggregate
    returns: List[float] = [r.get("total_return_pct", 0) for r in wfa_results]
    sharpes: List[float] = [r.get("sharpe_ratio", 0) for r in wfa_results]
    drawdowns: List[float] = [r.get("max_drawdown_pct", 0) for r in wfa_results]

    return {
        "method": "walk_forward",
        "folds": folds,
        "symbol": symbol,
        "strategy": strategy_class.__name__,
        "avg_return_pct": round(float(np.mean(returns)), 2) if returns else 0.0,
        "avg_sharpe": round(float(np.mean(sharpes)), 3) if sharpes else 0.0,
        "avg_max_dd_pct": round(float(np.mean(drawdowns)), 2) if drawdowns else 0.0,
        "fold_results": wfa_results,
    }


# ===========================================================================
# Monte-Carlo Simulation
# ===========================================================================

def run_monte_carlo(
    trades: List[Dict[str, Any]],
    runs: int = 1000,
    initial_balance: float = 10000.0,
) -> Dict[str, Any]:
    """
    Monte-Carlo simulation by reshuffling historical trades.

    Parameters
    ----------
    trades : list
        List of trade dicts with ``pnl`` key.
    runs : int
        Number of simulation runs.
    initial_balance : float

    Returns
    -------
    dict
        MC statistics: mean final equity, VaR, confidence intervals.
    """
    import numpy as np

    pnls: np.ndarray = np.array([t.get("pnl", 0) for t in trades])
    if len(pnls) == 0:
        return {"error": "No trades for MC simulation"}

    final_equities: List[float] = []
    for _ in range(runs):
        np.random.shuffle(pnls)
        equity_curve: np.ndarray = np.cumsum(pnls) + initial_balance
        final_equities.append(equity_curve[-1])

    equities_arr: np.ndarray = np.array(final_equities)
    var_95: float = float(np.percentile(equities_arr, 5))
    var_99: float = float(np.percentile(equities_arr, 1))

    return {
        "method": "monte_carlo",
        "runs": runs,
        "mean_final_equity": round(float(np.mean(equities_arr)), 2),
        "median_final_equity": round(float(np.median(equities_arr)), 2),
        "std_final_equity": round(float(np.std(equities_arr)), 2),
        "worst_case": round(float(np.min(equities_arr)), 2),
        "best_case": round(float(np.max(equities_arr)), 2),
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "prob_profit": round(float(np.mean(equities_arr > initial_balance)), 4),
    }


# ===========================================================================
# Report generation
# ===========================================================================

def generate_report(
    results: Dict[str, Any],
    fmt: str,
    output_dir: str,
) -> str:
    """
    Write the report to disk in the requested format.

    Returns
    -------
    str
        Path to the written report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        path: str = os.path.join(output_dir, f"backtest_report_{timestamp}.json")
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        return path

    elif fmt == "html":
        path = os.path.join(output_dir, f"backtest_report_{timestamp}.html")
        html_content: str = _build_html_report(results)
        with open(path, "w") as fh:
            fh.write(html_content)
        return path

    else:  # text
        path = os.path.join(output_dir, f"backtest_report_{timestamp}.txt")
        with open(path, "w") as fh:
            fh.write(_build_text_report(results))
        return path


def _build_text_report(results: Dict[str, Any]) -> str:
    """Build a plain-text report."""
    lines: List[str] = [
        "=" * 60,
        "BYBIT QUANT SYSTEM – BACKTEST REPORT",
        "=" * 60,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    for section, data in results.items():
        lines.append(f"--- {section.upper()} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    lines.append(f"{k}: ... ({type(v).__name__})")
                else:
                    lines.append(f"{k}: {v}")
        else:
            lines.append(str(data))
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _build_html_report(results: Dict[str, Any]) -> str:
    """Build an HTML report with tables."""
    rows: str = ""
    for section, data in results.items():
        rows += f"<tr><td colspan='2' style='background:#333;color:#fff;padding:8px;'><b>{section.upper()}</b></td></tr>\n"
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    v_str = f"... ({type(v).__name__})"
                else:
                    v_str = str(v)
                rows += f"<tr><td style='padding:6px;border:1px solid #ccc;'>{k}</td><td style='padding:6px;border:1px solid #ccc;'>{v_str}</td></tr>\n"
        else:
            rows += f"<tr><td colspan='2' style='padding:6px;border:1px solid #ccc;'>{str(data)}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Backtest Report</title></head>
<body style="font-family:Arial,sans-serif;margin:20px;">
<h2>Bybit Quant System – Backtest Report</h2>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
<table style="border-collapse:collapse;width:100%;max-width:800px;">
{rows}
</table>
</body></html>"""


# ===========================================================================
# Main async entry
# ===========================================================================

async def main() -> None:
    """Main entry: parse args, fetch data, run backtests, optionally optimise / WFA / MC."""
    args: argparse.Namespace = parse_args()

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== BACKTEST RUNNER START ===")
    logger.info(
        "symbols=%s days=%d leverage=%.1f strategies=%s",
        args.symbols, args.days, args.leverage, args.strategies,
    )

    # Configuration
    config: Config = Config()
    client: BybitClient = BybitClient(
        api_key=getattr(config, "BYBIT_API_KEY", ""),
        api_secret=getattr(config, "BYBIT_API_SECRET", ""),
        testnet=True,
    )

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",")]
    strategy_names: List[str] = [s.strip().lower() for s in args.strategies.split(",")]

    # Resolve strategy classes
    strategy_classes: List[Type] = []
    for name in strategy_names:
        if name in STRATEGIES:
            strategy_classes.append(STRATEGIES[name])
        else:
            logger.warning("Unknown strategy '%s', skipping", name)

    if not strategy_classes:
        logger.error("No valid strategies specified")
        sys.exit(1)

    # Fetch data
    data: Dict[str, Any] = {}
    for sym in symbols:
        try:
            df = await client.get_historical_klines(
                symbol=sym,
                interval=args.timeframe,
                days=args.days,
            )
            data[sym] = df
            logger.info("Fetched %d rows for %s", len(df), sym)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", sym, exc)

    await client.close()

    if not data:
        logger.error("No data fetched – aborting")
        sys.exit(1)

    # ---- Run backtests ---------------------------------------------------
    engine: BacktestEngine = BacktestEngine(initial_balance=args.initial_balance)
    all_results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "symbols": symbols,
            "days": args.days,
            "timeframe": args.timeframe,
            "leverage": args.leverage,
            "strategies": strategy_names,
        },
        "symbol_results": {},
    }

    for sym, df in data.items():
        sym_results: Dict[str, Any] = {}

        for strat_cls in strategy_classes:
            strat_name: str = strat_cls.__name__
            result: Dict[str, Any] = run_single_backtest(
                engine=engine,
                strategy_class=strat_cls,
                symbol=sym,
                df=df,
                leverage=args.leverage,
            )
            sym_results[strat_name] = result

            # ---- Walk-Forward Analysis -----------------------------
            if args.walk_forward:
                logger.info("Running WFA for %s / %s ...", sym, strat_name)
                wfa_result = run_walk_forward(
                    engine=engine,
                    strategy_class=strat_cls,
                    symbol=sym,
                    df=df,
                    leverage=args.leverage,
                    folds=args.folds,
                    train_ratio=args.train_ratio,
                )
                sym_results[f"{strat_name}_wfa"] = wfa_result

            # ---- Monte-Carlo ---------------------------------------
            if args.monte_carlo:
                logger.info("Running MC simulation for %s / %s ...", sym, strat_name)
                trades: List[Dict[str, Any]] = result.get("trades", [])
                if trades:
                    mc_result = run_monte_carlo(
                        trades=trades,
                        runs=args.mc_runs,
                        initial_balance=args.initial_balance,
                    )
                    sym_results[f"{strat_name}_mc"] = mc_result

        all_results["symbol_results"][sym] = sym_results

    # ---- Parameter Optimisation ------------------------------------------
    if args.optimize:
        logger.info("Running parameter optimisation with %d trials ...", args.trials)
        try:
            optimizer: StrategyOptimizer = StrategyOptimizer(
                engine=engine,
                data=data,
                leverage=args.leverage,
            )
            for strat_cls in strategy_classes:
                strat_name = strat_cls.__name__
                logger.info("Optimising %s ...", strat_name)
                best_params, best_value = optimizer.optimize(
                    strategy_class=strat_cls,
                    n_trials=args.trials,
                )
                all_results["symbol_results"][symbols[0]][f"{strat_name}_opt"] = {
                    "best_params": best_params,
                    "best_value": best_value,
                }
        except Exception as exc:
            logger.error("Optimization failed: %s", exc)

    # ---- Save report -----------------------------------------------------
    report_path: str = generate_report(all_results, args.report_format, args.output)
    logger.info("Report saved to: %s", report_path)
    logger.info("=== BACKTEST RUNNER DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
