#!/usr/bin/env python3
"""
Bybit Quantitative Trading System -- Main Entry Point
======================================================
Supports three modes:
    * paper-trading  -- simulated live trading
    * backtest       -- historical backtesting
    * optimization   -- hyper-parameter search via Optuna

Usage
-----
    python main.py --mode paper --symbols BTCUSDT,ETHUSDT
    python main.py --mode backtest --symbols BTCUSDT --days 90
    python main.py --mode optimization --symbols BTCUSDT --trials 200
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from bybit_quant_system.config.settings import Config
from bybit_quant_system.data.bybit_client import BybitClient
from bybit_quant_system.data.cache import DataCache
from bybit_quant_system.strategies.strategy_orchestrator import StrategyOrchestrator
from bybit_quant_system.risk.risk_manager import RiskManager
from bybit_quant_system.execution.trade_executor import TradeExecutor
from bybit_quant_system.execution.paper_trading import PaperTradingEngine
from bybit_quant_system.execution.monitor import TradingMonitor

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
logger = logging.getLogger("main")
_shutdown_event: asyncio.Event = asyncio.Event()


# ===========================================================================
# Logging setup
# ===========================================================================

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure root logging with file + console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # File
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"main_{datetime.now(timezone.utc):%Y%m%d}.log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_arguments() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Bybit Quantitative Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["paper", "backtest", "optimization"],
        default="paper",
        help="Execution mode (default: paper)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT",
        help="Comma-separated list of trading symbols (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="60",
        help="K-line timeframe in minutes (default: 60)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data to fetch (default: 30)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Main loop interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Optuna optimisation trials (default: 100)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="Default leverage (default: 3.0)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Paper-trading initial balance (default: 10000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for result outputs",
    )

    return parser.parse_args()


# ===========================================================================
# Graceful shutdown
# ===========================================================================

def _signal_handler(sig: int, frame: Any) -> None:
    """Handle SIGINT / SIGTERM for graceful exit."""
    logger.warning("Received signal %s -- initiating shutdown...", sig)
    _shutdown_event.set()


# ===========================================================================
# Helper: convert CombinedSignal to executor signal dict
# ===========================================================================

def _combined_signal_to_dict(combined: Any) -> Optional[Dict[str, Any]]:
    """Convert a CombinedSignal to a dict suitable for TradeExecutor."""
    if combined.action not in ("buy", "sell"):
        return None

    side = "Buy" if combined.action == "buy" else "Sell"
    return {
        "symbol": combined.symbol,
        "side": side,
        "confidence": combined.confidence,
        "volatility": combined.metadata.get("volatility", 0.02) if combined.metadata else 0.02,
        "sl_pct": combined.metadata.get("sl_pct", 0.02) if combined.metadata else 0.02,
        "tp_pct": combined.metadata.get("tp_pct", 0.04) if combined.metadata else 0.04,
    }


# ===========================================================================
# Paper Trading Mode
# ===========================================================================

async def run_paper_trading(config: Config, args: argparse.Namespace) -> None:
    """
    Run the paper-trading loop.

    Each cycle:
        1. Fetch latest k-line data.
        2. Run strategies to generate signals.
        3. Risk-check signals.
        4. Execute trades via paper engine.
        5. Manage open positions (SL / TP / trailing).
        6. Log equity and generate reports.
    """
    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",")]
    logger.info("=== PAPER TRADING START === symbols=%s", symbols)

    # ---- initialise components -------------------------------------------
    monitor: TradingMonitor = TradingMonitor(log_dir="logs/paper")
    monitor.send_alert("Paper trading session started", level="info")

    client: BybitClient = BybitClient(
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=config.bybit.testnet,
    )
    await client.connect()

    cache: DataCache = DataCache()
    risk_config: Dict[str, Any] = {
        "max_risk_per_trade": config.risk.max_risk_per_trade,
        "max_daily_drawdown": config.risk.max_daily_drawdown,
        "max_total_leverage": config.risk.max_total_leverage,
        "max_positions_per_symbol": config.risk.max_positions_per_symbol,
        "emergency_drawdown": config.risk.emergency_drawdown_pct,
        "min_risk_reward": config.risk.position_stop_loss_pct * 2,
    }
    risk_mgr: RiskManager = RiskManager(config=risk_config)
    paper: PaperTradingEngine = PaperTradingEngine(
        client=client,
        initial_balance=args.initial_balance,
    )
    executor: TradeExecutor = TradeExecutor(
        client=paper,
        risk_manager=risk_mgr,
        config={"max_positions": 10, "trailing_enabled": True},
    )
    orchestrator: StrategyOrchestrator = StrategyOrchestrator(
        config={"symbol_strategy_map": {sym: [] for sym in symbols}},
    )
    orchestrator.load_default_strategies()

    # Pre-load historical data
    logger.info("Fetching historical data...")
    for sym in symbols:
        try:
            df = await client.get_klines(sym, interval=args.timeframe, limit=500)
            cache.set_klines(sym, args.timeframe, df)
            logger.info("Loaded %d candles for %s", len(df), sym)
        except Exception as exc:
            logger.error("Failed to load %s: %s", sym, exc)
            monitor.log_error(exc, context=f"data_load_{sym}")

    cycle: int = 0
    while not _shutdown_event.is_set():
        cycle += 1
        loop_start: datetime = datetime.now(timezone.utc)
        logger.info("--- Cycle %d ---", cycle)

        try:
            # ---- 1. Fetch latest data --------------------------------
            price_data: Dict[str, Any] = {}
            df_dict_by_symbol: Dict[str, Dict[str, Any]] = {}
            for sym in symbols:
                try:
                    new_df = await client.get_klines(sym, interval=args.timeframe, limit=10)
                    cache.set_klines(sym, args.timeframe, new_df)
                    latest_price: float = float(new_df["close"].iloc[-1])
                    price_data[sym] = {"price": latest_price, "df": new_df}
                    df_dict_by_symbol[sym] = {args.timeframe: new_df}
                except Exception as exc:
                    logger.warning("Data fetch failed for %s: %s", sym, exc)
                    continue

            # ---- 2. Run strategies -----------------------------------
            all_signals: List[Dict[str, Any]] = []
            for sym, data in price_data.items():
                try:
                    signals = orchestrator.generate_signals(
                        symbol=sym,
                        df_dict=df_dict_by_symbol.get(sym, {args.timeframe: data["df"]}),
                    )
                    combined = orchestrator.combine_signals(signals)
                    sig_dict = _combined_signal_to_dict(combined)
                    if sig_dict is not None:
                        sig_dict["symbol"] = sym
                        sig_dict["current_price"] = data["price"]
                        all_signals.append(sig_dict)
                except Exception as exc:
                    logger.warning("Strategy failed for %s: %s", sym, exc)
                    continue

            logger.info("Generated %d actionable signals", len(all_signals))

            # ---- 3. Risk check & 4. Execute --------------------------
            equity: float = paper.get_equity()
            for sig in all_signals:
                sym: str = sig["symbol"]

                # External risk pre-check
                try:
                    position_dicts = [
                        {
                            "symbol": p.symbol,
                            "side": p.side,
                            "qty": p.qty,
                            "entry_price": p.entry_price,
                            "leverage": p.leverage,
                        }
                        for p in executor.active_positions.values()
                    ]
                    allowed, reason = risk_mgr.check_trade_allowed(
                        symbol=sym,
                        side=sig["side"].lower(),
                        qty=sig.get("qty", 0.0),
                        positions=position_dicts,
                        equity=equity,
                    )
                    monitor.log_signal(sym, sig, filtered=not allowed, filter_reason=reason)

                    if not allowed:
                        logger.info("Signal filtered: %s %s -- %s", sym, sig.get("side"), reason)
                        continue
                except Exception as exc:
                    logger.warning("Risk check failed for %s: %s", sym, exc)

                result = await executor.execute_signal(
                    signal=sig,
                    equity=equity,
                    current_price=price_data.get(sym, {}).get("price"),
                )
                if result.success:
                    logger.info(
                        "Executed %s %s qty=%.4f lev=%.1f",
                        result.side, result.symbol, result.qty, result.leverage,
                    )
                else:
                    logger.warning("Execution failed: %s", result.message)

            # ---- 5. Manage positions ---------------------------------
            await executor.manage_positions(
                {sym: data["price"] for sym, data in price_data.items()}
            )

            # Tick paper engine for PnL / SL / TP / liquidation
            tick_data: Dict[str, Any] = {
                sym: {"price": data["price"]} for sym, data in price_data.items()
            }
            events = await paper.tick(tick_data)
            if events:
                for sym_evt, evts in events.items():
                    for evt in evts:
                        monitor.send_alert(f"{sym_evt}: {evt}", level="warning")

            # ---- 6. Logging & equity ---------------------------------
            equity = paper.get_equity()
            monitor.log_equity(equity)

            if cycle % 10 == 0:
                report: Dict[str, Any] = monitor.generate_daily_report()
                logger.info("Daily report: %s", report)
                perf: Dict[str, Any] = paper.get_performance_report()
                logger.info("Performance: %s", perf)

        except Exception as exc:
            logger.exception("Cycle %d error: %s", cycle, exc)
            monitor.log_error(exc, context=f"cycle_{cycle}")

        # ---- Sleep until next interval -----------------------------
        elapsed: float = (datetime.now(timezone.utc) - loop_start).total_seconds()
        sleep_time: float = max(0, args.interval - elapsed)
        logger.info("Cycle %d finished in %.1fs; sleeping %.1fs", cycle, elapsed, sleep_time)
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=sleep_time)
        except asyncio.TimeoutError:
            pass

    # ---- Shutdown ---------------------------------------------------------
    logger.info("=== PAPER TRADING STOP ===")
    monitor.send_alert("Paper trading session stopped", level="info")
    perf = paper.get_performance_report()
    logger.info("Final performance: %s", perf)
    monitor.log_equity(paper.get_equity())

    # Save final report
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    report_path: str = os.path.join(args.output_dir, "paper_trading_report.json")
    with open(report_path, "w") as fh:
        json.dump(perf, fh, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    await client.disconnect()


# ===========================================================================
# Backtest Mode
# ===========================================================================

async def run_backtest_mode(config: Config, args: argparse.Namespace) -> None:
    """
    Historical backtest runner.

    For each symbol:
        1. Fetch historical k-lines.
        2. Run strategy orchestrator bar-by-bar.
        3. Execute signals on paper engine.
        4. Output performance report.
    """
    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",")]
    logger.info("=== BACKTEST MODE === symbols=%s days=%d", symbols, args.days)

    monitor: TradingMonitor = TradingMonitor(log_dir="logs/backtest")
    monitor.send_alert("Backtest started", level="info")

    client: BybitClient = BybitClient(
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=True,
    )
    await client.connect()

    results: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        logger.info("Backtesting %s ...", sym)
        try:
            df = await client.get_klines(
                symbol=sym,
                interval=args.timeframe,
                limit=min(args.days * 24, 1000),
            )
            logger.info("Loaded %d rows for %s", len(df), sym)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", sym, exc)
            continue

        risk_config: Dict[str, Any] = {
            "max_risk_per_trade": config.risk.max_risk_per_trade,
            "max_daily_drawdown": config.risk.max_daily_drawdown,
            "max_total_leverage": config.risk.max_total_leverage,
            "max_positions_per_symbol": config.risk.max_positions_per_symbol,
        }
        paper: PaperTradingEngine = PaperTradingEngine(initial_balance=args.initial_balance)
        risk_mgr: RiskManager = RiskManager(config=risk_config)
        executor: TradeExecutor = TradeExecutor(
            client=paper,
            risk_manager=risk_mgr,
            config={"trailing_enabled": True},
        )
        orchestrator: StrategyOrchestrator = StrategyOrchestrator(
            config={"symbol_strategy_map": {sym: []}},
        )
        orchestrator.load_default_strategies()

        # Bar-by-bar walk-forward
        warmup: int = min(100, len(df) // 4)
        for i in range(warmup, len(df)):
            slice_df = df.iloc[: i + 1]
            current_price: float = float(df["close"].iloc[i])
            timestamp = df.index[i] if hasattr(df, "index") else i

            # Generate combined signal
            df_dict = {args.timeframe: slice_df}
            try:
                signals = orchestrator.generate_signals(symbol=sym, df_dict=df_dict)
                combined = orchestrator.combine_signals(signals)
                sig_dict = _combined_signal_to_dict(combined)

                if sig_dict is not None:
                    sig_dict["symbol"] = sym
                    equity: float = paper.get_equity()
                    await executor.execute_signal(
                        signal=sig_dict,
                        equity=equity,
                        current_price=current_price,
                    )
            except Exception as exc:
                logger.warning("Signal/execution error at bar %d: %s", i, exc)

            # Manage positions
            await executor.manage_positions({sym: current_price})

            # Tick
            funding_rate: float = 0.0001  # placeholder
            await paper.tick({sym: {"price": current_price, "funding_rate": funding_rate}})

            # Periodic equity log
            if i % 100 == 0:
                monitor.log_equity(paper.get_equity(), timestamp=timestamp)

        perf: Dict[str, Any] = paper.get_performance_report()
        results[sym] = perf
        logger.info("%s backtest result: %s", sym, perf)

    await client.disconnect()

    # ---- Save results ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    import json

    summary: Dict[str, Any] = {
        "mode": "backtest",
        "symbols": symbols,
        "days": args.days,
        "timeframe": args.timeframe,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    report_path: str = os.path.join(args.output_dir, "backtest_report.json")
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Backtest report saved to %s", report_path)
    monitor.send_alert("Backtest completed", level="info")


# ===========================================================================
# Optimization Mode
# ===========================================================================

async def run_optimization_mode(config: Config, args: argparse.Namespace) -> None:
    """
    Hyper-parameter optimisation with Optuna.

    Each trial:
        1. Sample strategy parameters.
        2. Run a shortened backtest.
        3. Return Sharpe ratio as objective.
    """
    import optuna
    import nest_asyncio

    # Allow nested event loops for optuna sync objective inside async context
    nest_asyncio.apply()

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",")]
    symbol: str = symbols[0]  # Optimise on first symbol
    logger.info("=== OPTIMIZATION MODE === symbol=%s trials=%d", symbol, args.trials)

    monitor: TradingMonitor = TradingMonitor(log_dir="logs/optimize")
    monitor.send_alert("Optimization started", level="info")

    client: BybitClient = BybitClient(
        api_key=config.bybit.api_key,
        api_secret=config.bybit.api_secret,
        testnet=True,
    )
    await client.connect()

    # Fetch data once
    try:
        df = await client.get_klines(
            symbol=symbol,
            interval=args.timeframe,
            limit=min(args.days * 24, 1000),
        )
    except Exception as exc:
        logger.error("Failed to fetch data for optimisation: %s", exc)
        await client.disconnect()
        return

    logger.info("Loaded %d rows for optimisation", len(df))

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective -- run backtest with sampled params."""
        # Sample parameters
        params: Dict[str, Any] = {
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 85),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
            "ema_fast": trial.suggest_int("ema_fast", 5, 20),
            "ema_slow": trial.suggest_int("ema_slow", 30, 100),
            "atr_multiplier_sl": trial.suggest_float("atr_multiplier_sl", 1.0, 3.0),
            "atr_multiplier_tp": trial.suggest_float("atr_multiplier_tp", 2.0, 5.0),
            "sl_pct": trial.suggest_float("sl_pct", 0.01, 0.05),
            "tp_pct": trial.suggest_float("tp_pct", 0.02, 0.08),
        }

        # Build paper trading components for this trial
        trial_risk_config = {
            "max_risk_per_trade": config.risk.max_risk_per_trade,
            "max_daily_drawdown": config.risk.max_daily_drawdown,
            "max_total_leverage": config.risk.max_total_leverage,
        }
        paper: PaperTradingEngine = PaperTradingEngine(initial_balance=args.initial_balance)
        risk_mgr: RiskManager = RiskManager(config=trial_risk_config)
        executor: TradeExecutor = TradeExecutor(
            client=paper,
            risk_manager=risk_mgr,
            config={"trailing_enabled": False},
        )
        orchestrator: StrategyOrchestrator = StrategyOrchestrator(
            config={
                "symbol_strategy_map": {symbol: []},
                "strategy_configs": {"MomentumStrategy": params},
            },
        )
        orchestrator.load_default_strategies()

        # Shortened backtest (last 30% of data for speed)
        start_idx: int = int(len(df) * 0.7)
        warmup: int = start_idx + min(50, (len(df) - start_idx) // 4)

        async def _run_trial_backtest() -> Dict[str, Any]:
            for i in range(warmup, len(df)):
                slice_df = df.iloc[: i + 1]
                current_price: float = float(df["close"].iloc[i])

                df_dict = {args.timeframe: slice_df}
                signals = orchestrator.generate_signals(
                    symbol=symbol,
                    df_dict=df_dict,
                )
                combined = orchestrator.combine_signals(signals)
                sig_dict = _combined_signal_to_dict(combined)

                if sig_dict is not None:
                    sig_dict["symbol"] = symbol
                    sig_dict["sl_pct"] = params.get("sl_pct", 0.02)
                    sig_dict["tp_pct"] = params.get("tp_pct", 0.04)
                    try:
                        await executor.execute_signal(
                            signal=sig_dict,
                            equity=paper.get_equity(),
                            current_price=current_price,
                        )
                    except Exception:
                        pass

                try:
                    await executor.manage_positions({symbol: current_price})
                    await paper.tick({symbol: {"price": current_price}})
                except Exception:
                    pass

            return paper.get_performance_report()

        # Run the async backtest within the nested event loop
        try:
            loop = asyncio.get_event_loop()
            perf: Dict[str, Any] = loop.run_until_complete(_run_trial_backtest())
        except Exception as exc:
            logger.warning("Trial failed: %s", exc)
            return 0.0

        sharpe: float = perf.get("sharpe_ratio", 0.0)
        return sharpe if not math.isnan(sharpe) else 0.0

    study: optuna.Study = optuna.create_study(
        direction="maximize",
        study_name="bybit_quant_opt",
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    logger.info("Best trial: %.4f", study.best_trial.value)
    logger.info("Best params: %s", study.best_params)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    import json

    opt_result: Dict[str, Any] = {
        "mode": "optimization",
        "symbol": symbol,
        "trials": args.trials,
        "best_value": study.best_trial.value,
        "best_params": study.best_params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.output_dir, "optimization_result.json"), "w") as fh:
        json.dump(opt_result, fh, indent=2)

    monitor.send_alert(
        f"Optimization done. Best Sharpe={study.best_trial.value:.4f}",
        level="info",
    )
    await client.disconnect()


# ===========================================================================
# Main entry
# ===========================================================================

async def main() -> None:
    """Parse arguments, load config, route to the appropriate mode."""
    args: argparse.Namespace = parse_arguments()
    setup_logging(log_level=args.log_level)

    # Graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Starting Bybit Quant System | mode=%s", args.mode)

    # Load configuration
    config: Config = Config()
    if args.config and os.path.isfile(args.config):
        import json
        with open(args.config, "r") as fh:
            overrides: Dict[str, Any] = json.load(fh)
        for k, v in overrides.items():
            setattr(config, k, v)
            logger.debug("Config override: %s = %s", k, v)

    # Route to mode
    if args.mode == "paper":
        await run_paper_trading(config, args)
    elif args.mode == "backtest":
        await run_backtest_mode(config, args)
    elif args.mode == "optimization":
        await run_optimization_mode(config, args)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    logger.info("System shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
