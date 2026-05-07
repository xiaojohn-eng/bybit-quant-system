"""
Trading Monitor Module
======================
Multi-logger monitoring with rotating file handlers,
trade / signal / risk / equity / error / alert streams,
daily report generation and signal-quality statistics.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Colour formatter for console output
# ---------------------------------------------------------------------------
class _ColouredFormatter(logging.Formatter):
    """Simple colourised console formatter."""

    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"

    LEVEL_COLOUR = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOUR.get(record.levelno, self.RESET)
        record.levelname = f"{colour}{record.levelname}{self.RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# TradingMonitor
# ---------------------------------------------------------------------------

class TradingMonitor:
    """
    Centralised monitoring with six independent log channels:
    trade, signal, risk, equity, error, alert.

    Parameters
    ----------
    log_dir : str
        Root directory where log files are written.
    console : bool
        Whether to also emit coloured logs to stdout.
    """

    LOGGERS: List[str] = ["trade", "signal", "risk", "equity", "error", "alert"]
    MAX_BYTES: int = 10 * 1024 * 1024    # 10 MB per file
    BACKUP_COUNT: int = 5

    def __init__(
        self,
        log_dir: str = "logs",
        console: bool = True,
    ):
        self.log_dir: Path = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console: bool = console

        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: Dict[str, List[logging.Handler]] = defaultdict(list)

        # In-memory buffers for fast report generation
        self._trade_buffer: List[Dict[str, Any]] = []
        self._signal_buffer: List[Dict[str, Any]] = []
        self._risk_buffer: List[Dict[str, Any]] = []
        self._equity_buffer: List[Dict[str, Any]] = []
        self._error_buffer: List[Dict[str, Any]] = []

        self._init_loggers()

    def _init_loggers(self) -> None:
        """Create all six loggers with rotating file handlers."""
        for name in self.LOGGERS:
            logger: logging.Logger = logging.getLogger(f"monitor.{name}")
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            # Avoid duplicate handlers on re-init
            if logger.handlers:
                continue

            # Rotating file handler
            log_path: Path = self.log_dir / f"{name}.log"
            file_handler: RotatingFileHandler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=self.MAX_BYTES,
                backupCount=self.BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_fmt: logging.Formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
            self._handlers[name].append(file_handler)

            # Optional console handler
            if self.console:
                console_handler: logging.StreamHandler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG if name in ("error", "alert") else logging.INFO)
                console_handler.setFormatter(_ColouredFormatter(
                    "%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
                    datefmt="%H:%M:%S",
                ))
                logger.addHandler(console_handler)
                self._handlers[name].append(console_handler)

            self._loggers[name] = logger

    # ------------------------------------------------------------------ #
    # Public logging APIs
    # ------------------------------------------------------------------ #

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log a completed trade.

        Parameters
        ----------
        trade : dict
            Must contain at least ``symbol``, ``side``, ``qty``, ``pnl``.
        """
        ts: str = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            **trade,
        }
        self._trade_buffer.append(record)

        msg: str = (
            f"TRADE  {trade.get('symbol','?')} {trade.get('side','?')} "
            f"qty={trade.get('qty',0)} pnl={trade.get('pnl',0):.4f} "
            f"reason={trade.get('reason','close')}"
        )
        self._loggers["trade"].info(msg)

    def log_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        filtered: bool = False,
        filter_reason: Optional[str] = None,
    ) -> None:
        """
        Log a strategy signal.

        Parameters
        ----------
        symbol : str
        signal : dict
            Signal dict with ``side``, ``confidence``, etc.
        filtered : bool
            Whether the signal was filtered out by risk.
        filter_reason : str, optional
            Reason for filtering.
        """
        ts: str = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "symbol": symbol,
            "signal": signal,
            "filtered": filtered,
            "filter_reason": filter_reason,
        }
        self._signal_buffer.append(record)

        status: str = "FILTERED" if filtered else "PASSED"
        msg: str = (
            f"SIGNAL {symbol} {signal.get('side','?')} "
            f"conf={signal.get('confidence',0):.2f} status={status}"
        )
        if filtered and filter_reason:
            msg += f" reason={filter_reason}"
        self._loggers["signal"].info(msg)

    def log_risk_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a risk-management event."""
        ts: str = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "event_type": event_type,
            "details": details,
        }
        self._risk_buffer.append(record)
        self._loggers["risk"].warning("RISK %s: %s", event_type, json.dumps(details))

    def log_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """Log equity snapshot."""
        ts: datetime = timestamp or datetime.now(timezone.utc)
        record: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "equity": equity,
        }
        self._equity_buffer.append(record)
        self._loggers["equity"].info("EQUITY %.4f", equity)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an exception with context."""
        ts: str = datetime.now(timezone.utc).isoformat()
        record: Dict[str, Any] = {
            "timestamp": ts,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
        }
        self._error_buffer.append(record)
        self._loggers["error"].error("ERROR in %s: %s", context, error, exc_info=True)

    def send_alert(self, message: str, level: str = "info") -> None:
        """
        Send / log an alert message.

        Parameters
        ----------
        message : str
        level : str
            One of ``debug``, ``info``, ``warning``, ``error``, ``critical``.
        """
        log_method = getattr(self._loggers["alert"], level.lower(), self._loggers["alert"].info)
        log_method("ALERT [%s] %s", level.upper(), message)

    # ------------------------------------------------------------------ #
    # Report generation
    # ------------------------------------------------------------------ #

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a daily summary report.

        Returns
        -------
        dict
            signal_count, trade_count, gross_pnl, win_rate, max_drawdown,
            risk_events, error_count.
        """
        today: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Filter today's records
        today_signals: List[Dict] = [
            r for r in self._signal_buffer
            if r["timestamp"].startswith(today)
        ]
        today_trades: List[Dict] = [
            r for r in self._trade_buffer
            if r["timestamp"].startswith(today)
        ]
        today_risks: List[Dict] = [
            r for r in self._risk_buffer
            if r["timestamp"].startswith(today)
        ]
        today_errors: List[Dict] = [
            r for r in self._error_buffer
            if r["timestamp"].startswith(today)
        ]

        # Trade stats
        gross_pnl: float = sum(t.get("pnl", 0) for t in today_trades)
        wins: int = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
        win_rate: float = wins / len(today_trades) if today_trades else 0.0

        # Drawdown from today's equity points
        today_equity: List[Dict] = [
            r for r in self._equity_buffer
            if r["timestamp"].startswith(today)
        ]
        max_dd: float = self._compute_drawdown_from_equity(
            [e["equity"] for e in today_equity]
        )

        return {
            "date": today,
            "signal_count": len(today_signals),
            "signal_filtered": sum(1 for s in today_signals if s.get("filtered")),
            "trade_count": len(today_trades),
            "gross_pnl": round(gross_pnl, 4),
            "win_rate": round(win_rate, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "risk_events": len(today_risks),
            "error_count": len(today_errors),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_signal_quality_stats(self) -> Dict[str, Any]:
        """
        Compute signal-to-trade quality metrics.

        Returns
        -------
        dict
            total_signals, filtered_rate, avg_confidence, avg_pnl_by_confidence_band.
        """
        if not self._signal_buffer:
            return {
                "total_signals": 0,
                "filtered_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_pnl_low_conf": 0.0,
                "avg_pnl_mid_conf": 0.0,
                "avg_pnl_high_conf": 0.0,
            }

        total: int = len(self._signal_buffer)
        filtered: int = sum(1 for s in self._signal_buffer if s.get("filtered"))
        avg_conf: float = sum(
            s["signal"].get("confidence", 0) for s in self._signal_buffer
        ) / total

        # Map trades back to signals by symbol+timestamp proximity (simplified)
        pnl_by_band: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for sig in self._signal_buffer:
            conf: float = sig["signal"].get("confidence", 0)
            band: str = "low" if conf < 0.5 else "mid" if conf < 0.75 else "high"
            # Find matching trade
            for tr in self._trade_buffer:
                if (
                    tr.get("symbol") == sig["symbol"]
                    and not sig.get("filtered")
                ):
                    pnl_by_band[band].append(tr.get("pnl", 0))
                    break

        def _avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "total_signals": total,
            "filtered_rate": round(filtered / total, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_pnl_low_conf": round(_avg(pnl_by_band["low"]), 4),
            "avg_pnl_mid_conf": round(_avg(pnl_by_band["mid"]), 4),
            "avg_pnl_high_conf": round(_avg(pnl_by_band["high"]), 4),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_drawdown_from_equity(equities: List[float]) -> float:
        """Return max drawdown from a list of equity values."""
        if not equities:
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
