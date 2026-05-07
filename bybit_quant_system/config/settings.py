"""系统配置管理 - dataclass + 环境变量 + .env文件"""
from dataclasses import dataclass, field
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BybitConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    base_url: str = "https://api-testnet.bybit.com"
    ws_url: str = "wss://stream-testnet.bybit.com/v5/public"
    recv_window: int = 5000
    
    @classmethod
    def from_env(cls):
        testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
        return cls(
            api_key=os.getenv("BYBIT_API_KEY", ""),
            api_secret=os.getenv("BYBIT_API_SECRET", ""),
            testnet=testnet,
            base_url="https://api-testnet.bybit.com" if testnet else "https://api.bybit.com",
            ws_url="wss://stream-testnet.bybit.com/v5/public" if testnet else "wss://stream.bybit.com/v5/public",
        )


@dataclass
class RiskConfig:
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.05
    max_total_leverage: float = 10.0
    position_stop_loss_pct: float = 0.02
    position_take_profit_pct: float = 0.04
    funding_rate_threshold: float = 0.01
    emergency_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 5
    max_positions_per_symbol: int = 2


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: ["ETHUSDT", "BTCUSDT", "XRPUSDT", "SOLUSDT", "LINKUSDT"])
    symbol_weights: Dict[str, float] = field(default_factory=lambda: {
        "ETHUSDT": 0.30, "BTCUSDT": 0.25, "XRPUSDT": 0.15, "SOLUSDT": 0.15, "LINKUSDT": 0.15,
    })
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "MomentumStrategy": 0.30, "MeanReversionStrategy": 0.20,
        "BreakoutStrategy": 0.20, "FundingArbitrage": 0.20, "MLSignalEnhancer": 0.10,
    })
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        "momentum": "240", "mean_reversion": "15", "breakout": "60", "funding": "1",
    })
    max_leverage_per_symbol: Dict[str, int] = field(default_factory=lambda: {
        "ETHUSDT": 30, "BTCUSDT": 30, "XRPUSDT": 30, "SOLUSDT": 30, "LINKUSDT": 20,
    })
    trading_interval_seconds: int = 60


@dataclass
class MLConfig:
    model_path: str = "models/xgboost_enhancer.pkl"
    confidence_threshold: float = 0.6
    retrain_interval_hours: int = 24
    feature_lookback: int = 50


@dataclass
class Config:
    bybit: BybitConfig = field(default_factory=BybitConfig.from_env)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    log_level: str = "INFO"
    
    def validate(self) -> List[str]:
        errors = []
        if not self.bybit.api_key:
            errors.append("BYBIT_API_KEY not set")
        if not self.bybit.api_secret:
            errors.append("BYBIT_API_SECRET not set")
        if not self.bybit.testnet:
            errors.append("TESTNET must be True")
        if abs(sum(self.trading.symbol_weights.values()) - 1.0) > 0.01:
            errors.append("Symbol weights must sum to 1.0")
        if abs(sum(self.trading.strategy_weights.values()) - 1.0) > 0.01:
            errors.append("Strategy weights must sum to 1.0")
        return errors
