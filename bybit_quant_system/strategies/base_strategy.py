"""策略基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    action: str  # "buy", "sell", "hold"
    confidence: float = 0.0  # 0.0 ~ 1.0
    strategy_name: str = ""
    symbol: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseStrategy(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        pass

    def get_required_data(self) -> tuple:
        return (None, "15", 200)

    def get_parameters(self) -> dict:
        """返回策略参数字典（用于回测引擎）"""
        return {
            "name": self.name,
            "timeframe": self.get_required_data()[1],
            **self.config,
        }

    def __repr__(self):
        return f"{self.name}(config={self.config})"
