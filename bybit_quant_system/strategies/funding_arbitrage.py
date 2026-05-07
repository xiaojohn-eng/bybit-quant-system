"""Funding Rate Arbitrage - 资金费率套利策略

利用永续合约资金费率的正负进行方向性交易。
当资金费率极度为正时做空（收取资金费），
当资金费率极度为负时做多（收取资金费）。
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_strategy import BaseStrategy, Signal


class FundingArbitrage(BaseStrategy):
    """资金费率套利策略

    基于资金费率的均值回归特性。资金费率每小时/每8小时结算一次，
    当资金费率偏离正常范围时，产生反向交易信号。

    参数:
        threshold: 资金费率阈值 (默认 0.01 = 1%)
        hold_period: 持仓周期 (默认 8 个周期)
    """

    def __init__(self, config: dict = None):
        default_config = {
            "threshold": 0.01,  # 1% 阈值
            "hold_period": 8,   # 8 个周期
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def get_required_data(self) -> tuple:
        return (None, "1", 100)

    def generate_signal(
        self,
        df: pd.DataFrame,
        funding_rate: float = 0.0
    ) -> Signal:
        """生成交易信号

        买入条件:
            funding_rate < -threshold (资金费率为负，市场过度看空)
            -> 做多，收取资金费

        卖出条件:
            funding_rate > threshold (资金费率为正，市场过度看多)
            -> 做空，收取资金费

        置信度: min(|funding_rate| / threshold, 1.0)
            资金费率偏离越大，置信度越高

        Args:
            df: 价格数据 DataFrame (用于保持接口一致性)
            funding_rate: 当前资金费率

        Returns:
            Signal 交易信号
        """
        threshold = self.config["threshold"]

        action = "hold"
        confidence = 0.0

        # 1. 买入: 资金费率 < -threshold
        if funding_rate < -threshold:
            action = "buy"

        # 2. 卖出: 资金费率 > threshold
        elif funding_rate > threshold:
            action = "sell"

        # 3. 计算置信度
        if action in ("buy", "sell"):
            confidence = min(abs(funding_rate) / threshold, 1.0)
            confidence = max(0.0, min(1.0, confidence))

        return Signal(
            action=action,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "funding_rate": funding_rate,
                "threshold": threshold,
                "hold_period": self.config["hold_period"],
            }
        )
