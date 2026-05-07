"""Dynamic Leverage Manager - 动态杠杆管理器

基于 ATR (平均真实波幅) 的动态杠杆调整系统。
根据市场波动率自动选择合适的杠杆倍数，
在高波动时降低杠杆，低波动时提高杠杆。
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeverageTier:
    """杠杆档位"""
    max_atr_pct: float
    leverage: int
    label: str = ""


class DynamicLeverageManager:
    """动态杠杆管理器

    根据市场波动率 (ATR%) 自动调整杠杆倍数。
    波动率越低，杠杆越高；波动率越高，杠杆越低。

    档位配置:
        ATR% <= 1.0%  -> 30x 杠杆
        ATR% <= 2.0%  -> 20x 杠杆
        ATR% <= 3.0%  -> 10x 杠杆
        ATR% <= 5.0%  -> 5x 杠杆
        ATR% > 5.0%   -> 3x 杠杆
    """

    TIERS: List[Dict] = [
        {"max_atr_pct": 1.0, "leverage": 30},
        {"max_atr_pct": 2.0, "leverage": 20},
        {"max_atr_pct": 3.0, "leverage": 10},
        {"max_atr_pct": 5.0, "leverage": 5},
        {"max_atr_pct": float("inf"), "leverage": 3},
    ]

    # 连续亏损次数阈值，超过则暂停交易
    MAX_CONSECUTIVE_LOSSES = 3
    # ATR% 暂停交易阈值
    PAUSE_ATR_PCT_THRESHOLD = 8.0

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典，可覆盖默认档位
        """
        self.config = config or {}
        self.tiers = self.config.get("tiers", self.TIERS)
        self.max_consecutive_losses = self.config.get(
            "max_consecutive_losses", self.MAX_CONSECUTIVE_LOSSES
        )
        self.pause_atr_pct_threshold = self.config.get(
            "pause_atr_pct_threshold", self.PAUSE_ATR_PCT_THRESHOLD
        )

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算 ATR (平均真实波幅)

        TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
        ATR = SMA(TR, period)

        Args:
            df: OHLCV 数据
            period: ATR 周期

        Returns:
            最新 ATR 值
        """
        if len(df) < period + 1:
            logger.warning(f"Data length {len(df)} insufficient for ATR({period})")
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()

        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # ATR = SMA(TR, period)
        atr = tr.rolling(window=period, min_periods=period).mean()

        return atr.iloc[-1]

    def calculate_atr_percentage(self, atr: float, price: float) -> float:
        """计算 ATR 百分比

        ATR% = ATR / price * 100

        Args:
            atr: ATR 值
            price: 当前价格

        Returns:
            ATR 百分比
        """
        if price <= 0:
            return 0.0
        atr_pct = (atr / price) * 100.0
        return atr_pct

    def get_leverage_tier(self, atr_pct: float) -> Dict:
        """根据 ATR% 获取对应杠杆档位

        遍历档位，找到第一个满足 atr_pct <= max_atr_pct 的档位。

        Args:
            atr_pct: ATR 百分比

        Returns:
            档位字典 {"max_atr_pct": float, "leverage": int}
        """
        for tier in self.tiers:
            if atr_pct <= tier["max_atr_pct"]:
                return tier.copy()

        # 默认返回最后一档
        return self.tiers[-1].copy()

    def get_recommended_leverage(self, df: pd.DataFrame, price: float) -> int:
        """获取推荐杠杆倍数

        基于当前 ATR% 自动选择杠杆档位。

        Args:
            df: OHLCV 数据
            price: 当前价格

        Returns:
            推荐杠杆倍数
        """
        atr = self.calculate_atr(df)
        atr_pct = self.calculate_atr_percentage(atr, price)
        tier = self.get_leverage_tier(atr_pct)

        logger.debug(f"ATR={atr:.4f}, ATR%={atr_pct:.2f}%, Leverage={tier['leverage']}x")

        return tier["leverage"]

    def adjust_for_trend(self, base_leverage: int, adx: float) -> int:
        """根据趋势强度调整杠杆

        趋势越强 (ADX 越高)，杠杆可以适度增加；
        趋势越弱，杠杆降低。

        调整规则:
            ADX > 50: 保持基础杠杆
            ADX 25-50: 降低 10%
            ADX < 25: 降低 25%

        Args:
            base_leverage: 基础杠杆
            adx: ADX 值

        Returns:
            调整后的杠杆
        """
        if adx >= 50.0:
            # 强趋势，保持基础杠杆
            adjustment = 1.0
        elif adx >= 25.0:
            # 中等趋势，降低 10%
            adjustment = 0.9
        else:
            # 弱趋势/震荡，降低 25%
            adjustment = 0.75

        adjusted = int(base_leverage * adjustment)

        # 确保至少为 1
        adjusted = max(1, adjusted)

        logger.debug(
            f"ADX={adx:.2f}, BaseLeverage={base_leverage}x, "
            f"Adjusted={adjusted}x"
        )

        return adjusted

    def should_pause_trading(
        self,
        atr_pct: float,
        consecutive_losses: int = 0
    ) -> bool:
        """判断是否应该暂停交易

        暂停条件:
        1. ATR% 超过暂停阈值 (市场波动过大)
        2. 连续亏损次数超过限制

        Args:
            atr_pct: ATR 百分比
            consecutive_losses: 连续亏损次数

        Returns:
            True 如果应该暂停交易
        """
        # 检查波动率
        if atr_pct >= self.pause_atr_pct_threshold:
            logger.warning(
                f"ATR% ({atr_pct:.2f}%) exceeds threshold "
                f"({self.pause_atr_pct_threshold}%), pausing trading"
            )
            return True

        # 检查连续亏损
        if consecutive_losses >= self.max_consecutive_losses:
            logger.warning(
                f"Consecutive losses ({consecutive_losses}) exceed limit "
                f"({self.max_consecutive_losses}), pausing trading"
            )
            return True

        return False

    def get_all_tiers(self) -> List[Dict]:
        """获取所有杠杆档位"""
        return [tier.copy() for tier in self.tiers]

    def calculate_position_notional(
        self,
        equity: float,
        leverage: int,
        risk_pct: float = 0.01
    ) -> float:
        """计算仓位名义价值

        Args:
            equity: 账户权益
            leverage: 杠杆倍数
            risk_pct: 风险百分比

        Returns:
            仓位名义价值
        """
        margin = equity * risk_pct
        notional = margin * leverage
        return notional

    def __repr__(self):
        return (
            f"DynamicLeverageManager("
            f"tiers={len(self.tiers)}, "
            f"pause_threshold={self.pause_atr_pct_threshold}%)"
        )
