"""Breakout Strategy - 突破策略

基于布林带挤压 (Bollinger Squeeze) 后的波动性突破策略。
在低波动期后，等待价格突破布林带并伴随成交量放大时入场。
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_strategy import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """突破策略

    检测布林带挤压（低波动期），然后在价格突破布林带
    并伴随成交量放大时产生交易信号。

    参数:
        atr_period: ATR 计算周期
        squeeze_period: 挤压检测周期
        breakout_mult: 突破乘数，ATR 的倍数
        volume_mult: 成交量倍数阈值
    """

    def __init__(self, config: dict = None):
        default_config = {
            "atr_period": 14,
            "squeeze_period": 20,
            "breakout_mult": 1.5,
            "volume_mult": 1.5,
            "sl_atr_mult": 2.5,     # 新增：论文最优止损ATR倍数
            "tp_rr": 2.5,           # 新增：论文最优风险回报比
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def get_required_data(self) -> tuple:
        return (None, "60", 100)

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算平均真实波幅 (ATR)

        TR = max(high - low, abs(high - close_prev), abs(low - close_prev))
        ATR = SMA(TR, period)
        """
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

        return atr

    def _calculate_bollinger_bands(
        self, close: pd.Series, period: int, std_mult: float
    ) -> tuple:
        """计算布林带

        Returns:
            (upper_band, middle_band, lower_band, bb_width)
        """
        middle_band = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()

        upper_band = middle_band + std_mult * std
        lower_band = middle_band - std_mult * std

        # 带宽 = (上轨 - 下轨) / 中轨
        bb_width = (upper_band - lower_band) / middle_band

        return upper_band, middle_band, lower_band, bb_width

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """生成交易信号

        1. 计算 ATR 和布林带
        2. 检测挤压: 带宽 < 带宽的25%分位数
        3. 买入: 挤压后价格突破上轨 + breakout_mult*ATR 且成交量放大
        4. 卖出: 挤压后价格突破下轨 - breakout_mult*ATR 且成交量放大
        5. 置信度: min(ATR/ATR均值, 2.0) / 2.0
        """
        if len(df) < max(self.config["atr_period"], self.config["squeeze_period"]) + 20:
            return Signal(action="hold", confidence=0.0, strategy_name=self.name)

        close = df["close"].copy()
        high = df["high"].copy()
        low = df["low"].copy()
        volume = df["volume"].copy() if "volume" in df.columns else pd.Series(1.0, index=df.index)

        # 1. 计算 ATR
        atr = self._calculate_atr(df, self.config["atr_period"])

        # 2. 计算布林带
        upper_band, middle_band, lower_band, bb_width = \
            self._calculate_bollinger_bands(
                close, self.config["squeeze_period"], 2.0
            )

        # 3. 检测挤压 (Squeeze)
        # 带宽 < 带宽的 25% 分位数
        bb_width_quantile = bb_width.rolling(
            window=self.config["squeeze_period"], min_periods=self.config["squeeze_period"]
        ).quantile(0.25)

        is_squeeze = bb_width < bb_width_quantile

        # 4. 成交量均值
        volume_sma = volume.rolling(window=20, min_periods=10).mean()

        # 获取最新值
        latest_close = close.iloc[-1]
        latest_high = high.iloc[-1]
        latest_low = low.iloc[-1]
        latest_atr = atr.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        latest_volume = volume.iloc[-1]
        latest_vol_sma = volume_sma.iloc[-1]
        latest_is_squeeze = is_squeeze.iloc[-1]
        prev_is_squeeze = is_squeeze.iloc[-2] if len(is_squeeze) > 1 else False

        # 检测是否刚从挤压状态出来 (squeeze 结束)
        # 当前不是挤压，但前一个是挤压
        squeeze_release = (not latest_is_squeeze) and prev_is_squeeze

        # 或者更简单: 如果前 N 根中有挤压，现在突破了
        squeeze_recent = is_squeeze.tail(5).any()

        action = "hold"
        confidence = 0.0

        # 突破条件: 价格突破布林带 +/- ATR缓冲
        # 买入: 收盘 > 上轨 + 0.3*ATR
        # 卖出: 收盘 < 下轨 - 0.3*ATR
        breakout_level_upper = latest_upper + 0.3 * latest_atr
        breakout_level_lower = latest_lower - 0.3 * latest_atr
        volume_threshold = latest_vol_sma * self.config["volume_mult"]
        volume_ok = latest_vol_sma > 0 and latest_volume > volume_threshold

        if latest_close > breakout_level_upper and volume_ok:
            action = "buy"
        elif latest_close < breakout_level_lower and volume_ok:
            action = "sell"

        # 6. 计算置信度
        if action in ("buy", "sell"):
            atr_sma = atr.rolling(window=20, min_periods=10).mean().iloc[-1]
            if atr_sma > 0:
                confidence = min(latest_atr / atr_sma, 2.0) / 2.0
            else:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

        return Signal(
            action=action,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "atr": latest_atr,
                "upper_band": latest_upper,
                "lower_band": latest_lower,
                "bb_width": bb_width.iloc[-1] if len(bb_width) > 0 else None,
                "is_squeeze": latest_is_squeeze,
                "squeeze_recent": squeeze_recent,
                "squeeze_release": squeeze_release,
                "volume": latest_volume,
                "volume_sma": latest_vol_sma,
                "close": latest_close,
            }
        )
