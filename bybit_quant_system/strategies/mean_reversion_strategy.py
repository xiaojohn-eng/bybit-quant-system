"""Mean Reversion Strategy - 均值回归策略

基于 RSI 和布林带 (Bollinger Bands) 的均值回归策略。
在价格过度偏离均值时产生反向交易信号。
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_strategy import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略

    当 RSI 进入超买/超卖区域且价格触及布林带上下轨时，
    产生反向交易信号，预期价格回归均值。

    参数:
        rsi_period: RSI 计算周期
        overbought: RSI 超买阈值
        oversold: RSI 超卖阈值
        bb_period: 布林带周期
        bb_std: 布林带标准差倍数
    """

    def __init__(self, config: dict = None):
        default_config = {
            "rsi_period": 14,
            "overbought": 65,   # 从80下调 - 加密货币阈值调整为更实际的水平
            "oversold": 35,     # 从20上调
            "bb_period": 20,
            "bb_std": 2.0,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def get_required_data(self) -> tuple:
        return (None, "15", 100)

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """计算相对强弱指数 (RSI) - 纯 pandas 实现

        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度

        使用 Wilder 的平滑方法 (指数加权移动平均)
        """
        # 计算价格变化
        delta = close.diff(1)

        # 分离上涨和下跌
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # 使用指数加权移动平均 (Wilder's smoothing)
        alpha = 1.0 / period

        avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # 当 avg_loss 为 0 时，RSI = 100
        rsi = rsi.where(avg_loss > 0, 100.0)

        return rsi

    def _calculate_bollinger_bands(
        self, close: pd.Series, period: int, std_mult: float
    ) -> tuple:
        """计算布林带 (Bollinger Bands)

        中轨 = SMA(close, period)
        上轨 = 中轨 + std_mult * STD(close, period)
        下轨 = 中轨 - std_mult * STD(close, period)

        Returns:
            (upper_band, middle_band, lower_band, bandwidth, percent_b)
        """
        middle_band = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()

        upper_band = middle_band + std_mult * std
        lower_band = middle_band - std_mult * std

        # 带宽指标 (Bandwidth) = (上轨 - 下轨) / 中轨
        bandwidth = (upper_band - lower_band) / middle_band

        # %B 指标 = (close - 下轨) / (上轨 - 下轨)
        band_range = upper_band - lower_band
        percent_b = np.where(
            band_range > 0,
            (close - lower_band) / band_range,
            0.5
        )
        percent_b = pd.Series(percent_b, index=close.index)

        return upper_band, middle_band, lower_band, bandwidth, percent_b

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """生成交易信号

        买入条件 (超卖反弹):
            1. RSI < 超卖阈值 (价格过度下跌)
            2. 收盘价 <= 布林带下轨 (价格偏离均值)

        卖出条件 (超买回落):
            1. RSI > 超买阈值 (价格过度上涨)
            2. 收盘价 >= 布林带上轨 (价格偏离均值)

        置信度:
            买入: (超卖阈值 - RSI) / 超卖阈值，RSI 越低置信度越高
            卖出: (RSI - 超买阈值) / (100 - 超买阈值)，RSI 越高置信度越高
        """
        if len(df) < max(self.config["rsi_period"], self.config["bb_period"]) + 5:
            return Signal(action="hold", confidence=0.0, strategy_name=self.name)

        close = df["close"].copy()

        # 1. 计算 RSI
        rsi = self._calculate_rsi(close, self.config["rsi_period"])

        # 2. 计算布林带
        upper_band, middle_band, lower_band, bandwidth, percent_b = \
            self._calculate_bollinger_bands(
                close,
                self.config["bb_period"],
                self.config["bb_std"]
            )

        # 获取最新值
        latest_close = close.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]
        latest_middle = middle_band.iloc[-1]
        latest_bandwidth = bandwidth.iloc[-1]

        action = "hold"
        confidence = 0.0

        # 3. 判断买入信号 (RSI超卖 OR 价格跌破下轨)
        is_oversold = latest_rsi < self.config["oversold"]
        is_below_bb = latest_close <= latest_lower
        if is_oversold or is_below_bb:
            action = "buy"
            # 5. 买入置信度: 综合RSI偏离度和BB偏离度
            rsi_score = max(0, (self.config["oversold"] - latest_rsi) / self.config["oversold"]) if is_oversold else 0
            bb_score = max(0, (latest_lower - latest_close) / max(latest_upper - latest_lower, 1e-10)) if is_below_bb else 0
            confidence = min(max(rsi_score, bb_score, 0.3), 1.0)

        # 4. 判断卖出信号 (RSI超买 OR 价格突破上轨)
        is_overbought = latest_rsi > self.config["overbought"]
        is_above_bb = latest_close >= latest_upper
        if is_overbought or is_above_bb:
            action = "sell"
            # 5. 卖出置信度
            rsi_score = max(0, (latest_rsi - self.config["overbought"]) / (100.0 - self.config["overbought"])) if is_overbought else 0
            bb_score = max(0, (latest_close - latest_upper) / max(latest_upper - latest_lower, 1e-10)) if is_above_bb else 0
            confidence = min(max(rsi_score, bb_score, 0.3), 1.0)

        return Signal(
            action=action,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "rsi": latest_rsi,
                "upper_band": latest_upper,
                "lower_band": latest_lower,
                "middle_band": latest_middle,
                "bandwidth": latest_bandwidth,
                "percent_b": percent_b.iloc[-1],
                "close": latest_close,
            }
        )
