"""Momentum Strategy - 动量策略

基于 MACD + EMA + ADX 的动量交易策略。
在趋势确认且动量增强时产生交易信号。
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """动量策略

    使用 MACD 判断动量方向，EMA 判断趋势，ADX 确认趋势强度。
    只在趋势明确且动量增强时入场。

    参数:
        fast: MACD 快线周期
        slow: MACD 慢线周期
        signal: MACD 信号线周期
        adx_period: ADX 计算周期
        adx_threshold: ADX 阈值，高于此值认为趋势明确
        ema_period: EMA 周期
    """

    def __init__(self, config: dict = None):
        default_config = {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "adx_period": 14,
            "adx_threshold": 25.0,
            "ema_period": 20,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def get_required_data(self) -> tuple:
        return (None, "240", 200)

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线 (EMA)

        EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
        alpha = 2 / (period + 1)
        """
        alpha = 2.0 / (period + 1.0)
        ema = series.ewm(alpha=alpha, adjust=False).mean()
        return ema

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均线 (SMA)"""
        return series.rolling(window=period, min_periods=period).mean()

    def _calculate_macd(
        self, close: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple:
        """计算 MACD 指标

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = self._calculate_ema(close, fast)
        ema_slow = self._calculate_ema(close, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算平均趋向指数 (ADX)

        手动实现，不依赖外部库。

        计算步骤:
        1. TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
        2. +DM = high - high_prev (若 > 0 且 > low_prev - low)
        3. -DM = low_prev - low (若 > 0 且 > high - high_prev)
        4. ATR = SMA(TR, period)
        5. +DI = 100 * SMA(+DM, period) / ATR
        6. -DI = 100 * SMA(-DM, period) / ATR
        7. DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        8. ADX = SMA(DX, period)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 计算 True Range
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()

        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # 计算 +DM 和 -DM
        high_diff = high.diff(1)
        low_diff = low.diff(1)

        plus_dm = np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0.0
        )
        minus_dm = np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0.0
        )

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # 平滑计算 ATR, +DM, -DM
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        plus_di_smooth = plus_dm.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()
        minus_di_smooth = minus_dm.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        # 计算 +DI 和 -DI
        plus_di = 100.0 * plus_di_smooth / atr
        minus_di = 100.0 * minus_di_smooth / atr

        # 计算 DX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()

        dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, 0.0)
        dx = pd.Series(dx, index=df.index)

        # ADX = SMA(DX, period)
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return adx

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """生成交易信号

        买入条件:
            1. MACD 柱 > 0 (动量为正)
            2. MACD 柱在增加 (动量增强)
            3. 收盘价 > EMA (趋势向上)
            4. ADX > 阈值 (趋势明确)

        卖出条件:
            1. MACD 柱 < 0 (动量为负)
            2. MACD 柱在减少 (动量减弱)
            3. 收盘价 < EMA (趋势向下)
            4. ADX > 阈值 (趋势明确)

        置信度: 基于 MACD 柱相对强度和 ADX 水平
        """
        if len(df) < max(self.config["slow"], self.config["adx_period"], self.config["ema_period"]) + 20:
            return Signal(action="hold", confidence=0.0, strategy_name=self.name)

        close = df["close"].copy()

        # 1. 计算 EMA
        ema = self._calculate_ema(close, self.config["ema_period"])

        # 2. 计算 MACD
        macd_line, signal_line, macd_hist = self._calculate_macd(
            close,
            self.config["fast"],
            self.config["slow"],
            self.config["signal"]
        )

        # 3. 计算 ADX
        adx = self._calculate_adx(df, self.config["adx_period"])

        # 获取最新值
        latest_close = close.iloc[-1]
        latest_ema = ema.iloc[-1]
        latest_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2]
        latest_adx = adx.iloc[-1]

        action = "hold"
        confidence = 0.0

        # 4. 判断买入信号
        if (latest_macd_hist > 0 and
            latest_macd_hist > prev_macd_hist and
            latest_close > latest_ema and
            latest_adx > self.config["adx_threshold"]):
            action = "buy"

        # 5. 判断卖出信号
        elif (latest_macd_hist < 0 and
              latest_macd_hist < prev_macd_hist and
              latest_close < latest_ema and
              latest_adx > self.config["adx_threshold"]):
            action = "sell"

        # 6. 计算置信度
        if action in ("buy", "sell"):
            macd_hist_abs_max = macd_hist.abs().rolling(window=20, min_periods=1).max().iloc[-1]
            if macd_hist_abs_max > 0:
                macd_strength = abs(latest_macd_hist) / macd_hist_abs_max
            else:
                macd_strength = 0.0
            adx_factor = min(latest_adx / 50.0, 1.0)
            confidence = macd_strength * adx_factor
            confidence = max(0.0, min(1.0, confidence))

        return Signal(
            action=action,
            confidence=confidence,
            strategy_name=self.name,
            metadata={
                "ema": latest_ema,
                "macd_hist": latest_macd_hist,
                "adx": latest_adx,
                "close": latest_close,
            }
        )
