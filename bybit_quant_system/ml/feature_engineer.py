"""
Feature Engineer Module for Bybit Quantitative Trading System

Provides comprehensive feature engineering for cryptocurrency price data
using only pandas and numpy implementations.
"""

from __future__ import annotations

import logging
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for OHLCV cryptocurrency data.

    Generates price, trend, momentum, volatility, volume, and statistical
    features using pure pandas/numpy implementations. Also supports
    training data preparation with multi-class labels.
    """

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self._label_encoder: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical features from raw OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data with columns: open, high, low, close, volume.

        Returns
        -------
        pd.DataFrame
            DataFrame with all feature columns appended.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Work on a copy to avoid mutating the original
        data = df.copy()

        # Ensure required columns exist (case-insensitive)
        data = self._normalize_columns(data)

        # Generate feature groups
        self._add_price_features(data)
        self._add_lag_features(data)
        self._add_trend_features(data)
        self._add_momentum_features(data)
        self._add_volatility_features(data)
        self._add_volatility_features_enhanced(data)
        self._add_volume_features(data)
        self._add_statistical_features(data)

        # Record generated feature names (exclude original columns)
        self.feature_names = [
            c for c in data.columns
            if c not in {"open", "high", "low", "close", "volume"}
        ]

        logger.info(
            "Generated %d features for %d rows.",
            len(self.feature_names),
            len(data),
        )
        return data

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and multi-class labels y.

        Labels:
            0 = strong_down   (fwd return < -2%)
            1 = down          (-2% <= fwd return < -0.5%)
            2 = neutral       (-0.5% <= fwd return <= 0.5%)
            3 = up            (0.5% < fwd return <= 2%)
            4 = strong_up     (fwd return > 2%)

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame.
        forward_periods : int, default 5
            Number of periods ahead to compute forward returns.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) where X is the feature matrix and y are the labels.
        """
        # Generate all features first
        data = self.generate_all_features(df)

        # Compute forward returns
        future_return = (
            data["close"].shift(-forward_periods) - data["close"]
        ) / data["close"]

        # Create multi-class labels
        labels = pd.Series(np.nan, index=data.index, dtype="Int64")
        labels[future_return < -0.02] = 0                     # strong_down
        labels[(future_return >= -0.02) & (future_return < -0.005)] = 1  # down
        labels[(future_return >= -0.005) & (future_return <= 0.005)] = 2  # neutral
        labels[(future_return > 0.005) & (future_return <= 0.02)] = 3    # up
        labels[future_return > 0.02] = 4                      # strong_up

        # Build feature matrix (drop original OHLCV and rows with NaN in features or label)
        drop_cols = {"open", "high", "low", "close", "volume"}
        X = data.drop(columns=drop_cols, errors="ignore")
        y = labels

        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask].astype(np.float64)
        y = y[valid_mask].astype(int)

        logger.info(
            "Prepared training data: X.shape=%s, y distribution:\n%s",
            X.shape,
            y.value_counts().sort_index(),
        )
        return X, y

    # ------------------------------------------------------------------
    # Feature Groups
    # ------------------------------------------------------------------

    def _add_price_features(self, df: pd.DataFrame) -> None:
        """Add price-based features: returns, log returns, multi-period returns."""
        close = df["close"]

        # Basic returns
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        # Multi-period returns
        df["returns_5"] = close.pct_change(periods=5)
        df["returns_10"] = close.pct_change(periods=10)

        # Price range
        df["price_range"] = (df["high"] - df["low"]) / close

        # Body ratio (candlestick body relative to range)
        body = (close - df["open"]).abs()
        total_range = df["high"] - df["low"] + 1e-10
        df["body_ratio"] = body / total_range

        logger.debug("Added price features.")

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        滞后收益特征 (NeurIPS 2024 - 最重要的预测特征)
        lag_1: 1期前收益
        lag_5: 5期前收益
        lag_10: 10期前收益
        lag_20: 20期前收益
        """
        returns = df['close'].pct_change()
        df['lag_1'] = returns.shift(1)
        df['lag_5'] = returns.shift(5)
        df['lag_10'] = returns.shift(10)
        df['lag_20'] = returns.shift(20)

        # 累积收益特征
        df['cum_return_5'] = (1 + returns).rolling(5).apply(lambda x: x.prod()) - 1
        df['cum_return_10'] = (1 + returns).rolling(10).apply(lambda x: x.prod()) - 1
        df['cum_return_20'] = (1 + returns).rolling(20).apply(lambda x: x.prod()) - 1

        # 收益衰减特征 (近期 > 远期)
        df['return_decay'] = df['lag_1'] * 0.4 + df['lag_5'] * 0.3 + df['lag_10'] * 0.2 + df['lag_20'] * 0.1
        logger.debug("Added lag features.")
        return df

    def _add_volatility_features_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        增强波动率特征 (论文验证的关键预测因子)
        vol_5: 5期滚动波动率
        vol_10: 10期滚动波动率
        vol_20: 20期滚动波动率
        vol_ratio: 短期/长期波动率比
        """
        returns = df['close'].pct_change()
        df['vol_5'] = returns.rolling(5).std() * np.sqrt(365)
        df['vol_10'] = returns.rolling(10).std() * np.sqrt(365)
        df['vol_20'] = returns.rolling(20).std() * np.sqrt(365)
        df['vol_ratio'] = df['vol_5'] / df['vol_20'].replace(0, np.nan)

        # 波动率状态 (论文：低/中/高三档)
        vol_median = df['vol_20'].median()
        df['vol_regime'] = pd.cut(df['vol_20'],
                                   bins=[0, vol_median*0.7, vol_median*1.3, float('inf')],
                                   labels=[0, 1, 2]).astype(float)
        logger.debug("Added enhanced volatility features.")
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> None:
        """Add trend indicators: EMAs, MACD, ADX, trend direction."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        df["ema_10"] = self._ema(close, span=10)
        df["ema_20"] = self._ema(close, span=20)
        df["ema_50"] = self._ema(close, span=50)

        # EMA ratio
        df["ema_ratio"] = df["ema_10"] / df["ema_50"]

        # MACD
        ema_12 = self._ema(close, span=12)
        ema_26 = self._ema(close, span=26)
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = self._ema(df["macd"], span=9)
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ADX (14) - simplified Wilder's smoothing
        df["adx_14"] = self._adx(high, low, close, period=14)

        # Trend direction
        df["trend_direction"] = np.where(close > df["ema_20"], 1, -1)

        logger.debug("Added trend features.")

    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Add momentum oscillators: RSI, Stochastic, CCI, Williams %R."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI (14) and RSI (7)
        df["rsi_14"] = self._rsi(close, period=14)
        df["rsi_7"] = self._rsi(close, period=7)

        # Stochastic %K (smoothed over 3 periods)
        lowest_low_14 = low.rolling(window=14, min_periods=14).min()
        highest_high_14 = high.rolling(window=14, min_periods=14).max()
        raw_k = 100 * (close - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-10)
        df["stochastic_k"] = raw_k.rolling(window=3, min_periods=3).mean()

        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3.0
        tp_sma_20 = typical_price.rolling(window=20, min_periods=20).mean()
        tp_md_20 = typical_price.rolling(window=20, min_periods=20).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        df["cci_20"] = (typical_price - tp_sma_20) / (0.015 * tp_md_20 + 1e-10)

        # Williams %R
        highest_high_10 = high.rolling(window=10, min_periods=10).max()
        lowest_low_10 = low.rolling(window=10, min_periods=10).min()
        df["williams_r"] = -100 * (highest_high_10 - close) / (highest_high_10 - lowest_low_10 + 1e-10)

        logger.debug("Added momentum features.")

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility indicators: ATR, Bollinger Bands, historical volatility."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR (14)
        df["atr_14"] = self._atr(high, low, close, period=14)
        df["atr_ratio"] = df["atr_14"] / close

        # Bollinger Bands
        bb_middle = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std

        df["bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Historical volatility (annualized)
        returns = df["returns"] if "returns" in df.columns else close.pct_change()
        df["hv_20"] = returns.rolling(window=20, min_periods=20).std() * np.sqrt(365)

        logger.debug("Added volatility features.")

    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """Add volume-based features: volume SMA ratio, OBV, MFI."""
        close = df["close"]
        volume = df["volume"]

        # Volume relative to its 10-period SMA
        vol_sma_10 = volume.rolling(window=10, min_periods=10).mean()
        df["volume_sma_10"] = volume / (vol_sma_10 + 1e-10)

        # OBV (On-Balance Volume) - cumulative
        df["obv"] = self._obv(close, volume)

        # MFI (14)
        df["mfi_14"] = self._mfi(close, df["high"], df["low"], volume, period=14)

        logger.debug("Added volume features.")

    def _add_statistical_features(self, df: pd.DataFrame) -> None:
        """Add statistical features: z-score, skewness, rolling percentile."""
        close = df["close"]

        # Z-score relative to 20-period SMA
        sma_20 = close.rolling(window=20, min_periods=20).mean()
        std_20 = close.rolling(window=20, min_periods=20).std()
        df["z_score_20"] = (close - sma_20) / (std_20 + 1e-10)

        # Rolling skewness of returns (20 periods)
        returns = df["returns"] if "returns" in df.columns else close.pct_change()
        df["skew_20"] = returns.rolling(window=20, min_periods=20).skew()

        # Rolling percentile rank of close price (100 periods)
        df["percentile_100"] = close.rolling(window=100, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        logger.debug("Added statistical features.")

    # ------------------------------------------------------------------
    # Technical Indicator Helpers (pure pandas/numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential moving average using pandas ewm."""
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range using Wilder's smoothing."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        return atr

    def _adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average Directional Index (simplified).

        Uses +DM and -DM with Wilder's smoothing.
        """
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index,
        )

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        # Wilder's smoothing
        tr_smooth = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

        # Directional Indicators
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)

        # DX and ADX
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        return adx

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume (cumulative)."""
        direction = np.sign(close.diff())
        direction.iloc[0] = 0  # First element has no diff
        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def _mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Money Flow Index."""
        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume

        tp_diff = typical_price.diff()
        positive_flow = pd.Series(
            np.where(tp_diff > 0, raw_money_flow, 0.0), index=close.index
        )
        negative_flow = pd.Series(
            np.where(tp_diff < 0, raw_money_flow, 0.0), index=close.index
        )

        pos_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        neg_sum = negative_flow.rolling(window=period, min_periods=period).sum()

        money_ratio = pos_sum / (neg_sum + 1e-10)
        mfi = 100.0 - (100.0 / (1.0 + money_ratio))
        return mfi

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to lowercase standard names if needed.
        Handles common variants like 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        col_map = {}
        for c in df.columns:
            lower = c.lower().strip()
            if lower in {"open", "high", "low", "close", "volume"}:
                col_map[c] = lower
        if col_map:
            df = df.rename(columns=col_map)
        return df

    def get_feature_names(self) -> list[str]:
        """Return the list of generated feature names."""
        return self.feature_names.copy()
