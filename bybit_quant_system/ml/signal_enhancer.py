"""
XGBoost Signal Enhancer Module for Bybit Quantitative Trading System

Uses XGBoost classifier to enhance trading signal quality by predicting
directional probabilities from engineered features.
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional xgboost import with graceful fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    warnings.warn(
        "xgboost is not installed. XGBoostSignalEnhancer will operate in "
        "limited mode (training/prediction disabled).",
        ImportWarning,
        stacklevel=2,
    )

from .feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Helper dataclass for signal evaluation output
# ---------------------------------------------------------------------------

@dataclass
class SignalEvaluation:
    """Result of evaluating a single signal with ML enhancement."""

    quality_score: float
    ml_direction: int
    agreement: bool
    confidence: float
    recommendation: str
    probabilities: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class XGBoostSignalEnhancer:
    """
    XGBoost-based signal quality enhancer.

    Trains a multi-class classifier (strong_down / down / neutral / up /
    strong_up) on engineered features and uses the predicted probabilities
    to score the quality of trading signals.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str, optional
            Path to a pickled model to load at init.
        confidence_threshold : float
            Minimum confidence to accept a signal.
        """
        self.confidence_threshold = confidence_threshold
        self.model: Optional[Any] = None
        self.feature_engineer = FeatureEngineer()
        self._feature_names: list[str] = []
        self._trained = False

        if model_path is not None and os.path.exists(model_path):
            self.load_model(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        validation_split: float = 0.2,
        **xgb_params: Any,
    ) -> Dict[str, float]:
        """
        Train the XGBoost classifier on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame.
        forward_periods : int
            Forward-looking window for label generation.
        validation_split : float
            Fraction of data held out for validation (time-series split).
        **xgb_params
            Overrides for XGBClassifier hyper-parameters.

        Returns
        -------
        dict
            Dictionary with accuracy, precision_macro, recall_macro.
        """
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(
                "xgboost is not installed; training is unavailable."
            )

        # 1. Generate features and labels
        X, y = self.feature_engineer.prepare_training_data(df, forward_periods)
        self._feature_names = list(X.columns)

        if len(X) < 100:
            raise ValueError(
                f"Insufficient data after feature generation: {len(X)} rows."
            )

        # 2. Time-series train/test split (no shuffling)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_test) == 0:
            raise ValueError("Validation split yielded empty test set.")

        # 3. Default XGB parameters
        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "num_class": 5,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        default_params.update(xgb_params)

        # 4. Create and fit model
        self.model = xgb.XGBClassifier(**default_params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False,
            )

        self._trained = True

        # 5. Evaluate
        metrics = self._calculate_metrics(X_test, y_test)
        logger.info(
            "Training complete. Accuracy=%.4f, Precision(macro)=%.4f, Recall(macro)=%.4f",
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
        )
        return metrics

    def _calculate_metrics(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Calculate classification metrics on test set."""
        y_pred = self.model.predict(X_test)

        # Accuracy
        accuracy = np.mean(y_pred == y_test)

        # Macro precision and recall
        classes = np.arange(5)
        precisions = []
        recalls = []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_test == c))
            fp = np.sum((y_pred == c) & (y_test != c))
            fn = np.sum((y_pred != c) & (y_test == c))
            precisions.append(tp / (tp + fp + 1e-10))
            recalls.append(tp / (tp + fn + 1e-10))

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)),
        }
        return metrics

    # ------------------------------------------------------------------
    # Prediction & Evaluation
    # ------------------------------------------------------------------

    def predict_direction_probability(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict 5-class direction probabilities for the latest bar.

        Returns
        -------
        np.ndarray
            Probabilities for classes [strong_down, down, neutral, up, strong_up].
        """
        self._ensure_trained()
        features = self._prepare_features(df)
        latest = features.iloc[[-1]]
        probs = self.model.predict_proba(latest)[0]
        return probs

    def evaluate_signal(
        self,
        df: pd.DataFrame,
        current_signal: Dict[str, Any],
    ) -> SignalEvaluation:
        """
        Evaluate a trading signal using ML predictions.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data up to current bar.
        current_signal : dict
            Signal dict with at least 'direction' key (1=long, -1=short, 0=neutral).

        Returns
        -------
        SignalEvaluation
        """
        probs = self.predict_direction_probability(df)
        ml_direction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Map current signal direction to class
        signal_direction = current_signal.get("direction", 0)
        signal_class = self._direction_to_class(signal_direction)

        # Agreement between ML and signal
        agreement = ml_direction == signal_class

        # Quality score: high confidence + agreement = high score
        if agreement:
            quality_score = confidence
        else:
            # Penalize disagreement
            opposite_boost = probs[signal_class] if 0 <= signal_class < 5 else 0.0
            quality_score = max(0.0, confidence - 0.3 + opposite_boost * 0.3)

        # Recommendation
        if quality_score >= self.confidence_threshold and agreement:
            recommendation = "strong_confirm"
        elif quality_score >= self.confidence_threshold and not agreement:
            recommendation = "contrarian_opportunity"
        elif agreement:
            recommendation = "weak_confirm"
        else:
            recommendation = "avoid"

        return SignalEvaluation(
            quality_score=round(quality_score, 4),
            ml_direction=ml_direction,
            agreement=agreement,
            confidence=round(confidence, 4),
            recommendation=recommendation,
            probabilities=probs,
        )

    def filter_signals(
        self,
        signals: List[Dict[str, Any]],
        df_dict: Dict[str, pd.DataFrame],
        threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of signals, keeping only those with quality_score > threshold.

        Parameters
        ----------
        signals : list of dict
            Each dict has 'symbol' and signal keys.
        df_dict : dict
            Mapping of symbol -> OHLCV DataFrame.
        threshold : float
            Minimum quality score to retain.

        Returns
        -------
        list
            Filtered signals with added 'ml_quality' and 'ml_recommendation'.
        """
        filtered = []
        for sig in signals:
            symbol = sig.get("symbol")
            df = df_dict.get(symbol)
            if df is None or df.empty:
                continue

            try:
                eval_result = self.evaluate_signal(df, sig)
            except Exception as exc:  # pragma: no cover
                logger.warning("Signal evaluation failed for %s: %s", symbol, exc)
                continue

            sig = sig.copy()
            sig["ml_quality"] = eval_result.quality_score
            sig["ml_recommendation"] = eval_result.recommendation
            sig["ml_confidence"] = eval_result.confidence
            sig["ml_direction"] = eval_result.ml_direction

            if eval_result.quality_score >= threshold:
                filtered.append(sig)

        logger.info(
            "Filtered %d signals -> %d signals above threshold %.2f",
            len(signals),
            len(filtered),
            threshold,
        )
        return filtered

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save model and metadata using pickle."""
        self._ensure_trained()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self._feature_names,
            "confidence_threshold": self.confidence_threshold,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load model and metadata from pickle."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.model = payload["model"]
        self._feature_names = payload.get("feature_names", [])
        self.confidence_threshold = payload.get(
            "confidence_threshold", self.confidence_threshold
        )
        self._trained = True
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.Series:
        """
        Return feature importance as a pandas Series.

        Returns
        -------
        pd.Series
            Feature name -> importance value, sorted descending.
        """
        self._ensure_trained()
        importance = self.model.feature_importances_
        names = (
            self._feature_names
            if self._feature_names
            else [f"f{i}" for i in range(len(importance))]
        )
        series = pd.Series(importance, index=names).sort_values(ascending=False)
        return series

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate and align features to training columns."""
        data = self.feature_engineer.generate_all_features(df)
        drop_cols = {"open", "high", "low", "close", "volume"}
        X = data.drop(columns=drop_cols, errors="ignore")
        X = X.astype(np.float64).dropna()

        # Reindex to match training features if available
        if self._feature_names:
            for col in self._feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self._feature_names]
        return X

    def _ensure_trained(self) -> None:
        """Raise if model is not trained or loaded."""
        if self.model is None or not self._trained:
            raise RuntimeError(
                "Model has not been trained or loaded. Call train() or load_model() first."
            )

    @staticmethod
    def _direction_to_class(direction: int) -> int:
        """
        Map signal direction to class index.

        direction:  1 -> up (3)
                   -1 -> down (1)
                    0 -> neutral (2)
        """
        mapping = {1: 3, -1: 1, 0: 2}
        return mapping.get(direction, 2)


if __name__ == "__main__":
    # Minimal sanity-check
    logging.basicConfig(level=logging.INFO)
    print("XGBoostSignalEnhancer module loaded. XGBoost available:", XGBOOST_AVAILABLE)
