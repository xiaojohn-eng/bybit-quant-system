"""
Strategy Optimizer Module for Bybit Quantitative Trading System

Provides:
- Hyper-parameter optimisation via Optuna (TPE sampler)
- Walk-forward validation for stability assessment
- Monte-Carlo simulation for robustness analysis
- ``run_full_analysis`` convenience entry-point
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional optuna dependency
try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPTUNA_AVAILABLE = False
    warnings.warn(
        "optuna is not installed. StrategyOptimizer will fall back to "
        "grid-search-like behaviour (limited).",
        ImportWarning,
        stacklevel=2,
    )

# Local imports
from .engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """Result container for strategy optimization."""

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    optimization_history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    walk_forward_scores: List[float] = field(default_factory=list)
    walk_forward_stability: float = 0.0
    monte_carlo_results: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """
    Strategy parameter optimizer with cross-validation and Monte-Carlo analysis.
    """

    def __init__(self, engine: BacktestEngine) -> None:
        """
        Parameters
        ----------
        engine : BacktestEngine
            Backtest engine instance used for every trial.
        """
        self.engine = engine
        self._study: Optional[Any] = None

    # ------------------------------------------------------------------
    # 1. Hyper-parameter optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        strategy_class: Type,
        symbol: str,
        df: pd.DataFrame,
        param_space: Dict[str, Tuple[str, Any]],
        n_trials: int = 200,
        metric: str = "sharpe_ratio",
        direction: str = "maximize",
    ) -> OptimizeResult:
        """
        Optimise strategy parameters using Optuna (TPE sampler).

        Parameters
        ----------
        strategy_class : type
            Strategy class to instantiate per trial.
        symbol : str
            Trading symbol.
        df : pd.DataFrame
            OHLCV data.
        param_space : dict
            Mapping ``param_name -> (suggest_type, *args)``.
            Example::

                {
                    "fast_period": ("int", 5, 50),
                    "slow_period": ("int", 20, 200),
                    "threshold": ("float", 0.0, 1.0),
                    "use_filter": ("categorical", [True, False]),
                }

        n_trials : int
            Number of Optuna trials.
        metric : str
            Which metric from ``BacktestResult.metrics`` to optimise.
        direction : str
            ``"maximize"`` or ``"minimize"``.

        Returns
        -------
        OptimizeResult
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "optuna is not installed. Install it to use optimization."
            )

        if df.empty or len(df) < 200:
            raise ValueError("Insufficient data for optimization.")

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = self._sample_params(trial, param_space)

            try:
                strategy = strategy_class(**params)
            except Exception as exc:
                logger.debug("Strategy instantiation failed: %s", exc)
                return float("-inf") if direction == "maximize" else float("inf")

            try:
                result = self.engine.run(strategy, df, symbol)
                value = result.metrics.get(metric, 0.0)
            except Exception as exc:
                logger.debug("Backtest failed: %s", exc)
                return float("-inf") if direction == "maximize" else float("inf")

            # Guard against NaN / inf
            if not np.isfinite(value):
                return float("-inf") if direction == "maximize" else float("inf")

            return value

        # Suppress optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
        )
        self._study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Build history DataFrame
        history = pd.DataFrame(
            [
                {
                    "trial": t.number,
                    "value": t.value,
                    **t.params,
                }
                for t in self._study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and np.isfinite(t.value)
            ]
        )

        result = OptimizeResult(
            best_params=self._study.best_params.copy(),
            best_value=self._study.best_value,
            optimization_history=history,
        )
        logger.info(
            "Optimization complete. Best %s=%.4f with params=%s",
            metric,
            result.best_value,
            result.best_params,
        )
        return result

    # ------------------------------------------------------------------
    # 2. Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validation(
        self,
        strategy_class: Type,
        symbol: str,
        df: pd.DataFrame,
        best_params: Dict[str, Any],
        n_splits: int = 5,
        train_pct: float = 0.7,
    ) -> List[float]:
        """
        Walk-forward validation: split data into folds, train on first
        ``train_pct`` of each fold and validate on the remainder.

        Parameters
        ----------
        strategy_class : type
            Strategy class.
        symbol : str
            Trading symbol.
        df : pd.DataFrame
            Full OHLCV data.
        best_params : dict
            Best parameters from optimization.
        n_splits : int
            Number of folds.
        train_pct : float
            Fraction of each fold used for training (in-sample).

        Returns
        -------
        list of float
            Sharpe ratio per validation fold.
        """
        fold_size = len(df) // n_splits
        scores: List[float] = []

        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else len(df)
            fold = df.iloc[start:end]

            split_idx = int(len(fold) * train_pct)
            # Purge gap: 论文推荐 1-5 天，防止数据泄漏
            purge_gap = max(1, min(5, split_idx // 10))
            train_df = fold.iloc[:split_idx]
            test_df = fold.iloc[split_idx + purge_gap:]

            if len(train_df) < 50 or len(test_df) < 20:
                continue

            try:
                strategy = strategy_class(**best_params)
                # Train / warm-up on in-sample
                result = self.engine.run(strategy, train_df, symbol)

                # Test on out-of-sample
                result_test = self.engine.run(strategy, test_df, symbol)
                sharpe = result_test.metrics.get("sharpe_ratio", 0.0)
                scores.append(sharpe)
            except Exception as exc:
                logger.debug("Walk-forward fold %d failed: %s", i, exc)
                scores.append(0.0)

        logger.info(
            "Walk-forward validation (%d folds): mean_sharpe=%.4f, std=%.4f",
            len(scores),
            np.mean(scores) if scores else 0.0,
            np.std(scores) if scores else 0.0,
        )
        return scores

    # ------------------------------------------------------------------
    # CSCV PBO (Probability of Backtest Overfitting)
    # ------------------------------------------------------------------

    def calculate_pbo(self, results_matrix: np.ndarray, S: int = 16) -> float:
        """
        CSCV Probabilty of Backtest Overfitting (Lopez de Prado, 2024)

        PBO度量选择偏差的概率。
        PBO < 0.5: 可接受
        PBO < 0.3: 良好
        PBO < 0.1: 优秀
        PBO > 0.5: 高风险（过拟合）

        Args:
            results_matrix: shape (n_strategies, n_periods) 各策略各期IS Sharpe
            S: CSCV分割数，论文推荐S=16

        Returns:
            float: PBO值 [0, 1]
        """
        n_strategies, n_periods = results_matrix.shape
        if n_strategies < 2 or n_periods < S:
            return 1.0  # 数据不足，假设高PBO

        # 将period分为S组
        group_size = n_periods // S
        if group_size < 1:
            return 1.0

        logit_values = []

        # 生成所有C(S, S/2)组合
        half = S // 2

        # 简化版本：随机采样组合
        np.random.seed(42)
        n_combinations = min(100, max(10, S))

        for _ in range(n_combinations):
            # 随机分成两组
            indices = np.random.permutation(S)
            set1_indices = indices[:half]
            set2_indices = indices[half:]

            # 映射回period索引
            periods1 = []
            periods2 = []
            for gi in set1_indices:
                start = gi * group_size
                end = min(start + group_size, n_periods)
                periods1.extend(range(start, end))
            for gi in set2_indices:
                start = gi * group_size
                end = min(start + group_size, n_periods)
                periods2.extend(range(start, end))

            # 计算IS和OOS性能
            is_means = np.mean(results_matrix[:, periods1], axis=1)
            oos_means = np.mean(results_matrix[:, periods2], axis=1)

            # 找到IS最佳策略
            best_is_idx = np.argmax(is_means)

            # 计算logit
            if is_means[best_is_idx] != 0:
                rank = np.sum(oos_means < oos_means[best_is_idx])
                logit = np.log((rank + 0.5) / (n_strategies - rank + 0.5))
                logit_values.append(logit)

        if not logit_values:
            return 1.0

        # PBO = P(logit < 0)
        pbo = np.sum(np.array(logit_values) < 0) / len(logit_values)
        return pbo

    # ------------------------------------------------------------------
    # 3. Monte-Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo_simulation(
        self,
        result: BacktestResult,
        n_simulations: int = 1_000,
    ) -> Dict[str, float]:
        """
        Monte-Carlo robustness analysis by reshuffling trades.

        Parameters
        ----------
        result : BacktestResult
            A completed backtest result.
        n_simulations : int
            Number of random permutations.

        Returns
        -------
        dict
            MC statistics: ruin_probability, median_final_equity, etc.
        """
        trades = result.trades
        if trades.empty:
            return {
                "ruin_probability": 0.0,
                "median_final_equity": result.metrics.get("final_equity", 0.0),
                "worst_final_equity": result.metrics.get("final_equity", 0.0),
                "best_final_equity": result.metrics.get("final_equity", 0.0),
            }

        initial_capital = result.parameters.get(
            "initial_capital", self.engine.initial_capital
        )

        final_equities = []
        for _ in range(n_simulations):
            shuffled = trades.sample(frac=1.0, replace=False).reset_index(drop=True)
            equity = initial_capital
            min_equity = equity
            for _, row in shuffled.iterrows():
                equity += row["pnl"]
                if equity < min_equity:
                    min_equity = equity
                if equity <= initial_capital * 0.1:  # 90% drawdown = ruin
                    break
            final_equities.append(equity)

        arr = np.array(final_equities)
        ruin_threshold = initial_capital * 0.1
        ruin_count = np.sum(arr <= ruin_threshold)

        mc_results = {
            "ruin_probability_pct": round(100.0 * ruin_count / n_simulations, 4),
            "median_final_equity": round(float(np.median(arr)), 4),
            "mean_final_equity": round(float(np.mean(arr)), 4),
            "worst_final_equity": round(float(np.min(arr)), 4),
            "best_final_equity": round(float(np.max(arr)), 4),
            "pct_profitable_sims": round(100.0 * np.mean(arr > initial_capital), 4),
            "mc_sharpe_median": round(
                float(
                    np.median(arr - initial_capital)
                    / (np.std(arr) + 1e-10)
                ),
                4,
            ),
        }
        logger.info(
            "Monte-Carlo (%d sims): ruin_prob=%.2f%%, median_equity=%.2f",
            n_simulations,
            mc_results["ruin_probability_pct"],
            mc_results["median_final_equity"],
        )
        return mc_results

    # ------------------------------------------------------------------
    # 4. Full analysis pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        strategy_class: Type,
        symbol: str,
        df: pd.DataFrame,
        param_space: Dict[str, Tuple[str, Any]],
        n_trials: int = 200,
        n_splits: int = 5,
        n_simulations: int = 1_000,
    ) -> OptimizeResult:
        """
        One-shot pipeline: optimise -> walk-forward -> Monte-Carlo.

        Returns
        -------
        OptimizeResult
            Populated with all three analyses.
        """
        logger.info("=" * 60)
        logger.info("Running full analysis pipeline for %s", strategy_class.__name__)
        logger.info("=" * 60)

        # Step 1: Optimization
        opt_result = self.optimize(
            strategy_class=strategy_class,
            symbol=symbol,
            df=df,
            param_space=param_space,
            n_trials=n_trials,
        )

        # Step 2: Walk-forward validation
        wf_scores = self.walk_forward_validation(
            strategy_class=strategy_class,
            symbol=symbol,
            df=df,
            best_params=opt_result.best_params,
            n_splits=n_splits,
        )
        opt_result.walk_forward_scores = wf_scores
        if wf_scores:
            opt_result.walk_forward_stability = float(
                1.0 - (np.std(wf_scores) / (abs(np.mean(wf_scores)) + 1e-10))
            )

        # Step 3: Monte-Carlo (run on best-params backtest)
        try:
            best_strategy = strategy_class(**opt_result.best_params)
            best_result = self.engine.run(best_strategy, df, symbol)
            mc_results = self.monte_carlo_simulation(
                best_result, n_simulations=n_simulations
            )
            opt_result.monte_carlo_results = mc_results
        except Exception as exc:
            logger.warning("Monte-Carlo simulation failed: %s", exc)

        logger.info("Full analysis complete.")
        return opt_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_params(
        trial: Any, param_space: Dict[str, Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """Sample parameters from the defined space for an Optuna trial."""
        params: Dict[str, Any] = {}
        for name, spec in param_space.items():
            suggest_type = spec[0]
            args = spec[1:]

            if suggest_type == "int":
                params[name] = trial.suggest_int(name, args[0], args[1])
            elif suggest_type == "float":
                params[name] = trial.suggest_float(name, args[0], args[1])
            elif suggest_type == "categorical":
                params[name] = trial.suggest_categorical(name, args[0])
            elif suggest_type == "loguniform":
                params[name] = trial.suggest_float(
                    name, args[0], args[1], log=True
                )
            else:
                raise ValueError(f"Unknown suggest type: {suggest_type}")
        return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("StrategyOptimizer module loaded. Optuna available:", OPTUNA_AVAILABLE)
