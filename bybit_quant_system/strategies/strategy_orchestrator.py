"""Strategy Orchestrator - 策略编排器

管理多个交易品种的多个策略组合，负责信号生成和信号合并。
实现品种到策略的映射，以及多策略信号的加权合并。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from .base_strategy import BaseStrategy, Signal
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .funding_arbitrage import FundingArbitrage

logger = logging.getLogger(__name__)


@dataclass
class CombinedSignal:
    """合并后的信号"""
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 ~ 1.0
    symbol: str = ""
    signals: List[Signal] = field(default_factory=list)
    strategy_votes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class StrategyOrchestrator:
    """策略编排器

    管理不同交易品种的策略组合，并负责信号的生成和合并。

    品种-策略映射:
        ETHUSDT:   [MomentumStrategy, FundingArbitrage]
        BTCUSDT:   [MomentumStrategy, BreakoutStrategy]
        XRPUSDT:   [MeanReversionStrategy]
        SOLUSDT:   [MeanReversionStrategy, BreakoutStrategy]
        LINKUSDT:  [FundingArbitrage, MeanReversionStrategy]
    """

    # 默认品种-策略映射
    DEFAULT_SYMBOL_STRATEGY_MAP: Dict[str, List[type]] = {
        "ETHUSDT": [MomentumStrategy, FundingArbitrage],
        "BTCUSDT": [MomentumStrategy, BreakoutStrategy],
        "XRPUSDT": [MeanReversionStrategy],
        "SOLUSDT": [MeanReversionStrategy, BreakoutStrategy],
        "LINKUSDT": [FundingArbitrage, MeanReversionStrategy],
    }

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典，可包含:
                - symbol_strategy_map: 自定义品种-策略映射
                - strategy_configs: 各策略的独立配置
        """
        self.config = config or {}
        self.symbol_strategy_map: Dict[str, List[BaseStrategy]] = {}
        self._strategies: Dict[str, BaseStrategy] = {}  # 所有注册的策略实例

        # 加载品种-策略映射
        map_config = self.config.get("symbol_strategy_map", {})
        if map_config:
            # 使用自定义映射
            for symbol, strategy_classes in map_config.items():
                self.symbol_strategy_map[symbol] = []
                for strategy_class in strategy_classes:
                    self._register_strategy_for_symbol(symbol, strategy_class)
        else:
            # 使用默认映射
            self.load_default_strategies()

    def _register_strategy_for_symbol(
        self, symbol: str, strategy_class: type, strategy_config: dict = None
    ) -> None:
        """为指定品种注册策略实例"""
        strategy_name = strategy_class.__name__

        # 获取策略配置
        if strategy_config is None:
            strategy_configs = self.config.get("strategy_configs", {})
            strategy_config = strategy_configs.get(strategy_name, {})

        # 创建策略实例
        instance_key = f"{symbol}_{strategy_name}"
        if instance_key not in self._strategies:
            self._strategies[instance_key] = strategy_class(strategy_config)

        if symbol not in self.symbol_strategy_map:
            self.symbol_strategy_map[symbol] = []

        self.symbol_strategy_map[symbol].append(self._strategies[instance_key])
        logger.info(f"Registered {strategy_name} for {symbol}")

    def register_strategy(
        self,
        symbol: str,
        strategy: BaseStrategy
    ) -> None:
        """注册策略到指定品种

        Args:
            symbol: 交易品种，如 "BTCUSDT"
            strategy: 策略实例
        """
        if symbol not in self.symbol_strategy_map:
            self.symbol_strategy_map[symbol] = []

        self.symbol_strategy_map[symbol].append(strategy)

        instance_key = f"{symbol}_{strategy.name}"
        self._strategies[instance_key] = strategy

        logger.info(f"Registered {strategy.name} for {symbol}")

    def load_default_strategies(self) -> None:
        """加载默认的品种-策略映射"""
        for symbol, strategy_classes in self.DEFAULT_SYMBOL_STRATEGY_MAP.items():
            self.symbol_strategy_map[symbol] = []
            for strategy_class in strategy_classes:
                self._register_strategy_for_symbol(symbol, strategy_class)

        logger.info("Loaded default symbol-strategy mappings")

    def generate_signals(
        self,
        symbol: str,
        df_dict: Dict[str, pd.DataFrame],
        funding_rate: float = 0.0
    ) -> List[Signal]:
        """为指定品种生成所有策略的信号

        Args:
            symbol: 交易品种
            df_dict: 各时间周期数据的字典，如 {"15": df_15m, "60": df_60m, "240": df_240m, "1": df_1m}
            funding_rate: 资金费率 (用于 FundingArbitrage 策略)

        Returns:
            各策略生成的 Signal 列表
        """
        signals = []

        if symbol not in self.symbol_strategy_map:
            logger.warning(f"No strategies configured for {symbol}")
            return signals

        strategies = self.symbol_strategy_map[symbol]

        for strategy in strategies:
            try:
                # 获取策略所需的时间周期
                _, timeframe, _ = strategy.get_required_data()

                if timeframe not in df_dict or df_dict[timeframe] is None or df_dict[timeframe].empty:
                    logger.warning(
                        f"Strategy {strategy.name} requires timeframe {timeframe} "
                        f"data for {symbol}, but not available"
                    )
                    continue

                df = df_dict[timeframe]

                # FundingArbitrage 策略需要额外参数
                if isinstance(strategy, FundingArbitrage):
                    signal = strategy.generate_signal(df, funding_rate=funding_rate)
                else:
                    signal = strategy.generate_signal(df)

                signal.symbol = symbol
                signals.append(signal)

                logger.debug(
                    f"{symbol} | {strategy.name}: {signal.action} "
                    f"(confidence={signal.confidence:.4f})"
                )

            except Exception as e:
                logger.error(
                    f"Error generating signal for {symbol} with {strategy.name}: {e}",
                    exc_info=True
                )
                continue

        return signals

    def combine_signals(self, signals: List[Signal]) -> CombinedSignal:
        """合并多个策略的信号为单一信号

        合并规则:
        1. 统计各方向的投票数和总置信度
        2. 如果 buy 的加权置信度 > sell 的加权置信度，结果为 buy
        3. 反之结果为 sell
        4. 如果差距很小 (< 0.1)，结果为 hold (意见分歧)
        5. 最终置信度为获胜方向的平均置信度

        Args:
            signals: Signal 列表

        Returns:
            CombinedSignal 合并后的信号
        """
        if not signals:
            return CombinedSignal(action="hold", confidence=0.0)

        symbol = signals[0].symbol if signals[0].symbol else ""

        # 按方向分类统计
        buy_votes = []
        sell_votes = []
        hold_votes = []
        strategy_votes = {}

        for signal in signals:
            strategy_votes[signal.strategy_name] = signal.action
            if signal.action == "buy":
                buy_votes.append(signal)
            elif signal.action == "sell":
                sell_votes.append(signal)
            else:
                hold_votes.append(signal)

        # 计算加权置信度
        buy_weight = sum(s.confidence for s in buy_votes)
        sell_weight = sum(s.confidence for s in sell_votes)
        hold_weight = sum(s.confidence for s in hold_votes)

        total_weight = buy_weight + sell_weight + hold_weight

        # 判断最终方向
        action = "hold"
        confidence = 0.0

        if buy_weight > sell_weight and buy_weight > 0:
            # 买入方向获胜
            if buy_weight - sell_weight >= 0.1:  # 差距足够大
                action = "buy"
                confidence = buy_weight / max(len(buy_votes), 1)
            else:
                action = "hold"  # 意见分歧

        elif sell_weight > buy_weight and sell_weight > 0:
            # 卖出方向获胜
            if sell_weight - buy_weight >= 0.1:  # 差距足够大
                action = "sell"
                confidence = sell_weight / max(len(sell_votes), 1)
            else:
                action = "hold"  # 意见分歧

        # 限制置信度范围
        confidence = max(0.0, min(1.0, confidence))

        # 收集元数据
        metadata = {
            "buy_count": len(buy_votes),
            "sell_count": len(sell_votes),
            "hold_count": len(hold_votes),
            "buy_weight": buy_weight,
            "sell_weight": sell_weight,
            "hold_weight": hold_weight,
            "total_strategies": len(signals),
        }

        return CombinedSignal(
            action=action,
            confidence=confidence,
            symbol=symbol,
            signals=signals,
            strategy_votes=strategy_votes,
            metadata=metadata,
        )

    def generate_combined_signal(
        self,
        symbol: str,
        df_dict: Dict[str, pd.DataFrame],
        funding_rate: float = 0.0
    ) -> CombinedSignal:
        """一站式生成合并信号

        先为品种生成所有策略信号，然后合并。

        Args:
            symbol: 交易品种
            df_dict: 各时间周期数据
            funding_rate: 资金费率

        Returns:
            CombinedSignal 合并后的信号
        """
        signals = self.generate_signals(symbol, df_dict, funding_rate)
        combined = self.combine_signals(signals)
        combined.symbol = symbol
        return combined

    def get_registered_symbols(self) -> List[str]:
        """获取已注册的所有品种"""
        return list(self.symbol_strategy_map.keys())

    def get_symbol_strategies(self, symbol: str) -> List[str]:
        """获取指定品种注册的所有策略名称"""
        if symbol not in self.symbol_strategy_map:
            return []
        return [s.name for s in self.symbol_strategy_map[symbol]]

    def remove_symbol(self, symbol: str) -> None:
        """移除品种的所有策略配置"""
        if symbol in self.symbol_strategy_map:
            del self.symbol_strategy_map[symbol]
            # 清理策略实例
            keys_to_remove = [
                k for k in self._strategies if k.startswith(f"{symbol}_")
            ]
            for k in keys_to_remove:
                del self._strategies[k]
            logger.info(f"Removed all strategies for {symbol}")

    def __repr__(self):
        symbol_count = len(self.symbol_strategy_map)
        strategy_count = len(self._strategies)
        return (
            f"StrategyOrchestrator("
            f"symbols={symbol_count}, strategies={strategy_count})"
        )
