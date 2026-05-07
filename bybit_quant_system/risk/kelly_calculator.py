"""Kelly Criterion Calculator - 凯利公式计算器

用于根据历史交易数据和胜率计算最优仓位大小，
实现固定分数凯利公式以控制风险和最大化长期收益增长。
"""

import math
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class KellyResult:
    """凯利计算结果"""
    kelly_fraction: float = 0.0
    fractional_kelly: float = 0.0
    position_size: float = 0.0
    expected_growth_rate: float = 0.0
    win_rate: float = 0.0
    win_loss_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    confidence: str = "low"


class KellyCalculator:
    """凯利公式计算器

    凯利公式: f* = (bp - q) / b
    其中:
        f* = 最优资金比例
        b = 平均盈利/平均亏损 (盈亏比)
        p = 胜率
        q = 败率 = 1 - p
    """

    # 置信度阈值
    HIGH_CONFIDENCE_MIN_TRADES = 100
    MEDIUM_CONFIDENCE_MIN_TRADES = 30

    def __init__(self, default_kelly_fraction: float = 0.25):
        """
        Args:
            default_kelly_fraction: 默认凯利分数（保守系数），
                                  全凯利=1.0, 半凯利=0.5, 四分之一凯利=0.25
        """
        self.default_kelly_fraction = default_kelly_fraction

    def calculate_kelly_fraction(self, win_rate: float, win_loss_ratio: float) -> float:
        """计算完整凯利比例 f* = (bp - q) / b

        Args:
            win_rate: 胜率 (0.0 ~ 1.0)
            win_loss_ratio: 盈亏比 (平均盈利 / 平均亏损)

        Returns:
            凯利比例，范围 [-1.0, 1.0]，负值表示不应交易
        """
        if not (0 <= win_rate <= 1):
            raise ValueError(f"胜率必须在 [0, 1] 范围内，当前: {win_rate}")
        if win_loss_ratio <= 0:
            raise ValueError(f"盈亏比必须为正数，当前: {win_loss_ratio}")

        loss_rate = 1.0 - win_rate

        # 凯利公式: f* = (bp - q) / b
        kelly = (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio

        # 限制在 [-1.0, 1.0] 范围内
        kelly = max(-1.0, min(1.0, kelly))

        return kelly

    def calculate_fractional_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float,
        fraction: Optional[float] = None
    ) -> float:
        """计算分数凯利比例

        使用分数凯利（如 0.25 凯利）可以降低回撤和波动，
        同时保持较好的长期增长。

        Args:
            win_rate: 胜率 (0.0 ~ 1.0)
            win_loss_ratio: 盈亏比
            fraction: 凯利分数，None 使用默认值

        Returns:
            分数凯利仓位比例
        """
        fraction = fraction or self.default_kelly_fraction

        if not (0 < fraction <= 1.0):
            raise ValueError(f"凯利分数必须在 (0, 1] 范围内，当前: {fraction}")

        full_kelly = self.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # 如果全凯利为负，不交易
        if full_kelly <= 0:
            return 0.0

        return full_kelly * fraction

    def calculate_position_size(
        self,
        equity: float,
        win_rate: float,
        win_loss_ratio: float,
        stop_loss_pct: float,
        kelly_fraction: Optional[float] = None
    ) -> float:
        """根据凯利公式计算建议仓位大小

        仓位大小 = 权益 * 凯利分数 / 止损比例
        这样确保即使触发止损，损失也在凯利比例范围内。

        Args:
            equity: 账户权益
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            stop_loss_pct: 止损百分比 (0.0 ~ 1.0)
            kelly_fraction: 凯利分数

        Returns:
            建议仓位大小（以权益为单位）
        """
        if equity <= 0:
            return 0.0
        if stop_loss_pct <= 0:
            raise ValueError(f"止损百分比必须为正，当前: {stop_loss_pct}")

        kelly_fraction = kelly_fraction or self.default_kelly_fraction

        # 计算分数凯利仓位比例
        position_pct = self.calculate_fractional_kelly(
            win_rate, win_loss_ratio, kelly_fraction
        )

        if position_pct <= 0:
            return 0.0

        # 仓位 = 权益 * 凯利比例 / 止损比例
        position_size = equity * position_pct / stop_loss_pct

        return position_size

    def estimate_from_trades(self, trades: List[Dict]) -> Dict:
        """从历史交易列表估算凯利参数

        Args:
            trades: 交易列表，每项包含 'pnl' (盈亏金额)
                    或 'return_pct' (盈亏百分比)

        Returns:
            包含 win_rate, avg_win, avg_loss, win_loss_ratio 的字典
        """
        if not trades or len(trades) < 2:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 1.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
            }

        wins = []
        losses = []
        total_pnl = 0.0

        for trade in trades:
            pnl = trade.get("pnl", trade.get("return_pct", 0.0))
            total_pnl += pnl

            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))

        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)

        if total_trades == 0:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 1.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
            }

        win_rate = winning_trades / total_trades
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        # 盈亏比
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
        }

    def calculate_growth_rate(
        self,
        win_rate: float,
        win_loss_ratio: float,
        fraction: Optional[float] = None
    ) -> float:
        """计算预期增长率 G(f) = p*log(1+bf) + q*log(1-f)

        用于评估不同凯利分数下的预期复合增长速度。

        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            fraction: 仓位比例（凯利分数），None 使用默认值

        Returns:
            预期增长率，正值表示资金增长，负值表示资金衰减
        """
        fraction = fraction or self.default_kelly_fraction
        loss_rate = 1.0 - win_rate

        # 确保参数有效
        if win_rate <= 0 or loss_rate <= 0:
            return 0.0
        if fraction <= 0:
            return 0.0

        # G(f) = p * log(1 + b*f) + q * log(1 - f)
        # 其中 b 是盈亏比
        term1 = win_rate * math.log1p(win_loss_ratio * fraction)
        term2 = loss_rate * math.log1p(-fraction)

        growth_rate = term1 + term2

        return growth_rate

    def full_analysis(
        self,
        equity: float,
        trades: List[Dict],
        stop_loss_pct: float,
        kelly_fraction: Optional[float] = None
    ) -> KellyResult:
        """完整凯利分析

        从历史交易数据计算所有凯利相关指标。

        Args:
            equity: 当前权益
            trades: 历史交易列表
            stop_loss_pct: 止损百分比
            kelly_fraction: 凯利分数

        Returns:
            KellyResult 包含所有计算结果
        """
        kelly_fraction = kelly_fraction or self.default_kelly_fraction

        # 从历史交易估算参数
        est = self.estimate_from_trades(trades)

        # 判断置信度
        total_trades = est["total_trades"]
        if total_trades >= self.HIGH_CONFIDENCE_MIN_TRADES:
            confidence = "high"
        elif total_trades >= self.MEDIUM_CONFIDENCE_MIN_TRADES:
            confidence = "medium"
        else:
            confidence = "low"

        win_rate = est["win_rate"]
        win_loss_ratio = est["win_loss_ratio"]

        # 计算各项凯利指标
        kelly_f = self.calculate_kelly_fraction(win_rate, win_loss_ratio)
        frac_kelly = self.calculate_fractional_kelly(
            win_rate, win_loss_ratio, kelly_fraction
        )
        pos_size = self.calculate_position_size(
            equity, win_rate, win_loss_ratio, stop_loss_pct, kelly_fraction
        )
        growth = self.calculate_growth_rate(win_rate, win_loss_ratio, kelly_fraction)

        return KellyResult(
            kelly_fraction=kelly_f,
            fractional_kelly=frac_kelly,
            position_size=pos_size,
            expected_growth_rate=growth,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            avg_win=est["avg_win"],
            avg_loss=est["avg_loss"],
            total_trades=total_trades,
            confidence=confidence,
        )

    def get_optimal_fraction_grid(
        self,
        win_rate: float,
        win_loss_ratio: float,
        fractions: Optional[List[float]] = None
    ) -> List[Dict]:
        """计算不同凯利分数下的预期增长率网格

        用于可视化选择最优的凯利分数。

        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            fractions: 要测试的分数列表

        Returns:
            每个分数对应的预期增长率列表
        """
        if fractions is None:
            fractions = [0.1, 0.125, 0.25, 0.5, 0.75, 1.0]

        results = []
        for f in fractions:
            growth = self.calculate_growth_rate(win_rate, win_loss_ratio, f)
            results.append({
                "fraction": f,
                "growth_rate": growth,
                "position_pct": self.calculate_fractional_kelly(
                    win_rate, win_loss_ratio, f
                ),
            })

        return results
