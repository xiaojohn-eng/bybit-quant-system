"""Liquidation Guard - 清算保护

保护交易者免于被强制平仓，提供强平价格计算、
保证金健康检查、安全杠杆计算等功能。
"""

import math
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarginHealth:
    """保证金健康状态"""
    is_healthy: bool
    margin_ratio: float
    available_margin: float
    maintenance_margin: float
    position_value: float
    liquidation_price: float
    distance_to_liquidation_pct: float
    safety_status: str  # "safe", "warning", "danger"


class LiquidationGuard:
    """清算保护

    提供以下功能:
    1. 计算强平价格
    2. 计算安全区域
    3. 检查保证金健康
    4. 计算最大安全杠杆
    5. 获取维持保证金率
    """

    # Bybit 维持保证金率阶梯 (USDT永续)
    # 仓位价值越大，维持保证金率越高
    MAINTENANCE_MARGIN_TIERS = [
        {"max_value": 50000, "rate": 0.004, "leverage_cap": 125},
        {"max_value": 250000, "rate": 0.005, "leverage_cap": 100},
        {"max_value": 1000000, "rate": 0.01, "leverage_cap": 50},
        {"max_value": 5000000, "rate": 0.025, "leverage_cap": 20},
        {"max_value": 20000000, "rate": 0.05, "leverage_cap": 10},
        {"max_value": 50000000, "rate": 0.125, "leverage_cap": 4},
        {"max_value": 100000000, "rate": 0.25, "leverage_cap": 2},
        {"max_value": float("inf"), "rate": 0.5, "leverage_cap": 1},
    ]

    # 风险限额系数
    RISK_LIMIT_BUFFER = 0.005  # 0.5% 风险缓冲

    def __init__(self, config: dict = None):
        """
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.risk_buffer = self.config.get("risk_buffer", self.RISK_LIMIT_BUFFER)

    def calculate_liquidation_price(
        self,
        entry: float,
        leverage: int,
        side: str,
        margin_mode: str = "isolated",
        margin_amount: Optional[float] = None,
    ) -> float:
        """计算强平价格

        逐仓模式 (Isolated):
            多仓: liq = entry * (1 - 1/L + 0.005)
            空仓: liq = entry * (1 + 1/L - 0.005)

        全仓模式 (Cross):
            需要 wallet_balance 和 position_size 计算

        Args:
            entry: 入场价格
            leverage: 杠杆倍数
            side: "long" 或 "short"
            margin_mode: "isolated" (逐仓) 或 "cross" (全仓)
            margin_amount: 保证金金额 (逐仓模式使用)

        Returns:
            强平价格
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        if entry <= 0:
            raise ValueError("Entry price must be positive")

        side = side.lower()
        l = float(leverage)

        if margin_mode.lower() == "isolated":
            # 逐仓模式简化公式
            if side in ("long", "buy"):
                # 多仓强平: entry * (1 - 1/L + 0.005)
                liq = entry * (1.0 - 1.0 / l + self.risk_buffer)
            else:
                # 空仓强平: entry * (1 + 1/L - 0.005)
                liq = entry * (1.0 + 1.0 / l - self.risk_buffer)
        else:
            # 全仓模式需要更多参数
            if margin_amount is None:
                # 如果没有提供保证金，假设使用全部保证金
                margin_amount = entry / l

            if side in ("long", "buy"):
                liq = entry * margin_amount / (margin_amount / entry + entry * self.risk_buffer)
            else:
                liq = entry * margin_amount / (margin_amount / entry - entry * self.risk_buffer)

        return liq

    def get_safe_zone(
        self,
        entry: float,
        liquidation: float,
        side: str
    ) -> Dict:
        """计算安全区域

        将入场价到强平价之间的区域划分为:
        - 安全区 (绿色): 距离强平 > 50%
        - 警告区 (黄色): 距离强平 25% - 50%
        - 危险区 (红色): 距离强平 < 25%

        Args:
            entry: 入场价格
            liquidation: 强平价格
            side: "long" 或 "short"

        Returns:
            安全区域信息字典
        """
        side = side.lower()
        distance = abs(entry - liquidation)

        if distance <= 0:
            return {
                "safe_zone_lower": entry,
                "safe_zone_upper": entry,
                "warning_zone_lower": entry,
                "warning_zone_upper": entry,
                "danger_zone_lower": entry,
                "danger_zone_upper": entry,
            }

        if side in ("long", "buy"):
            # 多仓: 强平价在入场价下方
            safe_threshold = entry - distance * 0.5
            warning_threshold = entry - distance * 0.75

            return {
                "safe_zone_lower": safe_threshold,
                "safe_zone_upper": float("inf"),
                "warning_zone_lower": warning_threshold,
                "warning_zone_upper": safe_threshold,
                "danger_zone_lower": liquidation,
                "danger_zone_upper": warning_threshold,
            }
        else:
            # 空仓: 强平价在入场价上方
            safe_threshold = entry + distance * 0.5
            warning_threshold = entry + distance * 0.75

            return {
                "safe_zone_lower": 0.0,
                "safe_zone_upper": safe_threshold,
                "warning_zone_lower": safe_threshold,
                "warning_zone_upper": warning_threshold,
                "danger_zone_lower": warning_threshold,
                "danger_zone_upper": liquidation,
            }

    def check_margin_health(
        self,
        position: Dict,
        current_price: float,
    ) -> Dict:
        """检查保证金健康状态

        Args:
            position: 持仓信息字典，包含:
                - entry_price: 入场价
                - qty: 数量
                - leverage: 杠杆
                - side: "long" 或 "short"
                - margin_amount: 保证金金额
            current_price: 当前价格

        Returns:
            保证金健康状态字典
        """
        entry = position.get("entry_price", 0)
        leverage = position.get("leverage", 1)
        side = position.get("side", "long")
        margin_amount = position.get("margin_amount", 0)
        qty = position.get("qty", 0)

        # 计算仓位价值
        position_value = qty * current_price if current_price > 0 else qty * entry

        # 计算强平价格
        liq_price = self.calculate_liquidation_price(
            entry=entry,
            leverage=leverage,
            side=side,
            margin_amount=margin_amount if margin_amount > 0 else None,
        )

        # 计算到强平价的距离百分比
        distance_to_liq = abs(current_price - liq_price) / current_price if current_price > 0 else 0

        # 计算保证金率
        pnl = 0.0
        if side.lower() in ("long", "buy"):
            pnl = (current_price - entry) * qty
        else:
            pnl = (entry - current_price) * qty

        effective_margin = margin_amount + pnl
        margin_ratio = effective_margin / position_value if position_value > 0 else 0

        # 获取维持保证金率
        mmr = self.get_maintenance_margin_rate(position_value)
        maintenance_margin = position_value * mmr

        # 判断安全状态
        if distance_to_liq > 0.10:  # 距离强平 > 10%
            safety_status = "safe"
        elif distance_to_liq > 0.05:  # 距离强平 5% - 10%
            safety_status = "warning"
        else:  # 距离强平 < 5%
            safety_status = "danger"

        is_healthy = effective_margin > maintenance_margin and safety_status != "danger"

        return {
            "is_healthy": is_healthy,
            "margin_ratio": margin_ratio,
            "effective_margin": effective_margin,
            "available_margin": effective_margin - maintenance_margin,
            "maintenance_margin": maintenance_margin,
            "position_value": position_value,
            "liquidation_price": liq_price,
            "current_price": current_price,
            "distance_to_liquidation_pct": distance_to_liq,
            "safety_status": safety_status,
            "unrealized_pnl": pnl,
        }

    def calculate_max_safe_leverage(
        self,
        entry: float,
        stop_loss: float,
        side: str,
        safety_buffer: float = 0.02
    ) -> int:
        """计算最大安全杠杆

        基于止损距离和缓冲，计算不会触及强平的最大杠杆。

        推导:
        对于多仓: liq = entry * (1 - 1/L + 0.005)
        要求: liq < stop_loss
        entry * (1 - 1/L + 0.005) <= stop_loss * (1 - safety_buffer)
        1 - 1/L + 0.005 <= stop_loss * (1 - safety_buffer) / entry
        1/L >= 1 + 0.005 - stop_loss * (1 - safety_buffer) / entry
        L <= 1 / (1 + 0.005 - stop_loss * (1 - safety_buffer) / entry)

        Args:
            entry: 入场价格
            stop_loss: 止损价格
            side: "long" 或 "short"
            safety_buffer: 安全缓冲 (默认 2%)

        Returns:
            最大安全杠杆 (整数)
        """
        side = side.lower()

        # 止损距离
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 1  # 无止损，最低杠杆

        # 考虑缓冲后的有效止损距离
        if side in ("long", "buy"):
            # 多仓: 止损在下方
            effective_stop = stop_loss * (1.0 - safety_buffer)
            distance_ratio = (entry - effective_stop) / entry
        else:
            # 空仓: 止损在上方
            effective_stop = stop_loss * (1.0 + safety_buffer)
            distance_ratio = (effective_stop - entry) / entry

        if distance_ratio <= 0:
            return 1

        # 计算最大杠杆
        # liq_distance = 1/L - risk_buffer
        # L <= 1 / (distance_ratio + risk_buffer)
        max_leverage = 1.0 / (distance_ratio + self.risk_buffer)

        # 限制最大 125x
        max_leverage = min(max_leverage, 125.0)

        # 取整数并至少为 1
        return max(1, int(max_leverage))

    def get_maintenance_margin_rate(self, position_value: float) -> float:
        """获取维持保证金率

        根据仓位价值查找对应的维持保证金率。

        Args:
            position_value: 仓位名义价值 (USDT)

        Returns:
            维持保证金率
        """
        for tier in self.MAINTENANCE_MARGIN_TIERS:
            if position_value <= tier["max_value"]:
                return tier["rate"]

        # 默认最高档
        return self.MAINTENANCE_MARGIN_TIERS[-1]["rate"]

    def get_max_leverage_for_position_value(self, position_value: float) -> int:
        """根据仓位价值获取最大允许杠杆

        Args:
            position_value: 仓位名义价值 (USDT)

        Returns:
            最大允许杠杆
        """
        for tier in self.MAINTENANCE_MARGIN_TIERS:
            if position_value <= tier["max_value"]:
                return tier["leverage_cap"]

        return self.MAINTENANCE_MARGIN_TIERS[-1]["leverage_cap"]

    def calculate_position_buffer(
        self,
        entry: float,
        liquidation: float,
        current: float,
        side: str
    ) -> Dict:
        """计算当前仓位相对于强平价的缓冲状态

        Args:
            entry: 入场价
            liquidation: 强平价
            current: 当前价
            side: "long" 或 "short"

        Returns:
            缓冲状态字典
        """
        total_range = abs(entry - liquidation)
        current_distance = abs(current - liquidation)

        if total_range <= 0:
            return {"buffer_used_pct": 0, "buffer_remaining_pct": 0}

        buffer_used = (total_range - current_distance) / total_range
        buffer_remaining = current_distance / total_range

        return {
            "buffer_used_pct": max(0, min(1, buffer_used)),
            "buffer_remaining_pct": max(0, min(1, buffer_remaining)),
            "total_range": total_range,
            "current_distance": current_distance,
        }

    def __repr__(self):
        return f"LiquidationGuard(risk_buffer={self.risk_buffer})"
