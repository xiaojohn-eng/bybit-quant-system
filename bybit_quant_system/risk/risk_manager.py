"""Risk Manager - 风险管理器

核心风控模块，负责交易前的风险评估、仓位控制、
止损止盈计算、日度风控和紧急清仓判断。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    """风控检查结果"""
    passed: bool
    check_name: str
    message: str = ""
    severity: str = "info"  # "info", "warning", "critical"


@dataclass
class RiskConfig:
    """风控配置"""
    max_risk_per_trade: float = 0.01  # 单笔交易最大风险 1%
    max_daily_drawdown: float = 0.05  # 日最大回撤 5%
    max_total_leverage: float = 10.0  # 最大总杠杆 10x
    max_positions_per_symbol: int = 2  # 每个品种最大持仓数
    max_correlated_exposure: float = 0.03  # 相关品种最大敞口 3%
    emergency_drawdown: float = 0.10  # 紧急清仓回撤 10%
    min_risk_reward: float = 1.5  # 最低风险回报比
    max_portfolio_heat: float = 0.20  # 最大组合热度 20%


class RiskManager:
    """风险管理器

    交易前风控检查:
    1. 单笔交易风险限制
    2. 日度回撤限制
    3. 总杠杆限制
    4. 持仓数量限制
    5. 相关品种敞口限制
    6. 风险回报比检查

    交易后更新:
    - 更新日度盈亏
    - 更新连续亏损计数
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: 风控配置字典，可包含 RiskConfig 的所有字段
        """
        config = config or {}
        self.config = RiskConfig(
            max_risk_per_trade=config.get("max_risk_per_trade", 0.01),
            max_daily_drawdown=config.get("max_daily_drawdown", 0.05),
            max_total_leverage=config.get("max_total_leverage", 10.0),
            max_positions_per_symbol=config.get("max_positions_per_symbol", 2),
            max_correlated_exposure=config.get("max_correlated_exposure", 0.03),
            emergency_drawdown=config.get("emergency_drawdown", 0.10),
            min_risk_reward=config.get("min_risk_reward", 1.5),
            max_portfolio_heat=config.get("max_portfolio_heat", 0.20),
        )

        # 日度状态
        self._daily_pnl: float = 0.0  # 当日盈亏
        self._daily_trades: int = 0  # 当日交易次数
        self._consecutive_losses: int = 0  # 连续亏损次数
        self._current_date: date = date.today()
        self._daily_peak_equity: float = 0.0  # 当日权益峰值
        self._equity_history: List[float] = []  # 权益历史

    def check_trade_allowed(
        self,
        symbol: str,
        side: str,
        qty: float,
        positions: List[Dict],
        equity: float,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Tuple[bool, str]:
        """交易前风控检查

        检查6项风控规则，全部通过才允许交易。

        Args:
            symbol: 交易品种
            side: "buy" 或 "sell"
            qty: 交易数量
            positions: 当前持仓列表，每项包含 symbol, side, qty, leverage
            equity: 账户权益
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 止盈价格

        Returns:
            (是否允许, 原因说明)
        """
        # 重置日度状态（如果日期变化）
        self.reset_daily()

        checks: List[RiskCheck] = []

        # 1. 检查单笔交易风险
        if entry_price > 0 and stop_loss > 0:
            risk_per_trade = self._calculate_trade_risk(
                qty, entry_price, stop_loss, side, equity
            )
            if risk_per_trade > self.config.max_risk_per_trade:
                checks.append(RiskCheck(
                    passed=False,
                    check_name="单笔风险限制",
                    message=f"单笔风险 {risk_per_trade:.4f} > 限制 {self.config.max_risk_per_trade:.4f}",
                    severity="critical"
                ))
            else:
                checks.append(RiskCheck(
                    passed=True,
                    check_name="单笔风险限制",
                    message=f"单笔风险 {risk_per_trade:.4f} OK",
                ))

        # 2. 检查日度回撤
        current_drawdown = self._calculate_daily_drawdown(equity)
        if current_drawdown >= self.config.max_daily_drawdown:
            checks.append(RiskCheck(
                passed=False,
                check_name="日度回撤限制",
                message=f"日度回撤 {current_drawdown:.4f} >= 限制 {self.config.max_daily_drawdown:.4f}",
                severity="critical"
            ))
        else:
            checks.append(RiskCheck(
                passed=True,
                check_name="日度回撤限制",
                message=f"日度回撤 {current_drawdown:.4f} OK",
            ))

        # 3. 检查总杠杆
        total_leverage = self._calculate_total_leverage(positions, equity)
        trade_leverage = (qty * entry_price) / equity if equity > 0 else 0
        if total_leverage + trade_leverage > self.config.max_total_leverage:
            checks.append(RiskCheck(
                passed=False,
                check_name="总杠杆限制",
                message=f"总杠杆 {total_leverage + trade_leverage:.2f}x > 限制 {self.config.max_total_leverage:.1f}x",
                severity="critical"
            ))
        else:
            checks.append(RiskCheck(
                passed=True,
                check_name="总杠杆限制",
                message=f"总杠杆 {total_leverage + trade_leverage:.2f}x OK",
            ))

        # 4. 检查持仓数量
        symbol_positions = [
            p for p in positions if p.get("symbol") == symbol
        ]
        if len(symbol_positions) >= self.config.max_positions_per_symbol:
            checks.append(RiskCheck(
                passed=False,
                check_name="持仓数量限制",
                message=f"{symbol} 持仓数 {len(symbol_positions)} >= 限制 {self.config.max_positions_per_symbol}",
                severity="warning"
            ))
        else:
            checks.append(RiskCheck(
                passed=True,
                check_name="持仓数量限制",
                message=f"{symbol} 持仓数 {len(symbol_positions)} OK",
            ))

        # 5. 检查相关品种敞口
        correlated_exposure = self._calculate_correlated_exposure(symbol, positions, equity)
        if correlated_exposure > self.config.max_correlated_exposure:
            checks.append(RiskCheck(
                passed=False,
                check_name="相关品种敞口",
                message=f"相关敞口 {correlated_exposure:.4f} > 限制 {self.config.max_correlated_exposure:.4f}",
                severity="warning"
            ))
        else:
            checks.append(RiskCheck(
                passed=True,
                check_name="相关品种敞口",
                message=f"相关敞口 {correlated_exposure:.4f} OK",
            ))

        # 6. 检查风险回报比
        if entry_price > 0 and stop_loss > 0 and take_profit > 0:
            rr_ratio = self.calculate_risk_reward(
                entry_price, stop_loss, take_profit, side
            )
            if rr_ratio < self.config.min_risk_reward:
                checks.append(RiskCheck(
                    passed=False,
                    check_name="风险回报比",
                    message=f"R/R {rr_ratio:.2f} < 最低 {self.config.min_risk_reward:.1f}",
                    severity="warning"
                ))
            else:
                checks.append(RiskCheck(
                    passed=True,
                    check_name="风险回报比",
                    message=f"R/R {rr_ratio:.2f} OK",
                ))

        # 汇总结果
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            # 有关键检查失败
            critical_fails = [c for c in failed_checks if c.severity == "critical"]
            if critical_fails:
                reason = "; ".join([
                    f"[{c.check_name}] {c.message}" for c in critical_fails
                ])
                return False, f"风控阻止: {reason}"
            else:
                # 只有警告级别的失败，允许交易但记录
                reason = "; ".join([
                    f"[{c.check_name}] {c.message}" for c in failed_checks
                ])
                logger.warning(f"风控警告 (仍允许交易): {reason}")

        return True, "所有风控检查通过"

    def calculate_stop_loss(
        self,
        entry: float,
        side: str,
        atr: float,
        leverage: int = 1
    ) -> float:
        """计算止损价格

        止损距离 = max(ATR * 2, entry * 0.01 / leverage)
        确保止损不会因为高杠杆而过于接近入场价。

        Args:
            entry: 入场价格
            side: "buy" 或 "sell"
            atr: ATR 值
            leverage: 杠杆倍数

        Returns:
            止损价格
        """
        # 基于 ATR 的止损距离
        atr_stop = atr * 2.0

        # 基于入场价和杠杆的最小止损距离
        min_stop_distance = entry * 0.01 / max(leverage, 1)

        # 取较大值
        stop_distance = max(atr_stop, min_stop_distance)

        if side.lower() == "buy":
            stop_loss = entry - stop_distance
        else:
            stop_loss = entry + stop_distance

        return stop_loss

    def calculate_take_profit(
        self,
        entry: float,
        side: str,
        stop_loss: float,
        risk_reward: float = 2.0
    ) -> float:
        """计算止盈价格

        止盈距离 = 止损距离 * 风险回报比

        Args:
            entry: 入场价格
            side: "buy" 或 "sell"
            stop_loss: 止损价格
            risk_reward: 风险回报比

        Returns:
            止盈价格
        """
        stop_distance = abs(entry - stop_loss)
        take_distance = stop_distance * risk_reward

        if side.lower() == "buy":
            take_profit = entry + take_distance
        else:
            take_profit = entry - take_distance

        return take_profit

    def update_after_trade(self, trade_result: Dict) -> None:
        """交易后更新状态

        Args:
            trade_result: 交易结果字典，包含:
                - pnl: 盈亏金额
                - equity: 交易后权益
        """
        pnl = trade_result.get("pnl", 0.0)
        equity = trade_result.get("equity", 0.0)

        self._daily_pnl += pnl
        self._daily_trades += 1
        self._equity_history.append(equity)

        # 更新权益峰值
        if equity > self._daily_peak_equity:
            self._daily_peak_equity = equity

        # 更新连续亏损
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily(self) -> None:
        """重置日度状态（当日期变化时调用）"""
        today = date.today()
        if today != self._current_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._consecutive_losses = 0
            self._current_date = today
            self._daily_peak_equity = 0.0
            self._equity_history = []
            logger.info(f"Daily risk state reset for {today}")

    def get_portfolio_heat(self, positions: List[Dict], equity: float) -> Dict:
        """计算组合热度

        组合热度 = 所有持仓的风险敞口之和 / 权益
        反映当前组合面临的总风险水平。

        Args:
            positions: 持仓列表，每项包含 symbol, qty, entry_price, stop_loss, side
            equity: 账户权益

        Returns:
            组合热度信息字典
        """
        total_heat = 0.0
        position_heats = []

        for pos in positions:
            qty = pos.get("qty", 0)
            entry = pos.get("entry_price", 0)
            stop = pos.get("stop_loss", 0)
            side = pos.get("side", "buy")

            if entry <= 0 or stop <= 0 or equity <= 0:
                continue

            # 单个持仓的风险金额
            risk_amount = abs(entry - stop) * qty
            risk_pct = risk_amount / equity

            total_heat += risk_pct
            position_heats.append({
                "symbol": pos.get("symbol", ""),
                "risk_pct": risk_pct,
                "risk_amount": risk_amount,
            })

        is_overheated = total_heat > self.config.max_portfolio_heat

        return {
            "total_heat": total_heat,
            "max_heat": self.config.max_portfolio_heat,
            "is_overheated": is_overheated,
            "position_heats": position_heats,
            "available_heat": max(0, self.config.max_portfolio_heat - total_heat),
        }

    def should_emergency_close(
        self,
        positions: List[Dict],
        equity: float
    ) -> bool:
        """判断是否需要紧急清仓

        紧急清仓条件:
        1. 日度回撤达到紧急阈值
        2. 组合过度暴露且趋势逆转

        Args:
            positions: 当前持仓列表
            equity: 当前权益

        Returns:
            True 如果需要紧急清仓
        """
        # 检查日度回撤
        drawdown = self._calculate_daily_drawdown(equity)
        if drawdown >= self.config.emergency_drawdown:
            logger.critical(
                f"EMERGENCY CLOSE: Drawdown {drawdown:.4f} >= "
                f"threshold {self.config.emergency_drawdown:.4f}"
            )
            return True

        # 检查组合热度
        heat = self.get_portfolio_heat(positions, equity)
        if heat["is_overheated"] and self._consecutive_losses >= 2:
            logger.critical(
                f"EMERGENCY CLOSE: Portfolio overheated ({heat['total_heat']:.4f}) "
                f"with {self._consecutive_losses} consecutive losses"
            )
            return True

        return False

    def calculate_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float,
        side: str
    ) -> float:
        """计算风险回报比

        R/R = |止盈距离| / |止损距离|

        Args:
            entry: 入场价格
            stop: 止损价格
            target: 目标价格
            side: "buy" 或 "sell"

        Returns:
            风险回报比
        """
        risk = abs(entry - stop)
        reward = abs(target - entry)

        if risk <= 0:
            return 0.0

        return reward / risk

    # --- 内部辅助方法 ---

    def _calculate_trade_risk(
        self,
        qty: float,
        entry_price: float,
        stop_loss: float,
        side: str,
        equity: float
    ) -> float:
        """计算单笔交易的风险百分比"""
        if equity <= 0:
            return 1.0  # 最大风险
        risk_amount = abs(entry_price - stop_loss) * qty
        risk_pct = risk_amount / equity
        return risk_pct

    def _calculate_daily_drawdown(self, equity: float) -> float:
        """计算日度回撤"""
        if self._daily_peak_equity <= 0:
            self._daily_peak_equity = equity
            return 0.0

        if equity < self._daily_peak_equity:
            drawdown = (self._daily_peak_equity - equity) / self._daily_peak_equity
            return drawdown

        # 更新峰值
        if equity > self._daily_peak_equity:
            self._daily_peak_equity = equity

        return 0.0

    def _calculate_total_leverage(
        self, positions: List[Dict], equity: float
    ) -> float:
        """计算当前总杠杆"""
        if equity <= 0:
            return 0.0

        total_notional = sum(
            p.get("qty", 0) * p.get("mark_price", p.get("entry_price", 0))
            for p in positions
        )

        return total_notional / equity if equity > 0 else 0.0

    def _calculate_correlated_exposure(
        self, symbol: str, positions: List[Dict], equity: float
    ) -> float:
        """计算相关品种的总敞口"""
        # 相关品种组
        correlated_groups = [
            ["BTCUSDT", "ETHUSDT"],  # 主流币高度相关
            ["XRPUSDT", "SOLUSDT", "LINKUSDT"],  # 山寨币组
        ]

        # 找到 symbol 所属的相关组
        target_group = None
        for group in correlated_groups:
            if symbol in group:
                target_group = group
                break

        if target_group is None:
            return 0.0

        # 计算相关品种的总敞口
        total_exposure = 0.0
        for pos in positions:
            pos_symbol = pos.get("symbol", "")
            if pos_symbol in target_group:
                notional = pos.get("qty", 0) * pos.get("entry_price", 0)
                total_exposure += notional

        if equity <= 0:
            return 0.0

        return total_exposure / equity

    def get_status(self) -> Dict:
        """获取风控状态摘要"""
        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
            "current_date": self._current_date.isoformat(),
            "daily_peak_equity": self._daily_peak_equity,
            "config": {
                "max_risk_per_trade": self.config.max_risk_per_trade,
                "max_daily_drawdown": self.config.max_daily_drawdown,
                "max_total_leverage": self.config.max_total_leverage,
                "emergency_drawdown": self.config.emergency_drawdown,
            }
        }

    def __repr__(self):
        return (
            f"RiskManager("
            f"max_risk={self.config.max_risk_per_trade:.2%}, "
            f"max_dd={self.config.max_daily_drawdown:.2%}, "
            f"max_lev={self.config.max_total_leverage:.0f}x)"
        )
