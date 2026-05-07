"""数据缓存 - 内存缓存 + TTL 过期管理"""

import logging
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class _CacheEntry:
    """缓存条目内部结构"""

    __slots__ = ["value", "expires_at"]

    def __init__(self, value: Any, ttl: int) -> None:
        self.value = value
        self.expires_at = time.time() + ttl

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class DataCache:
    """线程安全的内存数据缓存

    支持：
    - 通用 key-value 存储（带 TTL）
    - K线数据专用存储（DataFrame）
    - 订单簿专用存储（dict，短 TTL）
    - 资金费率专用存储（dict）
    - 从订单簿计算中间价和价差
    - 自动清理过期条目
    """

    def __init__(self) -> None:
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 通用接口
    # ------------------------------------------------------------------
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值（任意类型）
            ttl: 过期时间（秒），默认 300
        """
        with self._lock:
            self._store[key] = _CacheEntry(value, ttl)
        logger.debug("Cache SET: key=%s, ttl=%s", key, ttl)

    def get(self, key: str) -> Any:
        """获取缓存值

        Returns:
            缓存值，如果不存在或已过期则返回 None
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            return entry.value

    def has(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._store[key]
                return False
            return True

    # ------------------------------------------------------------------
    # K线数据
    # ------------------------------------------------------------------
    def set_klines(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """存储 K 线 DataFrame

        Args:
            symbol: 交易对，如 "ETHUSDT"
            interval: 周期，如 "15", "60", "240"
            df: K 线 DataFrame
        """
        key = f"klines:{symbol}:{interval}"
        # K 线 TTL 设为 10 分钟（因为通过 REST 定期刷新）
        self.set(key, df, ttl=600)
        logger.debug("Cache SET klines: %s %s (rows=%s)", symbol, interval, len(df))

    def get_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """获取 K 线 DataFrame

        Returns:
            DataFrame 或 None
        """
        key = f"klines:{symbol}:{interval}"
        result = self.get(key)
        if result is not None and isinstance(result, pd.DataFrame):
            return result
        return None

    # ------------------------------------------------------------------
    # 订单簿
    # ------------------------------------------------------------------
    def set_orderbook(
        self,
        symbol: str,
        data: Dict[str, Any],
        ttl: int = 10,
    ) -> None:
        """存储订单簿数据（短 TTL，因为 WS 会推送更新）

        Args:
            symbol: 交易对
            data: {"bids": [...], "asks": [...], ...}
            ttl: 默认 10 秒
        """
        key = f"orderbook:{symbol}"
        self.set(key, data, ttl=ttl)
        logger.debug("Cache SET orderbook: %s", symbol)

    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取订单簿数据

        Returns:
            dict 或 None
        """
        key = f"orderbook:{symbol}"
        result = self.get(key)
        if result is not None and isinstance(result, dict):
            return result
        return None

    # ------------------------------------------------------------------
    # 资金费率
    # ------------------------------------------------------------------
    def set_funding(
        self,
        symbol: str,
        data: Dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """存储资金费率数据

        Args:
            symbol: 交易对
            data: {"fundingRate": ..., "fundingRateTimestamp": ...}
            ttl: 默认 1 小时
        """
        key = f"funding:{symbol}"
        self.set(key, data, ttl=ttl)
        logger.debug("Cache SET funding: %s = %s", symbol, data.get("fundingRate"))

    def get_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取资金费率数据

        Returns:
            dict 或 None
        """
        key = f"funding:{symbol}"
        result = self.get(key)
        if result is not None and isinstance(result, dict):
            return result
        return None

    # ------------------------------------------------------------------
    # 派生计算
    # ------------------------------------------------------------------
    def get_mid_price(self, symbol: str) -> Optional[float]:
        """从缓存的订单簿计算中间价（(best_bid + best_ask) / 2）

        Returns:
            中间价，如果订单簿不存在则返回 None
        """
        ob = self.get_orderbook(symbol)
        if ob is None:
            return None

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return None

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return (best_bid + best_ask) / 2.0
        except (IndexError, ValueError, TypeError):
            logger.warning("Cannot compute mid_price for %s", symbol)
            return None

    def get_spread(self, symbol: str) -> Optional[float]:
        """从缓存的订单簿计算价差（best_ask - best_bid）

        Returns:
            绝对价差，如果订单簿不存在则返回 None
        """
        ob = self.get_orderbook(symbol)
        if ob is None:
            return None

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return None

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return best_ask - best_bid
        except (IndexError, ValueError, TypeError):
            logger.warning("Cannot compute spread for %s", symbol)
            return None

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def clear_expired(self) -> int:
        """清理所有过期条目

        Returns:
            清理的条目数量
        """
        count = 0
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._store.items() if v.expires_at < now
            ]
            for k in expired_keys:
                del self._store[k]
                count += 1
        if count > 0:
            logger.info("Cache cleared %s expired entries", count)
        return count

    def clear_all(self) -> int:
        """清空所有缓存

        Returns:
            清空的条目数量
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
        logger.info("Cache cleared all %s entries", count)
        return count

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息"""
        with self._lock:
            total = len(self._store)
            expired = sum(1 for v in self._store.values() if v.is_expired())
            # 按前缀分类统计
            categories: Dict[str, int] = {}
            for key in self._store:
                prefix = key.split(":")[0] if ":" in key else "other"
                categories[prefix] = categories.get(prefix, 0) + 1

        return {
            "total_keys": total,
            "expired_keys": expired,
            "active_keys": total - expired,
            "categories": categories,
        }
