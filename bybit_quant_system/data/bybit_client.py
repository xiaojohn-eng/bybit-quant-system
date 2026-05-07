"""Bybit REST API 客户端 - 统一交易API V5"""

import asyncio
import logging
from typing import Optional, Dict, List, Any

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


class BybitAPIError(Exception):
    """Bybit API 自定义异常"""
    pass


class BybitClient:
    """Bybit 统一交易 API V5 客户端

    封装了 pybit.HTTP 的所有常用方法，包含：
    - 自动限频（asyncio.Semaphore(20)）
    - 自动重试（tenacity，指数退避）
    - 统一的异常处理和日志
    - K线数据自动转换为 pandas DataFrame
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        self._session: Optional[HTTP] = None
        # 全局并发限频：最多 20 个同时请求
        self._semaphore = asyncio.Semaphore(20)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """初始化 pybit HTTP 会话（同步调用包装为异步）"""
        loop = asyncio.get_event_loop()
        self._session = await loop.run_in_executor(
            None,
            lambda: HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
            ),
        )
        logger.info("BybitClient connected (testnet=%s)", self.testnet)

    async def disconnect(self) -> None:
        """关闭会话"""
        if self._session is not None:
            self._session = None
            logger.info("BybitClient disconnected")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    async def _request(self, method_name: str, **kwargs: Any) -> Any:
        """统一请求入口：限频 + 重试 + 异常处理

        所有公开方法最终都走这里。
        """
        async with self._semaphore:
            return await self._do_request(method_name, **kwargs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((BybitAPIError, ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _do_request(self, method_name: str, **kwargs: Any) -> Any:
        """实际执行请求（已被 tenacity 包装为可重试）"""
        if self._session is None:
            raise BybitAPIError("Session not connected. Call connect() first.")

        loop = asyncio.get_event_loop()
        try:
            method = getattr(self._session, method_name)
            result = await loop.run_in_executor(None, lambda: method(**kwargs))
        except Exception as exc:
            logger.error("Bybit API error [%s]: %s", method_name, exc)
            raise BybitAPIError(f"{method_name} failed: {exc}") from exc

        # pybit 返回的是 dict，包含 retCode / retMsg / result
        if isinstance(result, dict):
            ret_code = result.get("retCode", 0)
            ret_msg = result.get("retMsg", "")
            if ret_code != 0:
                logger.error(
                    "Bybit API returned error [retCode=%s, retMsg=%s]",
                    ret_code,
                    ret_msg,
                )
                raise BybitAPIError(
                    f"API error (retCode={ret_code}): {ret_msg}"
                )
        return result

    # ------------------------------------------------------------------
    # 市场数据
    # ------------------------------------------------------------------
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """获取 K 线数据

        Args:
            symbol: 交易对，如 "ETHUSDT"
            interval: K线周期，如 "15", "60", "240", "D"
            limit: 返回条数，最大 200

        Returns:
            DataFrame 列: [timestamp, open, high, low, close, volume, turnover]
        """
        try:
            resp = await self._request(
                "get_kline",
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit,
            )
            data_list = resp.get("result", {}).get("list", [])
            if not data_list:
                logger.warning("get_klines returned empty list for %s", symbol)
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
                )

            df = pd.DataFrame(
                data_list,
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )
            # Bybit 返回字符串，需要转换为数值
            numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as exc:
            logger.error("get_klines failed for %s: %s", symbol, exc)
            raise

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """获取订单簿

        Args:
            symbol: 交易对
            limit: 深度档位，如 1, 25, 50, 100, 200, 500

        Returns:
            {"bids": [[price, qty], ...], "asks": [[price, qty], ...], "ts": timestamp}
        """
        try:
            resp = await self._request(
                "get_orderbook",
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            result = resp.get("result", {})
            return {
                "symbol": symbol,
                "bids": result.get("b", []),
                "asks": result.get("a", []),
                "ts": result.get("ts"),
                "u": result.get("u"),  # update id
            }
        except Exception as exc:
            logger.error("get_orderbook failed for %s: %s", symbol, exc)
            raise

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """获取当前资金费率

        Returns:
            {"symbol": ..., "fundingRate": ..., "fundingRateTimestamp": ...}
        """
        try:
            resp = await self._request(
                "get_funding_rate",
                category="linear",
                symbol=symbol,
            )
            result = resp.get("result", {}).get("list", [])
            if result:
                return {
                    "symbol": symbol,
                    "fundingRate": float(result[0].get("fundingRate", 0)),
                    "fundingRateTimestamp": int(result[0].get("fundingRateTimestamp", 0)),
                }
            return {"symbol": symbol, "fundingRate": 0.0, "fundingRateTimestamp": 0}
        except Exception as exc:
            logger.error("get_funding_rate failed for %s: %s", symbol, exc)
            raise

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """获取历史资金费率

        Returns:
            DataFrame 列: [fundingRate, fundingRateTimestamp]
        """
        try:
            resp = await self._request(
                "get_funding_rate_history",
                category="linear",
                symbol=symbol,
                limit=limit,
            )
            data_list = resp.get("result", {}).get("list", [])
            if not data_list:
                logger.warning(
                    "get_funding_rate_history returned empty for %s", symbol
                )
                return pd.DataFrame(columns=["fundingRate", "fundingRateTimestamp"])

            df = pd.DataFrame(data_list)
            if "fundingRate" in df.columns:
                df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            if "fundingRateTimestamp" in df.columns:
                df["fundingRateTimestamp"] = pd.to_numeric(
                    df["fundingRateTimestamp"], errors="coerce"
                )
                df["datetime"] = pd.to_datetime(
                    df["fundingRateTimestamp"], unit="ms", utc=True
                )
            df.sort_values("fundingRateTimestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as exc:
            logger.error(
                "get_funding_rate_history failed for %s: %s", symbol, exc
            )
            raise

    async def get_tickers(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取最新行情 / 所有 ticker

        Args:
            symbol: 如果为 None，返回所有 linear 合约的 ticker

        Returns:
            如果指定 symbol: dict 包含 lastPrice, indexPrice, markPrice 等
            如果未指定: list[dict]
        """
        try:
            kwargs: Dict[str, Any] = {"category": "linear"}
            if symbol is not None:
                kwargs["symbol"] = symbol
            resp = await self._request("get_tickers", **kwargs)
            return resp.get("result", {})
        except Exception as exc:
            logger.error("get_tickers failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # 账户
    # ------------------------------------------------------------------
    async def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, Any]:
        """获取钱包余额

        Returns:
            {"coin": coin, "walletBalance": ..., "availableBalance": ...}
        """
        try:
            resp = await self._request(
                "get_wallet_balance",
                accountType="UNIFIED",
                coin=coin,
            )
            result = resp.get("result", {}).get("list", [])
            if result and "coin" in result[0]:
                coins = result[0]["coin"]
                if coins:
                    return {
                        "coin": coin,
                        "walletBalance": float(coins[0].get("walletBalance", 0)),
                        "availableBalance": float(coins[0].get("availableToWithdraw", 0)),
                        "equity": float(coins[0].get("equity", 0)),
                        "unrealisedPnl": float(coins[0].get("unrealisedPnl", 0)),
                        "cumRealisedPnl": float(coins[0].get("cumRealisedPnl", 0)),
                    }
            return {
                "coin": coin,
                "walletBalance": 0.0,
                "availableBalance": 0.0,
                "equity": 0.0,
                "unrealisedPnl": 0.0,
                "cumRealisedPnl": 0.0,
            }
        except Exception as exc:
            logger.error("get_wallet_balance failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # 交易
    # ------------------------------------------------------------------
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """设置杠杆倍数

        Args:
            symbol: 交易对
            leverage: 杠杆倍数 (1-100)

        Returns:
            API 原始响应 result
        """
        try:
            resp = await self._request(
                "set_leverage",
                category="linear",
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            logger.info("Leverage set: %s -> %sx", symbol, leverage)
            return resp.get("result", {})
        except Exception as exc:
            logger.error("set_leverage failed for %s: %s", symbol, exc)
            raise

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """下单

        Args:
            symbol: 交易对
            side: "Buy" or "Sell"
            qty: 下单数量
            order_type: "Market" or "Limit"
            price: 限价单价格（仅 Limit 单需要）
            stop_loss: 止损价格
            take_profit: 止盈价格

        Returns:
            {"orderId": ..., "orderLinkId": ..., "symbol": ...}
        """
        try:
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
            }
            if order_type == "Limit" and price is not None:
                params["price"] = str(price)
            if stop_loss is not None:
                params["stopLoss"] = str(stop_loss)
                params["slTriggerBy"] = "LastPrice"
            if take_profit is not None:
                params["takeProfit"] = str(take_profit)
                params["tpTriggerBy"] = "LastPrice"

            resp = await self._request("place_order", **params)
            result = resp.get("result", {})
            logger.info(
                "Order placed: %s %s %s %s @ %s",
                order_type, side, qty, symbol,
                price if price else "market",
            )
            return result

        except Exception as exc:
            logger.error(
                "place_order failed [%s %s %s]: %s",
                side, qty, symbol, exc,
            )
            raise

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓列表

        Args:
            symbol: 如果为 None，返回所有持仓

        Returns:
            持仓列表，每个元素包含 symbol, side, size, entryPrice, unrealisedPnl 等
        """
        try:
            kwargs: Dict[str, Any] = {
                "category": "linear",
                "settleCoin": "USDT",
            }
            if symbol is not None:
                kwargs["symbol"] = symbol

            resp = await self._request("get_positions", **kwargs)
            positions = resp.get("result", {}).get("list", [])

            # 过滤掉已平仓（size == 0）的持仓
            active = [
                pos for pos in positions
                if float(pos.get("size", 0)) != 0
            ]
            return active

        except Exception as exc:
            logger.error("get_positions failed: %s", exc)
            raise

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """市价平仓

        Args:
            symbol: 要平仓的交易对

        Returns:
            平仓订单结果
        """
        try:
            # 先查询持仓方向
            positions = await self.get_positions(symbol=symbol)
            if not positions:
                logger.warning("No active position to close for %s", symbol)
                return {"status": "no_position", "symbol": symbol}

            pos = positions[0]
            side = "Sell" if pos["side"] == "Buy" else "Buy"
            qty = pos["size"]

            result = await self.place_order(
                symbol=symbol,
                side=side,
                qty=float(qty),
                order_type="Market",
            )
            logger.info("Position closed: %s (was %s %s)", symbol, pos["side"], qty)
            return result

        except Exception as exc:
            logger.error("close_position failed for %s: %s", symbol, exc)
            raise

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """取消所有活动订单

        Args:
            symbol: 如果为 None，取消所有交易对的订单

        Returns:
            API 响应结果
        """
        try:
            params: Dict[str, Any] = {"category": "linear"}
            if symbol is not None:
                params["symbol"] = symbol

            resp = await self._request("cancel_all_orders", **params)
            logger.info("All orders cancelled for %s", symbol or "all symbols")
            return resp.get("result", {})

        except Exception as exc:
            logger.error("cancel_all_orders failed: %s", exc)
            raise
