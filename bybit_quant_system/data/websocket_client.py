"""Bybit WebSocket 客户端 - V5 实时数据流"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Callable, Dict, List, Optional, Any
from urllib.parse import urlencode, urlparse

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

logger = logging.getLogger(__name__)


class BybitWebSocket:
    """Bybit WebSocket 客户端（V5）

    功能：
    - 连接公共 WebSocket 流（tickers / orderbook）
    - 支持签名认证（私有流扩展）
    - 自动心跳保活（每 20 秒 ping）
    - 指数退避重连（1, 2, 4, 8, 16 秒）
    - 回调驱动：on_tick / on_orderbook / on_position
    """

    # 指数退避最大重连间隔（秒）
    MAX_RECONNECT_DELAY = 16

    def __init__(
        self,
        config: Any,
        symbols: List[str],
        on_tick: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_orderbook: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_position: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Args:
            config: 配置对象，需要包含 bybit.api_key / bybit.api_secret / bybit.ws_url
            symbols: 订阅的交易对列表
            on_tick: ticker 回调函数(data_dict)
            on_orderbook: 订单簿回调函数(data_dict)
            on_position: 持仓回调函数(data_dict) —— 预留
        """
        self.config = config
        self.symbols = symbols
        self.on_tick = on_tick
        self.on_orderbook = on_orderbook
        self.on_position = on_position

        self.ws_url: str = getattr(config.bybit, "ws_url", "wss://stream-testnet.bybit.com/v5/public")
        self.api_key: str = getattr(config.bybit, "api_key", "")
        self.api_secret: str = getattr(config.bybit, "api_secret", "")

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running: bool = False
        self._reconnect_delay: int = 1
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """连接到 WebSocket 服务器并订阅频道"""
        logger.info("WebSocket connecting to %s", self.ws_url)
        try:
            extra_headers = {}
            # 如果需要签名认证（私有频道）
            if self.api_key and self.api_secret:
                expires = str(int(time.time()) + 10)
                signature = self._generate_signature(expires)
                extra_headers = {
                    "api_key": self.api_key,
                    "expires": expires,
                    "signature": signature,
                }

            self._ws = await websockets.connect(
                self.ws_url,
                extra_headers=extra_headers if extra_headers else None,
                ping_interval=None,  # 我们自己管理心跳
                ping_timeout=None,
            )
            self._running = True
            self._reconnect_delay = 1  # 重置退避
            logger.info("WebSocket connected successfully")

            # 启动后台任务
            self._receive_task = asyncio.create_task(
                self._receive_loop(), name="ws-receive"
            )
            self._keepalive_task = asyncio.create_task(
                self._keep_alive(), name="ws-keepalive"
            )

            # 订阅频道
            await self.subscribe_tickers(self.symbols)
            await self.subscribe_orderbook(self.symbols, depth=50)

        except Exception as exc:
            logger.error("WebSocket connect failed: %s", exc)
            await self._reconnect()

    async def disconnect(self) -> None:
        """主动断开连接"""
        logger.info("WebSocket disconnecting...")
        self._running = False

        # 取消后台任务
        for task in (self._receive_task, self._keepalive_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("WebSocket disconnected")

    # ------------------------------------------------------------------
    # 订阅
    # ------------------------------------------------------------------
    async def subscribe_tickers(self, symbols: List[str]) -> None:
        """订阅 tickers 频道（最新成交价、24h 统计等）"""
        args = [f"tickers.{s}" for s in symbols]
        msg = {"op": "subscribe", "args": args}
        await self._send(msg)
        logger.info("Subscribed tickers: %s", symbols)

    async def subscribe_orderbook(
        self,
        symbols: List[str],
        depth: int = 50,
    ) -> None:
        """订阅订单簿频道

        Args:
            symbols: 交易对列表
            depth: 深度档位 1 / 25 / 50 / 100 / 200 / 500
        """
        args = [f"orderbook.{depth}.{s}" for s in symbols]
        msg = {"op": "subscribe", "args": args}
        await self._send(msg)
        logger.info("Subscribed orderbook (depth=%s): %s", depth, symbols)

    # ------------------------------------------------------------------
    # 发送 / 接收
    # ------------------------------------------------------------------
    async def _send(self, msg: Dict[str, Any]) -> None:
        """发送消息到 WebSocket 服务器"""
        if self._ws is None or self._ws.closed:
            logger.warning("Cannot send: WebSocket not connected")
            return
        try:
            payload = json.dumps(msg)
            await self._ws.send(payload)
        except Exception as exc:
            logger.error("WebSocket send error: %s", exc)

    async def _receive_loop(self) -> None:
        """消息接收主循环"""
        logger.debug("Receive loop started")
        try:
            while self._running and self._ws is not None:
                try:
                    raw = await asyncio.wait_for(
                        self._ws.recv(), timeout=30.0
                    )
                    await self._handle_message(raw)
                except asyncio.TimeoutError:
                    logger.warning("WebSocket receive timeout")
                    break
                except ConnectionClosed as exc:
                    logger.warning("WebSocket connection closed: %s", exc)
                    break
                except Exception as exc:
                    logger.error("WebSocket receive error: %s", exc)
                    break
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        finally:
            if self._running:
                asyncio.create_task(self._reconnect())

    async def _handle_message(self, raw: str) -> None:
        """解析并分发消息到回调函数"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to decode WebSocket message: %s", raw[:200])
            return

        # 处理操作响应（subscribe / pong 等）
        if "op" in msg:
            op = msg.get("op")
            if op == "pong":
                logger.debug("Received pong")
            elif op in ("subscribe", "unsubscribe"):
                if msg.get("success"):
                    logger.debug("WS op success: %s", op)
                else:
                    logger.warning("WS op failed: %s -> %s", op, msg.get("ret_msg"))
            return

        # 处理数据推送
        topic = msg.get("topic", "")
        data = msg.get("data", {})
        ts = msg.get("ts", int(time.time() * 1000))

        if not topic:
            return

        # ticker 数据
        if topic.startswith("tickers."):
            symbol = topic.replace("tickers.", "")
            payload = {
                "type": "tick",
                "symbol": symbol,
                "data": data,
                "timestamp": ts,
            }
            if self.on_tick is not None:
                try:
                    if asyncio.iscoroutinefunction(self.on_tick):
                        asyncio.create_task(self.on_tick(payload))
                    else:
                        self.on_tick(payload)
                except Exception as exc:
                    logger.error("on_tick callback error: %s", exc)

        # 订单簿数据
        elif topic.startswith("orderbook."):
            parts = topic.split(".")
            if len(parts) >= 3:
                symbol = parts[2]
                payload = {
                    "type": "orderbook",
                    "symbol": symbol,
                    "data": data,
                    "timestamp": ts,
                }
                if self.on_orderbook is not None:
                    try:
                        if asyncio.iscoroutinefunction(self.on_orderbook):
                            asyncio.create_task(self.on_orderbook(payload))
                        else:
                            self.on_orderbook(payload)
                    except Exception as exc:
                        logger.error("on_orderbook callback error: %s", exc)

    # ------------------------------------------------------------------
    # 心跳保活
    # ------------------------------------------------------------------
    async def _keep_alive(self) -> None:
        """每 20 秒发送一次 ping，保持连接活跃"""
        logger.debug("Keep-alive task started")
        try:
            while self._running and self._ws is not None:
                await asyncio.sleep(20.0)
                if self._ws is not None and not self._ws.closed:
                    try:
                        await self._ws.send(json.dumps({"op": "ping"}))
                        logger.debug("Ping sent")
                    except ConnectionClosed:
                        logger.warning("Keep-alive: connection closed")
                        break
                    except Exception as exc:
                        logger.error("Keep-alive ping error: %s", exc)
                        break
                else:
                    break
        except asyncio.CancelledError:
            logger.debug("Keep-alive task cancelled")
        finally:
            if self._running:
                asyncio.create_task(self._reconnect())

    # ------------------------------------------------------------------
    # 重连
    # ------------------------------------------------------------------
    async def _reconnect(self) -> None:
        """指数退避重连"""
        if not self._running:
            return

        delay = min(self._reconnect_delay, self.MAX_RECONNECT_DELAY)
        logger.info(
            "WebSocket reconnecting in %s seconds (delay=%s)...",
            delay,
            self._reconnect_delay,
        )
        await asyncio.sleep(delay)

        # 指数退避：1, 2, 4, 8, 16
        self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

        try:
            await self.connect()
        except Exception as exc:
            logger.error("WebSocket reconnect failed: %s", exc)
            # 如果重连也失败，继续下一次退避
            asyncio.create_task(self._reconnect())

    # ------------------------------------------------------------------
    # 签名工具
    # ------------------------------------------------------------------
    def _generate_signature(self, expires: str) -> str:
        """生成 WebSocket 认证签名"""
        payload = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    # ------------------------------------------------------------------
    # 上下文管理
    # ------------------------------------------------------------------
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
