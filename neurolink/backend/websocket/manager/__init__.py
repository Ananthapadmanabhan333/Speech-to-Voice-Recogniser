from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

from neurolink.backend.core.config import settings
from neurolink.backend.core.logging import get_logger
from neurolink.backend.core.security import SecurityManager

logger = get_logger("ws.manager")


class WebSocketConnection:
    """Represents a single authenticated WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        connection_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.user_id = user_id
        self.connection_id = connection_id or uuid.uuid4().hex
        self.channels: set[str] = set()
        self.connected_at = time.monotonic()
        self.last_heartbeat = time.monotonic()

    async def send_json(self, data: dict[str, Any]) -> None:
        try:
            await self.websocket.send_json(data)
        except Exception:
            pass

    async def send_text(self, text: str) -> None:
        try:
            await self.websocket.send_text(text)
        except Exception:
            pass

    async def send_bytes(self, data: bytes) -> None:
        try:
            await self.websocket.send_bytes(data)
        except Exception:
            pass

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.connected_at

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_heartbeat

    def refresh_heartbeat(self) -> None:
        self.last_heartbeat = time.monotonic()


class ConnectionManager:
    """Manages all active WebSocket connections, channels, and broadcasting."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocketConnection] = {}
        self._user_connections: dict[str, set[str]] = {}
        self._channels: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._logger = get_logger("ws.manager")

    # ── Connection lifecycle ───────────────────────────────────────────────

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        channels: list[str] | None = None,
    ) -> WebSocketConnection:
        # Enforce per-user connection limit
        async with self._lock:
            existing = self._user_connections.get(user_id, set())
            if len(existing) >= settings.WS_MAX_CONNECTIONS_PER_USER:
                raise ConnectionError(f"Max {settings.WS_MAX_CONNECTIONS_PER_USER} connections per user")

            connection = WebSocketConnection(websocket, user_id)
            self._connections[connection.connection_id] = connection
            self._user_connections.setdefault(user_id, set()).add(connection.connection_id)

            if channels:
                for ch in channels:
                    self._channels.setdefault(ch, set()).add(connection.connection_id)
                    connection.channels.add(ch)

        self._logger.info("ws_connected", user_id=user_id, conn_id=connection.connection_id)
        return connection

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            connection = self._connections.pop(connection_id, None)
            if not connection:
                return

            # Remove from user index
            user_conns = self._user_connections.get(connection.user_id)
            if user_conns:
                user_conns.discard(connection_id)
                if not user_conns:
                    del self._user_connections[connection.user_id]

            # Remove from channels
            for ch in list(connection.channels):
                channel_set = self._channels.get(ch)
                if channel_set:
                    channel_set.discard(connection_id)
                    if not channel_set:
                        del self._channels[ch]

            connection.channels.clear()

        self._logger.info("ws_disconnected", conn_id=connection_id, user_id=connection.user_id)

    async def disconnect_user(self, user_id: str) -> int:
        async with self._lock:
            conn_ids = list(self._user_connections.get(user_id, set()))
        count = 0
        for cid in conn_ids:
            conn = self._connections.get(cid)
            if conn:
                try:
                    await conn.websocket.close()
                except Exception:
                    pass
                await self.disconnect(cid)
                count += 1
        return count

    # ── Subscriptions ──────────────────────────────────────────────────────

    async def subscribe(self, connection_id: str, channel: str) -> None:
        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return
            self._channels.setdefault(channel, set()).add(connection_id)
            connection.channels.add(channel)

    async def unsubscribe(self, connection_id: str, channel: str) -> None:
        async with self._lock:
            connection = self._connections.get(connection_id)
            if not connection:
                return
            connection.channels.discard(channel)
            channel_set = self._channels.get(channel)
            if channel_set:
                channel_set.discard(connection_id)
                if not channel_set:
                    del self._channels[channel]

    # ── Broadcasting ───────────────────────────────────────────────────────

    async def broadcast_to_channel(
        self,
        channel: str,
        message: dict[str, Any],
    ) -> int:
        async with self._lock:
            conn_ids = list(self._channels.get(channel, set()))

        sent = 0
        for cid in conn_ids:
            conn = self._connections.get(cid)
            if conn:
                await conn.send_json(message)
                sent += 1
        return sent

    async def broadcast_to_user(
        self,
        user_id: str,
        message: dict[str, Any],
    ) -> int:
        async with self._lock:
            conn_ids = list(self._user_connections.get(user_id, set()))

        sent = 0
        for cid in conn_ids:
            conn = self._connections.get(cid)
            if conn:
                await conn.send_json(message)
                sent += 1
        return sent

    async def broadcast_all(self, message: dict[str, Any]) -> int:
        async with self._lock:
            conn_ids = list(self._connections.keys())

        sent = 0
        for cid in conn_ids:
            conn = self._connections.get(cid)
            if conn:
                await conn.send_json(message)
                sent += 1
        return sent

    # ── Queries ────────────────────────────────────────────────────────────

    def get_connection(self, connection_id: str) -> WebSocketConnection | None:
        return self._connections.get(connection_id)

    def get_user_connections(self, user_id: str) -> list[WebSocketConnection]:
        conn_ids = self._user_connections.get(user_id, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_channel_connections(self, channel: str) -> list[WebSocketConnection]:
        conn_ids = self._channels.get(channel, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    @property
    def active_connections(self) -> int:
        return len(self._connections)

    @property
    def active_users(self) -> int:
        return len(self._user_connections)

    @property
    def active_channels(self) -> int:
        return len(self._channels)

    # ── Heartbeat ──────────────────────────────────────────────────────────

    async def start_heartbeat(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            return

        async def _heartbeat_loop() -> None:
            while True:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
                await self._check_heartbeats()

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())
        self._logger.info("ws_heartbeat_started", interval=settings.WS_HEARTBEAT_INTERVAL)

    async def stop_heartbeat(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        self._logger.info("ws_heartbeat_stopped")

    async def _check_heartbeats(self) -> None:
        timeout = settings.WS_HEARTBEAT_TIMEOUT
        stale: list[str] = []
        async with self._lock:
            for cid, conn in list(self._connections.items()):
                if conn.idle_seconds > timeout:
                    stale.append(cid)

        for cid in stale:
            conn = self._connections.get(cid)
            if conn:
                try:
                    await conn.send_json({"type": "heartbeat_timeout"})
                    await conn.websocket.close()
                except Exception:
                    pass
                await self.disconnect(cid)
                self._logger.warning("ws_stale_connection_closed", conn_id=cid)

    # ── Stats ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "active_connections": self.active_connections,
            "active_users": self.active_users,
            "active_channels": self.active_channels,
        }


# Singleton
connection_manager = ConnectionManager()
