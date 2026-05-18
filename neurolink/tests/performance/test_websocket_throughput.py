from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from neurolink.backend.main import app
from neurolink.backend.core.security import SecurityManager


def _make_token(user_id: str = "perf-user") -> str:
    return SecurityManager.create_access_token(subject=user_id)


class TestMessageThroughput:
    """Test WebSocket message throughput."""

    def test_single_connection_throughput(self) -> None:
        token = _make_token("throughput-1")
        client = TestClient(app)
        messages_sent = 0
        start = time.perf_counter()

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                for i in range(100):
                    try:
                        ws.send_json({
                            "type": "message:new",
                            "data": {
                                "content": f"Message {i}",
                                "seq": i,
                                "timestamp": time.time(),
                            },
                        })
                        messages_sent += 1
                    except Exception:
                        break

                elapsed = time.perf_counter() - start
                throughput = messages_sent / elapsed if elapsed > 0 else 0
                print(f"\nSingle Connection Throughput: {throughput:.2f} msg/s ({messages_sent} in {elapsed:.2f}s)")

    def test_burst_throughput(self) -> None:
        token = _make_token("burst-user")
        client = TestClient(app)
        messages_sent = 0

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                start = time.perf_counter()
                for i in range(50):
                    try:
                        ws.send_json({
                            "type": "gesture:frame",
                            "data": {
                                "frame_id": i,
                                "landmarks": [[0.5, 0.3, 0.0]] * 21,
                            },
                        })
                        messages_sent += 1
                    except Exception:
                        break

                elapsed = time.perf_counter() - start
                throughput = messages_sent / elapsed if elapsed > 0 else 0
                print(f"\nBurst Throughput: {throughput:.2f} msg/s ({messages_sent} in {elapsed:.2f}s)")

    def test_mixed_message_types(self) -> None:
        token = _make_token("mixed-user")
        client = TestClient(app)
        messages_sent = 0

        message_types = [
            {"type": "gesture:frame", "data": {"landmarks": []}},
            {"type": "speech:audio", "data": {"audio": [0.0] * 160}},
            {"type": "session:start", "data": {"type": "multimodal"}},
            {"type": "message:new", "data": {"content": "test"}},
        ]

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                start = time.perf_counter()
                for i in range(40):
                    try:
                        msg = message_types[i % len(message_types)]
                        ws.send_json(msg)
                        messages_sent += 1
                    except Exception:
                        break

                elapsed = time.perf_counter() - start
                throughput = messages_sent / elapsed if elapsed > 0 else 0
                print(f"\nMixed Message Throughput: {throughput:.2f} msg/s")


class TestConcurrentConnections:
    """Test concurrent WebSocket connections."""

    def test_multiple_concurrent_connections(self) -> None:
        client = TestClient(app)
        tokens = [_make_token(f"concurrent-{i}") for i in range(5)]
        connections = []

        for token in tokens:
            with pytest.raises(WebSocketDisconnect):
                conn = client.websocket_connect(f"/ws?token={token}")
                connections.append(conn)

        print(f"\nConcurrent Connections Established: {len(connections)}")

    def test_concurrent_message_sending(self) -> None:
        client = TestClient(app)
        tokens = [_make_token(f"multi-send-{i}") for i in range(3)]
        total_sent = 0

        for token in tokens:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"/ws?token={token}") as ws:
                    for i in range(10):
                        try:
                            ws.send_json({"type": "ping"})
                            total_sent += 1
                        except Exception:
                            break

        print(f"\nTotal Concurrent Messages: {total_sent}")


class TestMessageSizeLimits:
    """Test WebSocket message size handling."""

    def test_large_message(self) -> None:
        token = _make_token("large-msg")
        client = TestClient(app)

        large_data = {
            "type": "gesture:frame",
            "data": {
                "landmarks": [[0.5, 0.3, 0.0] * 100] * 21,
            },
        }

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json(large_data)

    def test_small_message(self) -> None:
        token = _make_token("small-msg")
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                for _ in range(5):
                    ws.send_json({"type": "ping"})

    def test_binary_message_size(self) -> None:
        token = _make_token("binary-size")
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_bytes(b"x" * 1024)  # 1KB
                ws.send_bytes(b"x" * 10240)  # 10KB
                ws.send_bytes(b"x" * 102400)  # 100KB


class TestReconnectionStress:
    """Test reconnection under stress."""

    def test_rapid_reconnect(self) -> None:
        client = TestClient(app)

        for i in range(10):
            token = _make_token(f"rapid-reconnect-{i}")
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"/ws?token={token}") as ws:
                    ws.send_json({"type": "ping"})

    def test_connect_disconnect_loop(self) -> None:
        client = TestClient(app)

        for i in range(5):
            token = _make_token(f"loop-{i}")
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"/ws?token={token}"):
                    pass

    def test_interleaved_connect_disconnect(self) -> None:
        client = TestClient(app)
        tokens = [_make_token(f"interleave-{i}") for i in range(3)]

        for _ in range(3):
            for token in tokens:
                with pytest.raises(WebSocketDisconnect):
                    with client.websocket_connect(f"/ws?token={token}") as ws:
                        ws.send_json({"type": "ping"})
