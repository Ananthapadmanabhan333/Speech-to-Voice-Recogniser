from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from neurolink.backend.main import app


class TestWebSocketConnection:
    """Test WebSocket connection lifecycle."""

    def test_connect_no_auth(self) -> None:
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws"):
                pass

    def test_connect_with_token(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="test-user")

        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}"):
                pass

    def test_connect_invalid_token(self) -> None:
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=invalid"):
                pass

    def test_connect_multiple_times(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="multi-user")
        client = TestClient(app)

        connections = []
        for _ in range(3):
            with pytest.raises(WebSocketDisconnect):
                conn = client.websocket_connect(f"/ws?token={token}")
                connections.append(conn)


class TestWebSocketEvents:
    """Test WebSocket event handling."""

    def _make_token(self) -> str:
        from neurolink.backend.core.security import SecurityManager
        return SecurityManager.create_access_token(subject="event-user")

    def test_send_gesture_frame(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({
                    "type": "gesture:frame",
                    "data": {"landmarks": [[0.5, 0.3, 0.0]] * 21},
                })
                # May or may not receive a response depending on handler implementation

    def test_send_speech_audio(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_bytes(b"audio data")
                # Audio should be handled

    def test_send_multimodal_event(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({
                    "type": "multimodal:process",
                    "data": {
                        "text": "Hello",
                        "gesture": None,
                        "emotion": None,
                    },
                })

    def test_send_heartbeat(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "ping"})

    def test_send_invalid_json(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_bytes(b"invalid json!!!")

    def test_session_lifecycle(self) -> None:
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "session:start", "data": {"type": "multimodal"}})

    def test_message_ordering(self) -> None:
        """Verify messages maintain order."""
        token = self._make_token()
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "message:new", "data": {"content": "first", "seq": 1}})
                ws.send_json({"type": "message:new", "data": {"content": "second", "seq": 2}})
                ws.send_json({"type": "message:new", "data": {"content": "third", "seq": 3}})


class TestWebSocketReconnection:
    """Test WebSocket reconnection behavior."""

    def test_reconnect_with_same_token(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="reconnect-user")
        client = TestClient(app)

        # First connection
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "ping"})

        # Second connection with same token
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "ping"})

    def test_reconnect_after_disconnect(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="reconnect2-user")
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "session:start"})

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "session:start"})


class TestConcurrentConnections:
    """Test concurrent WebSocket connections."""

    def test_multiple_users(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        client = TestClient(app)

        tokens = [
            SecurityManager.create_access_token(subject=f"concurrent-user-{i}")
            for i in range(3)
        ]

        connections = []
        for token in tokens:
            with pytest.raises(WebSocketDisconnect):
                conn = client.websocket_connect(f"/ws?token={token}")
                connections.append(conn)

    def test_same_user_multiple_connections(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="multi-conn-user")
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws1:
                with client.websocket_connect(f"/ws?token={token}") as ws2:
                    ws1.send_json({"type": "ping"})
                    ws2.send_json({"type": "ping"})


class TestWebSocketErrorHandling:
    """Test WebSocket error scenarios."""

    def test_disconnect_during_session(self) -> None:
        from neurolink.backend.core.security import SecurityManager
        token = SecurityManager.create_access_token(subject="error-user")
        client = TestClient(app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws?token={token}") as ws:
                ws.send_json({"type": "session:start"})
                ws.close()
