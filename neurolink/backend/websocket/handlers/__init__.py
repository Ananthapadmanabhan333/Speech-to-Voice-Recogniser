from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from neurolink.backend.core.config import settings
from neurolink.backend.core.logging import get_logger
from neurolink.backend.core.security import SecurityManager
from neurolink.backend.db import get_session_factory
from neurolink.backend.db.models import GestureHistory
from neurolink.backend.websocket.manager import WebSocketConnection, connection_manager

logger = get_logger("ws.handlers")


async def handle_gesture_frame(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Process a gesture video frame sent over WebSocket."""
    import time
    start = time.monotonic()

    frame_data = data.get("frame_data")
    if not frame_data:
        return {"type": "error", "error": "Missing frame_data"}

    try:
        from neurolink.backend.ml.gesture_recognizer import GestureRecognizer
        recognizer = GestureRecognizer()
        result = await recognizer.predict(
            frame_data,
            model_version=data.get("model_version"),
        )
        elapsed = (time.monotonic() - start) * 1000
        logger.info("ws_gesture_frame_processed", gesture=result["gesture_type"], elapsed_ms=round(elapsed, 2))

        return {
            "type": "gesture_result",
            "gesture_type": result["gesture_type"],
            "confidence": result["confidence"],
            "landmarks": result.get("landmarks"),
            "processing_time_ms": round(elapsed, 2),
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.error("ws_gesture_frame_failed", error=str(exc))
        return {"type": "error", "error": f"Gesture processing failed: {exc}"}


async def handle_speech_audio(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Process a speech audio chunk sent over WebSocket."""
    import time
    start = time.monotonic()

    audio_data = data.get("audio_data")
    if not audio_data:
        return {"type": "error", "error": "Missing audio_data"}

    try:
        from neurolink.backend.speech.stt_engine import STTEngine
        engine = STTEngine()

        import base64
        audio_bytes = base64.b64decode(audio_data) if isinstance(audio_data, str) else audio_data

        result = await engine.transcribe(
            audio_bytes=audio_bytes,
            language=data.get("language", "en"),
            sample_rate=data.get("sample_rate", 16000),
        )
        elapsed = (time.monotonic() - start) * 1000
        logger.info("ws_speech_audio_processed", text_len=len(result["text"]), elapsed_ms=round(elapsed, 2))

        return {
            "type": "stt_result",
            "text": result["text"],
            "confidence": result["confidence"],
            "language": result.get("language", "en"),
            "is_final": data.get("is_final", True),
            "processing_time_ms": round(elapsed, 2),
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.error("ws_speech_audio_failed", error=str(exc))
        return {"type": "error", "error": f"Speech processing failed: {exc}"}


async def handle_multimodal_event(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Process a multimodal event combining gesture, speech, and emotion data."""
    import time
    start = time.monotonic()

    try:
        from neurolink.backend.ml.multimodal_fusion import MultimodalFusionEngine
        fusion_engine = MultimodalFusionEngine()
        result = await fusion_engine.fuse(
            gesture_data=data.get("gesture"),
            speech_data=data.get("speech"),
            emotion_data=data.get("emotion"),
            context=data.get("context"),
        )
        elapsed = (time.monotonic() - start) * 1000

        return {
            "type": "multimodal_result",
            "interpretation": result.get("interpretation"),
            "confidence": result.get("confidence"),
            "modalities": result.get("modalities", []),
            "processing_time_ms": round(elapsed, 2),
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.error("ws_multimodal_fusion_failed", error=str(exc))
        return {"type": "error", "error": f"Multimodal fusion failed: {exc}"}


async def handle_communication_message(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Handle a communication message (text or translated)."""
    message_type = data.get("message_type", "text")
    content = data.get("content", "")
    target_channel = data.get("channel", "general")

    message = {
        "type": "communication",
        "message_type": message_type,
        "content": content,
        "sender_id": conn.user_id,
        "sender_connection": conn.connection_id,
        "timestamp": time.time(),
        "message_id": uuid.uuid4().hex,
    }

    # Broadcast to the specified channel
    sent_count = await connection_manager.broadcast_to_channel(target_channel, message)

    message["broadcast_count"] = sent_count
    return message


async def handle_analytics_feed(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Handle subscription to real-time analytics feed."""
    action = data.get("action", "subscribe")
    metric_types = data.get("metric_types", ["all"])

    if action == "subscribe":
        for metric in metric_types:
            channel = f"analytics:{metric}"
            await connection_manager.subscribe(conn.connection_id, channel)

        return {
            "type": "analytics_subscribed",
            "metric_types": metric_types,
            "timestamp": time.time(),
        }
    elif action == "unsubscribe":
        for metric in metric_types:
            channel = f"analytics:{metric}"
            await connection_manager.unsubscribe(conn.connection_id, channel)

        return {
            "type": "analytics_unsubscribed",
            "metric_types": metric_types,
            "timestamp": time.time(),
        }

    return {"type": "error", "error": f"Unknown action: {action}"}


async def handle_personalization_event(conn: WebSocketConnection, data: dict[str, Any]) -> dict[str, Any]:
    """Handle personalization events – memory store/recall, preferences update."""
    import time
    start = time.monotonic()

    action = data.get("action", "recall")
    memory_type = data.get("memory_type", "general")
    key = data.get("key", "")
    value = data.get("value")

    try:
        from neurolink.backend.personalization.memory_manager import MemoryManager
        manager = MemoryManager()

        if action == "store" and key and value is not None:
            result = await manager.store(
                user_id=conn.user_id,
                memory_type=memory_type,
                key=key,
                value=value,
            )
        elif action == "recall" and key:
            result = await manager.recall(
                user_id=conn.user_id,
                memory_type=memory_type,
                key=key,
            )
        elif action == "forget" and key:
            result = await manager.forget(
                user_id=conn.user_id,
                memory_type=memory_type,
                key=key,
            )
        elif action == "search":
            query = data.get("query", "")
            limit = data.get("limit", 5)
            result = await manager.search(
                user_id=conn.user_id,
                query=query,
                memory_type=memory_type,
                limit=limit,
            )
        else:
            return {"type": "error", "error": "Invalid personalization action or missing parameters"}

        elapsed = (time.monotonic() - start) * 1000
        return {
            "type": "personalization_result",
            "action": action,
            "memory_type": memory_type,
            "key": key,
            "result": result,
            "processing_time_ms": round(elapsed, 2),
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.error("ws_personalization_failed", error=str(exc))
        return {"type": "error", "error": f"Personalization failed: {exc}"}


# ── Handler registry ──────────────────────────────────────────────────────

EVENT_HANDLERS: dict[str, Callable[[WebSocketConnection, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "gesture_frame": handle_gesture_frame,
    "speech_audio": handle_speech_audio,
    "multimodal_event": handle_multimodal_event,
    "communication": handle_communication_message,
    "analytics_feed": handle_analytics_feed,
    "personalization": handle_personalization_event,
}


async def handle_websocket_message(conn: WebSocketConnection, raw: str | bytes) -> None:
    """Parse and route an incoming WebSocket message to the correct handler."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (json.JSONDecodeError, TypeError) as exc:
        await conn.send_json({"type": "error", "error": f"Invalid JSON: {exc}"})
        return

    event_type = data.get("type", "")
    handler = EVENT_HANDLERS.get(event_type)
    if not handler:
        await conn.send_json({"type": "error", "error": f"Unknown event type: {event_type}"})
        return

    try:
        response = await handler(conn, data)
        # Send response back to the sender
        if response:
            await conn.send_json(response)
    except Exception as exc:
        logger.error("ws_handler_error", event_type=event_type, error=str(exc))
        await conn.send_json({"type": "error", "error": f"Handler error: {exc}"})


async def websocket_endpoint_handler(websocket: WebSocket) -> None:
    """Main WebSocket endpoint handler – authenticate, connect, and process messages."""
    # Authenticate via query param token
    token = websocket.query_params.get("token")
    user_id = SecurityManager.authenticate_websocket(token)
    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("ws_accepted", user_id=user_id)

    channels = websocket.query_params.get("channels", "").split(",") if websocket.query_params.get("channels") else []

    try:
        conn = await connection_manager.connect(websocket, user_id, channels)

        # Send welcome message
        await conn.send_json({
            "type": "connected",
            "connection_id": conn.connection_id,
            "user_id": user_id,
            "channels": list(conn.channels),
        })

        # Message processing loop
        async for message in websocket.iter_text():
            conn.refresh_heartbeat()
            await handle_websocket_message(conn, message)

    except WebSocketDisconnect:
        logger.info("ws_disconnected", user_id=user_id)
    except Exception as exc:
        logger.error("ws_error", user_id=user_id, error=str(exc))
    finally:
        # Cleanup on disconnect
        conn_to_remove = None
        for cid, c in connection_manager._connections.items():
            if c.user_id == user_id and c.websocket == websocket:
                conn_to_remove = cid
                break
        if conn_to_remove:
            await connection_manager.disconnect(conn_to_remove)
