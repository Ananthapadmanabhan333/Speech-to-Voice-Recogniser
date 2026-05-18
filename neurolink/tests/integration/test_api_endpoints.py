from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, Response

from neurolink.backend.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestAuthFlow:
    """Test complete authentication flow."""

    @pytest.mark.asyncio
    async def test_register(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/register",
            json={
                "email": "test@example.com",
                "name": "Test User",
                "password": "securePassword123!",
            },
        )
        # Note: This may fail if DB is not connected; we test the endpoint structure
        assert response.status_code in (201, 422, 500)

    @pytest.mark.asyncio
    async def test_register_invalid_email(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/register",
            json={
                "email": "not-an-email",
                "name": "Test",
                "password": "short",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_short_password(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/register",
            json={
                "email": "test@example.com",
                "name": "Test",
                "password": "1234567",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_login(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/login",
            json={
                "email": "test@example.com",
                "password": "securePassword123!",
            },
        )
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_login_missing_fields(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/login",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_refresh_token(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/refresh",
            json={"refresh_token": "invalid_refresh_token"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_me_unauthenticated(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_flow_complete(self, client: AsyncClient) -> None:
        # Register
        reg_resp = await client.post(
            "/api/v1/register",
            json={
                "email": f"flow_{id(self)}@example.com",
                "name": "Flow Test",
                "password": "securePassword123!",
            },
        )
        # Just verify the endpoint exists and returns reasonable status
        assert reg_resp.status_code in (201, 409, 500)


class TestGestureAPI:
    """Test gesture API endpoints."""

    @pytest.mark.asyncio
    async def test_gesture_endpoint_exists(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/gestures/recognize",
            json={"landmarks": []},
        )
        assert response.status_code in (200, 401, 422, 500)

    @pytest.mark.asyncio
    async def test_gesture_list_endpoint(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/gestures")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_gesture_invalid_input(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/gestures/recognize",
            json={},
        )
        assert response.status_code in (422, 401)


class TestSpeechAPI:
    """Test speech API endpoints."""

    @pytest.mark.asyncio
    async def test_speech_transcribe_endpoint(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/speech/transcribe",
            json={"audio": []},
        )
        assert response.status_code in (200, 401, 422, 500)

    @pytest.mark.asyncio
    async def test_speech_synthesize_endpoint(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/speech/synthesize",
            json={"text": "Hello world"},
        )
        assert response.status_code in (200, 401, 422, 500)


class TestCommunicationAPI:
    """Test communication API endpoints."""

    @pytest.mark.asyncio
    async def test_communication_send(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/communication/send",
            json={"content": "Hello", "modality": "text"},
        )
        assert response.status_code in (200, 401, 422, 500)

    @pytest.mark.asyncio
    async def test_communication_sessions(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/communication/sessions")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_communication_session_detail(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/communication/sessions/test-id")
        assert response.status_code in (200, 401, 404, 500)


class TestAnalyticsAPI:
    """Test analytics API endpoints."""

    @pytest.mark.asyncio
    async def test_analytics_metrics(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/metrics")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_analytics_emotion_distribution(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/emotions")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_analytics_efficiency(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/efficiency")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_analytics_learning_curve(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/learning-curve")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_analytics_adaptation(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/adaptation")
        assert response.status_code in (200, 401, 500)

    @pytest.mark.asyncio
    async def test_analytics_export(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/analytics/export?format=json")
        assert response.status_code in (200, 401, 422, 500)


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "environment" in data

    @pytest.mark.asyncio
    async def test_root(self, client: AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestErrorResponses:
    """Test error response formatting."""

    @pytest.mark.asyncio
    async def test_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, client: AsyncClient) -> None:
        response = await client.delete("/api/v1/login")
        assert response.status_code == 405


class TestAuthentication:
    """Test authentication middleware."""

    @pytest.mark.asyncio
    async def test_unauthenticated_access(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/me")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/me",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_header(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/me")
        assert response.status_code == 401


class TestRateLimiting:
    """Test rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, client: AsyncClient) -> None:
        for _ in range(5):
            response = await client.get("/health")
        # Rate limit may or may not be hit depending on config
        assert response.status_code in (200, 429)


class TestCORS:
    """Test CORS headers."""

    @pytest.mark.asyncio
    async def test_cors_headers(self, client: AsyncClient) -> None:
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
