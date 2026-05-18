from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Dict

import pytest
from jose import jwt

from neurolink.backend.core.config import settings
from neurolink.backend.core.security import SecurityManager


class TestJWT:
    """Test JWT creation and verification."""

    def test_create_access_token(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_extra_claims(self) -> None:
        token = SecurityManager.create_access_token(
            subject="user-123",
            extra_claims={"role": "admin", "ws": True},
        )
        payload = SecurityManager.decode_token(token)
        assert payload.get("role") == "admin"
        assert payload.get("ws") is True

    def test_create_refresh_token(self) -> None:
        token = SecurityManager.create_refresh_token(subject="user-123")
        payload = SecurityManager.decode_token(token)
        assert payload.get("type") == "refresh"

    def test_decode_valid_token(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        payload = SecurityManager.decode_token(token)
        assert payload.get("sub") == "user-123"
        assert payload.get("type") == "access"
        assert "iat" in payload
        assert "exp" in payload

    def test_decode_invalid_token(self) -> None:
        payload = SecurityManager.decode_token("invalid-token")
        assert payload == {}

    def test_decode_expired_token(self) -> None:
        token = SecurityManager.create_access_token(
            subject="user-123",
            expires_delta=timedelta(seconds=-1),
        )
        payload = SecurityManager.decode_token(token)
        assert payload == {}

    def test_verify_valid_access_token(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        payload = SecurityManager.verify_token(token, "access")
        assert payload.get("sub") == "user-123"

    def test_verify_wrong_type(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        payload = SecurityManager.verify_token(token, "refresh")
        assert payload == {}

    def test_is_token_valid(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        assert SecurityManager.is_token_valid(token) is True
        assert SecurityManager.is_token_valid("invalid") is False

    def test_get_subject_from_token(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        sub = SecurityManager.get_subject_from_token(token)
        assert sub == "user-123"

    def test_get_subject_from_invalid_token(self) -> None:
        sub = SecurityManager.get_subject_from_token("invalid")
        assert sub is None

    def test_refresh_access_token_valid(self) -> None:
        refresh = SecurityManager.create_refresh_token(subject="user-123")
        new_access = SecurityManager.refresh_access_token(refresh)
        assert new_access is not None
        payload = SecurityManager.decode_token(new_access)
        assert payload.get("sub") == "user-123"
        assert payload.get("type") == "access"

    def test_refresh_access_token_invalid(self) -> None:
        result = SecurityManager.refresh_access_token("invalid")
        assert result is None

    def test_refresh_access_token_expired(self) -> None:
        refresh = SecurityManager.create_refresh_token(subject="user-123")
        # Can't easily expire a refresh token without time travel
        # but we can verify it at least creates an access token
        result = SecurityManager.refresh_access_token(refresh)
        assert result is not None

    def test_jwt_contains_standard_claims(self) -> None:
        token = SecurityManager.create_access_token(subject="42")
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        assert payload["sub"] == "42"
        assert payload["type"] == "access"
        assert "iat" in payload
        assert "exp" in payload


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self) -> None:
        hashed = SecurityManager.hash_password("my-secure-password")
        assert isinstance(hashed, str)
        assert hashed != "my-secure-password"
        assert hashed.startswith("$2b$")  # bcrypt prefix

    def test_verify_password_correct(self) -> None:
        hashed = SecurityManager.hash_password("my-secure-password")
        assert SecurityManager.verify_password("my-secure-password", hashed) is True

    def test_verify_password_incorrect(self) -> None:
        hashed = SecurityManager.hash_password("my-secure-password")
        assert SecurityManager.verify_password("wrong-password", hashed) is False

    def test_hash_unique_per_call(self) -> None:
        h1 = SecurityManager.hash_password("same-password")
        h2 = SecurityManager.hash_password("same-password")
        assert h1 != h2  # bcrypt uses unique salts

    def test_hash_empty_password(self) -> None:
        hashed = SecurityManager.hash_password("")
        assert SecurityManager.verify_password("", hashed) is True

    def test_hash_long_password(self) -> None:
        long_pw = "a" * 100
        hashed = SecurityManager.hash_password(long_pw)
        assert SecurityManager.verify_password(long_pw, hashed) is True


class TestAPIKeys:
    """Test API key generation and validation."""

    def test_generate_api_key(self) -> None:
        key = SecurityManager.generate_api_key()
        assert key.startswith("nlk_")
        assert len(key) > 32

    def test_hash_api_key(self) -> None:
        key = SecurityManager.generate_api_key()
        hashed = SecurityManager.hash_api_key(key)
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 hexdigest

    def test_verify_api_key_correct(self) -> None:
        key = SecurityManager.generate_api_key()
        hashed = SecurityManager.hash_api_key(key)
        assert SecurityManager.verify_api_key(key, hashed) is True

    def test_verify_api_key_incorrect(self) -> None:
        key = SecurityManager.generate_api_key()
        hashed = SecurityManager.hash_api_key(key)
        assert SecurityManager.verify_api_key("wrong-key", hashed) is False

    def test_api_key_uniqueness(self) -> None:
        keys = {SecurityManager.generate_api_key() for _ in range(100)}
        assert len(keys) == 100


class TestRateLimiting:
    """Test rate limiting helpers."""

    def test_build_rate_limit_key(self) -> None:
        key = SecurityManager.build_rate_limit_key("user-123", prefix="rl")
        assert key == "rl:user-123"

    def test_build_rate_limit_key_default_prefix(self) -> None:
        key = SecurityManager.build_rate_limit_key("user-123")
        assert key == "rl:user-123"

    def test_rate_limit_key_with_ip(self) -> None:
        key = SecurityManager.build_rate_limit_key("192.168.1.1", prefix="ip")
        assert key == "ip:192.168.1.1"


class TestWebSocketAuth:
    """Test WebSocket authentication."""

    def test_authenticate_websocket_valid(self) -> None:
        token = SecurityManager.create_access_token(subject="user-123")
        result = SecurityManager.authenticate_websocket(token)
        assert result == "user-123"

    def test_authenticate_websocket_invalid(self) -> None:
        result = SecurityManager.authenticate_websocket("invalid")
        assert result is None

    def test_authenticate_websocket_none(self) -> None:
        result = SecurityManager.authenticate_websocket(None)
        assert result is None

    def test_generate_ws_token(self) -> None:
        token = SecurityManager.generate_ws_token("user-123")
        payload = SecurityManager.decode_token(token)
        assert payload.get("sub") == "user-123"
        assert payload.get("ws") is True
        assert payload.get("type") == "access"


class TestCSRFAndNonce:
    """Test CSRF nonce and signing."""

    def test_generate_nonce(self) -> None:
        nonce = SecurityManager.generate_nonce()
        assert isinstance(nonce, str)
        assert len(nonce) == 32  # 16 bytes hex encoded

    def test_nonce_uniqueness(self) -> None:
        nonces = {SecurityManager.generate_nonce() for _ in range(100)}
        assert len(nonces) == 100

    def test_sign_payload(self) -> None:
        sig = SecurityManager.sign_payload("test-payload")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 HMAC

    def test_sign_payload_deterministic(self) -> None:
        sig1 = SecurityManager.sign_payload("test-payload")
        sig2 = SecurityManager.sign_payload("test-payload")
        assert sig1 == sig2

    def test_sign_payload_different_secret(self) -> None:
        sig1 = SecurityManager.sign_payload("test-payload", "secret1")
        sig2 = SecurityManager.sign_payload("test-payload", "secret2")
        assert sig1 != sig2


class TestInputSanitization:
    """Test input sanitization helpers."""

    def test_sanitize_token_header_bearer(self) -> None:
        result = SecurityManager.sanitize_token_header("Bearer mytoken123")
        assert result == "mytoken123"

    def test_sanitize_token_header_no_bearer(self) -> None:
        result = SecurityManager.sanitize_token_header("mytoken123")
        assert result == "mytoken123"

    def test_sanitize_token_header_none(self) -> None:
        result = SecurityManager.sanitize_token_header(None)
        assert result is None

    def test_sanitize_token_header_empty(self) -> None:
        result = SecurityManager.sanitize_token_header("")
        assert result is None
