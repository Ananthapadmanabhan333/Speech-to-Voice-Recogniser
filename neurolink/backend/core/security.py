from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from neurolink.backend.core.config import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Centralised security operations – JWT, hashing, rate-limiting, API keys."""

    # ── Password hashing ──────────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        return _pwd_ctx.hash(password, rounds=settings.BCRYPT_ROUNDS)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return _pwd_ctx.verify(plain, hashed)

    # ── JWT ───────────────────────────────────────────────────────────────

    @staticmethod
    def create_access_token(
        subject: str | int,
        extra_claims: dict[str, Any] | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        delta = expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": str(subject),
            "iat": now,
            "exp": now + delta,
            "type": "access",
        }
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def create_refresh_token(subject: str | int) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(subject),
            "iat": now,
            "exp": now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            "type": "refresh",
        }
        return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
            return payload
        except JWTError:
            return {}

    @staticmethod
    def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:
        payload = SecurityManager.decode_token(token)
        if not payload or payload.get("type") != expected_type:
            return {}
        return payload

    @staticmethod
    def is_token_valid(token: str) -> bool:
        return bool(SecurityManager.verify_token(token))

    @staticmethod
    def get_subject_from_token(token: str) -> str | None:
        payload = SecurityManager.decode_token(token)
        return payload.get("sub") if payload else None

    @staticmethod
    def refresh_access_token(refresh_token: str) -> str | None:
        payload = SecurityManager.verify_token(refresh_token, "refresh")
        if not payload:
            return None
        return SecurityManager.create_access_token(subject=payload["sub"])

    # ── API keys ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_api_key() -> str:
        return f"nlk_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def verify_api_key(api_key: str, hashed: str) -> bool:
        expected = SecurityManager.hash_api_key(api_key)
        return hmac.compare_digest(expected, hashed)

    # ── Rate limiting helpers ─────────────────────────────────────────────

    @staticmethod
    def build_rate_limit_key(identifier: str, prefix: str = "rl") -> str:
        return f"{prefix}:{identifier}"

    # ── WebSocket auth ────────────────────────────────────────────────────

    @staticmethod
    def authenticate_websocket(token: str | None) -> str | None:
        if not token:
            return None
        payload = SecurityManager.verify_token(token, "access")
        return payload.get("sub") if payload else None

    @staticmethod
    def generate_ws_token(user_id: str | int) -> str:
        return SecurityManager.create_access_token(
            subject=user_id,
            extra_claims={"ws": True},
            expires_delta=timedelta(hours=1),
        )

    # ── CSRF / nonce ──────────────────────────────────────────────────────

    @staticmethod
    def generate_nonce() -> str:
        return secrets.token_hex(16)

    @staticmethod
    def sign_payload(payload: str, secret: str | None = None) -> str:
        key = (secret or settings.JWT_SECRET).encode()
        return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()

    # ── Input sanitisation helpers ────────────────────────────────────────

    @staticmethod
    def sanitize_token_header(header: str | None) -> str | None:
        if not header:
            return None
        if header.startswith("Bearer "):
            return header[7:]
        return header
