from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from neurolink.backend.core.config import settings
from neurolink.backend.core.logging import get_logger
from neurolink.backend.db.models import Base

logger = get_logger("db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def create_engine() -> AsyncEngine:
    """Create and configure the async SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        return _engine

    _engine = create_async_engine(
        str(settings.DATABASE_URL),
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_MAX_SIZE,
        max_overflow=settings.DB_POOL_MAX_SIZE // 2,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    logger.info("db_engine_created", url=str(settings.DATABASE_URL).replace("://", "://<redacted>@"))
    return _engine


async def create_session_factory() -> async_sessionmaker[AsyncSession]:
    """Build or retrieve the async session factory."""
    global _session_factory
    if _session_factory is not None:
        return _session_factory

    engine = await create_engine()
    _session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    logger.info("db_session_factory_created")
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    """FastAPI dependency – yield an async DB session."""
    factory = await get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global session factory, creating it if necessary."""
    if _session_factory is None:
        return await create_session_factory()
    return _session_factory


async def get_engine() -> AsyncEngine:
    """Return the global engine, creating it if necessary."""
    if _engine is None:
        return await create_engine()
    return _engine


async def init_db() -> None:
    """Create all tables (safe for first run / testing)."""
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db_tables_created")


async def drop_db() -> None:
    """Drop all tables (use with extreme caution)."""
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("db_tables_dropped")


async def health_check() -> dict[str, Any]:
    """Run a lightweight health check against the database."""
    result: dict[str, Any] = {"status": "unknown", "latency_ms": None}
    try:
        engine = await get_engine()
        start = __import__("time").monotonic()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        result["latency_ms"] = round((__import__("time").monotonic() - start) * 1000, 2)
        result["status"] = "healthy"
    except Exception as exc:
        result["status"] = "unhealthy"
        result["error"] = str(exc)
        logger.error("db_health_check_failed", error=str(exc))
    return result


async def close_db() -> None:
    """Dispose of the engine and release all connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("db_engine_disposed")


__all__ = [
    "Base",
    "create_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
    "init_db",
    "drop_db",
    "health_check",
    "close_db",
]
