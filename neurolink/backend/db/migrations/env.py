"""Alembic migration environment configuration for async PostgreSQL.

This module configures Alembic to work with SQLAlchemy's async engine,
auto-detects model changes from the declarative Base, and supports
both online and offline migration generation.
"""

from __future__ import annotations

import asyncio
import re
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from neurolink.backend.core.config import settings
from neurolink.backend.db.models import Base

# ── Alembic Config ─────────────────────────────────────────────────────────

config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from application settings
database_url = str(settings.DATABASE_URL)
config.set_main_option("sqlalchemy.url", database_url)

# Metadata for autogenerate support
target_metadata = Base.metadata

# ── Naming convention for constraints ─────────────────────────────────────

target_metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# ── Exclude tables from autogenerate ──────────────────────────────────────

exclude_tables: list[str] = []


def include_object(obj: Any, name: str, type_: str, *args: Any, **kwargs: Any) -> bool:
    """Filter objects for autogenerate. Excludes specified tables."""
    if type_ == "table" and name in exclude_tables:
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.
    Calls to context.execute() emit the SQL to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an async engine and runs migrations within a connection.
    Supports both sync and async driver URLs.
    """
    # Handle the case where URL might be sync (replace driver for async)
    url = config.get_main_option("sqlalchemy.url")

    # Ensure we use an async driver
    if "+asyncpg" not in url:
        url = re.sub(r"postgresql://", "postgresql+asyncpg://", url)

    connectable: AsyncEngine = create_async_engine(
        url,
        poolclass=pool.NullPool,
        future=True,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def process_revision_directives(context: Any, revision: Any, directives: Any) -> None:
    """Add custom header to generated migration files."""
    if config.cmd_opts and getattr(config.cmd_opts, "autogenerate", False):
        script = directives[0]
        if script.upgrade_ops.is_empty():
            directives[:] = []
            print("No changes detected — skipping migration generation.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
