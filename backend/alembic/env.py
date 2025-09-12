"""
Alembic environment configuration for ML Evaluation Platform.

This module configures Alembic for database migrations with support for:
- Async SQLAlchemy operations
- Environment variable configuration
- PostgreSQL compatibility
- Structured logging integration
"""

import asyncio
import os
from logging.config import fileConfig
from typing import Optional

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import Connection

from alembic import context
import structlog

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Import models to ensure they're registered with SQLAlchemy
from src.database.connection import Base

# Import all model modules to register with Base
try:
    from src.models import (
        image,
        annotation,
        model,
        dataset,
        training_job,
        inference_job,
        performance_metric,
        deployment,
    )
except ImportError:
    # Models might not be converted to SQLAlchemy yet
    pass

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Configure structured logging for migrations
logger = structlog.get_logger("alembic.migrations")

# Add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """
    Get database URL from environment variables.

    Returns:
        Database URL with async driver
    """
    # Try to get from environment first
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        # Fallback to config file
        database_url = config.get_main_option("sqlalchemy.url")

    if not database_url:
        raise ValueError(
            "Database URL not found. Set DATABASE_URL environment variable "
            "or configure sqlalchemy.url in alembic.ini"
        )

    # Convert to async driver for migrations
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

    return database_url


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()

    # Remove async driver for offline mode
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect server default changes
        render_as_batch=False,  # Set to True for SQLite compatibility
    )

    logger.info(
        "Running migrations in offline mode",
        database_url=url.split("@")[1] if "@" in url else "***",
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Detect column type changes
        compare_server_default=True,  # Detect server default changes
        render_as_batch=False,  # Set to True for SQLite compatibility
        # Custom migration options
        transaction_per_migration=True,  # Use separate transaction per migration
        # Include object filters for better control
        include_object=include_object,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def include_object(
    object, name: str, type_: str, reflected: bool, compare_to: Optional[object]
) -> bool:
    """
    Filter objects to include in migrations.

    This function allows filtering of database objects during autogenerate.
    Return True to include the object, False to exclude it.
    """
    # Include all objects by default
    if type_ == "table":
        # Exclude any system or temporary tables
        if name.startswith("_") or name.startswith("temp_"):
            logger.debug("Excluding table from migrations", table_name=name)
            return False

    return True


def include_name(name: Optional[str], type_: str, parent_names: dict) -> bool:
    """
    Filter names to include in migrations.

    This function allows filtering of database object names during autogenerate.
    Return True to include the name, False to exclude it.
    """
    if name is None:
        return True

    # Exclude system schemas and objects
    if type_ == "schema" and name in ("information_schema", "pg_catalog", "pg_toast"):
        return False

    # Exclude system tables and views
    if type_ in ("table", "view") and name.startswith(("pg_", "information_schema")):
        return False

    return True


async def run_async_migrations() -> None:
    """Run migrations in async mode using asyncpg."""
    database_url = get_database_url()

    logger.info(
        "Starting async migrations",
        database_url=database_url.split("@")[1] if "@" in database_url else "***",
    )

    # Create async engine for migrations
    connectable = create_async_engine(
        database_url,
        poolclass=pool.NullPool,  # Don't use connection pooling for migrations
        echo=os.getenv("LOG_LEVEL") == "DEBUG",
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()
    logger.info("Async migrations completed successfully")


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Run async migrations
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
