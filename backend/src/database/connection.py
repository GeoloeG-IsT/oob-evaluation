"""
Database connection module for ML Evaluation Platform.

Provides Supabase connection management with SQLAlchemy, connection pooling,
async operations, and robust error handling with retry logic.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import os
from contextlib import asynccontextmanager
import time

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from supabase import create_client, Client
import structlog

# Configure structured logger for database operations
logger = structlog.get_logger("database.connection")


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseConfig:
    """Database configuration management."""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
        
        # Retry settings
        self.max_retries = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("DB_RETRY_DELAY", "1.0"))
        self.retry_backoff = float(os.getenv("DB_RETRY_BACKOFF", "2.0"))
        
    def _get_database_url(self) -> str:
        """Get database URL with async driver."""
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        
        # Convert to async driver if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        
        return database_url


class DatabaseConnection:
    """
    Database connection manager with connection pooling, 
    async operations, and error handling.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self._supabase_client = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            logger.info("Database connection already initialized")
            return
            
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Validate connections before use
                echo=os.getenv("LOG_LEVEL") == "DEBUG",  # Log SQL in debug mode
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Initialize Supabase client if credentials are available
            if self.config.supabase_url and self.config.supabase_service_key:
                self._supabase_client = create_client(
                    self.config.supabase_url,
                    self.config.supabase_service_key
                )
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info(
                "Database connection initialized successfully",
                database_url=self.config.database_url.split("@")[1] if "@" in self.config.database_url else "***",
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize database connection",
                error=str(e),
                database_url=self.config.database_url.split("@")[1] if "@" in self.config.database_url else "***",
            )
            raise
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        async with self.session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            logger.debug("Database connection test successful")
    
    async def close(self) -> None:
        """Close database connection and clean up resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get database session with automatic transaction management and error handling.
        
        Usage:
            async with db.get_session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()
    
    async def execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute database operation with retry logic for transient failures.
        
        Args:
            operation: Async function to execute
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result of the operation
            
        Raises:
            SQLAlchemyError: After max retries exceeded
        """
        last_exception = None
        retry_delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
                
            except (DisconnectionError, ConnectionError) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    logger.warning(
                        "Database connection error, retrying",
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        error=str(e),
                        retry_delay=retry_delay,
                    )
                    
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.config.retry_backoff
                else:
                    logger.error(
                        "Database connection failed after max retries",
                        max_retries=self.config.max_retries,
                        error=str(e),
                    )
                    
            except SQLAlchemyError as e:
                # Don't retry for non-connection errors
                logger.error("Database operation failed", error=str(e))
                raise
        
        if last_exception:
            raise last_exception
    
    def get_supabase_client(self) -> Optional[Client]:
        """Get Supabase client instance."""
        return self._supabase_client
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information for monitoring."""
        if not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "status": "connected" if self._initialized else "disconnected",
            "pool_size": pool.size(),
            "pool_checked_in": pool.checkedin(),
            "pool_checked_out": pool.checkedout(),
            "pool_overflow": pool.overflow(),
            "engine_url": str(self.engine.url).split("@")[1] if "@" in str(self.engine.url) else "***",
        }


# Global database connection instance
db_connection = DatabaseConnection()


async def init_db() -> None:
    """Initialize database connection (called on application startup)."""
    await db_connection.initialize()


async def close_db() -> None:
    """Close database connection (called on application shutdown)."""
    await db_connection.close()


@asynccontextmanager
async def get_db_session():
    """
    Dependency for getting database session in FastAPI endpoints.
    
    Usage:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(query)
    """
    async with db_connection.get_session() as session:
        yield session


def get_supabase() -> Optional[Client]:
    """Get Supabase client dependency for FastAPI endpoints."""
    return db_connection.get_supabase_client()


async def health_check() -> Dict[str, Any]:
    """Database health check for monitoring endpoints."""
    try:
        connection_info = await db_connection.get_connection_info()
        
        # Test a simple query if connected
        if connection_info.get("status") == "connected":
            start_time = time.time()
            async with db_connection.get_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            query_time = time.time() - start_time
            
            connection_info["query_time_ms"] = round(query_time * 1000, 2)
        
        return {
            "database": connection_info,
            "supabase": {
                "available": db_connection.get_supabase_client() is not None,
                "url_configured": bool(db_connection.config.supabase_url),
                "key_configured": bool(db_connection.config.supabase_service_key),
            }
        }
        
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return {
            "database": {"status": "error", "error": str(e)},
            "supabase": {"available": False}
        }


# Import text here to avoid circular imports
from sqlalchemy import text