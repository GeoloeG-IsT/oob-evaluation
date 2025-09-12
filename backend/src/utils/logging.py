"""
Structured logging configuration for ML Evaluation Platform.

Provides comprehensive logging with correlation IDs, request tracking, 
performance metrics, error logging, and integration with FastAPI and Celery.
"""

import logging
import sys
import time
import uuid
import json
import traceback
from typing import Any, Dict, Optional, Union, Callable
from contextlib import contextmanager
from contextvars import ContextVar
import os
from datetime import datetime, timezone

import structlog
from structlog.typing import FilteringBoundLogger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Context variables for correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class LoggingConfig:
    """Logging configuration management."""
    
    def __init__(self):
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.structured_logging = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
        self.log_format = os.getenv("LOG_FORMAT", "json")  # json or console
        self.include_source = os.getenv("LOG_INCLUDE_SOURCE", "false").lower() == "true"
        self.performance_logging = os.getenv("LOG_PERFORMANCE", "true").lower() == "true"
        
        # Service identification
        self.service_name = os.getenv("SERVICE_NAME", "ml-eval-platform-backend")
        self.service_version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")


def add_correlation_context(logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add correlation context to log events.
    
    This processor adds correlation_id, request_id, and user_id from context
    to every log event for request tracing.
    """
    # Add correlation context
    if correlation_id.get():
        event_dict["correlation_id"] = correlation_id.get()
    
    if request_id.get():
        event_dict["request_id"] = request_id.get()
        
    if user_id.get():
        event_dict["user_id"] = user_id.get()
    
    # Add service context
    config = LoggingConfig()
    event_dict.update({
        "service": config.service_name,
        "version": config.service_version,
        "environment": config.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    return event_dict


def add_performance_context(logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add performance metrics to log events when available."""
    # This will be populated by the performance tracking decorator
    if hasattr(logger, '_performance_metrics'):
        event_dict.update(logger._performance_metrics)
    
    return event_dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation context
        if correlation_id.get():
            log_data["correlation_id"] = correlation_id.get()
        if request_id.get():
            log_data["request_id"] = request_id.get()
        if user_id.get():
            log_data["user_id"] = user_id.get()
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage'
            }:
                log_data[key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Configure structured logging for the application."""
    config = LoggingConfig()
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        add_correlation_context,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.performance_logging:
        processors.append(add_performance_context)
    
    if config.structured_logging and config.log_format == "json":
        # JSON output for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ])
    else:
        # Console output for development
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    
    # Configure standard logging
    handler = logging.StreamHandler(sys.stdout)
    
    if config.structured_logging and config.log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    # Configure root logger
    logging.root.handlers = [handler]
    logging.root.setLevel(getattr(logging, config.log_level))
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if config.log_level == "DEBUG" else logging.WARNING
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """Set correlation ID in context and return it."""
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    
    correlation_id.set(corr_id)
    return corr_id


def set_request_id(req_id: Optional[str] = None) -> str:
    """Set request ID in context and return it."""
    if req_id is None:
        req_id = str(uuid.uuid4())
    
    request_id.set(req_id)
    return req_id


def set_user_id(uid: Optional[str]) -> None:
    """Set user ID in context."""
    user_id.set(uid)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id.get()


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id.get()


def get_user_id() -> Optional[str]:
    """Get current user ID from context."""
    return user_id.get()


@contextmanager
def correlation_context(corr_id: Optional[str] = None, req_id: Optional[str] = None, uid: Optional[str] = None):
    """
    Context manager for setting correlation context.
    
    Usage:
        with correlation_context("corr-123", "req-456", "user-789"):
            logger.info("This will include correlation context")
    """
    old_correlation_id = correlation_id.get()
    old_request_id = request_id.get()
    old_user_id = user_id.get()
    
    try:
        if corr_id:
            correlation_id.set(corr_id)
        if req_id:
            request_id.set(req_id)
        if uid:
            user_id.set(uid)
        
        yield
        
    finally:
        correlation_id.set(old_correlation_id)
        request_id.set(old_request_id)
        user_id.set(old_user_id)


def performance_timer(operation_name: str):
    """
    Decorator for tracking operation performance.
    
    Usage:
        @performance_timer("database_query")
        async def query_data():
            # operation
            return result
    """
    def decorator(func: Callable) -> Callable:
        if hasattr(func, '__wrapped__'):
            # Already wrapped, return original
            return func
            
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "Operation completed successfully",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    function=func.__name__,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info(
                    "Operation completed successfully",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    function=func.__name__,
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        import asyncio
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.__wrapped__ = func
        return wrapper
        
    return decorator


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request/response logging with correlation tracking.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("http.middleware")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate correlation and request IDs
        corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Extract user ID from headers if available
        uid = request.headers.get("X-User-ID")
        
        start_time = time.time()
        
        with correlation_context(corr_id, req_id, uid):
            # Log request
            self.logger.info(
                "Request started",
                method=request.method,
                url=str(request.url),
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                content_length=request.headers.get("Content-Length"),
            )
            
            try:
                response = await call_next(request)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log response
                self.logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                    response_size=response.headers.get("Content-Length"),
                )
                
                # Add correlation headers to response
                response.headers["X-Correlation-ID"] = corr_id
                response.headers["X-Request-ID"] = req_id
                
                return response
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.error(
                    "Request failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(duration_ms, 2),
                )
                raise


# Celery logging integration
def configure_celery_logging():
    """Configure structured logging for Celery workers."""
    from celery.utils.log import get_task_logger
    from celery.signals import before_task_publish, task_prerun, task_postrun, task_failure
    
    @before_task_publish.connect
    def task_sent_handler(sender=None, headers=None, body=None, **kwargs):
        """Log when task is sent."""
        logger = get_logger("celery.task")
        task_id = headers.get('id') if headers else None
        
        with correlation_context(task_id):
            logger.info(
                "Task sent",
                task_name=sender,
                task_id=task_id,
                **kwargs
            )
    
    @task_prerun.connect
    def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
        """Log before task execution."""
        logger = get_logger("celery.task")
        
        with correlation_context(task_id):
            logger.info(
                "Task started",
                task_name=sender.__name__ if sender else task.name,
                task_id=task_id,
                args=args,
                kwargs=kwargs,
            )
    
    @task_postrun.connect
    def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
        """Log after task execution."""
        logger = get_logger("celery.task")
        
        with correlation_context(task_id):
            logger.info(
                "Task completed",
                task_name=sender.__name__ if sender else task.name,
                task_id=task_id,
                state=state,
                result_type=type(retval).__name__ if retval is not None else None,
            )
    
    @task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
        """Log task failures."""
        logger = get_logger("celery.task")
        
        with correlation_context(task_id):
            logger.error(
                "Task failed",
                task_name=sender.__name__ if sender else None,
                task_id=task_id,
                error=str(exception),
                error_type=type(exception).__name__ if exception else None,
                traceback=str(traceback) if traceback else None,
            )


# Health check for logging system
def logging_health_check() -> Dict[str, Any]:
    """Health check for logging system."""
    try:
        logger = get_logger("health_check")
        logger.debug("Logging health check")
        
        config = LoggingConfig()
        
        return {
            "status": "healthy",
            "log_level": config.log_level,
            "structured_logging": config.structured_logging,
            "log_format": config.log_format,
            "service_name": config.service_name,
            "correlation_id": get_correlation_id(),
            "request_id": get_request_id(),
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }