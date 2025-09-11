"""
ML Evaluation Platform Backend API

FastAPI application for object detection and segmentation model evaluation.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from .routers import images, annotations, models, inference, training, evaluation, deployments, export
from .database.connection import init_db, close_db, health_check as db_health_check
from .utils.logging import configure_logging, LoggingMiddleware, get_logger, logging_health_check

# Load environment variables
load_dotenv()

# Configure structured logging
configure_logging()
logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("ML Evaluation Platform Backend starting")
    try:
        await init_db()
        logger.info("Database connection initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("ML Evaluation Platform Backend shutting down")
    try:
        await close_db()
        logger.info("Database connection closed successfully")
    except Exception as e:
        logger.error("Error closing database connection", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="ML Evaluation Platform API",
    description="API for object detection and segmentation model evaluation platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add logging middleware for request/response tracking
app.add_middleware(LoggingMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler to flatten error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # If detail is a dict with an error key, flatten it to match API contract
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            headers={"content-type": "application/json"}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)},
        headers={"content-type": "application/json"}
    )

# Include routers
app.include_router(images.router)
app.include_router(annotations.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(deployments.router)
app.include_router(export.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ML Evaluation Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Get database health
        db_health = await db_health_check()
        
        # Get logging health
        logging_health = logging_health_check()
        
        # Determine overall status
        overall_status = "healthy"
        if db_health.get("database", {}).get("status") != "connected":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "version": "1.0.0",
            "timestamp": logger._context.get("timestamp") if hasattr(logger, '_context') else None,
            "database": db_health,
            "logging": logging_health,
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "version": "1.0.0",
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )