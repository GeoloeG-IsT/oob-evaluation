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

from .routers import images, annotations

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 ML Evaluation Platform Backend starting...")
    yield
    # Shutdown
    print("🛑 ML Evaluation Platform Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="ML Evaluation Platform API",
    description="API for object detection and segmentation model evaluation platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
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