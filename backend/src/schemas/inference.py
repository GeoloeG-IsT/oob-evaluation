"""
Pydantic schemas for inference-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class SingleInferenceRequest(BaseModel):
    image_id: str
    model_id: str
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    nms_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(100, ge=1, le=1000)
    metadata: Optional[Dict[str, Any]] = None


class BatchInferenceRequest(BaseModel):
    image_ids: List[str] = Field(..., min_items=1)
    model_id: str
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    nms_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0)
    batch_size: Optional[int] = Field(10, ge=1, le=100)
    max_detections: Optional[int] = Field(100, ge=1, le=1000)
    priority: Optional[Priority] = Priority.NORMAL
    metadata: Optional[Dict[str, Any]] = None


class Detection(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4)  # [x, y, width, height]
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Segmentation(BaseModel):
    mask: List[List[float]]  # Polygon points as [x, y] pairs
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    image_id: str
    detections: Optional[List[Detection]] = []
    segmentations: Optional[List[Segmentation]] = []
    inference_time_ms: float
    model_confidence: float
    timestamp: str


class SingleInferenceResponse(InferenceResult):
    pass


class InferenceJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    image_ids: List[str]
    model_id: str
    total_images: int
    processed_images: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    results: Optional[List[InferenceResult]] = []
    error_message: Optional[str] = None
    progress_percentage: Optional[float] = None
    estimated_completion_time: Optional[str] = None
    priority: Optional[Priority] = None
    metadata: Optional[Dict[str, Any]] = None


class InferenceJobListResponse(BaseModel):
    jobs: List[InferenceJobResponse]
    total_count: int
    limit: int
    offset: int