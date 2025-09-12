"""
Pydantic schemas for model-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModelFramework(str, Enum):
    YOLO11 = "YOLO11"
    YOLO12 = "YOLO12"
    RT_DETR = "RT-DETR"
    SAM2 = "SAM2"


class ModelType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class TrainingStatus(str, Enum):
    PRE_TRAINED = "pre-trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class ModelResponse(BaseModel):
    id: str
    name: str
    framework: ModelFramework
    type: ModelType
    variant: str
    description: Optional[str] = None
    version: str
    created_at: str
    training_status: TrainingStatus = TrainingStatus.PRE_TRAINED
    model_path: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    training_info: Optional[Dict[str, Any]] = None
    supported_formats: Optional[List[str]] = None
    class_labels: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_pretrained: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class ModelListResponse(BaseModel):
    models: List[ModelResponse]
    total_count: int
    limit: int
    offset: int


class ModelCreate(BaseModel):
    name: str
    framework: ModelFramework
    type: ModelType
    variant: str
    description: Optional[str] = None
    version: str
    model_path: str
    metadata: Optional[Dict[str, Any]] = None