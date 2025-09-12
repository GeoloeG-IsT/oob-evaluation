"""
Pydantic schemas for training-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Hyperparameters(BaseModel):
    epochs: int = Field(..., ge=1, le=1000)
    batch_size: int = Field(..., ge=1, le=512)
    learning_rate: float = Field(..., gt=0.0, le=1.0)
    optimizer: str = Field("Adam", pattern="^(Adam|SGD|AdamW|RMSprop)$")
    weight_decay: Optional[float] = Field(0.0001, ge=0.0, le=1.0)
    momentum: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    patience: Optional[int] = Field(10, ge=1, le=100)
    warmup_epochs: Optional[int] = Field(0, ge=0, le=50)
    image_size: Optional[int] = Field(640, ge=128, le=2048)


class TrainingJobRequest(BaseModel):
    base_model_id: str
    dataset_id: str
    hyperparameters: Hyperparameters
    experiment_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float
    timestamp: str


class TrainingJobResponse(BaseModel):
    id: str
    base_model_id: str
    dataset_id: str
    status: TrainingStatus
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    hyperparameters: Hyperparameters
    created_at: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result_model_id: Optional[str] = None
    execution_logs: Optional[str] = None
    training_metrics: Optional[List[TrainingMetrics]] = []
    error_message: Optional[str] = None
    estimated_completion_time: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    best_metric_value: Optional[float] = None
    experiment_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingJobListResponse(BaseModel):
    jobs: List[TrainingJobResponse]
    total_count: int
    limit: int
    offset: int


class TrainingJobUpdateRequest(BaseModel):
    status: Optional[TrainingStatus] = None
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    execution_logs: Optional[str] = None
    error_message: Optional[str] = None