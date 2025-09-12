"""
Pydantic schemas for evaluation-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MetricType(str, Enum):
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAP = "mAP"
    MAP50 = "mAP@0.5"
    MAP75 = "mAP@0.75"
    IOU = "IoU"
    ACCURACY = "accuracy"
    INFERENCE_TIME = "inference_time_ms"
    MEMORY_USAGE = "memory_usage_mb"


class EvaluationRequest(BaseModel):
    model_id: str
    dataset_id: str
    ground_truth_annotations: List[str]  # Annotation IDs
    predicted_annotations: List[str]  # Annotation IDs
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    metrics_to_calculate: Optional[List[MetricType]] = None
    metadata: Optional[Dict[str, Any]] = None


class ClassMetrics(BaseModel):
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    support: int  # Number of true instances


class BoundingBoxMetrics(BaseModel):
    iou_threshold: float
    map: float
    map50: float
    map75: float
    precision: float
    recall: float
    f1_score: float


class SegmentationMetrics(BaseModel):
    iou_threshold: float
    mean_iou: float
    pixel_accuracy: float
    dice_coefficient: float
    precision: float
    recall: float
    f1_score: float


class PerformanceMetrics(BaseModel):
    inference_time_ms: float
    memory_usage_mb: float
    throughput_fps: float
    model_size_mb: float


class EvaluationResult(BaseModel):
    evaluation_id: str
    model_id: str
    dataset_id: str
    overall_metrics: Dict[str, float]
    class_metrics: List[ClassMetrics]
    bounding_box_metrics: Optional[BoundingBoxMetrics] = None
    segmentation_metrics: Optional[SegmentationMetrics] = None
    performance_metrics: PerformanceMetrics
    confusion_matrix: Optional[List[List[int]]] = None
    created_at: str
    execution_time_seconds: float
    total_images_evaluated: int
    metadata: Optional[Dict[str, Any]] = None


class EvaluationListResponse(BaseModel):
    evaluations: List[EvaluationResult]
    total_count: int
    limit: int
    offset: int


class MetricComparisonRequest(BaseModel):
    evaluation_ids: List[str] = Field(..., min_items=2, max_items=10)
    metric_types: Optional[List[MetricType]] = None


class ModelComparison(BaseModel):
    model_id: str
    model_name: str
    metrics: Dict[str, float]
    rank: int


class MetricComparisonResponse(BaseModel):
    comparison_id: str
    metric_comparisons: Dict[str, List[ModelComparison]]
    created_at: str
    evaluation_count: int