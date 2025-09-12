"""
Pydantic schemas for export-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ExportFormat(str, Enum):
    COCO = "COCO"
    YOLO = "YOLO"
    PASCAL_VOC = "PASCAL_VOC"
    CSV = "CSV"
    JSON = "JSON"


class ExportType(str, Enum):
    ANNOTATIONS = "annotations"
    IMAGES = "images"
    DATASET = "dataset"
    MODELS = "models"
    EVALUATION_RESULTS = "evaluation_results"


class ExportStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnnotationExportRequest(BaseModel):
    format: ExportFormat
    image_ids: Optional[List[str]] = None
    annotation_ids: Optional[List[str]] = None
    dataset_split: Optional[str] = None  # train, validation, test
    include_images: bool = False
    include_metadata: bool = True
    class_mapping: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetExportRequest(BaseModel):
    dataset_id: str
    format: ExportFormat
    include_images: bool = True
    include_annotations: bool = True
    splits_to_include: Optional[List[str]] = ["train", "validation", "test"]
    class_mapping: Optional[Dict[str, str]] = None
    compression: Optional[str] = "zip"  # zip, tar, tar.gz
    metadata: Optional[Dict[str, Any]] = None


class EvaluationExportRequest(BaseModel):
    evaluation_ids: List[str] = Field(..., min_items=1)
    format: ExportFormat = ExportFormat.JSON
    include_metrics: bool = True
    include_confusion_matrix: bool = True
    include_class_metrics: bool = True
    include_performance_data: bool = True
    metadata: Optional[Dict[str, Any]] = None


class ModelExportRequest(BaseModel):
    model_id: str
    export_weights: bool = True
    export_config: bool = True
    export_metadata: bool = True
    format: Optional[str] = "pytorch"  # pytorch, onnx, tensorrt
    optimization_level: Optional[str] = "none"  # none, basic, aggressive
    target_platform: Optional[str] = None  # cpu, gpu, mobile
    metadata: Optional[Dict[str, Any]] = None


class ExportFileInfo(BaseModel):
    filename: str
    file_size_bytes: int
    file_path: str
    checksum: str
    content_type: str


class ExportJobResponse(BaseModel):
    job_id: str
    export_type: ExportType
    format: ExportFormat
    status: ExportStatus
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    download_url: Optional[str] = None
    expires_at: Optional[str] = None
    file_info: Optional[ExportFileInfo] = None
    total_items: Optional[int] = None
    processed_items: Optional[int] = None
    error_message: Optional[str] = None
    estimated_completion_time: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExportJobListResponse(BaseModel):
    jobs: List[ExportJobResponse]
    total_count: int
    limit: int
    offset: int


class ExportStatsResponse(BaseModel):
    total_exports: int
    successful_exports: int
    failed_exports: int
    total_files_size_bytes: int
    most_popular_format: str
    exports_by_format: Dict[str, int]
    exports_by_type: Dict[str, int]


class BulkExportRequest(BaseModel):
    export_requests: List[Dict[str, Any]] = Field(..., min_items=1, max_items=20)
    batch_name: Optional[str] = None
    notification_email: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BulkExportResponse(BaseModel):
    batch_id: str
    total_jobs: int
    created_jobs: List[str]  # List of job IDs
    failed_requests: List[Dict[str, Any]]
    created_at: str
    metadata: Optional[Dict[str, Any]] = None