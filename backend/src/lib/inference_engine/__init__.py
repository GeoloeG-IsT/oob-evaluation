"""
Inference Engine Library (T046)

This library provides inference capabilities for ML models including:
- Real-time single image inference (< 2 seconds)
- Batch processing capabilities  
- Support for all model variants (YOLO11/12, RT-DETR, SAM2)
- Performance monitoring and result formatting
- Concurrent processing support

Features:
- Single image inference
- Batch image processing
- Inference job management
- Performance metrics collection
- Result formatting and standardization
- Memory-efficient processing
"""

from .engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
    BatchInferenceJob,
    InferenceStatus,
    PerformanceMetrics,
)
from .processors import (
    SingleImageProcessor,
    BatchImageProcessor,
    InferenceJobManager,
)
from .formatters import (
    ResultFormatter,
    COCOFormatter,
    YOLOFormatter,
    StandardFormatter,
)
from .monitoring import (
    PerformanceMonitor,
    InferenceMetrics,
)

__all__ = [
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResult", 
    "BatchInferenceJob",
    "InferenceStatus",
    "PerformanceMetrics",
    "SingleImageProcessor",
    "BatchImageProcessor",
    "InferenceJobManager",
    "ResultFormatter",
    "COCOFormatter",
    "YOLOFormatter",
    "StandardFormatter",
    "PerformanceMonitor",
    "InferenceMetrics",
]