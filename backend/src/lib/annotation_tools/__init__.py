"""
Annotation Tools Library (T048)

This library provides assisted annotation capabilities including:
- Assisted annotation using pre-trained models
- Integration with SAM2 for segmentation assistance
- Confidence threshold handling
- Format conversion utilities
- Manual annotation validation
- Batch annotation processing

Features:
- SAM2-based segmentation assistance
- YOLO/RT-DETR detection assistance
- Interactive annotation refinement
- Confidence-based filtering
- Multi-format annotation export
- Annotation quality validation
- Batch processing capabilities
"""

from .assistants import (
    AnnotationAssistant,
    SAM2Assistant,
    DetectionAssistant,
    AssistantConfig,
    AssistanceRequest,
    AssistanceResult,
)
from .tools import (
    AnnotationTool,
    BoundingBoxTool,
    SegmentationTool,
    PolygonTool,
    PointTool,
    AnnotationValidator,
)
from .converters import (
    AnnotationConverter,
    COCOConverter,
    YOLOConverter,
    PascalVOCConverter,
    FormatConverter,
)
from .processors import (
    BatchAnnotationProcessor,
    AnnotationPipeline,
    ProcessingConfig,
    QualityChecker,
)
from .utils import (
    AnnotationUtils,
    GeometryUtils,
    VisualizationUtils,
    StatisticsCalculator,
)

__all__ = [
    "AnnotationAssistant",
    "SAM2Assistant", 
    "DetectionAssistant",
    "AssistantConfig",
    "AssistanceRequest",
    "AssistanceResult",
    "AnnotationTool",
    "BoundingBoxTool",
    "SegmentationTool",
    "PolygonTool", 
    "PointTool",
    "AnnotationValidator",
    "AnnotationConverter",
    "COCOConverter",
    "YOLOConverter",
    "PascalVOCConverter",
    "FormatConverter",
    "BatchAnnotationProcessor",
    "AnnotationPipeline",
    "ProcessingConfig",
    "QualityChecker",
    "AnnotationUtils",
    "GeometryUtils",
    "VisualizationUtils",
    "StatisticsCalculator",
]