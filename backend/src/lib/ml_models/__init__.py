"""
ML Models Library (T045)

This library provides unified access to ML models including:
- YOLO11/12 variants (nano, small, medium, large, extra-large)
- RT-DETR variants (R18, R34, R50, R101, RF-DETR Nano/Small/Medium)
- SAM2 variants (Tiny, Small, Base+, Large)

Features:
- Model loading and configuration
- Metadata management
- Model variant support
- Unified interface for all model types
"""

from .models import (
    ModelRegistry,
    ModelVariant,
    ModelType,
    ModelConfig,
    BaseModelWrapper,
    YOLOModelWrapper,
    RTDETRModelWrapper,
    SAM2ModelWrapper,
)
from .registry import get_model_registry
from .factory import ModelFactory

__all__ = [
    "ModelRegistry",
    "ModelVariant", 
    "ModelType",
    "ModelConfig",
    "BaseModelWrapper",
    "YOLOModelWrapper",
    "RTDETRModelWrapper", 
    "SAM2ModelWrapper",
    "get_model_registry",
    "ModelFactory",
]