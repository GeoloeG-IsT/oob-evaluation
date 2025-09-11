"""
Core model definitions and wrappers for ML models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid
from datetime import datetime, timezone


class ModelType(str, Enum):
    """Supported model types."""
    YOLO11 = "yolo11"
    YOLO12 = "yolo12"
    RT_DETR = "rt_detr"
    SAM2 = "sam2"


class ModelVariant(str, Enum):
    """Model variants for different model types."""
    # YOLO variants
    YOLO_NANO = "nano"
    YOLO_SMALL = "small"
    YOLO_MEDIUM = "medium"
    YOLO_LARGE = "large"
    YOLO_EXTRA_LARGE = "extra_large"
    
    # RT-DETR variants
    RT_DETR_R18 = "r18"
    RT_DETR_R34 = "r34"
    RT_DETR_R50 = "r50"
    RT_DETR_R101 = "r101"
    RT_DETR_RF_NANO = "rf_nano"
    RT_DETR_RF_SMALL = "rf_small"
    RT_DETR_RF_MEDIUM = "rf_medium"
    
    # SAM2 variants
    SAM2_TINY = "tiny"
    SAM2_SMALL = "small"
    SAM2_BASE_PLUS = "base_plus"
    SAM2_LARGE = "large"


@dataclass
class ModelConfig:
    """Configuration for a ML model."""
    model_id: str
    model_type: ModelType
    variant: ModelVariant
    name: str
    description: str
    input_size: tuple[int, int]  # (width, height)
    num_classes: int
    weights_path: Optional[str] = None
    config_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


class BaseModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model from weights."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def predict(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Make predictions on an image."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def model_id(self) -> str:
        """Get model ID."""
        return self.config.model_id


class YOLOModelWrapper(BaseModelWrapper):
    """Wrapper for YOLO11/12 models."""
    
    def load_model(self) -> None:
        """Load YOLO model."""
        try:
            # Placeholder for actual YOLO model loading
            # In real implementation, this would use ultralytics
            self._model = f"yolo_{self.config.variant}_model"
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model {self.config.model_id}: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload YOLO model."""
        self._model = None
        self._loaded = False
    
    def predict(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Make YOLO predictions."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        confidence = kwargs.get("confidence", 0.5)
        iou_threshold = kwargs.get("iou_threshold", 0.5)
        
        # Placeholder prediction result
        return {
            "model_id": self.config.model_id,
            "predictions": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 300]
                }
            ],
            "inference_time": 0.045,
            "image_path": image_path,
            "parameters": {
                "confidence": confidence,
                "iou_threshold": iou_threshold
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get YOLO model information."""
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "variant": self.config.variant,
            "name": self.config.name,
            "description": self.config.description,
            "input_size": self.config.input_size,
            "num_classes": self.config.num_classes,
            "performance_metrics": self.config.performance_metrics,
            "loaded": self._loaded
        }


class RTDETRModelWrapper(BaseModelWrapper):
    """Wrapper for RT-DETR models."""
    
    def load_model(self) -> None:
        """Load RT-DETR model."""
        try:
            # Placeholder for actual RT-DETR model loading
            self._model = f"rtdetr_{self.config.variant}_model"
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load RT-DETR model {self.config.model_id}: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload RT-DETR model."""
        self._model = None
        self._loaded = False
    
    def predict(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Make RT-DETR predictions."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        confidence = kwargs.get("confidence", 0.5)
        
        # Placeholder prediction result
        return {
            "model_id": self.config.model_id,
            "predictions": [
                {
                    "class_id": 1,
                    "class_name": "car",
                    "confidence": 0.92,
                    "bbox": [150, 50, 350, 200]
                }
            ],
            "inference_time": 0.038,
            "image_path": image_path,
            "parameters": {
                "confidence": confidence
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get RT-DETR model information."""
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "variant": self.config.variant,
            "name": self.config.name,
            "description": self.config.description,
            "input_size": self.config.input_size,
            "num_classes": self.config.num_classes,
            "performance_metrics": self.config.performance_metrics,
            "loaded": self._loaded
        }


class SAM2ModelWrapper(BaseModelWrapper):
    """Wrapper for SAM2 models."""
    
    def load_model(self) -> None:
        """Load SAM2 model."""
        try:
            # Placeholder for actual SAM2 model loading
            self._model = f"sam2_{self.config.variant}_model"
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model {self.config.model_id}: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload SAM2 model."""
        self._model = None
        self._loaded = False
    
    def predict(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Make SAM2 predictions."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        points = kwargs.get("points", [])
        boxes = kwargs.get("boxes", [])
        
        # Placeholder prediction result for segmentation
        return {
            "model_id": self.config.model_id,
            "predictions": [
                {
                    "mask": [[0, 0, 1, 1], [0, 1, 1, 0]],  # Simplified mask representation
                    "confidence": 0.88,
                    "area": 1250
                }
            ],
            "inference_time": 0.125,
            "image_path": image_path,
            "parameters": {
                "points": points,
                "boxes": boxes
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get SAM2 model information."""
        return {
            "model_id": self.config.model_id,
            "model_type": self.config.model_type,
            "variant": self.config.variant,
            "name": self.config.name,
            "description": self.config.description,
            "input_size": self.config.input_size,
            "num_classes": self.config.num_classes,
            "performance_metrics": self.config.performance_metrics,
            "loaded": self._loaded
        }


class ModelRegistry:
    """Registry for managing ML models."""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._loaded_models: Dict[str, BaseModelWrapper] = {}
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._models[config.model_id] = config
    
    def unregister_model(self, model_id: str) -> None:
        """Unregister a model."""
        if model_id in self._loaded_models:
            self._loaded_models[model_id].unload_model()
            del self._loaded_models[model_id]
        if model_id in self._models:
            del self._models[model_id]
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self._models.get(model_id)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelConfig]:
        """List all registered models, optionally filtered by type."""
        models = list(self._models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models
    
    def get_model_wrapper(self, model_id: str) -> Optional[BaseModelWrapper]:
        """Get a loaded model wrapper."""
        return self._loaded_models.get(model_id)
    
    def load_model(self, model_id: str) -> BaseModelWrapper:
        """Load a model and return its wrapper."""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        config = self.get_model_config(model_id)
        if not config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Create appropriate wrapper based on model type
        if config.model_type in [ModelType.YOLO11, ModelType.YOLO12]:
            wrapper = YOLOModelWrapper(config)
        elif config.model_type == ModelType.RT_DETR:
            wrapper = RTDETRModelWrapper(config)
        elif config.model_type == ModelType.SAM2:
            wrapper = SAM2ModelWrapper(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        wrapper.load_model()
        self._loaded_models[model_id] = wrapper
        return wrapper
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model to free memory."""
        if model_id in self._loaded_models:
            self._loaded_models[model_id].unload_model()
            del self._loaded_models[model_id]