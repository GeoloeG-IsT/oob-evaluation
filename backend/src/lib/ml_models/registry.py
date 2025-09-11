"""
Model registry management and initialization.
"""
from typing import Dict, Any
from .models import ModelRegistry, ModelConfig, ModelType, ModelVariant


def create_default_models() -> Dict[str, ModelConfig]:
    """Create default model configurations for all supported models."""
    
    models = {}
    
    # YOLO11 models
    yolo11_variants = {
        ModelVariant.YOLO_NANO: {
            "name": "YOLO11 Nano",
            "description": "Ultra-lightweight YOLO11 model for edge deployment",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.375, "speed_ms": 1.2}
        },
        ModelVariant.YOLO_SMALL: {
            "name": "YOLO11 Small", 
            "description": "Compact YOLO11 model balancing speed and accuracy",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.465, "speed_ms": 2.1}
        },
        ModelVariant.YOLO_MEDIUM: {
            "name": "YOLO11 Medium",
            "description": "Medium-sized YOLO11 model for general purpose detection",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.505, "speed_ms": 4.2}
        },
        ModelVariant.YOLO_LARGE: {
            "name": "YOLO11 Large",
            "description": "Large YOLO11 model for high-accuracy detection",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.535, "speed_ms": 6.8}
        },
        ModelVariant.YOLO_EXTRA_LARGE: {
            "name": "YOLO11 Extra Large",
            "description": "Largest YOLO11 model for maximum accuracy",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.550, "speed_ms": 9.5}
        }
    }
    
    for variant, config in yolo11_variants.items():
        model_id = f"yolo11_{variant.value}"
        models[model_id] = ModelConfig(
            model_id=model_id,
            model_type=ModelType.YOLO11,
            variant=variant,
            name=config["name"],
            description=config["description"],
            input_size=config["input_size"],
            num_classes=80,  # COCO dataset classes
            performance_metrics=config["performance_metrics"]
        )
    
    # YOLO12 models (similar structure to YOLO11 but with updated performance)
    yolo12_variants = {
        ModelVariant.YOLO_NANO: {
            "name": "YOLO12 Nano",
            "description": "Next-gen ultra-lightweight YOLO12 model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.395, "speed_ms": 1.0}
        },
        ModelVariant.YOLO_SMALL: {
            "name": "YOLO12 Small",
            "description": "Improved compact YOLO12 model",
            "input_size": (640, 640), 
            "performance_metrics": {"mAP": 0.485, "speed_ms": 1.8}
        },
        ModelVariant.YOLO_MEDIUM: {
            "name": "YOLO12 Medium",
            "description": "Enhanced medium-sized YOLO12 model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.525, "speed_ms": 3.8}
        },
        ModelVariant.YOLO_LARGE: {
            "name": "YOLO12 Large",
            "description": "Advanced large YOLO12 model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.555, "speed_ms": 6.2}
        },
        ModelVariant.YOLO_EXTRA_LARGE: {
            "name": "YOLO12 Extra Large",
            "description": "State-of-the-art largest YOLO12 model", 
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.570, "speed_ms": 8.8}
        }
    }
    
    for variant, config in yolo12_variants.items():
        model_id = f"yolo12_{variant.value}"
        models[model_id] = ModelConfig(
            model_id=model_id,
            model_type=ModelType.YOLO12,
            variant=variant,
            name=config["name"],
            description=config["description"],
            input_size=config["input_size"],
            num_classes=80,
            performance_metrics=config["performance_metrics"]
        )
    
    # RT-DETR models
    rtdetr_variants = {
        ModelVariant.RT_DETR_R18: {
            "name": "RT-DETR ResNet-18",
            "description": "RT-DETR with ResNet-18 backbone",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.468, "speed_ms": 4.8}
        },
        ModelVariant.RT_DETR_R34: {
            "name": "RT-DETR ResNet-34", 
            "description": "RT-DETR with ResNet-34 backbone",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.482, "speed_ms": 6.2}
        },
        ModelVariant.RT_DETR_R50: {
            "name": "RT-DETR ResNet-50",
            "description": "RT-DETR with ResNet-50 backbone",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.514, "speed_ms": 8.5}
        },
        ModelVariant.RT_DETR_R101: {
            "name": "RT-DETR ResNet-101",
            "description": "RT-DETR with ResNet-101 backbone",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.528, "speed_ms": 12.1}
        },
        ModelVariant.RT_DETR_RF_NANO: {
            "name": "RF-DETR Nano",
            "description": "Lightweight RF-DETR model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.395, "speed_ms": 3.2}
        },
        ModelVariant.RT_DETR_RF_SMALL: {
            "name": "RF-DETR Small",
            "description": "Small RF-DETR model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.425, "speed_ms": 4.8}
        },
        ModelVariant.RT_DETR_RF_MEDIUM: {
            "name": "RF-DETR Medium",
            "description": "Medium RF-DETR model",
            "input_size": (640, 640),
            "performance_metrics": {"mAP": 0.455, "speed_ms": 7.2}
        }
    }
    
    for variant, config in rtdetr_variants.items():
        model_id = f"rtdetr_{variant.value}"
        models[model_id] = ModelConfig(
            model_id=model_id,
            model_type=ModelType.RT_DETR,
            variant=variant,
            name=config["name"],
            description=config["description"],
            input_size=config["input_size"],
            num_classes=80,
            performance_metrics=config["performance_metrics"]
        )
    
    # SAM2 models
    sam2_variants = {
        ModelVariant.SAM2_TINY: {
            "name": "SAM2 Tiny",
            "description": "Lightweight SAM2 model for segmentation",
            "input_size": (1024, 1024),
            "performance_metrics": {"mIoU": 0.725, "speed_ms": 85}
        },
        ModelVariant.SAM2_SMALL: {
            "name": "SAM2 Small",
            "description": "Small SAM2 model for efficient segmentation",
            "input_size": (1024, 1024),
            "performance_metrics": {"mIoU": 0.745, "speed_ms": 125}
        },
        ModelVariant.SAM2_BASE_PLUS: {
            "name": "SAM2 Base+",
            "description": "Enhanced base SAM2 model",
            "input_size": (1024, 1024),
            "performance_metrics": {"mIoU": 0.768, "speed_ms": 185}
        },
        ModelVariant.SAM2_LARGE: {
            "name": "SAM2 Large",
            "description": "Large SAM2 model for high-quality segmentation",
            "input_size": (1024, 1024),
            "performance_metrics": {"mIoU": 0.785, "speed_ms": 285}
        }
    }
    
    for variant, config in sam2_variants.items():
        model_id = f"sam2_{variant.value}"
        models[model_id] = ModelConfig(
            model_id=model_id,
            model_type=ModelType.SAM2,
            variant=variant,
            name=config["name"],
            description=config["description"],
            input_size=config["input_size"],
            num_classes=1,  # SAM2 is class-agnostic
            performance_metrics=config["performance_metrics"]
        )
    
    return models


# Global registry instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
        # Register default models
        default_models = create_default_models()
        for config in default_models.values():
            _model_registry.register_model(config)
    return _model_registry


def reset_model_registry() -> None:
    """Reset the global model registry (useful for testing)."""
    global _model_registry
    _model_registry = None