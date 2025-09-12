"""
Model factory for creating and managing model instances.
"""
from typing import Dict, Any, Optional
from .models import ModelConfig, ModelType, ModelVariant, BaseModelWrapper
from .registry import get_model_registry


class ModelFactory:
    """Factory class for creating and managing model instances."""
    
    @staticmethod
    def create_model_config(
        model_type: ModelType,
        variant: ModelVariant,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> ModelConfig:
        """Create a custom model configuration."""
        
        model_id = f"{model_type.value}_{variant.value}"
        if custom_config and "model_id" in custom_config:
            model_id = custom_config["model_id"]
        
        # Base configuration
        config = {
            "model_id": model_id,
            "model_type": model_type,
            "variant": variant,
            "name": f"{model_type.value.upper()} {variant.value.title()}",
            "description": f"Custom {model_type.value.upper()} {variant.value} model",
            "input_size": (640, 640),
            "num_classes": 80
        }
        
        # Apply custom configuration overrides
        if custom_config:
            config.update(custom_config)
        
        return ModelConfig(**config)
    
    @staticmethod
    def get_available_models(model_type: Optional[ModelType] = None) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        registry = get_model_registry()
        models = registry.list_models(model_type)
        
        return {
            model.model_id: {
                "model_id": model.model_id,
                "model_type": model.model_type,
                "variant": model.variant,
                "name": model.name,
                "description": model.description,
                "input_size": model.input_size,
                "num_classes": model.num_classes,
                "performance_metrics": model.performance_metrics,
                "created_at": model.created_at
            }
            for model in models
        }
    
    @staticmethod
    def get_model_variants(model_type: ModelType) -> list[ModelVariant]:
        """Get supported variants for a model type."""
        variant_mapping = {
            ModelType.YOLO11: [
                ModelVariant.YOLO_NANO,
                ModelVariant.YOLO_SMALL,
                ModelVariant.YOLO_MEDIUM,
                ModelVariant.YOLO_LARGE,
                ModelVariant.YOLO_EXTRA_LARGE
            ],
            ModelType.YOLO12: [
                ModelVariant.YOLO_NANO,
                ModelVariant.YOLO_SMALL,
                ModelVariant.YOLO_MEDIUM,
                ModelVariant.YOLO_LARGE,
                ModelVariant.YOLO_EXTRA_LARGE
            ],
            ModelType.RT_DETR: [
                ModelVariant.RT_DETR_R18,
                ModelVariant.RT_DETR_R34,
                ModelVariant.RT_DETR_R50,
                ModelVariant.RT_DETR_R101,
                ModelVariant.RT_DETR_RF_NANO,
                ModelVariant.RT_DETR_RF_SMALL,
                ModelVariant.RT_DETR_RF_MEDIUM
            ],
            ModelType.SAM2: [
                ModelVariant.SAM2_TINY,
                ModelVariant.SAM2_SMALL,
                ModelVariant.SAM2_BASE_PLUS,
                ModelVariant.SAM2_LARGE
            ]
        }
        
        return variant_mapping.get(model_type, [])
    
    @staticmethod
    def load_model(model_id: str) -> BaseModelWrapper:
        """Load a model by ID."""
        registry = get_model_registry()
        return registry.load_model(model_id)
    
    @staticmethod
    def unload_model(model_id: str) -> None:
        """Unload a model by ID."""
        registry = get_model_registry()
        registry.unload_model(model_id)
    
    @staticmethod
    def register_custom_model(config: ModelConfig) -> None:
        """Register a custom model configuration."""
        registry = get_model_registry()
        registry.register_model(config)
    
    @staticmethod
    def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        registry = get_model_registry()
        config = registry.get_model_config(model_id)
        if not config:
            return None
        
        wrapper = registry.get_model_wrapper(model_id)
        is_loaded = wrapper is not None and wrapper.is_loaded
        
        return {
            "model_id": config.model_id,
            "model_type": config.model_type,
            "variant": config.variant,
            "name": config.name,
            "description": config.description,
            "input_size": config.input_size,
            "num_classes": config.num_classes,
            "weights_path": config.weights_path,
            "config_path": config.config_path,
            "metadata": config.metadata,
            "performance_metrics": config.performance_metrics,
            "created_at": config.created_at,
            "is_loaded": is_loaded
        }
    
    @staticmethod
    def predict(model_id: str, image_path: str, **kwargs) -> Dict[str, Any]:
        """Make predictions using a model."""
        registry = get_model_registry()
        
        # Load model if not already loaded
        if model_id not in registry._loaded_models:
            registry.load_model(model_id)
        
        wrapper = registry.get_model_wrapper(model_id)
        if not wrapper:
            raise ValueError(f"Model {model_id} not found or failed to load")
        
        return wrapper.predict(image_path, **kwargs)