"""
Model management and selection service.
"""
from typing import List, Optional, Dict, Any, Tuple
import uuid

from ..models.model import ModelModel, model_storage
from ..lib.ml_models import ModelType, ModelVariant, get_model_registry, ModelFactory
from ..lib.ml_models.registry import create_default_models


class ModelService:
    """Service for handling model operations."""
    
    def __init__(self):
        self.storage = model_storage
        self.registry = get_model_registry()
        
    def get_model(self, model_id: str) -> Optional[ModelModel]:
        """Get a model by ID."""
        return self.storage.get_by_id(model_id)
    
    def list_models(self, type: Optional[str] = None, framework: Optional[str] = None,
                   training_status: Optional[str] = None, limit: int = 50, 
                   offset: int = 0) -> Tuple[List[ModelModel], int]:
        """List models with optional filtering and pagination."""
        return self.storage.list_models(type, framework, training_status, limit, offset)
    
    def get_available_model_types(self) -> List[str]:
        """Get list of available model types."""
        return [model_type.value for model_type in ModelType]
    
    def get_available_variants(self, model_type: Optional[str] = None) -> List[str]:
        """Get list of available model variants, optionally filtered by type."""
        variants = []
        for variant in ModelVariant:
            # Filter by model type if specified
            if model_type:
                if model_type.upper() == "YOLO11" and variant.value.startswith("yolo_"):
                    variants.append(variant.value)
                elif model_type.upper() == "YOLO12" and variant.value.startswith("yolo_"):
                    variants.append(variant.value)
                elif model_type.upper() == "RT-DETR" and variant.value.startswith("rt_detr_"):
                    variants.append(variant.value)
                elif model_type.upper() == "SAM2" and variant.value.startswith("sam2_"):
                    variants.append(variant.value)
            else:
                variants.append(variant.value)
        return variants
    
    def get_registry_models(self) -> List[Dict[str, Any]]:
        """Get all models from the registry with their configurations."""
        registry_models = []
        for model_id, config in self.registry.list_models().items():
            registry_models.append({
                "model_id": model_id,
                "name": config.name,
                "type": config.model_type.value,
                "variant": config.variant.value,
                "description": config.description,
                "input_size": config.input_size,
                "num_classes": config.num_classes,
                "performance_metrics": config.performance_metrics,
                "is_loaded": model_id in self.registry._loaded_models
            })
        return registry_models
    
    def load_model(self, model_id: str) -> bool:
        """Load a model into memory."""
        try:
            self.registry.load_model(model_id)
            return True
        except Exception as e:
            print(f"Failed to load model {model_id}: {str(e)}")
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        try:
            self.registry.unload_model(model_id)
            return True
        except Exception as e:
            print(f"Failed to unload model {model_id}: {str(e)}")
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self.registry._loaded_models.keys())
    
    def create_custom_model(self, name: str, type: str, variant: str, 
                           version: str, framework: str, model_path: str,
                           performance_metrics: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ModelModel:
        """Create a new custom model entry."""
        # Validate required fields
        if not name:
            raise ValueError("name is required")
        if not type:
            raise ValueError("type is required")
        if not variant:
            raise ValueError("variant is required")
        if not version:
            raise ValueError("version is required")
        if not framework:
            raise ValueError("framework is required")
        if not model_path:
            raise ValueError("model_path is required")
        
        # Validate type and framework
        valid_types = ["detection", "segmentation"]
        if type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}")
        
        valid_frameworks = ["YOLO11", "YOLO12", "RT-DETR", "SAM2"]
        if framework not in valid_frameworks:
            raise ValueError(f"framework must be one of {valid_frameworks}")
        
        # Check if model with same name and version already exists
        existing_model = self.storage.get_by_name_and_version(name, version)
        if existing_model:
            raise ValueError(f"Model with name '{name}' and version '{version}' already exists")
        
        # Create model
        model = ModelModel(
            name=name,
            type=type,
            variant=variant,
            version=version,
            framework=framework,
            model_path=model_path,
            training_status="custom",
            performance_metrics=performance_metrics,
            metadata=metadata
        )
        
        return self.storage.save(model)
    
    def update_model(self, model_id: str, **kwargs) -> Optional[ModelModel]:
        """Update model properties."""
        model = self.storage.get_by_id(model_id)
        if not model:
            return None
        
        # Update allowed fields
        updateable_fields = [
            "name", "type", "variant", "version", "framework", 
            "model_path", "training_status", "performance_metrics", "metadata"
        ]
        
        for field, value in kwargs.items():
            if field in updateable_fields and hasattr(model, field):
                setattr(model, field, value)
        
        return self.storage.save(model)
    
    def get_model_recommendations(self, use_case: str, 
                                performance_priority: str = "balanced") -> List[Dict[str, Any]]:
        """Get model recommendations based on use case and performance priorities."""
        recommendations = []
        registry_models = self.get_registry_models()
        
        # Filter by use case
        if use_case.lower() == "detection":
            filtered_models = [m for m in registry_models if m["type"] in ["YOLO11", "YOLO12", "RT-DETR"]]
        elif use_case.lower() == "segmentation":
            filtered_models = [m for m in registry_models if m["type"] == "SAM2"]
        else:
            filtered_models = registry_models
        
        # Sort by performance priority
        if performance_priority == "speed":
            # Prioritize models with lower inference time
            filtered_models.sort(key=lambda x: x["performance_metrics"].get("speed_ms", 999))
        elif performance_priority == "accuracy":
            # Prioritize models with higher mAP or mIoU
            filtered_models.sort(key=lambda x: x["performance_metrics"].get("mAP", 
                               x["performance_metrics"].get("mIoU", 0)), reverse=True)
        elif performance_priority == "balanced":
            # Balance between speed and accuracy
            for model in filtered_models:
                metrics = model["performance_metrics"]
                accuracy = metrics.get("mAP", metrics.get("mIoU", 0))
                speed_score = 1000 / max(metrics.get("speed_ms", 1), 1)  # Higher is better
                model["balance_score"] = (accuracy + speed_score / 100) / 2
            filtered_models.sort(key=lambda x: x.get("balance_score", 0), reverse=True)
        
        # Format recommendations
        for i, model in enumerate(filtered_models[:5]):  # Top 5 recommendations
            recommendation = {
                "rank": i + 1,
                "model_id": model["model_id"],
                "name": model["name"],
                "type": model["type"],
                "variant": model["variant"],
                "description": model["description"],
                "performance_metrics": model["performance_metrics"],
                "is_loaded": model["is_loaded"],
                "recommendation_reason": self._get_recommendation_reason(model, performance_priority, use_case)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_recommendation_reason(self, model: Dict[str, Any], priority: str, use_case: str) -> str:
        """Generate recommendation reason for a model."""
        reasons = []
        
        if priority == "speed":
            speed = model["performance_metrics"].get("speed_ms", 0)
            if speed < 2:
                reasons.append("extremely fast inference")
            elif speed < 5:
                reasons.append("fast inference")
            else:
                reasons.append("good performance")
        
        if priority == "accuracy":
            accuracy = model["performance_metrics"].get("mAP", model["performance_metrics"].get("mIoU", 0))
            if accuracy > 0.75:
                reasons.append("high accuracy")
            elif accuracy > 0.5:
                reasons.append("good accuracy")
            else:
                reasons.append("decent accuracy")
        
        if use_case == "detection" and model["type"] in ["YOLO11", "YOLO12"]:
            reasons.append("optimized for object detection")
        elif use_case == "segmentation" and model["type"] == "SAM2":
            reasons.append("specialized for segmentation tasks")
        
        if model["variant"].endswith("nano"):
            reasons.append("lightweight for edge deployment")
        elif model["variant"].endswith("large") or model["variant"].endswith("extra_large"):
            reasons.append("maximum performance capability")
        
        return ", ".join(reasons) if reasons else "general purpose model"
    
    def validate_model_compatibility(self, model_id: str, 
                                   target_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate if a model is compatible with target use case."""
        config = self.registry.get_model_config(model_id)
        if not config:
            return {
                "is_compatible": False,
                "error": f"Model {model_id} not found in registry"
            }
        
        compatibility_info = {
            "is_compatible": True,
            "model_id": model_id,
            "model_type": config.model_type.value,
            "supports_classes": config.num_classes,
            "input_size": config.input_size,
            "warnings": [],
            "recommendations": []
        }
        
        # Check class compatibility
        if target_classes:
            if len(target_classes) > config.num_classes:
                compatibility_info["warnings"].append(
                    f"Model supports {config.num_classes} classes but {len(target_classes)} were requested"
                )
                compatibility_info["recommendations"].append(
                    "Consider using a model with more classes or fine-tuning this model"
                )
        
        # Type-specific recommendations
        if config.model_type == ModelType.SAM2 and target_classes and len(target_classes) > 1:
            compatibility_info["warnings"].append(
                "SAM2 is class-agnostic and works best for general segmentation tasks"
            )
        
        return compatibility_info
    
    def get_model_performance_comparison(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance metrics between multiple models."""
        if not model_ids:
            return {"error": "No model IDs provided"}
        
        comparison_data = {
            "models": [],
            "metrics_summary": {},
            "best_performers": {}
        }
        
        # Collect model data
        for model_id in model_ids:
            config = self.registry.get_model_config(model_id)
            if config:
                model_data = {
                    "model_id": model_id,
                    "name": config.name,
                    "type": config.model_type.value,
                    "variant": config.variant.value,
                    "metrics": config.performance_metrics
                }
                comparison_data["models"].append(model_data)
        
        if not comparison_data["models"]:
            return {"error": "No valid models found"}
        
        # Calculate metrics summary
        all_metrics = [model["metrics"] for model in comparison_data["models"]]
        
        # Identify best performers for key metrics
        speed_metric = "speed_ms"
        accuracy_metrics = ["mAP", "mIoU"]
        
        # Find fastest model (lowest speed_ms)
        speeds = [(model["model_id"], model["metrics"].get(speed_metric, float('inf'))) 
                 for model in comparison_data["models"]]
        if speeds:
            fastest_model = min(speeds, key=lambda x: x[1])
            comparison_data["best_performers"]["fastest"] = {
                "model_id": fastest_model[0],
                "value": fastest_model[1],
                "metric": speed_metric
            }
        
        # Find most accurate model
        for accuracy_metric in accuracy_metrics:
            accuracies = [(model["model_id"], model["metrics"].get(accuracy_metric, 0)) 
                         for model in comparison_data["models"] 
                         if accuracy_metric in model["metrics"]]
            if accuracies:
                most_accurate = max(accuracies, key=lambda x: x[1])
                comparison_data["best_performers"][f"most_accurate_{accuracy_metric}"] = {
                    "model_id": most_accurate[0],
                    "value": most_accurate[1],
                    "metric": accuracy_metric
                }
                break
        
        return comparison_data