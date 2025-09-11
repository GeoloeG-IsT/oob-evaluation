"""
In-memory storage for models (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class ModelModel:
    """Temporary in-memory model model for TDD GREEN phase."""
    
    def __init__(self, name: str, type: str, variant: str, version: str, 
                 framework: str, model_path: str, training_status: str = "pre-trained",
                 performance_metrics: Optional[Dict[str, Any]] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = type  # detection or segmentation
        self.variant = variant  # nano, small, medium, large, xl
        self.version = version
        self.framework = framework  # YOLO11, YOLO12, RT-DETR, SAM2
        self.model_path = model_path
        self.training_status = training_status  # pre-trained, training, trained, failed
        self.performance_metrics = performance_metrics or {}
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class ModelStorage:
    """Temporary in-memory storage for models."""
    
    def __init__(self):
        self._models: Dict[str, ModelModel] = {}
    
    def save(self, model: ModelModel) -> ModelModel:
        """Save a model to storage."""
        model.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._models[model.id] = model
        return model
    
    def get_by_id(self, model_id: str) -> Optional[ModelModel]:
        """Get a model by ID."""
        return self._models.get(model_id)
    
    def list_models(self, type: Optional[str] = None, framework: Optional[str] = None, 
                   training_status: Optional[str] = None, limit: int = 50, 
                   offset: int = 0) -> tuple[List[ModelModel], int]:
        """List models with optional filtering and pagination."""
        models = list(self._models.values())
        
        # Apply filters
        if type:
            models = [m for m in models if m.type == type]
        if framework:
            models = [m for m in models if m.framework == framework]
        if training_status:
            models = [m for m in models if m.training_status == training_status]
        
        total_count = len(models)
        
        # Apply pagination
        paginated_models = models[offset:offset + limit]
        
        return paginated_models, total_count
    
    def get_by_name_and_version(self, name: str, version: str) -> Optional[ModelModel]:
        """Get a model by name and version."""
        for model in self._models.values():
            if model.name == name and model.version == version:
                return model
        return None


# Global storage instance (temporary for TDD)
model_storage = ModelStorage()