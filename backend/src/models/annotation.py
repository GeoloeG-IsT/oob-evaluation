"""
In-memory storage for annotations (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class AnnotationModel:
    """Temporary in-memory annotation model for TDD GREEN phase."""
    
    def __init__(self, image_id: str, bounding_boxes: List[Dict] = None, segments: List[Dict] = None,
                 class_labels: List[str] = None, confidence_scores: List[float] = None, 
                 user_tag: str = None, metadata: Dict[str, Any] = None):
        self.id = str(uuid.uuid4())
        self.image_id = image_id
        self.bounding_boxes = bounding_boxes or []
        self.segments = segments or []
        self.class_labels = class_labels or []
        self.confidence_scores = confidence_scores or []
        self.creation_method = "user"
        self.user_tag = user_tag
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class AnnotationStorage:
    """Temporary in-memory storage for annotations."""
    
    def __init__(self):
        self._annotations: Dict[str, AnnotationModel] = {}
    
    def save(self, annotation: AnnotationModel) -> AnnotationModel:
        """Save an annotation to storage."""
        self._annotations[annotation.id] = annotation
        return annotation
    
    def get_by_id(self, annotation_id: str) -> Optional[AnnotationModel]:
        """Get an annotation by ID."""
        return self._annotations.get(annotation_id)
    
    def get_by_image_id(self, image_id: str) -> List[AnnotationModel]:
        """Get all annotations for an image."""
        return [ann for ann in self._annotations.values() if ann.image_id == image_id]


# Global storage instance (temporary for TDD)
annotation_storage = AnnotationStorage()