"""
Annotation processing and storage service.
"""
from typing import List, Dict, Any
import uuid

from ..models.annotation import AnnotationModel, annotation_storage
from ..schemas.annotation import AnnotationCreate


class AnnotationService:
    """Service for handling annotation operations."""
    
    def __init__(self):
        self.storage = annotation_storage
        
    def create_annotation(self, annotation_data: AnnotationCreate) -> AnnotationModel:
        """Create a new annotation."""
        # Validate required fields
        if not annotation_data.image_id:
            raise ValueError("image_id is required")
        
        if not annotation_data.class_labels:
            raise ValueError("class_labels is required")
        
        if len(annotation_data.class_labels) == 0:
            raise ValueError("class_labels cannot be empty")
        
        # Validate image_id is a proper UUID
        try:
            uuid.UUID(annotation_data.image_id)
        except ValueError:
            raise ValueError("Invalid UUID format for image_id")
        
        # Validate confidence_scores length if provided
        if annotation_data.confidence_scores:
            if len(annotation_data.confidence_scores) != len(annotation_data.class_labels):
                raise ValueError("confidence_scores length must match class_labels length")
        
        # Convert pydantic models to dicts for storage
        bounding_boxes_data = []
        if annotation_data.bounding_boxes:
            bounding_boxes_data = [bbox.model_dump() for bbox in annotation_data.bounding_boxes]
        
        segments_data = []
        if annotation_data.segments:
            segments_data = [segment.model_dump() for segment in annotation_data.segments]
        
        # Create annotation model
        annotation_model = AnnotationModel(
            image_id=annotation_data.image_id,
            bounding_boxes=bounding_boxes_data,
            segments=segments_data,
            class_labels=annotation_data.class_labels,
            confidence_scores=annotation_data.confidence_scores,
            user_tag=annotation_data.user_tag,
            metadata=annotation_data.metadata
        )
        
        # Save to storage
        return self.storage.save(annotation_model)
    
    def get_annotation(self, annotation_id: str) -> AnnotationModel:
        """Get an annotation by ID."""
        return self.storage.get_by_id(annotation_id)
    
    def get_annotations_by_image(self, image_id: str) -> List[AnnotationModel]:
        """Get all annotations for an image."""
        return self.storage.get_by_image_id(image_id)