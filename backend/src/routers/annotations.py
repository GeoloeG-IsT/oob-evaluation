"""
Annotation management API routes.
"""
from typing import List
from fastapi import APIRouter, HTTPException

from ..services.annotation_service import AnnotationService
from ..schemas.annotation import AnnotationCreate, AnnotationResponse, BoundingBox, Segment


router = APIRouter(prefix="/api/v1", tags=["annotations"])
annotation_service = AnnotationService()


@router.post("/annotations", response_model=AnnotationResponse, status_code=201)
async def create_annotation(annotation_data: AnnotationCreate):
    """Create a new annotation."""
    try:
        # Create the annotation
        annotation = annotation_service.create_annotation(annotation_data)
        
        # Convert to response format
        # Convert stored dicts back to pydantic models
        bounding_boxes = []
        if annotation.bounding_boxes:
            bounding_boxes = [BoundingBox(**bbox) for bbox in annotation.bounding_boxes]
        
        segments = []
        if annotation.segments:
            segments = [Segment(**seg) for seg in annotation.segments]
        
        return AnnotationResponse(
            id=annotation.id,
            image_id=annotation.image_id,
            bounding_boxes=bounding_boxes,
            segments=segments,
            class_labels=annotation.class_labels,
            confidence_scores=annotation.confidence_scores,
            creation_method=annotation.creation_method,
            user_tag=annotation.user_tag,
            created_at=annotation.created_at,
            metadata=annotation.metadata
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in create_annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})