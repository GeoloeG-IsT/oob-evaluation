"""
Pydantic schemas for annotation-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    class_id: Optional[int] = None
    confidence: Optional[float] = None


class Segment(BaseModel):
    polygon: List[List[float]]  # List of [x, y] coordinate pairs
    class_id: Optional[int] = None
    confidence: Optional[float] = None


class AnnotationCreate(BaseModel):
    image_id: Optional[str] = None
    bounding_boxes: Optional[List[BoundingBox]] = []
    segments: Optional[List[Segment]] = []
    class_labels: Optional[List[str]] = None
    confidence_scores: Optional[List[float]] = None
    user_tag: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    bounding_boxes: Optional[List[BoundingBox]] = []
    segments: Optional[List[Segment]] = []
    class_labels: List[str]
    confidence_scores: Optional[List[float]] = None
    creation_method: str = "user"
    user_tag: Optional[str] = None
    created_at: str
    metadata: Optional[Dict[str, Any]] = None