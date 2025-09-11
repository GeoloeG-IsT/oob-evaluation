"""
Pydantic schemas for image-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation" 
    TEST = "test"


class ImageResponse(BaseModel):
    id: str
    filename: str
    file_path: str
    file_size: int
    format: str
    width: int
    height: int
    dataset_split: DatasetSplit
    upload_timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class ImageListResponse(BaseModel):
    images: List[ImageResponse]
    total_count: int
    limit: int
    offset: int


class ImageUploadResponse(BaseModel):
    uploaded_images: List[ImageResponse]
    total_count: int
    success_count: int
    failed_count: int