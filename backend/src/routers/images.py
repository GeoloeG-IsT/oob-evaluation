"""
Image management API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from ..services.image_service import ImageService
from ..schemas.image import ImageUploadResponse, ImageListResponse, ImageResponse, DatasetSplit


router = APIRouter(prefix="/api/v1", tags=["images"])
image_service = ImageService()


@router.post("/images", response_model=ImageUploadResponse, status_code=201)
async def upload_images(
    files: List[UploadFile] = File(None),
    dataset_split: str = Form("train")
):
    """Upload one or more images to the platform."""
    if not files or all(not f.filename for f in files):
        raise HTTPException(status_code=400, detail={"error": "No files provided"})
    
    # Validate dataset split first
    if dataset_split not in [split.value for split in DatasetSplit]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid dataset split: {dataset_split}"})
    
    try:
        # Process the uploads
        successful_images, success_count, failed_count = await image_service.process_upload(
            files, dataset_split
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in upload_images: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})
    
    # Check if all files failed - this suggests invalid file types
    if success_count == 0 and failed_count > 0:
        raise HTTPException(status_code=400, detail={"error": "No valid image files provided"})
        
    # Convert to response format
    uploaded_images = [
        ImageResponse(
            id=img.id,
            filename=img.filename,
            file_path=img.file_path,
            file_size=img.file_size,
            format=img.format,
            width=img.width,
            height=img.height,
            dataset_split=img.dataset_split,
            upload_timestamp=img.upload_timestamp,
            metadata=img.metadata
        )
        for img in successful_images
    ]
    
    return ImageUploadResponse(
        uploaded_images=uploaded_images,
        total_count=len(files),
        success_count=success_count,
        failed_count=failed_count
    )


@router.get("/images", response_model=ImageListResponse)
async def list_images(
    dataset_split: Optional[str] = Query(None, description="Filter by dataset split"),
    limit: int = Query(50, description="Number of images to return"),
    offset: int = Query(0, description="Number of images to skip")
):
    """List images with optional filtering and pagination."""
    if dataset_split and dataset_split not in [split.value for split in DatasetSplit]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid dataset split: {dataset_split}"})
    
    if offset < 0:
        raise HTTPException(status_code=400, detail={"error": "Offset must be non-negative"})
    
    if limit < 1:
        raise HTTPException(status_code=400, detail={"error": "Limit must be positive"})
    
    # Cap limit at maximum
    if limit > 1000:
        limit = 1000
    
    try:
        images, total_count = image_service.list_images(dataset_split, limit, offset)
        
        # Convert to response format
        image_responses = [
            ImageResponse(
                id=img.id,
                filename=img.filename,
                file_path=img.file_path,
                file_size=img.file_size,
                format=img.format,
                width=img.width,
                height=img.height,
                dataset_split=img.dataset_split,
                upload_timestamp=img.upload_timestamp,
                metadata=img.metadata
            )
            for img in images
        ]
        
        return ImageListResponse(
            images=image_responses,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/images/{image_id}", response_model=ImageResponse)
async def get_image(image_id: str):
    """Get a single image by ID."""
    # Basic UUID validation
    try:
        import uuid
        uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    image = image_service.get_image(image_id)
    if not image:
        raise HTTPException(
            status_code=404, 
            detail={
                "error": "Image not found",
                "message": f"No image found with ID: {image_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    return ImageResponse(
        id=image.id,
        filename=image.filename,
        file_path=image.file_path,
        file_size=image.file_size,
        format=image.format,
        width=image.width,
        height=image.height,
        dataset_split=image.dataset_split,
        upload_timestamp=image.upload_timestamp,
        metadata=image.metadata
    )