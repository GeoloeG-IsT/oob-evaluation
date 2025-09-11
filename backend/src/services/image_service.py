"""
Image processing and storage service.
"""
import os
import tempfile
from typing import List, Tuple
from PIL import Image
from fastapi import UploadFile

from ..models.image import ImageModel, image_storage
from ..schemas.image import DatasetSplit


class ImageService:
    """Service for handling image operations."""
    
    def __init__(self):
        self.storage = image_storage
        
    async def process_upload(self, files: List[UploadFile], 
                           dataset_split: str = "train") -> Tuple[List[ImageModel], int, int]:
        """
        Process uploaded image files.
        Returns: (successful_images, success_count, failed_count)
        """
        successful_images = []
        failed_count = 0
        
        
        for file in files:
            try:
                # Read file content
                file_content = await file.read()
                
                # Validate that we received actual content
                if not file_content:
                    failed_count += 1
                    continue
                
                # Create temporary file to process with PIL
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                try:
                    # Process image with PIL to get metadata
                    with Image.open(temp_path) as img:
                        width, height = img.size
                        image_format = img.format
                        
                        # For TDD GREEN phase, simulate file storage
                        # In real implementation, this would be uploaded to cloud storage
                        file_path = f"/storage/{dataset_split}/{file.filename}"
                        file_size = len(file_content)
                        
                        # Create image model
                        image_model = ImageModel(
                            filename=file.filename,
                            file_path=file_path,
                            file_size=file_size,
                            format=image_format,
                            width=width,
                            height=height,
                            dataset_split=dataset_split
                        )
                        
                        # Save to storage
                        saved_image = self.storage.save(image_model)
                        successful_images.append(saved_image)
                        
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                failed_count += 1
                print(f"Failed to process {file.filename}: {str(e)}")
                
        success_count = len(successful_images)
        return successful_images, success_count, failed_count
    
    def get_image(self, image_id: str) -> ImageModel:
        """Get a single image by ID."""
        return self.storage.get_by_id(image_id)
    
    def list_images(self, dataset_split: str = None, limit: int = 50, 
                   offset: int = 0) -> Tuple[List[ImageModel], int]:
        """List images with filtering and pagination."""
        return self.storage.list_images(dataset_split, limit, offset)