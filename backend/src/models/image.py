"""
In-memory storage for images (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid


class ImageModel:
    """Temporary in-memory image model for TDD GREEN phase."""
    
    def __init__(self, filename: str, file_path: str, file_size: int, 
                 format: str, width: int, height: int, dataset_split: str = "train", 
                 metadata: Optional[Dict] = None):
        self.id = str(uuid.uuid4())
        self.filename = filename
        self.file_path = file_path
        self.file_size = file_size
        self.format = format
        self.width = width
        self.height = height
        self.dataset_split = dataset_split
        self.upload_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class ImageStorage:
    """Temporary in-memory storage for TDD GREEN phase."""
    
    def __init__(self):
        self._images: Dict[str, ImageModel] = {}
    
    def save(self, image: ImageModel) -> ImageModel:
        """Save an image to storage."""
        self._images[image.id] = image
        return image
    
    def get_by_id(self, image_id: str) -> Optional[ImageModel]:
        """Get an image by ID."""
        return self._images.get(image_id)
    
    def list_images(self, dataset_split: Optional[str] = None, 
                   limit: int = 50, offset: int = 0) -> tuple[List[ImageModel], int]:
        """List images with optional filtering and pagination."""
        images = list(self._images.values())
        
        # Filter by dataset split if provided
        if dataset_split:
            images = [img for img in images if img.dataset_split == dataset_split]
        
        total_count = len(images)
        
        # Apply pagination
        paginated_images = images[offset:offset + limit]
        
        return paginated_images, total_count


# Global storage instance (temporary for TDD)
image_storage = ImageStorage()