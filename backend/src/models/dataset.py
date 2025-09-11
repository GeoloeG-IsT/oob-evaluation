"""
In-memory storage for datasets (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class DatasetModel:
    """Temporary in-memory dataset model for TDD GREEN phase."""
    
    def __init__(self, name: str, description: Optional[str] = None,
                 train_count: int = 0, validation_count: int = 0, test_count: int = 0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.train_count = train_count
        self.validation_count = validation_count
        self.test_count = test_count
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class DatasetStorage:
    """Temporary in-memory storage for datasets."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetModel] = {}
        # Track many-to-many relationships with images
        self._dataset_images: Dict[str, List[str]] = {}  # dataset_id -> [image_id]
    
    def save(self, dataset: DatasetModel) -> DatasetModel:
        """Save a dataset to storage."""
        dataset.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._datasets[dataset.id] = dataset
        if dataset.id not in self._dataset_images:
            self._dataset_images[dataset.id] = []
        return dataset
    
    def get_by_id(self, dataset_id: str) -> Optional[DatasetModel]:
        """Get a dataset by ID."""
        return self._datasets.get(dataset_id)
    
    def get_by_name(self, name: str) -> Optional[DatasetModel]:
        """Get a dataset by name."""
        for dataset in self._datasets.values():
            if dataset.name == name:
                return dataset
        return None
    
    def list_datasets(self, limit: int = 50, offset: int = 0) -> tuple[List[DatasetModel], int]:
        """List datasets with pagination."""
        datasets = list(self._datasets.values())
        total_count = len(datasets)
        
        # Apply pagination
        paginated_datasets = datasets[offset:offset + limit]
        
        return paginated_datasets, total_count
    
    def add_image_to_dataset(self, dataset_id: str, image_id: str) -> bool:
        """Add an image to a dataset."""
        if dataset_id in self._dataset_images:
            if image_id not in self._dataset_images[dataset_id]:
                self._dataset_images[dataset_id].append(image_id)
            return True
        return False
    
    def remove_image_from_dataset(self, dataset_id: str, image_id: str) -> bool:
        """Remove an image from a dataset."""
        if dataset_id in self._dataset_images:
            if image_id in self._dataset_images[dataset_id]:
                self._dataset_images[dataset_id].remove(image_id)
            return True
        return False
    
    def get_dataset_images(self, dataset_id: str) -> List[str]:
        """Get all image IDs for a dataset."""
        return self._dataset_images.get(dataset_id, [])
    
    def update_counts(self, dataset_id: str, train_count: int = None, 
                      validation_count: int = None, test_count: int = None):
        """Update dataset split counts."""
        dataset = self.get_by_id(dataset_id)
        if dataset:
            if train_count is not None:
                dataset.train_count = train_count
            if validation_count is not None:
                dataset.validation_count = validation_count
            if test_count is not None:
                dataset.test_count = test_count
            dataset.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Global storage instance (temporary for TDD)
dataset_storage = DatasetStorage()