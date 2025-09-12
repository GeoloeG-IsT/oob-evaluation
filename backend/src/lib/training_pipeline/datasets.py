"""
Dataset management and validation for training pipeline.
"""
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import mimetypes


class DatasetSplit(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    CUSTOM = "custom"


@dataclass
class AnnotationInfo:
    """Information about an annotation."""
    annotation_id: str
    image_id: str
    category_id: int
    bbox: List[float]  # [x, y, width, height] for COCO format
    area: float
    segmentation: Optional[List[List[float]]] = None
    is_crowd: bool = False
    confidence: float = 1.0


@dataclass
class ImageInfo:
    """Information about an image in the dataset."""
    image_id: str
    filename: str
    width: int
    height: int
    file_path: str
    file_size_bytes: int
    format: str
    dataset_split: DatasetSplit
    annotations: List[AnnotationInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_annotation(self, annotation: AnnotationInfo) -> None:
        """Add an annotation to this image."""
        annotation.image_id = self.image_id
        self.annotations.append(annotation)
    
    @property
    def annotation_count(self) -> int:
        """Get number of annotations for this image."""
        return len(self.annotations)


@dataclass
class CategoryInfo:
    """Information about a dataset category/class."""
    category_id: int
    name: str
    supercategory: str = "object"
    description: str = ""
    color: Optional[Tuple[int, int, int]] = None  # RGB color for visualization


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset_id: str
    name: str
    description: str
    dataset_path: str
    format: DatasetFormat
    
    # Dataset splits
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0
    
    # Categories/classes
    categories: List[CategoryInfo] = field(default_factory=list)
    
    # Data augmentation settings
    enable_augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Validation settings
    min_images_per_class: int = 1
    max_images_per_class: Optional[int] = None
    min_annotations_per_image: int = 0
    max_annotations_per_image: Optional[int] = None
    
    # File filters
    supported_image_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    max_image_size_mb: float = 100.0
    min_image_dimensions: Tuple[int, int] = (32, 32)
    max_image_dimensions: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        # Validate split ratios
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError("Dataset splits must sum to 1.0")
    
    @property
    def num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return len(self.categories)
    
    @property 
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return [cat.name for cat in self.categories]


@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool = True
    total_images: int = 0
    total_annotations: int = 0
    
    # Split statistics
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0
    
    # Per-class statistics
    class_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Issues found
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # File issues
    missing_files: List[str] = field(default_factory=list)
    invalid_files: List[str] = field(default_factory=list)
    oversized_files: List[str] = field(default_factory=list)
    
    # Annotation issues
    invalid_annotations: List[str] = field(default_factory=list)
    empty_images: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "is_valid": self.is_valid,
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "dataset_splits": {
                "train": self.train_images,
                "validation": self.val_images,
                "test": self.test_images
            },
            "class_distribution": self.class_distribution,
            "issues": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "missing_files": len(self.missing_files),
                "invalid_files": len(self.invalid_files),
                "oversized_files": len(self.oversized_files),
                "invalid_annotations": len(self.invalid_annotations),
                "empty_images": len(self.empty_images)
            }
        }


class DatasetValidator:
    """Validates dataset structure and content."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_path = Path(config.dataset_path)
    
    def validate_dataset(self) -> ValidationResult:
        """Perform comprehensive dataset validation."""
        result = ValidationResult()
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            result.add_error(f"Dataset path does not exist: {self.dataset_path}")
            return result
        
        # Load and validate dataset structure
        try:
            images = self._load_dataset_images()
            result.total_images = len(images)
            
            # Validate images and annotations
            self._validate_images(images, result)
            self._validate_annotations(images, result)
            self._validate_splits(images, result)
            self._validate_class_distribution(images, result)
            
        except Exception as e:
            result.add_error(f"Failed to load dataset: {str(e)}")
        
        return result
    
    def _load_dataset_images(self) -> List[ImageInfo]:
        """Load dataset images based on format."""
        if self.config.format == DatasetFormat.COCO:
            return self._load_coco_dataset()
        elif self.config.format == DatasetFormat.YOLO:
            return self._load_yolo_dataset()
        elif self.config.format == DatasetFormat.PASCAL_VOC:
            return self._load_pascal_voc_dataset()
        else:
            return self._load_custom_dataset()
    
    def _load_coco_dataset(self) -> List[ImageInfo]:
        """Load COCO format dataset."""
        images = []
        
        # Look for annotation files
        for split in ["train", "val", "test"]:
            annotation_file = self.dataset_path / f"annotations/instances_{split}.json"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                # Process images
                for img_data in data.get("images", []):
                    image_info = ImageInfo(
                        image_id=str(img_data["id"]),
                        filename=img_data["file_name"],
                        width=img_data["width"],
                        height=img_data["height"],
                        file_path=str(self.dataset_path / f"images/{split}" / img_data["file_name"]),
                        file_size_bytes=0,  # Will be calculated later
                        format=Path(img_data["file_name"]).suffix,
                        dataset_split=DatasetSplit(split)
                    )
                    images.append(image_info)
                
                # Process annotations
                for ann_data in data.get("annotations", []):
                    # Find corresponding image
                    image_id = str(ann_data["image_id"])
                    image = next((img for img in images if img.image_id == image_id), None)
                    
                    if image:
                        annotation = AnnotationInfo(
                            annotation_id=str(ann_data["id"]),
                            image_id=image_id,
                            category_id=ann_data["category_id"],
                            bbox=ann_data["bbox"],
                            area=ann_data["area"],
                            segmentation=ann_data.get("segmentation"),
                            is_crowd=ann_data.get("iscrowd", False)
                        )
                        image.add_annotation(annotation)
        
        return images
    
    def _load_yolo_dataset(self) -> List[ImageInfo]:
        """Load YOLO format dataset."""
        images = []
        
        # Look for images and labels in typical YOLO structure
        for split in ["train", "val", "test"]:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            if not images_dir.exists():
                continue
            
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in self.config.supported_image_formats:
                    # Get corresponding label file
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    
                    image_info = ImageInfo(
                        image_id=img_file.stem,
                        filename=img_file.name,
                        width=0,  # Will be determined later
                        height=0,  # Will be determined later
                        file_path=str(img_file),
                        file_size_bytes=img_file.stat().st_size if img_file.exists() else 0,
                        format=img_file.suffix,
                        dataset_split=DatasetSplit(split)
                    )
                    
                    # Load annotations if label file exists
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            for line_num, line in enumerate(f):
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    annotation = AnnotationInfo(
                                        annotation_id=f"{img_file.stem}_{line_num}",
                                        image_id=image_info.image_id,
                                        category_id=int(parts[0]),
                                        bbox=[float(x) for x in parts[1:5]],  # YOLO format: x_center, y_center, width, height
                                        area=float(parts[3]) * float(parts[4]),  # width * height
                                        confidence=float(parts[5]) if len(parts) > 5 else 1.0
                                    )
                                    image_info.add_annotation(annotation)
                    
                    images.append(image_info)
        
        return images
    
    def _load_pascal_voc_dataset(self) -> List[ImageInfo]:
        """Load Pascal VOC format dataset."""
        # Simplified Pascal VOC loading - would need XML parsing in real implementation
        images = []
        
        images_dir = self.dataset_path / "JPEGImages"
        annotations_dir = self.dataset_path / "Annotations"
        
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                xml_file = annotations_dir / f"{img_file.stem}.xml"
                
                image_info = ImageInfo(
                    image_id=img_file.stem,
                    filename=img_file.name,
                    width=0,  # Would be parsed from XML
                    height=0,  # Would be parsed from XML
                    file_path=str(img_file),
                    file_size_bytes=img_file.stat().st_size,
                    format=img_file.suffix,
                    dataset_split=DatasetSplit.TRAIN  # Would need to determine split
                )
                
                images.append(image_info)
        
        return images
    
    def _load_custom_dataset(self) -> List[ImageInfo]:
        """Load custom format dataset."""
        images = []
        
        # Simple directory-based loading
        for img_file in self.dataset_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in self.config.supported_image_formats:
                # Determine split from directory structure
                split = DatasetSplit.TRAIN
                if "val" in str(img_file) or "validation" in str(img_file):
                    split = DatasetSplit.VALIDATION
                elif "test" in str(img_file):
                    split = DatasetSplit.TEST
                
                image_info = ImageInfo(
                    image_id=img_file.stem,
                    filename=img_file.name,
                    width=0,
                    height=0,
                    file_path=str(img_file),
                    file_size_bytes=img_file.stat().st_size,
                    format=img_file.suffix,
                    dataset_split=split
                )
                
                images.append(image_info)
        
        return images
    
    def _validate_images(self, images: List[ImageInfo], result: ValidationResult) -> None:
        """Validate image files."""
        for image in images:
            image_path = Path(image.file_path)
            
            # Check if file exists
            if not image_path.exists():
                result.missing_files.append(image.file_path)
                continue
            
            # Check file size
            if image.file_size_bytes > self.config.max_image_size_mb * 1024 * 1024:
                result.oversized_files.append(image.file_path)
            
            # Check file format
            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith('image/'):
                result.invalid_files.append(image.file_path)
            
            # Check dimensions (would need actual image loading)
            if (image.width < self.config.min_image_dimensions[0] or 
                image.height < self.config.min_image_dimensions[1]):
                result.add_warning(f"Image {image.filename} has small dimensions: {image.width}x{image.height}")
            
            if (self.config.max_image_dimensions and
                (image.width > self.config.max_image_dimensions[0] or
                 image.height > self.config.max_image_dimensions[1])):
                result.add_warning(f"Image {image.filename} has large dimensions: {image.width}x{image.height}")
    
    def _validate_annotations(self, images: List[ImageInfo], result: ValidationResult) -> None:
        """Validate annotations."""
        total_annotations = 0
        
        for image in images:
            annotation_count = len(image.annotations)
            total_annotations += annotation_count
            
            # Check for empty images
            if annotation_count == 0:
                result.empty_images.append(image.filename)
            
            # Check annotation count limits
            if annotation_count < self.config.min_annotations_per_image:
                result.add_warning(f"Image {image.filename} has too few annotations: {annotation_count}")
            
            if (self.config.max_annotations_per_image and 
                annotation_count > self.config.max_annotations_per_image):
                result.add_warning(f"Image {image.filename} has too many annotations: {annotation_count}")
            
            # Validate individual annotations
            for annotation in image.annotations:
                if not self._validate_annotation(annotation, image):
                    result.invalid_annotations.append(
                        f"{image.filename}:{annotation.annotation_id}"
                    )
        
        result.total_annotations = total_annotations
    
    def _validate_annotation(self, annotation: AnnotationInfo, image: ImageInfo) -> bool:
        """Validate a single annotation."""
        # Check bbox validity
        if len(annotation.bbox) != 4:
            return False
        
        x, y, w, h = annotation.bbox
        
        # Check for negative dimensions
        if w <= 0 or h <= 0:
            return False
        
        # Check bounds (assuming COCO format)
        if (x < 0 or y < 0 or 
            x + w > image.width or 
            y + h > image.height):
            return False
        
        # Check category ID
        if annotation.category_id < 0:
            return False
        
        return True
    
    def _validate_splits(self, images: List[ImageInfo], result: ValidationResult) -> None:
        """Validate dataset splits."""
        split_counts = {
            DatasetSplit.TRAIN: 0,
            DatasetSplit.VALIDATION: 0,
            DatasetSplit.TEST: 0
        }
        
        for image in images:
            split_counts[image.dataset_split] += 1
        
        result.train_images = split_counts[DatasetSplit.TRAIN]
        result.val_images = split_counts[DatasetSplit.VALIDATION]  
        result.test_images = split_counts[DatasetSplit.TEST]
        
        total_images = sum(split_counts.values())
        
        if total_images == 0:
            result.add_error("No images found in dataset")
            return
        
        # Check split ratios
        actual_train_ratio = result.train_images / total_images
        actual_val_ratio = result.val_images / total_images
        
        if abs(actual_train_ratio - self.config.train_split) > 0.1:
            result.add_warning(f"Train split ratio mismatch: expected {self.config.train_split:.2f}, got {actual_train_ratio:.2f}")
        
        if abs(actual_val_ratio - self.config.val_split) > 0.1:
            result.add_warning(f"Validation split ratio mismatch: expected {self.config.val_split:.2f}, got {actual_val_ratio:.2f}")
    
    def _validate_class_distribution(self, images: List[ImageInfo], result: ValidationResult) -> None:
        """Validate class distribution."""
        class_counts = {}
        
        for image in images:
            for annotation in image.annotations:
                class_id = annotation.category_id
                class_name = f"class_{class_id}"
                
                # Try to get actual class name from config
                for category in self.config.categories:
                    if category.category_id == class_id:
                        class_name = category.name
                        break
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        result.class_distribution = class_counts
        
        # Check class count requirements
        for class_name, count in class_counts.items():
            if count < self.config.min_images_per_class:
                result.add_warning(f"Class '{class_name}' has too few examples: {count}")
            
            if (self.config.max_images_per_class and 
                count > self.config.max_images_per_class):
                result.add_warning(f"Class '{class_name}' has too many examples: {count}")


class DatasetManager:
    """Manages datasets for training pipeline."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetConfig] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}
    
    def register_dataset(self, config: DatasetConfig) -> None:
        """Register a dataset configuration."""
        self._datasets[config.dataset_id] = config
        
        # Clear validation cache for this dataset
        if config.dataset_id in self._validation_cache:
            del self._validation_cache[config.dataset_id]
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetConfig]:
        """Get dataset configuration by ID."""
        return self._datasets.get(dataset_id)
    
    def list_datasets(self) -> List[DatasetConfig]:
        """List all registered datasets."""
        return list(self._datasets.values())
    
    def validate_dataset(self, dataset_id: str, use_cache: bool = True) -> Optional[ValidationResult]:
        """Validate a dataset."""
        config = self.get_dataset(dataset_id)
        if not config:
            return None
        
        # Check cache
        if use_cache and dataset_id in self._validation_cache:
            return self._validation_cache[dataset_id]
        
        # Perform validation
        validator = DatasetValidator(config)
        result = validator.validate_dataset()
        
        # Cache result
        self._validation_cache[dataset_id] = result
        
        return result
    
    def create_dataset_splits(
        self,
        dataset_id: str,
        output_dir: str,
        random_seed: int = 42
    ) -> Dict[str, List[str]]:
        """Create dataset splits from a dataset."""
        config = self.get_dataset(dataset_id)
        if not config:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        validator = DatasetValidator(config)
        images = validator._load_dataset_images()
        
        # Shuffle images
        random.seed(random_seed)
        random.shuffle(images)
        
        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * config.train_split)
        val_size = int(total_images * config.val_split)
        
        # Create splits
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Create output directories and file lists
        output_path = Path(output_dir)
        splits = {
            "train": [img.file_path for img in train_images],
            "val": [img.file_path for img in val_images],
            "test": [img.file_path for img in test_images]
        }
        
        # Save split files
        for split_name, image_paths in splits.items():
            split_file = output_path / f"{split_name}.txt"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(split_file, 'w') as f:
                for path in image_paths:
                    f.write(f"{path}\n")
        
        return splits
    
    def get_dataset_statistics(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive dataset statistics."""
        validation_result = self.validate_dataset(dataset_id)
        if not validation_result:
            return None
        
        config = self.get_dataset(dataset_id)
        
        return {
            "dataset_id": dataset_id,
            "name": config.name,
            "format": config.format,
            "total_images": validation_result.total_images,
            "total_annotations": validation_result.total_annotations,
            "splits": {
                "train": validation_result.train_images,
                "validation": validation_result.val_images,
                "test": validation_result.test_images
            },
            "classes": {
                "count": config.num_classes,
                "names": config.class_names,
                "distribution": validation_result.class_distribution
            },
            "validation": validation_result.get_summary(),
            "configuration": {
                "train_split": config.train_split,
                "val_split": config.val_split,
                "test_split": config.test_split,
                "augmentation_enabled": config.enable_augmentation,
                "supported_formats": config.supported_image_formats,
                "max_image_size_mb": config.max_image_size_mb
            }
        }