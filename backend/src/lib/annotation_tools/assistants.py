"""
Annotation assistance using pre-trained models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

from ..ml_models import ModelFactory, ModelType, ModelVariant


class AssistanceMode(str, Enum):
    """Modes of annotation assistance."""
    DETECTION = "detection"  # Object detection assistance
    SEGMENTATION = "segmentation"  # Segmentation assistance
    INTERACTIVE = "interactive"  # Interactive segmentation with prompts
    REFINEMENT = "refinement"  # Refine existing annotations


class PromptType(str, Enum):
    """Types of prompts for interactive annotation."""
    POINTS = "points"  # Click points (positive/negative)
    BOXES = "boxes"  # Bounding box prompts
    MASKS = "masks"  # Mask prompts
    TEXT = "text"  # Text prompts (future)


@dataclass
class InteractivePrompt:
    """Prompt for interactive annotation."""
    prompt_type: PromptType
    points: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) coordinates
    point_labels: List[int] = field(default_factory=list)  # 1 for positive, 0 for negative
    boxes: List[List[float]] = field(default_factory=list)  # [x1, y1, x2, y2] format
    masks: List[List[List[int]]] = field(default_factory=list)  # Binary mask arrays
    text_prompts: List[str] = field(default_factory=list)  # Text descriptions


@dataclass
class AssistantConfig:
    """Configuration for annotation assistant."""
    model_id: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    
    # Segmentation-specific
    points_per_side: Optional[int] = None
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0
    
    # Quality control
    min_confidence: float = 0.1
    max_confidence: float = 1.0
    filter_small_objects: bool = True
    min_object_area: int = 100
    
    # Processing options
    batch_size: int = 1
    use_multimask: bool = True
    output_format: str = "coco"  # coco, yolo, polygon


@dataclass
class AssistanceRequest:
    """Request for annotation assistance."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    
    # Assistance configuration
    mode: AssistanceMode = AssistanceMode.DETECTION
    config: AssistantConfig = field(default_factory=AssistantConfig)
    
    # Interactive prompts (for interactive mode)
    prompts: Optional[InteractivePrompt] = None
    
    # Existing annotations (for refinement mode)
    existing_annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class AssistanceResult:
    """Result of annotation assistance."""
    request_id: str
    status: str = "completed"  # completed, failed, partial
    
    # Generated annotations
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_annotations: int = 0
    processing_time_ms: float = 0.0
    model_used: str = ""
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    quality_score: Optional[float] = None
    quality_issues: List[str] = field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def calculate_stats(self) -> None:
        """Calculate statistics for the result."""
        self.total_annotations = len(self.annotations)
        
        if self.annotations:
            confidences = []
            for ann in self.annotations:
                if 'confidence' in ann or 'score' in ann:
                    confidence = ann.get('confidence', ann.get('score', 0.0))
                    confidences.append(confidence)
            
            if confidences:
                self.confidence_stats = {
                    "min": min(confidences),
                    "max": max(confidences),
                    "mean": sum(confidences) / len(confidences),
                    "count": len(confidences)
                }


class AnnotationAssistant(ABC):
    """Base class for annotation assistants."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.model_wrapper = None
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the underlying ML model."""
        pass
    
    @abstractmethod
    async def assist(self, request: AssistanceRequest) -> AssistanceResult:
        """Provide annotation assistance."""
        pass
    
    def _validate_request(self, request: AssistanceRequest) -> List[str]:
        """Validate assistance request."""
        issues = []
        
        if not request.image_path:
            issues.append("Image path is required")
        
        if request.image_width <= 0 or request.image_height <= 0:
            issues.append("Valid image dimensions are required")
        
        if not request.config.model_id:
            issues.append("Model ID is required")
        
        return issues
    
    def _filter_annotations(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter annotations based on configuration."""
        filtered = []
        
        for ann in annotations:
            # Confidence filtering
            confidence = ann.get('confidence', ann.get('score', 1.0))
            if confidence < self.config.confidence_threshold:
                continue
            
            # Size filtering
            if self.config.filter_small_objects:
                area = ann.get('area', 0)
                if area < self.config.min_object_area:
                    continue
            
            filtered.append(ann)
        
        # Limit number of detections
        if len(filtered) > self.config.max_detections:
            # Sort by confidence and keep top detections
            filtered.sort(key=lambda x: x.get('confidence', x.get('score', 0)), reverse=True)
            filtered = filtered[:self.config.max_detections]
        
        return filtered


class DetectionAssistant(AnnotationAssistant):
    """Assistant for object detection using YOLO/RT-DETR models."""
    
    async def load_model(self) -> None:
        """Load detection model."""
        try:
            self.model_wrapper = ModelFactory.load_model(self.config.model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load detection model {self.config.model_id}: {str(e)}")
    
    async def assist(self, request: AssistanceRequest) -> AssistanceResult:
        """Provide detection assistance."""
        result = AssistanceResult(request_id=request.request_id, model_used=self.config.model_id)
        
        # Validate request
        issues = self._validate_request(request)
        if issues:
            result.status = "failed"
            result.error_message = "; ".join(issues)
            return result
        
        try:
            # Load model if not already loaded
            if not self.model_wrapper:
                await self.load_model()
            
            # Prepare inference parameters
            inference_params = {
                "confidence": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold
            }
            
            # Run inference
            import time
            start_time = time.time()
            
            predictions = ModelFactory.predict(
                self.config.model_id,
                request.image_path,
                **inference_params
            )
            
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            # Convert predictions to annotation format
            raw_annotations = predictions.get("predictions", [])
            annotations = self._convert_detections_to_annotations(
                raw_annotations,
                request.image_width,
                request.image_height
            )
            
            # Filter annotations
            filtered_annotations = self._filter_annotations(annotations)
            result.annotations = filtered_annotations
            
            # Calculate statistics
            result.calculate_stats()
            
            # Quality assessment
            result.quality_score = self._assess_detection_quality(filtered_annotations)
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
        
        return result
    
    def _convert_detections_to_annotations(
        self,
        detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """Convert model predictions to annotation format."""
        annotations = []
        
        for i, detection in enumerate(detections):
            # Extract bounding box
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Convert to COCO format [x, y, width, height]
            coco_bbox = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)
            
            # Create annotation
            annotation = {
                "id": i + 1,
                "image_id": 1,  # Would be actual image ID
                "category_id": detection.get("class_id", 0),
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "confidence": detection.get("confidence", 1.0),
                "class_name": detection.get("class_name", f"class_{detection.get('class_id', 0)}")
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def _assess_detection_quality(self, annotations: List[Dict[str, Any]]) -> float:
        """Assess quality of detections."""
        if not annotations:
            return 0.0
        
        # Simple quality score based on confidence distribution
        confidences = [ann.get("confidence", 0.0) for ann in annotations]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Quality factors
        confidence_factor = min(avg_confidence, 1.0)
        count_factor = min(len(annotations) / 10.0, 1.0)  # Normalize by expected count
        
        return (confidence_factor * 0.7 + count_factor * 0.3)


class SAM2Assistant(AnnotationAssistant):
    """Assistant for segmentation using SAM2 models."""
    
    async def load_model(self) -> None:
        """Load SAM2 model."""
        try:
            self.model_wrapper = ModelFactory.load_model(self.config.model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2 model {self.config.model_id}: {str(e)}")
    
    async def assist(self, request: AssistanceRequest) -> AssistanceResult:
        """Provide segmentation assistance."""
        result = AssistanceResult(request_id=request.request_id, model_used=self.config.model_id)
        
        # Validate request
        issues = self._validate_request(request)
        if issues:
            result.status = "failed"
            result.error_message = "; ".join(issues)
            return result
        
        try:
            # Load model if not already loaded
            if not self.model_wrapper:
                await self.load_model()
            
            # Prepare inference parameters based on mode
            inference_params = {}
            
            if request.mode == AssistanceMode.INTERACTIVE and request.prompts:
                # Interactive segmentation with prompts
                if request.prompts.points:
                    inference_params["points"] = request.prompts.points
                if request.prompts.boxes:
                    inference_params["boxes"] = request.prompts.boxes
                if request.prompts.masks:
                    inference_params["masks"] = request.prompts.masks
            
            elif request.mode == AssistanceMode.SEGMENTATION:
                # Automatic segmentation
                if self.config.points_per_side:
                    inference_params["points_per_side"] = self.config.points_per_side
            
            # Run inference
            import time
            start_time = time.time()
            
            predictions = ModelFactory.predict(
                self.config.model_id,
                request.image_path,
                **inference_params
            )
            
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            # Convert predictions to annotation format
            raw_masks = predictions.get("predictions", [])
            annotations = self._convert_masks_to_annotations(
                raw_masks,
                request.image_width,
                request.image_height
            )
            
            # Filter annotations
            filtered_annotations = self._filter_annotations(annotations)
            result.annotations = filtered_annotations
            
            # Calculate statistics
            result.calculate_stats()
            
            # Quality assessment
            result.quality_score = self._assess_segmentation_quality(filtered_annotations)
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
        
        return result
    
    def _convert_masks_to_annotations(
        self,
        masks: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """Convert mask predictions to annotation format."""
        annotations = []
        
        for i, mask_data in enumerate(masks):
            # Extract mask information
            mask = mask_data.get("mask", [])
            confidence = mask_data.get("confidence", 1.0)
            area = mask_data.get("area", 0)
            
            if not mask or area < self.config.min_mask_region_area:
                continue
            
            # Convert mask to polygon or RLE format
            segmentation = self._mask_to_segmentation(mask)
            
            # Calculate bounding box from mask
            bbox = self._mask_to_bbox(mask, image_width, image_height)
            
            # Create annotation
            annotation = {
                "id": i + 1,
                "image_id": 1,  # Would be actual image ID
                "category_id": 0,  # SAM2 is class-agnostic
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "confidence": confidence
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def _mask_to_segmentation(self, mask: List[List[int]]) -> List[List[float]]:
        """Convert binary mask to polygon segmentation."""
        # Simplified polygon extraction
        # In real implementation, would use cv2.findContours or similar
        
        # For now, return a simple rectangular approximation
        if not mask or not mask[0]:
            return [[]]
        
        height = len(mask)
        width = len(mask[0])
        
        # Find bounding rectangle of mask
        min_y, max_y = height, 0
        min_x, max_x = width, 0
        
        for y in range(height):
            for x in range(width):
                if mask[y][x] == 1:
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
        
        # Create simple rectangular polygon
        polygon = [
            float(min_x), float(min_y),  # Top-left
            float(max_x), float(min_y),  # Top-right
            float(max_x), float(max_y),  # Bottom-right
            float(min_x), float(max_y)   # Bottom-left
        ]
        
        return [polygon]
    
    def _mask_to_bbox(self, mask: List[List[int]], image_width: int, image_height: int) -> List[float]:
        """Calculate bounding box from mask."""
        if not mask or not mask[0]:
            return [0.0, 0.0, 0.0, 0.0]
        
        height = len(mask)
        width = len(mask[0])
        
        # Find bounding rectangle
        min_y, max_y = height, 0
        min_x, max_x = width, 0
        
        for y in range(height):
            for x in range(width):
                if mask[y][x] == 1:
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
        
        # Convert to COCO format [x, y, width, height]
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        return [float(min_x), float(min_y), float(bbox_width), float(bbox_height)]
    
    def _assess_segmentation_quality(self, annotations: List[Dict[str, Any]]) -> float:
        """Assess quality of segmentations."""
        if not annotations:
            return 0.0
        
        # Quality factors
        confidences = [ann.get("confidence", 0.0) for ann in annotations]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Check for reasonable mask sizes
        areas = [ann.get("area", 0) for ann in annotations]
        avg_area = sum(areas) / len(areas) if areas else 0.0
        
        # Normalize area (assume typical object is 1-10% of image)
        total_image_area = 640 * 640  # Default size
        area_factor = min(avg_area / (total_image_area * 0.05), 1.0)
        
        return (avg_confidence * 0.6 + area_factor * 0.4)


class HybridAssistant(AnnotationAssistant):
    """Hybrid assistant combining detection and segmentation."""
    
    def __init__(self, config: AssistantConfig):
        super().__init__(config)
        self.detection_assistant = None
        self.segmentation_assistant = None
    
    async def load_model(self) -> None:
        """Load both detection and segmentation models."""
        # This would typically use separate model configs
        # For now, assume the model_id specifies a capable model
        try:
            self.model_wrapper = ModelFactory.load_model(self.config.model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load hybrid model {self.config.model_id}: {str(e)}")
    
    async def assist(self, request: AssistanceRequest) -> AssistanceResult:
        """Provide hybrid detection + segmentation assistance."""
        result = AssistanceResult(request_id=request.request_id, model_used=self.config.model_id)
        
        # Validate request
        issues = self._validate_request(request)
        if issues:
            result.status = "failed"
            result.error_message = "; ".join(issues)
            return result
        
        try:
            # Load model if not already loaded
            if not self.model_wrapper:
                await self.load_model()
            
            # First, get detections to identify objects
            detection_params = {
                "confidence": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold
            }
            
            import time
            start_time = time.time()
            
            # Get detections
            detection_predictions = ModelFactory.predict(
                self.config.model_id,
                request.image_path,
                **detection_params
            )
            
            # For each detection, get segmentation
            annotations = []
            detections = detection_predictions.get("predictions", [])
            
            for i, detection in enumerate(detections):
                # Use detection bbox as prompt for segmentation
                bbox = detection.get("bbox", [])
                if len(bbox) == 4:
                    # Create segmentation prompt
                    seg_params = {"boxes": [bbox]}
                    
                    # Get segmentation for this detection
                    seg_predictions = ModelFactory.predict(
                        self.config.model_id,
                        request.image_path,
                        **seg_params
                    )
                    
                    # Combine detection and segmentation info
                    masks = seg_predictions.get("predictions", [])
                    if masks:
                        mask_data = masks[0]  # Take first/best mask
                        
                        annotation = {
                            "id": i + 1,
                            "image_id": 1,
                            "category_id": detection.get("class_id", 0),
                            "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # Convert to COCO format
                            "area": mask_data.get("area", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                            "segmentation": self._mask_to_segmentation(mask_data.get("mask", [])),
                            "iscrowd": 0,
                            "confidence": detection.get("confidence", 1.0),
                            "class_name": detection.get("class_name", f"class_{detection.get('class_id', 0)}")
                        }
                        
                        annotations.append(annotation)
            
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            # Filter annotations
            filtered_annotations = self._filter_annotations(annotations)
            result.annotations = filtered_annotations
            
            # Calculate statistics
            result.calculate_stats()
            
            # Quality assessment (combine detection and segmentation quality)
            result.quality_score = self._assess_hybrid_quality(filtered_annotations)
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
        
        return result
    
    def _assess_hybrid_quality(self, annotations: List[Dict[str, Any]]) -> float:
        """Assess quality of hybrid annotations."""
        if not annotations:
            return 0.0
        
        # Combine detection and segmentation quality metrics
        confidences = [ann.get("confidence", 0.0) for ann in annotations]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Check if segmentations exist and are reasonable
        has_segmentation = all("segmentation" in ann for ann in annotations)
        
        quality_factors = [
            avg_confidence * 0.5,  # Detection confidence
            1.0 if has_segmentation else 0.5,  # Segmentation presence
            min(len(annotations) / 5.0, 1.0) * 0.3  # Reasonable number of objects
        ]
        
        return sum(quality_factors) / len(quality_factors)