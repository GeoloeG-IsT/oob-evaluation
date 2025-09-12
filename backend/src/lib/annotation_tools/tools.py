"""
Manual annotation tools and validation utilities.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
import math


class AnnotationType(str, Enum):
    """Types of annotations."""
    BOUNDING_BOX = "bbox"
    POLYGON = "polygon"
    SEGMENTATION = "segmentation"
    POINT = "point"
    POLYLINE = "polyline"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"


class ValidationLevel(str, Enum):
    """Levels of annotation validation."""
    BASIC = "basic"      # Basic geometric validation
    STANDARD = "standard"  # Standard quality checks
    STRICT = "strict"    # Strict quality requirements
    CUSTOM = "custom"    # Custom validation rules


@dataclass
class AnnotationPoint:
    """A point in an annotation."""
    x: float
    y: float
    label: Optional[str] = None  # For labeled points
    confidence: float = 1.0
    
    def distance_to(self, other: "AnnotationPoint") -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "confidence": self.confidence
        }


@dataclass
class AnnotationShape:
    """Base annotation shape."""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    annotation_type: AnnotationType = AnnotationType.BOUNDING_BOX
    category_id: int = 0
    category_name: str = ""
    confidence: float = 1.0
    is_crowd: bool = False
    
    # Geometric properties
    points: List[AnnotationPoint] = field(default_factory=list)
    bbox: Optional[List[float]] = None  # [x, y, width, height]
    area: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "manual"  # manual, assisted, automatic
    validated: bool = False
    validation_issues: List[str] = field(default_factory=list)
    
    def calculate_bbox(self) -> List[float]:
        """Calculate bounding box from points."""
        if not self.points:
            return [0.0, 0.0, 0.0, 0.0]
        
        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        return [min_x, min_y, width, height]
    
    def calculate_area(self) -> float:
        """Calculate area of the annotation."""
        if self.annotation_type == AnnotationType.BOUNDING_BOX:
            if self.bbox and len(self.bbox) >= 4:
                return self.bbox[2] * self.bbox[3]
            else:
                bbox = self.calculate_bbox()
                return bbox[2] * bbox[3]
        
        elif self.annotation_type == AnnotationType.POLYGON:
            # Shoelace formula for polygon area
            if len(self.points) < 3:
                return 0.0
            
            area = 0.0
            n = len(self.points)
            
            for i in range(n):
                j = (i + 1) % n
                area += self.points[i].x * self.points[j].y
                area -= self.points[j].x * self.points[i].y
            
            return abs(area) / 2.0
        
        elif self.annotation_type == AnnotationType.CIRCLE:
            # Assume first point is center, second point defines radius
            if len(self.points) >= 2:
                radius = self.points[0].distance_to(self.points[1])
                return math.pi * radius * radius
            return 0.0
        
        return 0.0
    
    def is_valid(self) -> bool:
        """Check if annotation is geometrically valid."""
        if not self.points:
            return False
        
        if self.annotation_type == AnnotationType.BOUNDING_BOX:
            return len(self.points) >= 2
        elif self.annotation_type == AnnotationType.POLYGON:
            return len(self.points) >= 3
        elif self.annotation_type == AnnotationType.POINT:
            return len(self.points) >= 1
        elif self.annotation_type == AnnotationType.CIRCLE:
            return len(self.points) >= 2
        
        return True
    
    def to_coco_format(self, image_id: int = 1) -> Dict[str, Any]:
        """Convert to COCO annotation format."""
        if not self.bbox:
            self.bbox = self.calculate_bbox()
        
        if self.area == 0.0:
            self.area = self.calculate_area()
        
        annotation = {
            "id": hash(self.annotation_id) % (10 ** 10),  # Convert to numeric ID
            "image_id": image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "area": self.area,
            "iscrowd": 1 if self.is_crowd else 0
        }
        
        # Add segmentation for polygons
        if self.annotation_type == AnnotationType.POLYGON:
            segmentation = []
            for point in self.points:
                segmentation.extend([point.x, point.y])
            annotation["segmentation"] = [segmentation]
        
        return annotation
    
    def to_yolo_format(self, image_width: int, image_height: int) -> str:
        """Convert to YOLO annotation format."""
        if not self.bbox:
            self.bbox = self.calculate_bbox()
        
        # Convert to normalized coordinates
        x_center = (self.bbox[0] + self.bbox[2] / 2) / image_width
        y_center = (self.bbox[1] + self.bbox[3] / 2) / image_height
        width = self.bbox[2] / image_width
        height = self.bbox[3] / image_height
        
        return f"{self.category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


class AnnotationTool(ABC):
    """Base class for annotation tools."""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.current_annotation: Optional[AnnotationShape] = None
        self.annotations: List[AnnotationShape] = []
    
    @abstractmethod
    def start_annotation(self, category_id: int, category_name: str = "") -> str:
        """Start creating a new annotation."""
        pass
    
    @abstractmethod
    def add_point(self, x: float, y: float) -> bool:
        """Add a point to the current annotation."""
        pass
    
    @abstractmethod
    def finish_annotation(self) -> Optional[AnnotationShape]:
        """Finish the current annotation."""
        pass
    
    def cancel_annotation(self) -> None:
        """Cancel the current annotation."""
        self.current_annotation = None
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation by ID."""
        for i, ann in enumerate(self.annotations):
            if ann.annotation_id == annotation_id:
                del self.annotations[i]
                return True
        return False
    
    def get_annotations(self) -> List[AnnotationShape]:
        """Get all annotations."""
        return self.annotations.copy()
    
    def clear_annotations(self) -> None:
        """Clear all annotations."""
        self.annotations.clear()
        self.current_annotation = None


class BoundingBoxTool(AnnotationTool):
    """Tool for creating bounding box annotations."""
    
    def __init__(self):
        super().__init__("BoundingBox")
    
    def start_annotation(self, category_id: int, category_name: str = "") -> str:
        """Start creating a bounding box annotation."""
        self.current_annotation = AnnotationShape(
            annotation_type=AnnotationType.BOUNDING_BOX,
            category_id=category_id,
            category_name=category_name
        )
        return self.current_annotation.annotation_id
    
    def add_point(self, x: float, y: float) -> bool:
        """Add a corner point to the bounding box."""
        if not self.current_annotation:
            return False
        
        point = AnnotationPoint(x=x, y=y)
        self.current_annotation.points.append(point)
        
        # Bounding box needs exactly 2 points (opposite corners)
        return len(self.current_annotation.points) <= 2
    
    def finish_annotation(self) -> Optional[AnnotationShape]:
        """Finish the bounding box annotation."""
        if not self.current_annotation or len(self.current_annotation.points) < 2:
            return None
        
        # Calculate bounding box
        self.current_annotation.bbox = self.current_annotation.calculate_bbox()
        self.current_annotation.area = self.current_annotation.calculate_area()
        
        # Add to annotations list
        self.annotations.append(self.current_annotation)
        
        # Return and clear current annotation
        annotation = self.current_annotation
        self.current_annotation = None
        return annotation


class PolygonTool(AnnotationTool):
    """Tool for creating polygon annotations."""
    
    def __init__(self):
        super().__init__("Polygon")
        self.min_points = 3
        self.max_points = 1000
    
    def start_annotation(self, category_id: int, category_name: str = "") -> str:
        """Start creating a polygon annotation."""
        self.current_annotation = AnnotationShape(
            annotation_type=AnnotationType.POLYGON,
            category_id=category_id,
            category_name=category_name
        )
        return self.current_annotation.annotation_id
    
    def add_point(self, x: float, y: float) -> bool:
        """Add a vertex to the polygon."""
        if not self.current_annotation:
            return False
        
        # Check if we're at the maximum points
        if len(self.current_annotation.points) >= self.max_points:
            return False
        
        point = AnnotationPoint(x=x, y=y)
        self.current_annotation.points.append(point)
        return True
    
    def can_finish(self) -> bool:
        """Check if polygon can be finished."""
        return (self.current_annotation and 
                len(self.current_annotation.points) >= self.min_points)
    
    def finish_annotation(self) -> Optional[AnnotationShape]:
        """Finish the polygon annotation."""
        if not self.can_finish():
            return None
        
        # Calculate bounding box and area
        self.current_annotation.bbox = self.current_annotation.calculate_bbox()
        self.current_annotation.area = self.current_annotation.calculate_area()
        
        # Add to annotations list
        self.annotations.append(self.current_annotation)
        
        # Return and clear current annotation
        annotation = self.current_annotation
        self.current_annotation = None
        return annotation


class SegmentationTool(AnnotationTool):
    """Tool for creating segmentation masks."""
    
    def __init__(self):
        super().__init__("Segmentation")
        self.brush_size = 5
        self.mask_points: List[Tuple[int, int]] = []
    
    def start_annotation(self, category_id: int, category_name: str = "") -> str:
        """Start creating a segmentation annotation."""
        self.current_annotation = AnnotationShape(
            annotation_type=AnnotationType.SEGMENTATION,
            category_id=category_id,
            category_name=category_name
        )
        self.mask_points.clear()
        return self.current_annotation.annotation_id
    
    def set_brush_size(self, size: int) -> None:
        """Set brush size for segmentation."""
        self.brush_size = max(1, min(size, 50))
    
    def add_point(self, x: float, y: float) -> bool:
        """Add a point to the segmentation mask."""
        if not self.current_annotation:
            return False
        
        # Add brush stroke points around the clicked point
        brush_points = self._get_brush_points(int(x), int(y))
        self.mask_points.extend(brush_points)
        
        # Also add to annotation points for bbox calculation
        point = AnnotationPoint(x=x, y=y)
        self.current_annotation.points.append(point)
        
        return True
    
    def _get_brush_points(self, center_x: int, center_y: int) -> List[Tuple[int, int]]:
        """Get all points covered by brush stroke."""
        points = []
        radius = self.brush_size // 2
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    points.append((center_x + dx, center_y + dy))
        
        return points
    
    def finish_annotation(self) -> Optional[AnnotationShape]:
        """Finish the segmentation annotation."""
        if not self.current_annotation or not self.mask_points:
            return None
        
        # Convert mask points to segmentation format
        # This is simplified - real implementation would create proper mask
        self.current_annotation.metadata["mask_points"] = self.mask_points
        
        # Calculate bounding box and area
        self.current_annotation.bbox = self.current_annotation.calculate_bbox()
        self.current_annotation.area = len(self.mask_points)  # Approximate area
        
        # Add to annotations list
        self.annotations.append(self.current_annotation)
        
        # Return and clear current annotation
        annotation = self.current_annotation
        self.current_annotation = None
        self.mask_points.clear()
        return annotation


class PointTool(AnnotationTool):
    """Tool for creating point annotations."""
    
    def __init__(self):
        super().__init__("Point")
    
    def start_annotation(self, category_id: int, category_name: str = "") -> str:
        """Start creating a point annotation."""
        self.current_annotation = AnnotationShape(
            annotation_type=AnnotationType.POINT,
            category_id=category_id,
            category_name=category_name
        )
        return self.current_annotation.annotation_id
    
    def add_point(self, x: float, y: float) -> bool:
        """Add the point location."""
        if not self.current_annotation:
            return False
        
        # Point annotation only needs one point
        if len(self.current_annotation.points) >= 1:
            return False
        
        point = AnnotationPoint(x=x, y=y)
        self.current_annotation.points.append(point)
        return True
    
    def finish_annotation(self) -> Optional[AnnotationShape]:
        """Finish the point annotation."""
        if not self.current_annotation or len(self.current_annotation.points) != 1:
            return None
        
        # Create small bounding box around point
        point = self.current_annotation.points[0]
        self.current_annotation.bbox = [point.x - 2, point.y - 2, 4, 4]
        self.current_annotation.area = 4.0
        
        # Add to annotations list
        self.annotations.append(self.current_annotation)
        
        # Return and clear current annotation
        annotation = self.current_annotation
        self.current_annotation = None
        return annotation


@dataclass
class ValidationRule:
    """Rule for validating annotations."""
    rule_name: str
    rule_type: str  # geometry, size, position, quality
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    warning_message: str = ""
    is_critical: bool = True  # Critical rules must pass


class AnnotationValidator:
    """Validates annotation quality and correctness."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules based on level."""
        # Basic rules (always active)
        self.rules.extend([
            ValidationRule(
                rule_name="valid_geometry",
                rule_type="geometry",
                error_message="Annotation has invalid geometry",
                is_critical=True
            ),
            ValidationRule(
                rule_name="minimum_size",
                rule_type="size",
                parameters={"min_area": 4},
                error_message="Annotation is too small",
                warning_message="Annotation may be too small for reliable detection"
            ),
            ValidationRule(
                rule_name="within_image_bounds",
                rule_type="position",
                error_message="Annotation extends outside image bounds",
                is_critical=True
            )
        ])
        
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            self.rules.extend([
                ValidationRule(
                    rule_name="reasonable_aspect_ratio",
                    rule_type="geometry",
                    parameters={"max_ratio": 20.0},
                    warning_message="Annotation has unusual aspect ratio"
                ),
                ValidationRule(
                    rule_name="minimum_confidence",
                    rule_type="quality",
                    parameters={"min_confidence": 0.5},
                    warning_message="Annotation has low confidence"
                )
            ])
        
        if self.validation_level == ValidationLevel.STRICT:
            self.rules.extend([
                ValidationRule(
                    rule_name="polygon_complexity",
                    rule_type="geometry",
                    parameters={"max_vertices": 100},
                    error_message="Polygon is too complex"
                ),
                ValidationRule(
                    rule_name="high_quality_required",
                    rule_type="quality",
                    parameters={"min_confidence": 0.8},
                    error_message="Annotation does not meet quality standards"
                )
            ])
    
    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)
    
    def validate_annotation(
        self,
        annotation: AnnotationShape,
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """Validate a single annotation."""
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "rule_results": {}
        }
        
        for rule in self.rules:
            rule_result = self._apply_rule(rule, annotation, image_width, image_height)
            result["rule_results"][rule.rule_name] = rule_result
            
            if not rule_result["passed"]:
                if rule.is_critical:
                    result["errors"].append(rule_result["message"])
                    result["is_valid"] = False
                else:
                    result["warnings"].append(rule_result["message"])
        
        return result
    
    def _apply_rule(
        self,
        rule: ValidationRule,
        annotation: AnnotationShape,
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """Apply a validation rule to an annotation."""
        result = {"passed": True, "message": ""}
        
        try:
            if rule.rule_name == "valid_geometry":
                if not annotation.is_valid():
                    result["passed"] = False
                    result["message"] = rule.error_message
            
            elif rule.rule_name == "minimum_size":
                min_area = rule.parameters.get("min_area", 4)
                if annotation.area < min_area:
                    result["passed"] = False
                    result["message"] = rule.warning_message or rule.error_message
            
            elif rule.rule_name == "within_image_bounds":
                if annotation.bbox:
                    x, y, w, h = annotation.bbox
                    if (x < 0 or y < 0 or 
                        x + w > image_width or y + h > image_height):
                        result["passed"] = False
                        result["message"] = rule.error_message
            
            elif rule.rule_name == "reasonable_aspect_ratio":
                if annotation.bbox and len(annotation.bbox) >= 4:
                    width, height = annotation.bbox[2], annotation.bbox[3]
                    if height > 0:
                        ratio = max(width / height, height / width)
                        max_ratio = rule.parameters.get("max_ratio", 20.0)
                        if ratio > max_ratio:
                            result["passed"] = False
                            result["message"] = rule.warning_message
            
            elif rule.rule_name == "minimum_confidence":
                min_conf = rule.parameters.get("min_confidence", 0.5)
                if annotation.confidence < min_conf:
                    result["passed"] = False
                    result["message"] = rule.warning_message or rule.error_message
            
            elif rule.rule_name == "polygon_complexity":
                if annotation.annotation_type == AnnotationType.POLYGON:
                    max_vertices = rule.parameters.get("max_vertices", 100)
                    if len(annotation.points) > max_vertices:
                        result["passed"] = False
                        result["message"] = rule.error_message
            
            elif rule.rule_name == "high_quality_required":
                min_conf = rule.parameters.get("min_confidence", 0.8)
                if annotation.confidence < min_conf:
                    result["passed"] = False
                    result["message"] = rule.error_message
        
        except Exception as e:
            result["passed"] = False
            result["message"] = f"Validation error: {str(e)}"
        
        return result
    
    def validate_annotations(
        self,
        annotations: List[AnnotationShape],
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """Validate multiple annotations."""
        results = {
            "overall_valid": True,
            "total_annotations": len(annotations),
            "valid_annotations": 0,
            "invalid_annotations": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "annotation_results": []
        }
        
        for annotation in annotations:
            ann_result = self.validate_annotation(annotation, image_width, image_height)
            results["annotation_results"].append({
                "annotation_id": annotation.annotation_id,
                "validation_result": ann_result
            })
            
            if ann_result["is_valid"]:
                results["valid_annotations"] += 1
            else:
                results["invalid_annotations"] += 1
                results["overall_valid"] = False
            
            results["total_errors"] += len(ann_result["errors"])
            results["total_warnings"] += len(ann_result["warnings"])
        
        return results