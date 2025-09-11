"""
Annotation format converters for different dataset standards.
"""
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .tools import AnnotationShape, AnnotationType


@dataclass
class ConversionConfig:
    """Configuration for annotation conversion."""
    source_format: str
    target_format: str
    image_width: int
    image_height: int
    
    # Format-specific options
    coco_category_mapping: Optional[Dict[int, str]] = None
    yolo_class_names: Optional[List[str]] = None
    pascal_voc_database: str = "Unknown"
    
    # Conversion options
    normalize_coordinates: bool = True
    include_confidence: bool = True
    merge_overlapping: bool = False
    simplify_polygons: bool = False
    
    # Quality control
    min_area_threshold: float = 0.0
    max_vertices_polygon: int = 1000


class AnnotationConverter(ABC):
    """Base class for annotation format converters."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
    
    @abstractmethod
    def convert(self, annotations: List[AnnotationShape]) -> Any:
        """Convert annotations to target format."""
        pass
    
    @abstractmethod
    def parse(self, data: Any) -> List[AnnotationShape]:
        """Parse annotations from source format."""
        pass
    
    def _filter_annotations(self, annotations: List[AnnotationShape]) -> List[AnnotationShape]:
        """Filter annotations based on conversion config."""
        filtered = []
        
        for ann in annotations:
            # Area threshold
            if ann.area < self.config.min_area_threshold:
                continue
            
            # Polygon complexity
            if (ann.annotation_type == AnnotationType.POLYGON and 
                len(ann.points) > self.config.max_vertices_polygon):
                continue
            
            # Bounds check
            if ann.bbox:
                x, y, w, h = ann.bbox
                if (x < 0 or y < 0 or 
                    x + w > self.config.image_width or 
                    y + h > self.config.image_height):
                    continue
            
            filtered.append(ann)
        
        return filtered
    
    def _merge_overlapping_annotations(self, annotations: List[AnnotationShape]) -> List[AnnotationShape]:
        """Merge overlapping annotations if enabled."""
        if not self.config.merge_overlapping:
            return annotations
        
        # Simple implementation - merge annotations with same category and high overlap
        merged = []
        processed = set()
        
        for i, ann1 in enumerate(annotations):
            if i in processed:
                continue
            
            current_group = [ann1]
            processed.add(i)
            
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check if they should be merged
                if (ann1.category_id == ann2.category_id and 
                    self._calculate_overlap(ann1, ann2) > 0.7):
                    current_group.append(ann2)
                    processed.add(j)
            
            # Merge annotations in group
            merged_ann = self._merge_annotation_group(current_group)
            merged.append(merged_ann)
        
        return merged
    
    def _calculate_overlap(self, ann1: AnnotationShape, ann2: AnnotationShape) -> float:
        """Calculate IoU overlap between two annotations."""
        if not ann1.bbox or not ann2.bbox:
            return 0.0
        
        x1, y1, w1, h1 = ann1.bbox
        x2, y2, w2, h2 = ann2.bbox
        
        # Calculate intersection
        int_x1 = max(x1, x2)
        int_y1 = max(y1, y2)
        int_x2 = min(x1 + w1, x2 + w2)
        int_y2 = min(y1 + h1, y2 + h2)
        
        if int_x2 <= int_x1 or int_y2 <= int_y1:
            return 0.0
        
        int_area = (int_x2 - int_x1) * (int_y2 - int_y1)
        union_area = w1 * h1 + w2 * h2 - int_area
        
        return int_area / union_area if union_area > 0 else 0.0
    
    def _merge_annotation_group(self, annotations: List[AnnotationShape]) -> AnnotationShape:
        """Merge a group of annotations into one."""
        if len(annotations) == 1:
            return annotations[0]
        
        # Use the first annotation as base
        merged = annotations[0]
        
        # Merge bounding boxes
        if all(ann.bbox for ann in annotations):
            x_coords = []
            y_coords = []
            
            for ann in annotations:
                x, y, w, h = ann.bbox
                x_coords.extend([x, x + w])
                y_coords.extend([y, y + h])
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            merged.bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            merged.area = (max_x - min_x) * (max_y - min_y)
        
        # Average confidence
        confidences = [ann.confidence for ann in annotations]
        merged.confidence = sum(confidences) / len(confidences)
        
        return merged


class COCOConverter(AnnotationConverter):
    """Converter for COCO format."""
    
    def convert(self, annotations: List[AnnotationShape]) -> Dict[str, Any]:
        """Convert annotations to COCO format."""
        # Filter and process annotations
        filtered_annotations = self._filter_annotations(annotations)
        if self.config.merge_overlapping:
            filtered_annotations = self._merge_overlapping_annotations(filtered_annotations)
        
        # Create COCO structure
        coco_data = {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Converted annotations",
                "contributor": "Annotation Tools",
                "url": "",
                "date_created": "2024-01-01T00:00:00+00:00"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [
                {
                    "id": 1,
                    "width": self.config.image_width,
                    "height": self.config.image_height,
                    "file_name": "image.jpg",
                    "license": 1,
                    "date_captured": "2024-01-01T00:00:00+00:00"
                }
            ],
            "categories": self._create_coco_categories(filtered_annotations),
            "annotations": []
        }
        
        # Convert annotations
        for i, annotation in enumerate(filtered_annotations):
            coco_ann = annotation.to_coco_format(image_id=1)
            coco_ann["id"] = i + 1
            
            # Add confidence if enabled
            if self.config.include_confidence:
                coco_ann["confidence"] = annotation.confidence
            
            coco_data["annotations"].append(coco_ann)
        
        return coco_data
    
    def parse(self, data: Union[Dict[str, Any], str]) -> List[AnnotationShape]:
        """Parse COCO format annotations."""
        if isinstance(data, str):
            with open(data, 'r') as f:
                coco_data = json.load(f)
        else:
            coco_data = data
        
        annotations = []
        
        # Create category mapping
        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
        
        # Parse annotations
        for coco_ann in coco_data.get("annotations", []):
            annotation = AnnotationShape()
            annotation.category_id = coco_ann["category_id"]
            annotation.category_name = categories.get(coco_ann["category_id"], "")
            annotation.bbox = coco_ann["bbox"]
            annotation.area = coco_ann["area"]
            annotation.is_crowd = bool(coco_ann.get("iscrowd", 0))
            annotation.confidence = coco_ann.get("confidence", 1.0)
            
            # Handle segmentation
            if "segmentation" in coco_ann:
                segmentation = coco_ann["segmentation"]
                if segmentation and isinstance(segmentation[0], list):
                    # Polygon format
                    annotation.annotation_type = AnnotationType.POLYGON
                    coords = segmentation[0]
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            from .tools import AnnotationPoint
                            point = AnnotationPoint(x=coords[i], y=coords[i+1])
                            annotation.points.append(point)
                else:
                    # RLE format (not fully supported)
                    annotation.annotation_type = AnnotationType.SEGMENTATION
            else:
                annotation.annotation_type = AnnotationType.BOUNDING_BOX
                # Create points from bbox
                x, y, w, h = annotation.bbox
                from .tools import AnnotationPoint
                annotation.points = [
                    AnnotationPoint(x=x, y=y),
                    AnnotationPoint(x=x+w, y=y+h)
                ]
            
            annotations.append(annotation)
        
        return annotations
    
    def _create_coco_categories(self, annotations: List[AnnotationShape]) -> List[Dict[str, Any]]:
        """Create COCO categories from annotations."""
        categories = {}
        
        for annotation in annotations:
            if annotation.category_id not in categories:
                category_name = annotation.category_name or f"class_{annotation.category_id}"
                categories[annotation.category_id] = {
                    "id": annotation.category_id,
                    "name": category_name,
                    "supercategory": "object"
                }
        
        # Use provided mapping if available
        if self.config.coco_category_mapping:
            for cat_id, cat_name in self.config.coco_category_mapping.items():
                categories[cat_id] = {
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": "object"
                }
        
        return list(categories.values())


class YOLOConverter(AnnotationConverter):
    """Converter for YOLO format."""
    
    def convert(self, annotations: List[AnnotationShape]) -> Dict[str, List[str]]:
        """Convert annotations to YOLO format."""
        # Filter and process annotations
        filtered_annotations = self._filter_annotations(annotations)
        if self.config.merge_overlapping:
            filtered_annotations = self._merge_overlapping_annotations(filtered_annotations)
        
        yolo_lines = []
        
        for annotation in filtered_annotations:
            yolo_line = annotation.to_yolo_format(
                self.config.image_width,
                self.config.image_height
            )
            
            # Add confidence if enabled
            if self.config.include_confidence:
                yolo_line += f" {annotation.confidence:.6f}"
            
            yolo_lines.append(yolo_line)
        
        return {"annotations": yolo_lines}
    
    def parse(self, data: Union[List[str], str]) -> List[AnnotationShape]:
        """Parse YOLO format annotations."""
        if isinstance(data, str):
            # Assume it's a file path
            with open(data, 'r') as f:
                lines = f.readlines()
        else:
            lines = data
        
        annotations = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            annotation = AnnotationShape()
            annotation.annotation_type = AnnotationType.BOUNDING_BOX
            annotation.category_id = int(parts[0])
            
            # Parse normalized coordinates
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to absolute coordinates
            abs_width = width * self.config.image_width
            abs_height = height * self.config.image_height
            abs_x = (x_center * self.config.image_width) - (abs_width / 2)
            abs_y = (y_center * self.config.image_height) - (abs_height / 2)
            
            annotation.bbox = [abs_x, abs_y, abs_width, abs_height]
            annotation.area = abs_width * abs_height
            
            # Parse confidence if available
            if len(parts) > 5:
                annotation.confidence = float(parts[5])
            
            # Set category name if available
            if self.config.yolo_class_names and annotation.category_id < len(self.config.yolo_class_names):
                annotation.category_name = self.config.yolo_class_names[annotation.category_id]
            
            # Create points from bbox
            from .tools import AnnotationPoint
            annotation.points = [
                AnnotationPoint(x=abs_x, y=abs_y),
                AnnotationPoint(x=abs_x + abs_width, y=abs_y + abs_height)
            ]
            
            annotations.append(annotation)
        
        return annotations


class PascalVOCConverter(AnnotationConverter):
    """Converter for Pascal VOC XML format."""
    
    def convert(self, annotations: List[AnnotationShape]) -> str:
        """Convert annotations to Pascal VOC XML format."""
        # Filter and process annotations
        filtered_annotations = self._filter_annotations(annotations)
        if self.config.merge_overlapping:
            filtered_annotations = self._merge_overlapping_annotations(filtered_annotations)
        
        # Create XML structure
        root = ET.Element("annotation")
        
        # Add folder and filename
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = "image.jpg"
        ET.SubElement(root, "path").text = "/path/to/image.jpg"
        
        # Add database info
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = self.config.pascal_voc_database
        
        # Add image size
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(self.config.image_width)
        ET.SubElement(size, "height").text = str(self.config.image_height)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(root, "segmented").text = "0"
        
        # Add objects
        for annotation in filtered_annotations:
            obj = ET.SubElement(root, "object")
            
            # Object name
            class_name = annotation.category_name or f"class_{annotation.category_id}"
            ET.SubElement(obj, "name").text = class_name
            
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            # Add confidence if enabled
            if self.config.include_confidence:
                ET.SubElement(obj, "confidence").text = f"{annotation.confidence:.6f}"
            
            # Bounding box
            if annotation.bbox:
                bndbox = ET.SubElement(obj, "bndbox")
                x, y, w, h = annotation.bbox
                ET.SubElement(bndbox, "xmin").text = str(int(x))
                ET.SubElement(bndbox, "ymin").text = str(int(y))
                ET.SubElement(bndbox, "xmax").text = str(int(x + w))
                ET.SubElement(bndbox, "ymax").text = str(int(y + h))
        
        # Convert to string
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding='unicode')
    
    def parse(self, data: Union[str, ET.Element]) -> List[AnnotationShape]:
        """Parse Pascal VOC XML annotations."""
        if isinstance(data, str):
            if Path(data).exists():
                # File path
                tree = ET.parse(data)
                root = tree.getroot()
            else:
                # XML string
                root = ET.fromstring(data)
        else:
            root = data
        
        annotations = []
        
        # Get image dimensions
        size_elem = root.find("size")
        if size_elem is not None:
            width_elem = size_elem.find("width")
            height_elem = size_elem.find("height")
            
            if width_elem is not None and height_elem is not None:
                self.config.image_width = int(width_elem.text)
                self.config.image_height = int(height_elem.text)
        
        # Parse objects
        for obj in root.findall("object"):
            annotation = AnnotationShape()
            annotation.annotation_type = AnnotationType.BOUNDING_BOX
            
            # Get class name
            name_elem = obj.find("name")
            if name_elem is not None:
                annotation.category_name = name_elem.text
                # Try to map to category ID
                if self.config.coco_category_mapping:
                    for cat_id, cat_name in self.config.coco_category_mapping.items():
                        if cat_name == annotation.category_name:
                            annotation.category_id = cat_id
                            break
            
            # Get confidence if available
            conf_elem = obj.find("confidence")
            if conf_elem is not None:
                annotation.confidence = float(conf_elem.text)
            
            # Get bounding box
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                
                width = xmax - xmin
                height = ymax - ymin
                
                annotation.bbox = [xmin, ymin, width, height]
                annotation.area = width * height
                
                # Create points from bbox
                from .tools import AnnotationPoint
                annotation.points = [
                    AnnotationPoint(x=xmin, y=ymin),
                    AnnotationPoint(x=xmax, y=ymax)
                ]
            
            annotations.append(annotation)
        
        return annotations


class FormatConverter:
    """High-level format converter that delegates to specific converters."""
    
    SUPPORTED_FORMATS = {
        "coco": COCOConverter,
        "yolo": YOLOConverter,
        "pascal_voc": PascalVOCConverter,
        "voc": PascalVOCConverter  # Alias
    }
    
    @classmethod
    def convert(
        self,
        annotations: List[AnnotationShape],
        source_format: str,
        target_format: str,
        config: ConversionConfig
    ) -> Any:
        """Convert annotations from one format to another."""
        
        # Validate formats
        if target_format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        # Update config
        config.source_format = source_format.lower()
        config.target_format = target_format.lower()
        
        # Get appropriate converter
        converter_class = self.SUPPORTED_FORMATS[target_format.lower()]
        converter = converter_class(config)
        
        # Convert annotations
        return converter.convert(annotations)
    
    @classmethod
    def parse(
        self,
        data: Any,
        source_format: str,
        config: ConversionConfig
    ) -> List[AnnotationShape]:
        """Parse annotations from a specific format."""
        
        # Validate format
        if source_format.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported source format: {source_format}")
        
        # Update config
        config.source_format = source_format.lower()
        
        # Get appropriate converter
        converter_class = self.SUPPORTED_FORMATS[source_format.lower()]
        converter = converter_class(config)
        
        # Parse annotations
        return converter.parse(data)
    
    @classmethod
    def convert_file_to_file(
        self,
        input_file: str,
        output_file: str,
        source_format: str,
        target_format: str,
        config: ConversionConfig
    ) -> None:
        """Convert annotations from one file format to another."""
        
        # Parse source file
        annotations = self.parse(input_file, source_format, config)
        
        # Convert to target format
        converted_data = self.convert(annotations, source_format, target_format, config)
        
        # Write to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if target_format.lower() in ["coco"]:
            with open(output_file, 'w') as f:
                json.dump(converted_data, f, indent=2)
                
        elif target_format.lower() in ["yolo"]:
            with open(output_file, 'w') as f:
                for line in converted_data["annotations"]:
                    f.write(line + '\n')
                    
        elif target_format.lower() in ["pascal_voc", "voc"]:
            with open(output_file, 'w') as f:
                f.write(converted_data)
        
        else:
            raise ValueError(f"Don't know how to write {target_format} format to file")
    
    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported annotation formats."""
        return list(cls.SUPPORTED_FORMATS.keys())
    
    @classmethod
    def validate_format(cls, format_name: str) -> bool:
        """Check if a format is supported."""
        return format_name.lower() in cls.SUPPORTED_FORMATS