"""
Export service for annotation data in various formats.
"""
import json
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
import os

from ..models.annotation import annotation_storage
from ..models.image import image_storage
from ..models.dataset import dataset_storage
from ..lib.annotation_tools.converters import AnnotationConverter, ConversionConfig
from ..lib.annotation_tools.tools import AnnotationShape, AnnotationType


class ExportService:
    """Service for handling annotation data exports."""
    
    def __init__(self):
        self.annotation_storage = annotation_storage
        self.image_storage = image_storage
        self.dataset_storage = dataset_storage
        self.supported_formats = ["coco", "yolo", "pascal_voc", "labelme", "csv"]
    
    def export_annotations(self, export_format: str, 
                         image_ids: Optional[List[str]] = None,
                         dataset_id: Optional[str] = None,
                         class_filter: Optional[List[str]] = None,
                         include_images: bool = False) -> Dict[str, Any]:
        """Export annotations in the specified format."""
        # Validate format
        if export_format not in self.supported_formats:
            raise ValueError(f"Unsupported export format. Supported: {self.supported_formats}")
        
        # Collect annotations
        annotations_data = self._collect_annotations(image_ids, dataset_id, class_filter)
        
        if not annotations_data:
            return {
                "export_format": export_format,
                "status": "no_data",
                "message": "No annotations found matching the criteria"
            }
        
        # Create export based on format
        export_result = None
        
        if export_format == "coco":
            export_result = self._export_coco_format(annotations_data, include_images)
        elif export_format == "yolo":
            export_result = self._export_yolo_format(annotations_data, include_images)
        elif export_format == "pascal_voc":
            export_result = self._export_pascal_voc_format(annotations_data, include_images)
        elif export_format == "labelme":
            export_result = self._export_labelme_format(annotations_data, include_images)
        elif export_format == "csv":
            export_result = self._export_csv_format(annotations_data)
        
        return {
            "export_format": export_format,
            "status": "completed",
            "total_images": len(annotations_data),
            "total_annotations": sum(len(data["annotations"]) for data in annotations_data.values()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **export_result
        }
    
    def _collect_annotations(self, image_ids: Optional[List[str]] = None,
                           dataset_id: Optional[str] = None,
                           class_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect annotations and image data for export."""
        annotations_data = {}
        
        # Determine which images to process
        if image_ids:
            target_image_ids = image_ids
        elif dataset_id:
            # Get all images in dataset
            dataset = self.dataset_storage.get_by_id(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            target_image_ids = dataset.image_ids
        else:
            # Get all images
            all_images, _ = self.image_storage.list_images(limit=10000)
            target_image_ids = [img.id for img in all_images]
        
        # Collect annotations for each image
        for image_id in target_image_ids:
            # Get image info
            image = self.image_storage.get_by_id(image_id)
            if not image:
                continue
            
            # Get annotations for this image
            image_annotations = self.annotation_storage.get_annotations_by_image(image_id)
            
            if not image_annotations:
                continue
            
            # Filter by class if specified
            filtered_annotations = []
            for annotation in image_annotations:
                if class_filter:
                    # Check if any of the annotation's classes match the filter
                    matching_classes = [cls for cls in annotation.class_labels if cls in class_filter]
                    if not matching_classes:
                        continue
                    
                    # Create filtered annotation with only matching classes
                    filtered_annotation = self._filter_annotation_by_classes(annotation, class_filter)
                    if filtered_annotation:
                        filtered_annotations.append(filtered_annotation)
                else:
                    filtered_annotations.append(annotation)
            
            if filtered_annotations:
                annotations_data[image_id] = {
                    "image": image,
                    "annotations": filtered_annotations
                }
        
        return annotations_data
    
    def _filter_annotation_by_classes(self, annotation, class_filter: List[str]):
        """Filter annotation to only include specified classes."""
        filtered_labels = []
        filtered_boxes = []
        filtered_segments = []
        filtered_confidences = []
        
        for i, label in enumerate(annotation.class_labels):
            if label in class_filter:
                filtered_labels.append(label)
                
                if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                    filtered_boxes.append(annotation.bounding_boxes[i])
                
                if annotation.segments and i < len(annotation.segments):
                    filtered_segments.append(annotation.segments[i])
                
                if annotation.confidence_scores and i < len(annotation.confidence_scores):
                    filtered_confidences.append(annotation.confidence_scores[i])
        
        if not filtered_labels:
            return None
        
        # Create filtered annotation (simplified for TDD GREEN phase)
        return type(annotation)(
            image_id=annotation.image_id,
            bounding_boxes=filtered_boxes,
            segments=filtered_segments,
            class_labels=filtered_labels,
            confidence_scores=filtered_confidences if filtered_confidences else None,
            user_tag=annotation.user_tag,
            metadata=annotation.metadata
        )
    
    def _export_coco_format(self, annotations_data: Dict[str, Any], 
                          include_images: bool) -> Dict[str, Any]:
        """Export annotations in COCO format."""
        # Build COCO structure
        coco_data = {
            "info": {
                "description": "Exported annotations from ML Evaluation Platform",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ML Evaluation Platform",
                "date_created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Collect all unique classes
        all_classes = set()
        for data in annotations_data.values():
            for annotation in data["annotations"]:
                all_classes.update(annotation.class_labels)
        
        # Create category mappings
        class_to_id = {cls: idx + 1 for idx, cls in enumerate(sorted(all_classes))}
        coco_data["categories"] = [
            {"id": class_id, "name": class_name, "supercategory": "object"}
            for class_name, class_id in class_to_id.items()
        ]
        
        # Process each image
        annotation_id = 1
        for image_id, data in annotations_data.items():
            image = data["image"]
            annotations = data["annotations"]
            
            # Add image info
            coco_data["images"].append({
                "id": image_id,
                "width": image.width,
                "height": image.height,
                "file_name": image.filename,
                "license": 1,
                "date_captured": image.created_at if hasattr(image, 'created_at') else ""
            })
            
            # Add annotations
            for annotation in annotations:
                for i, class_label in enumerate(annotation.class_labels):
                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_to_id[class_label],
                        "iscrowd": 0
                    }
                    
                    # Add bounding box if available
                    if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                        bbox = annotation.bounding_boxes[i]
                        # Convert to COCO format [x, y, width, height]
                        coco_annotation["bbox"] = [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
                        coco_annotation["area"] = bbox["width"] * bbox["height"]
                    
                    # Add segmentation if available
                    if annotation.segments and i < len(annotation.segments):
                        segment = annotation.segments[i]
                        # Convert to COCO polygon format
                        if segment.get("points"):
                            coco_annotation["segmentation"] = [
                                [coord for point in segment["points"] for coord in [point["x"], point["y"]]]
                            ]
                    
                    coco_data["annotations"].append(coco_annotation)
                    annotation_id += 1
        
        # Create export file
        export_filename = f"coco_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_content = json.dumps(coco_data, indent=2)
        
        return {
            "format": "coco",
            "filename": export_filename,
            "content": export_content,
            "content_type": "application/json",
            "categories": list(all_classes)
        }
    
    def _export_yolo_format(self, annotations_data: Dict[str, Any], 
                          include_images: bool) -> Dict[str, Any]:
        """Export annotations in YOLO format."""
        # Collect all unique classes
        all_classes = set()
        for data in annotations_data.values():
            for annotation in data["annotations"]:
                all_classes.update(annotation.class_labels)
        
        class_names = sorted(all_classes)
        class_to_id = {cls: idx for idx, cls in enumerate(class_names)}
        
        # Create YOLO format files
        yolo_files = {}
        
        # Create classes.txt
        yolo_files["classes.txt"] = "\n".join(class_names)
        
        # Create annotation files for each image
        for image_id, data in annotations_data.items():
            image = data["image"]
            annotations = data["annotations"]
            
            yolo_lines = []
            for annotation in annotations:
                for i, class_label in enumerate(annotation.class_labels):
                    if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                        bbox = annotation.bounding_boxes[i]
                        
                        # Convert to YOLO format (normalized center x, center y, width, height)
                        center_x = (bbox["x"] + bbox["width"] / 2) / image.width
                        center_y = (bbox["y"] + bbox["height"] / 2) / image.height
                        norm_width = bbox["width"] / image.width
                        norm_height = bbox["height"] / image.height
                        
                        class_id = class_to_id[class_label]
                        yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                        yolo_lines.append(yolo_line)
            
            # Create annotation file for this image
            annotation_filename = f"{Path(image.filename).stem}.txt"
            yolo_files[annotation_filename] = "\n".join(yolo_lines)
        
        # Create zip file content
        zip_filename = f"yolo_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_content = self._create_zip_from_files(yolo_files)
        
        return {
            "format": "yolo",
            "filename": zip_filename,
            "content": zip_content,
            "content_type": "application/zip",
            "categories": class_names,
            "files": list(yolo_files.keys())
        }
    
    def _export_pascal_voc_format(self, annotations_data: Dict[str, Any], 
                                include_images: bool) -> Dict[str, Any]:
        """Export annotations in Pascal VOC format."""
        voc_files = {}
        
        for image_id, data in annotations_data.items():
            image = data["image"]
            annotations = data["annotations"]
            
            # Create XML structure
            root = ET.Element("annotation")
            
            # Add folder and filename
            ET.SubElement(root, "folder").text = "images"
            ET.SubElement(root, "filename").text = image.filename
            
            # Add source
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "ML Evaluation Platform"
            
            # Add size
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(image.width)
            ET.SubElement(size, "height").text = str(image.height)
            ET.SubElement(size, "depth").text = "3"
            
            # Add segmented
            ET.SubElement(root, "segmented").text = "0"
            
            # Add objects
            for annotation in annotations:
                for i, class_label in enumerate(annotation.class_labels):
                    if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                        bbox = annotation.bounding_boxes[i]
                        
                        obj = ET.SubElement(root, "object")
                        ET.SubElement(obj, "name").text = class_label
                        ET.SubElement(obj, "pose").text = "Unspecified"
                        ET.SubElement(obj, "truncated").text = "0"
                        ET.SubElement(obj, "difficult").text = "0"
                        
                        bndbox = ET.SubElement(obj, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = str(int(bbox["x"]))
                        ET.SubElement(bndbox, "ymin").text = str(int(bbox["y"]))
                        ET.SubElement(bndbox, "xmax").text = str(int(bbox["x"] + bbox["width"]))
                        ET.SubElement(bndbox, "ymax").text = str(int(bbox["y"] + bbox["height"]))
            
            # Convert to string
            xml_filename = f"{Path(image.filename).stem}.xml"
            xml_content = ET.tostring(root, encoding='unicode')
            voc_files[xml_filename] = xml_content
        
        # Create zip file content
        zip_filename = f"pascal_voc_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_content = self._create_zip_from_files(voc_files)
        
        return {
            "format": "pascal_voc",
            "filename": zip_filename,
            "content": zip_content,
            "content_type": "application/zip",
            "files": list(voc_files.keys())
        }
    
    def _export_labelme_format(self, annotations_data: Dict[str, Any], 
                             include_images: bool) -> Dict[str, Any]:
        """Export annotations in LabelMe format."""
        labelme_files = {}
        
        for image_id, data in annotations_data.items():
            image = data["image"]
            annotations = data["annotations"]
            
            # Create LabelMe structure
            labelme_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [],
                "imagePath": image.filename,
                "imageData": None,  # Would contain base64 encoded image data
                "imageHeight": image.height,
                "imageWidth": image.width
            }
            
            # Add shapes
            for annotation in annotations:
                for i, class_label in enumerate(annotation.class_labels):
                    # Add bounding box as rectangle
                    if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                        bbox = annotation.bounding_boxes[i]
                        
                        shape = {
                            "label": class_label,
                            "points": [
                                [bbox["x"], bbox["y"]],
                                [bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
                            ],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {}
                        }
                        labelme_data["shapes"].append(shape)
                    
                    # Add polygon if available
                    if annotation.segments and i < len(annotation.segments):
                        segment = annotation.segments[i]
                        if segment.get("points"):
                            shape = {
                                "label": class_label,
                                "points": [[p["x"], p["y"]] for p in segment["points"]],
                                "group_id": None,
                                "shape_type": "polygon",
                                "flags": {}
                            }
                            labelme_data["shapes"].append(shape)
            
            # Create JSON file for this image
            json_filename = f"{Path(image.filename).stem}.json"
            labelme_files[json_filename] = json.dumps(labelme_data, indent=2)
        
        # Create zip file content
        zip_filename = f"labelme_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_content = self._create_zip_from_files(labelme_files)
        
        return {
            "format": "labelme",
            "filename": zip_filename,
            "content": zip_content,
            "content_type": "application/zip",
            "files": list(labelme_files.keys())
        }
    
    def _export_csv_format(self, annotations_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export annotations in CSV format."""
        csv_lines = [
            "image_id,image_filename,class_label,bbox_x,bbox_y,bbox_width,bbox_height,confidence,annotation_id"
        ]
        
        for image_id, data in annotations_data.items():
            image = data["image"]
            annotations = data["annotations"]
            
            for annotation in annotations:
                for i, class_label in enumerate(annotation.class_labels):
                    bbox_info = ["", "", "", ""]  # x, y, width, height
                    
                    if annotation.bounding_boxes and i < len(annotation.bounding_boxes):
                        bbox = annotation.bounding_boxes[i]
                        bbox_info = [
                            str(bbox["x"]),
                            str(bbox["y"]),
                            str(bbox["width"]),
                            str(bbox["height"])
                        ]
                    
                    confidence = ""
                    if annotation.confidence_scores and i < len(annotation.confidence_scores):
                        confidence = str(annotation.confidence_scores[i])
                    
                    csv_line = ",".join([
                        image_id,
                        image.filename,
                        class_label,
                        *bbox_info,
                        confidence,
                        annotation.id
                    ])
                    csv_lines.append(csv_line)
        
        csv_content = "\n".join(csv_lines)
        csv_filename = f"annotations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return {
            "format": "csv",
            "filename": csv_filename,
            "content": csv_content,
            "content_type": "text/csv"
        }
    
    def _create_zip_from_files(self, files_dict: Dict[str, str]) -> bytes:
        """Create zip file content from dictionary of filename -> content."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for filename, content in files_dict.items():
                    zip_file.writestr(filename, content)
            
            # Read the zip file content
            temp_file.seek(0)
            return temp_file.read()
    
    def get_export_formats(self) -> List[Dict[str, Any]]:
        """Get list of supported export formats with descriptions."""
        formats = [
            {
                "format": "coco",
                "name": "COCO JSON",
                "description": "Common Objects in Context format with JSON structure",
                "supports": ["detection", "segmentation"],
                "file_extension": ".json",
                "use_case": "Best for object detection and instance segmentation tasks"
            },
            {
                "format": "yolo",
                "name": "YOLO TXT",
                "description": "YOLO format with normalized coordinates",
                "supports": ["detection"],
                "file_extension": ".txt",
                "use_case": "Best for YOLO model training and inference"
            },
            {
                "format": "pascal_voc",
                "name": "Pascal VOC XML",
                "description": "Pascal VOC format with XML annotation files",
                "supports": ["detection"],
                "file_extension": ".xml",
                "use_case": "Best for traditional object detection frameworks"
            },
            {
                "format": "labelme",
                "name": "LabelMe JSON",
                "description": "LabelMe format for image annotation",
                "supports": ["detection", "segmentation"],
                "file_extension": ".json",
                "use_case": "Best for annotation tools and custom workflows"
            },
            {
                "format": "csv",
                "name": "CSV",
                "description": "Simple CSV format for annotation data",
                "supports": ["detection"],
                "file_extension": ".csv",
                "use_case": "Best for data analysis and custom processing"
            }
        ]
        return formats
    
    def validate_export_request(self, export_format: str, 
                              image_ids: Optional[List[str]] = None,
                              dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate export request parameters."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate format
        if export_format not in self.supported_formats:
            validation_result["errors"].append(f"Unsupported format: {export_format}")
            validation_result["is_valid"] = False
        
        # Validate data selection
        if not image_ids and not dataset_id:
            validation_result["warnings"].append("No specific images or dataset selected - will export all annotations")
        
        if image_ids and dataset_id:
            validation_result["warnings"].append("Both image_ids and dataset_id specified - image_ids will take precedence")
        
        # Validate dataset if specified
        if dataset_id:
            dataset = self.dataset_storage.get_by_id(dataset_id)
            if not dataset:
                validation_result["errors"].append(f"Dataset {dataset_id} not found")
                validation_result["is_valid"] = False
            elif not dataset.image_ids:
                validation_result["warnings"].append("Dataset contains no images")
        
        # Validate images if specified
        if image_ids:
            invalid_images = []
            for image_id in image_ids:
                image = self.image_storage.get_by_id(image_id)
                if not image:
                    invalid_images.append(image_id)
            
            if invalid_images:
                validation_result["errors"].extend([f"Image {img_id} not found" for img_id in invalid_images])
                validation_result["is_valid"] = False
        
        # Format-specific validations
        if export_format == "yolo":
            validation_result["warnings"].append("YOLO format only supports bounding box annotations")
        elif export_format == "csv":
            validation_result["warnings"].append("CSV format has limited support for complex annotations")
        
        return validation_result
    
    def get_export_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get export usage statistics (simulated for TDD GREEN phase)."""
        # In real implementation, this would track actual export requests
        return {
            "period_days": days,
            "total_exports": 45,
            "format_breakdown": {
                "coco": 18,
                "yolo": 15,
                "pascal_voc": 8,
                "labelme": 3,
                "csv": 1
            },
            "most_popular_format": "coco",
            "average_annotations_per_export": 156,
            "largest_export_annotations": 1250
        }