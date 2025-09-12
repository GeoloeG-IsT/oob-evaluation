"""
Result formatters for different output formats and standards.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime, timezone

from .engine import InferenceResult, BatchInferenceJob


class ResultFormatter(ABC):
    """Abstract base class for result formatters."""
    
    @abstractmethod
    def format_single_result(self, result: InferenceResult) -> Dict[str, Any]:
        """Format a single inference result."""
        pass
    
    @abstractmethod
    def format_batch_results(self, job: BatchInferenceJob) -> Dict[str, Any]:
        """Format batch inference results."""
        pass
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format predictions list. Override in subclasses for custom formatting."""
        return predictions


class StandardFormatter(ResultFormatter):
    """Standard formatter for ML Evaluation Platform."""
    
    def format_single_result(self, result: InferenceResult) -> Dict[str, Any]:
        """Format single result in standard format."""
        return {
            "request_id": result.request_id,
            "model_id": result.model_id,
            "status": result.status,
            "predictions": self.format_predictions(result.predictions),
            "performance": {
                "inference_time_ms": result.performance_metrics.inference_time_ms,
                "preprocessing_time_ms": result.performance_metrics.preprocessing_time_ms,
                "postprocessing_time_ms": result.performance_metrics.postprocessing_time_ms,
                "total_time_ms": result.performance_metrics.total_time_ms,
                "throughput_fps": result.performance_metrics.throughput_fps,
                "memory_usage_mb": result.performance_metrics.memory_usage_mb
            },
            "error_message": result.error_message,
            "timestamps": {
                "created_at": result.created_at,
                "completed_at": result.completed_at
            }
        }
    
    def format_batch_results(self, job: BatchInferenceJob) -> Dict[str, Any]:
        """Format batch results in standard format."""
        # Calculate aggregate performance metrics
        successful_results = [r for r in job.results if r.status == "completed"]
        
        aggregate_performance = {}
        if successful_results:
            total_inference_time = sum(r.performance_metrics.inference_time_ms for r in successful_results)
            total_processing_time = sum(r.performance_metrics.total_time_ms for r in successful_results)
            
            aggregate_performance = {
                "average_inference_time_ms": total_inference_time / len(successful_results),
                "average_total_time_ms": total_processing_time / len(successful_results),
                "total_inference_time_ms": total_inference_time,
                "total_processing_time_ms": total_processing_time,
                "average_throughput_fps": sum(r.performance_metrics.throughput_fps for r in successful_results) / len(successful_results)
            }
        
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status,
            "progress": {
                "total_images": job.total_images,
                "completed_images": job.completed_images,
                "failed_images": job.failed_images,
                "progress_percentage": job.progress_percentage
            },
            "results": [self.format_single_result(result) for result in job.results],
            "aggregate_performance": aggregate_performance,
            "parameters": job.parameters,
            "error_message": job.error_message,
            "timestamps": {
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
        }


class COCOFormatter(ResultFormatter):
    """COCO dataset format formatter."""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
        self.category_mapping = {
            0: {"id": 1, "name": "person", "supercategory": "person"},
            1: {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            2: {"id": 3, "name": "car", "supercategory": "vehicle"},
            3: {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
            # Add more categories as needed
        }
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format predictions in COCO format."""
        coco_annotations = []
        
        for i, pred in enumerate(predictions):
            annotation = {
                "id": i + 1,
                "image_id": 1,  # Would be actual image ID in real implementation
                "category_id": pred.get("class_id", 0) + 1,  # COCO uses 1-based indexing
                "score": pred.get("confidence", 0.0)
            }
            
            # Handle bounding box format
            if "bbox" in pred:
                bbox = pred["bbox"]
                if len(bbox) == 4:
                    # Convert from [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = bbox
                    annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    annotation["area"] = (x2 - x1) * (y2 - y1)
            
            # Handle segmentation mask
            if "mask" in pred:
                annotation["segmentation"] = pred["mask"]
                annotation["area"] = pred.get("area", 0)
            
            coco_annotations.append(annotation)
        
        return coco_annotations
    
    def format_single_result(self, result: InferenceResult) -> Dict[str, Any]:
        """Format single result in COCO format."""
        coco_result = {
            "images": [{
                "id": 1,
                "file_name": "image.jpg",  # Would be actual filename
                "width": 640,  # Would be actual dimensions
                "height": 640
            }],
            "annotations": self.format_predictions(result.predictions),
            "categories": list(self.category_mapping.values())
        }
        
        if self.include_metadata:
            coco_result["info"] = {
                "description": "ML Evaluation Platform Results",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ML Evaluation Platform",
                "date_created": result.created_at
            }
            
            coco_result["metadata"] = {
                "request_id": result.request_id,
                "model_id": result.model_id,
                "status": result.status,
                "performance_metrics": {
                    "inference_time_ms": result.performance_metrics.inference_time_ms,
                    "total_time_ms": result.performance_metrics.total_time_ms,
                    "throughput_fps": result.performance_metrics.throughput_fps
                }
            }
        
        return coco_result
    
    def format_batch_results(self, job: BatchInferenceJob) -> Dict[str, Any]:
        """Format batch results in COCO format."""
        # Combine all annotations from all results
        all_images = []
        all_annotations = []
        annotation_id = 1
        
        for image_idx, result in enumerate(job.results):
            if result.status != "completed":
                continue
                
            # Add image info
            image_info = {
                "id": image_idx + 1,
                "file_name": f"image_{image_idx + 1}.jpg",  # Would be actual filename
                "width": 640,  # Would be actual dimensions
                "height": 640
            }
            all_images.append(image_info)
            
            # Add annotations for this image
            for pred in result.predictions:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_idx + 1,
                    "category_id": pred.get("class_id", 0) + 1,
                    "score": pred.get("confidence", 0.0)
                }
                
                if "bbox" in pred:
                    bbox = pred["bbox"]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        annotation["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                        annotation["area"] = (x2 - x1) * (y2 - y1)
                
                if "mask" in pred:
                    annotation["segmentation"] = pred["mask"]
                    annotation["area"] = pred.get("area", 0)
                
                all_annotations.append(annotation)
                annotation_id += 1
        
        coco_result = {
            "images": all_images,
            "annotations": all_annotations,
            "categories": list(self.category_mapping.values())
        }
        
        if self.include_metadata:
            coco_result["info"] = {
                "description": f"ML Evaluation Platform Batch Results - Job {job.job_id}",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ML Evaluation Platform",
                "date_created": job.created_at
            }
            
            coco_result["metadata"] = {
                "job_id": job.job_id,
                "model_id": job.model_id,
                "status": job.status,
                "total_images": job.total_images,
                "completed_images": job.completed_images,
                "failed_images": job.failed_images
            }
        
        return coco_result


class YOLOFormatter(ResultFormatter):
    """YOLO dataset format formatter."""
    
    def __init__(self, image_width: int = 640, image_height: int = 640):
        self.image_width = image_width
        self.image_height = image_height
    
    def format_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format predictions in YOLO format."""
        yolo_predictions = []
        
        for pred in predictions:
            if "bbox" in pred:
                bbox = pred["bbox"]
                if len(bbox) == 4:
                    # Convert from [x1, y1, x2, y2] to normalized [x_center, y_center, width, height]
                    x1, y1, x2, y2 = bbox
                    x_center = ((x1 + x2) / 2) / self.image_width
                    y_center = ((y1 + y2) / 2) / self.image_height
                    width = (x2 - x1) / self.image_width
                    height = (y2 - y1) / self.image_height
                    
                    yolo_pred = {
                        "class_id": pred.get("class_id", 0),
                        "confidence": pred.get("confidence", 0.0),
                        "bbox_normalized": [x_center, y_center, width, height],
                        "bbox_absolute": bbox
                    }
                    
                    # YOLO format string: class_id x_center y_center width height confidence
                    yolo_pred["yolo_format"] = f"{pred.get('class_id', 0)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {pred.get('confidence', 0.0):.6f}"
                    
                    yolo_predictions.append(yolo_pred)
        
        return yolo_predictions
    
    def format_single_result(self, result: InferenceResult) -> Dict[str, Any]:
        """Format single result in YOLO format."""
        return {
            "request_id": result.request_id,
            "model_id": result.model_id,
            "status": result.status,
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height
            },
            "predictions": self.format_predictions(result.predictions),
            "yolo_lines": [pred["yolo_format"] for pred in self.format_predictions(result.predictions) if "yolo_format" in pred],
            "performance_metrics": {
                "inference_time_ms": result.performance_metrics.inference_time_ms,
                "total_time_ms": result.performance_metrics.total_time_ms,
                "throughput_fps": result.performance_metrics.throughput_fps
            },
            "timestamps": {
                "created_at": result.created_at,
                "completed_at": result.completed_at
            }
        }
    
    def format_batch_results(self, job: BatchInferenceJob) -> Dict[str, Any]:
        """Format batch results in YOLO format."""
        all_predictions = []
        yolo_files = {}
        
        for i, result in enumerate(job.results):
            if result.status != "completed":
                continue
                
            formatted_preds = self.format_predictions(result.predictions)
            all_predictions.extend(formatted_preds)
            
            # Create YOLO format file content for each image
            yolo_lines = [pred["yolo_format"] for pred in formatted_preds if "yolo_format" in pred]
            if yolo_lines:
                yolo_files[f"image_{i:04d}.txt"] = "\n".join(yolo_lines)
        
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status,
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height
            },
            "total_predictions": len(all_predictions),
            "predictions": all_predictions,
            "yolo_files": yolo_files,
            "progress": {
                "total_images": job.total_images,
                "completed_images": job.completed_images,
                "failed_images": job.failed_images
            },
            "timestamps": {
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
        }


class PascalVOCFormatter(ResultFormatter):
    """Pascal VOC XML format formatter."""
    
    def __init__(self, image_width: int = 640, image_height: int = 640, image_depth: int = 3):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic_light"
        ]  # Simplified class list
    
    def create_voc_xml(self, predictions: List[Dict[str, Any]], filename: str = "image.jpg") -> str:
        """Create Pascal VOC XML format string."""
        xml_lines = [
            "<annotation>",
            f"    <filename>{filename}</filename>",
            "    <size>",
            f"        <width>{self.image_width}</width>",
            f"        <height>{self.image_height}</height>",
            f"        <depth>{self.image_depth}</depth>",
            "    </size>",
            "    <segmented>0</segmented>"
        ]
        
        for pred in predictions:
            if "bbox" in pred:
                bbox = pred["bbox"]
                class_id = pred.get("class_id", 0)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    xml_lines.extend([
                        "    <object>",
                        f"        <name>{class_name}</name>",
                        "        <pose>Unspecified</pose>",
                        "        <truncated>0</truncated>",
                        "        <difficult>0</difficult>",
                        "        <bndbox>",
                        f"            <xmin>{int(x1)}</xmin>",
                        f"            <ymin>{int(y1)}</ymin>",
                        f"            <xmax>{int(x2)}</xmax>",
                        f"            <ymax>{int(y2)}</ymax>",
                        "        </bndbox>",
                        f"        <confidence>{pred.get('confidence', 0.0):.6f}</confidence>",
                        "    </object>"
                    ])
        
        xml_lines.append("</annotation>")
        return "\n".join(xml_lines)
    
    def format_single_result(self, result: InferenceResult) -> Dict[str, Any]:
        """Format single result in Pascal VOC format."""
        voc_xml = self.create_voc_xml(result.predictions)
        
        return {
            "request_id": result.request_id,
            "model_id": result.model_id,
            "status": result.status,
            "voc_xml": voc_xml,
            "predictions": result.predictions,
            "performance_metrics": {
                "inference_time_ms": result.performance_metrics.inference_time_ms,
                "total_time_ms": result.performance_metrics.total_time_ms,
                "throughput_fps": result.performance_metrics.throughput_fps
            },
            "timestamps": {
                "created_at": result.created_at,
                "completed_at": result.completed_at
            }
        }
    
    def format_batch_results(self, job: BatchInferenceJob) -> Dict[str, Any]:
        """Format batch results in Pascal VOC format."""
        voc_files = {}
        
        for i, result in enumerate(job.results):
            if result.status != "completed":
                continue
                
            filename = f"image_{i:04d}.xml"
            voc_xml = self.create_voc_xml(result.predictions, f"image_{i:04d}.jpg")
            voc_files[filename] = voc_xml
        
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status,
            "voc_files": voc_files,
            "progress": {
                "total_images": job.total_images,
                "completed_images": job.completed_images,
                "failed_images": job.failed_images
            },
            "timestamps": {
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at
            }
        }


# Formatter factory for easy access
def get_formatter(format_type: str = "standard", **kwargs) -> ResultFormatter:
    """Get formatter instance by type."""
    formatters = {
        "standard": StandardFormatter,
        "coco": COCOFormatter,
        "yolo": YOLOFormatter,
        "pascal_voc": PascalVOCFormatter,
        "voc": PascalVOCFormatter  # Alias
    }
    
    formatter_class = formatters.get(format_type.lower())
    if not formatter_class:
        raise ValueError(f"Unknown formatter type: {format_type}. Available: {list(formatters.keys())}")
    
    return formatter_class(**kwargs)