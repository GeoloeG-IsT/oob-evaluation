"""
Utility functions for annotation tools.
"""
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from pathlib import Path

from .tools import AnnotationShape, AnnotationType, AnnotationPoint


class GeometryUtils:
    """Geometric utility functions for annotations."""
    
    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        int_x1 = max(x1, x2)
        int_y1 = max(y1, y2)
        int_x2 = min(x1 + w1, x2 + w2)
        int_y2 = min(y1 + h1, y2 + h2)
        
        if int_x2 <= int_x1 or int_y2 <= int_y1:
            return 0.0
        
        intersection = (int_x2 - int_x1) * (int_y2 - int_y1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_distance(point1: AnnotationPoint, point2: AnnotationPoint) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    
    @staticmethod
    def point_in_polygon(point: AnnotationPoint, polygon: List[AnnotationPoint]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        if len(polygon) < 3:
            return False
        
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0].x, polygon[0].y
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n].x, polygon[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def simplify_polygon(
        points: List[AnnotationPoint],
        tolerance: float = 1.0
    ) -> List[AnnotationPoint]:
        """Simplify polygon using Douglas-Peucker algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line between first and last
        start = points[0]
        end = points[-1]
        max_distance = 0.0
        index = 0
        
        for i in range(1, len(points) - 1):
            distance = GeometryUtils._point_line_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursive call
            left_simplified = GeometryUtils.simplify_polygon(points[:index+1], tolerance)
            right_simplified = GeometryUtils.simplify_polygon(points[index:], tolerance)
            
            # Combine results (remove duplicate point)
            result = left_simplified[:-1] + right_simplified
            return result
        else:
            # Base case: return endpoints
            return [start, end]
    
    @staticmethod
    def _point_line_distance(point: AnnotationPoint, line_start: AnnotationPoint, line_end: AnnotationPoint) -> float:
        """Calculate perpendicular distance from point to line segment."""
        x0, y0 = point.x, point.y
        x1, y1 = line_start.x, line_start.y
        x2, y2 = line_end.x, line_end.y
        
        # Calculate line length
        line_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_length_squared == 0:
            # Line is actually a point
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        
        # Calculate perpendicular distance
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt(line_length_squared)
        
        return numerator / denominator
    
    @staticmethod
    def calculate_polygon_centroid(points: List[AnnotationPoint]) -> AnnotationPoint:
        """Calculate centroid of a polygon."""
        if not points:
            return AnnotationPoint(0, 0)
        
        if len(points) == 1:
            return points[0]
        
        # Calculate centroid using shoelace formula
        area = 0.0
        cx = 0.0
        cy = 0.0
        
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            cross = points[i].x * points[j].y - points[j].x * points[i].y
            area += cross
            cx += (points[i].x + points[j].x) * cross
            cy += (points[i].y + points[j].y) * cross
        
        area *= 0.5
        
        if abs(area) < 1e-10:
            # Degenerate polygon, return geometric center
            avg_x = sum(p.x for p in points) / len(points)
            avg_y = sum(p.y for p in points) / len(points)
            return AnnotationPoint(avg_x, avg_y)
        
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        
        return AnnotationPoint(cx, cy)
    
    @staticmethod
    def rotate_point(point: AnnotationPoint, center: AnnotationPoint, angle_degrees: float) -> AnnotationPoint:
        """Rotate a point around a center point."""
        angle_rad = math.radians(angle_degrees)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Translate to origin
        dx = point.x - center.x
        dy = point.y - center.y
        
        # Rotate
        rotated_x = dx * cos_angle - dy * sin_angle
        rotated_y = dx * sin_angle + dy * cos_angle
        
        # Translate back
        return AnnotationPoint(
            rotated_x + center.x,
            rotated_y + center.y
        )
    
    @staticmethod
    def scale_annotation(
        annotation: AnnotationShape,
        scale_x: float,
        scale_y: float,
        center: Optional[AnnotationPoint] = None
    ) -> AnnotationShape:
        """Scale an annotation by given factors."""
        import copy
        scaled = copy.deepcopy(annotation)
        
        if center is None:
            center = AnnotationPoint(0, 0)
        
        # Scale points
        for point in scaled.points:
            # Translate to center
            dx = point.x - center.x
            dy = point.y - center.y
            
            # Scale
            dx *= scale_x
            dy *= scale_y
            
            # Translate back
            point.x = dx + center.x
            point.y = dy + center.y
        
        # Update bbox and area
        if scaled.bbox:
            scaled.bbox = scaled.calculate_bbox()
        scaled.area = scaled.calculate_area()
        
        return scaled


class VisualizationUtils:
    """Utilities for visualizing annotations."""
    
    @staticmethod
    def generate_colors(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        random.seed(seed)
        colors = []
        
        for i in range(num_colors):
            # Use HSV color space for better color distribution
            hue = (i * 137.508) % 360  # Golden angle approximation
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            rgb = VisualizationUtils._hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV color to RGB."""
        h = h / 60.0
        c = v * s
        x = c * (1 - abs((h % 2) - 1))
        m = v - c
        
        if 0 <= h < 1:
            rgb = (c, x, 0)
        elif 1 <= h < 2:
            rgb = (x, c, 0)
        elif 2 <= h < 3:
            rgb = (0, c, x)
        elif 3 <= h < 4:
            rgb = (0, x, c)
        elif 4 <= h < 5:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)
        
        return (
            int((rgb[0] + m) * 255),
            int((rgb[1] + m) * 255),
            int((rgb[2] + m) * 255)
        )
    
    @staticmethod
    def create_annotation_overlay(
        image_width: int,
        image_height: int,
        annotations: List[AnnotationShape],
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> List[Dict[str, Any]]:
        """Create overlay data for annotations."""
        
        if colors is None:
            # Generate colors for each category
            category_ids = list(set(ann.category_id for ann in annotations))
            colors = VisualizationUtils.generate_colors(len(category_ids))
            color_map = dict(zip(category_ids, colors))
        else:
            category_ids = list(set(ann.category_id for ann in annotations))
            color_map = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(category_ids)}
        
        overlay_data = []
        
        for annotation in annotations:
            color = color_map.get(annotation.category_id, (255, 0, 0))
            
            overlay_item = {
                "annotation_id": annotation.annotation_id,
                "category_id": annotation.category_id,
                "category_name": annotation.category_name,
                "color": color,
                "confidence": annotation.confidence,
                "type": annotation.annotation_type
            }
            
            if annotation.annotation_type == AnnotationType.BOUNDING_BOX:
                overlay_item["bbox"] = annotation.bbox
                
            elif annotation.annotation_type == AnnotationType.POLYGON:
                overlay_item["polygon"] = [(p.x, p.y) for p in annotation.points]
                
            elif annotation.annotation_type == AnnotationType.POINT:
                overlay_item["point"] = (annotation.points[0].x, annotation.points[0].y) if annotation.points else (0, 0)
            
            overlay_data.append(overlay_item)
        
        return overlay_data
    
    @staticmethod
    def create_heatmap_data(
        annotations: List[AnnotationShape],
        image_width: int,
        image_height: int,
        grid_size: int = 32
    ) -> List[List[float]]:
        """Create heatmap data showing annotation density."""
        
        # Create grid
        grid_width = image_width // grid_size
        grid_height = image_height // grid_size
        heatmap = [[0.0 for _ in range(grid_width)] for _ in range(grid_height)]
        
        # Count annotations in each grid cell
        for annotation in annotations:
            if annotation.bbox:
                x, y, w, h = annotation.bbox
                
                # Find grid cells that overlap with annotation
                start_col = max(0, int(x // grid_size))
                end_col = min(grid_width - 1, int((x + w) // grid_size))
                start_row = max(0, int(y // grid_size))
                end_row = min(grid_height - 1, int((y + h) // grid_size))
                
                # Add weight to overlapping cells
                weight = annotation.confidence if annotation.confidence > 0 else 1.0
                
                for row in range(start_row, end_row + 1):
                    for col in range(start_col, end_col + 1):
                        heatmap[row][col] += weight
        
        # Normalize heatmap
        max_value = max(max(row) for row in heatmap) if heatmap else 1.0
        if max_value > 0:
            heatmap = [[cell / max_value for cell in row] for row in heatmap]
        
        return heatmap


class StatisticsCalculator:
    """Calculate statistics for annotations."""
    
    @staticmethod
    def calculate_annotation_stats(annotations: List[AnnotationShape]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for annotations."""
        
        if not annotations:
            return {
                "total_annotations": 0,
                "area_stats": {},
                "confidence_stats": {},
                "category_distribution": {},
                "type_distribution": {}
            }
        
        # Basic counts
        total_annotations = len(annotations)
        
        # Area statistics
        areas = [ann.area for ann in annotations if ann.area > 0]
        area_stats = {}
        if areas:
            area_stats = {
                "min": min(areas),
                "max": max(areas),
                "mean": sum(areas) / len(areas),
                "median": sorted(areas)[len(areas) // 2],
                "std": StatisticsCalculator._calculate_std(areas),
                "total": sum(areas)
            }
        
        # Confidence statistics
        confidences = [ann.confidence for ann in annotations]
        confidence_stats = {}
        if confidences:
            confidence_stats = {
                "min": min(confidences),
                "max": max(confidences),
                "mean": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2],
                "std": StatisticsCalculator._calculate_std(confidences)
            }
        
        # Category distribution
        category_counts = {}
        for ann in annotations:
            category_name = ann.category_name or f"class_{ann.category_id}"
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        # Type distribution
        type_counts = {}
        for ann in annotations:
            type_counts[ann.annotation_type] = type_counts.get(ann.annotation_type, 0) + 1
        
        return {
            "total_annotations": total_annotations,
            "area_stats": area_stats,
            "confidence_stats": confidence_stats,
            "category_distribution": category_counts,
            "type_distribution": type_counts
        }
    
    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    @staticmethod
    def calculate_overlap_matrix(annotations: List[AnnotationShape]) -> List[List[float]]:
        """Calculate pairwise overlap matrix for annotations."""
        n = len(annotations)
        overlap_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    overlap_matrix[i][j] = 1.0
                else:
                    if annotations[i].bbox and annotations[j].bbox:
                        iou = GeometryUtils.calculate_iou(
                            annotations[i].bbox,
                            annotations[j].bbox
                        )
                        overlap_matrix[i][j] = iou
                        overlap_matrix[j][i] = iou
        
        return overlap_matrix
    
    @staticmethod
    def find_duplicate_annotations(
        annotations: List[AnnotationShape],
        iou_threshold: float = 0.9,
        confidence_threshold: float = 0.01
    ) -> List[List[int]]:
        """Find potentially duplicate annotations."""
        
        duplicates = []
        processed = set()
        
        for i, ann1 in enumerate(annotations):
            if i in processed:
                continue
            
            duplicate_group = [i]
            
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check if they're likely duplicates
                if (ann1.category_id == ann2.category_id and
                    abs(ann1.confidence - ann2.confidence) < confidence_threshold):
                    
                    if ann1.bbox and ann2.bbox:
                        iou = GeometryUtils.calculate_iou(ann1.bbox, ann2.bbox)
                        if iou > iou_threshold:
                            duplicate_group.append(j)
                            processed.add(j)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                for idx in duplicate_group:
                    processed.add(idx)
        
        return duplicates
    
    @staticmethod
    def calculate_class_balance(annotations: List[AnnotationShape]) -> Dict[str, Any]:
        """Calculate class balance statistics."""
        
        category_counts = {}
        for ann in annotations:
            category_name = ann.category_name or f"class_{ann.category_id}"
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        if not category_counts:
            return {"balance_score": 0.0, "imbalance_ratio": 0.0}
        
        total_annotations = sum(category_counts.values())
        num_classes = len(category_counts)
        
        # Calculate balance score (higher is more balanced)
        expected_per_class = total_annotations / num_classes
        balance_score = 1.0 - (
            sum(abs(count - expected_per_class) for count in category_counts.values()) /
            (2 * total_annotations)
        )
        
        # Calculate imbalance ratio (max class count / min class count)
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            "balance_score": balance_score,
            "imbalance_ratio": imbalance_ratio,
            "class_counts": category_counts,
            "total_classes": num_classes,
            "most_frequent_class": max(category_counts.items(), key=lambda x: x[1]),
            "least_frequent_class": min(category_counts.items(), key=lambda x: x[1])
        }


class AnnotationUtils:
    """General utility functions for annotations."""
    
    @staticmethod
    def merge_annotations(
        annotations: List[AnnotationShape],
        merge_threshold: float = 0.7
    ) -> List[AnnotationShape]:
        """Merge overlapping annotations with same category."""
        
        if len(annotations) <= 1:
            return annotations
        
        merged = []
        processed = set()
        
        for i, ann1 in enumerate(annotations):
            if i in processed:
                continue
            
            # Find all annotations to merge with this one
            merge_group = [ann1]
            processed.add(i)
            
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if j in processed:
                    continue
                
                if (ann1.category_id == ann2.category_id and
                    ann1.bbox and ann2.bbox):
                    
                    iou = GeometryUtils.calculate_iou(ann1.bbox, ann2.bbox)
                    if iou > merge_threshold:
                        merge_group.append(ann2)
                        processed.add(j)
            
            # Merge the group
            merged_annotation = AnnotationUtils._merge_annotation_group(merge_group)
            merged.append(merged_annotation)
        
        return merged
    
    @staticmethod
    def _merge_annotation_group(annotations: List[AnnotationShape]) -> AnnotationShape:
        """Merge a group of annotations into one."""
        if len(annotations) == 1:
            return annotations[0]
        
        import copy
        merged = copy.deepcopy(annotations[0])
        
        # Merge bounding boxes (union)
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
        merged.confidence = sum(ann.confidence for ann in annotations) / len(annotations)
        
        # Merge metadata
        for ann in annotations[1:]:
            if ann.metadata:
                merged.metadata.update(ann.metadata)
        
        return merged
    
    @staticmethod
    def filter_annotations(
        annotations: List[AnnotationShape],
        min_area: float = 0.0,
        max_area: float = float('inf'),
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        allowed_categories: Optional[List[Union[int, str]]] = None,
        image_bounds: Optional[Tuple[int, int]] = None
    ) -> List[AnnotationShape]:
        """Filter annotations based on various criteria."""
        
        filtered = []
        
        for ann in annotations:
            # Area filter
            if not (min_area <= ann.area <= max_area):
                continue
            
            # Confidence filter
            if not (min_confidence <= ann.confidence <= max_confidence):
                continue
            
            # Category filter
            if allowed_categories is not None:
                category_match = (
                    ann.category_id in allowed_categories or
                    ann.category_name in allowed_categories
                )
                if not category_match:
                    continue
            
            # Bounds filter
            if image_bounds and ann.bbox:
                image_width, image_height = image_bounds
                x, y, w, h = ann.bbox
                
                if (x < 0 or y < 0 or 
                    x + w > image_width or 
                    y + h > image_height):
                    continue
            
            filtered.append(ann)
        
        return filtered
    
    @staticmethod
    def sort_annotations(
        annotations: List[AnnotationShape],
        sort_by: str = "confidence",
        ascending: bool = False
    ) -> List[AnnotationShape]:
        """Sort annotations by specified criteria."""
        
        sort_key_map = {
            "confidence": lambda ann: ann.confidence,
            "area": lambda ann: ann.area,
            "category_id": lambda ann: ann.category_id,
            "category_name": lambda ann: ann.category_name,
            "x": lambda ann: ann.bbox[0] if ann.bbox else 0,
            "y": lambda ann: ann.bbox[1] if ann.bbox else 0
        }
        
        if sort_by not in sort_key_map:
            raise ValueError(f"Invalid sort key: {sort_by}")
        
        return sorted(annotations, key=sort_key_map[sort_by], reverse=not ascending)
    
    @staticmethod
    def convert_coordinates(
        annotation: AnnotationShape,
        from_format: str,
        to_format: str,
        image_width: int,
        image_height: int
    ) -> AnnotationShape:
        """Convert annotation coordinates between different formats."""
        
        import copy
        converted = copy.deepcopy(annotation)
        
        if not annotation.bbox or len(annotation.bbox) < 4:
            return converted
        
        if from_format == "coco" and to_format == "yolo":
            # COCO: [x, y, width, height] (absolute)
            # YOLO: [x_center, y_center, width, height] (normalized)
            x, y, w, h = annotation.bbox
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            norm_width = w / image_width
            norm_height = h / image_height
            
            converted.bbox = [x_center, y_center, norm_width, norm_height]
        
        elif from_format == "yolo" and to_format == "coco":
            # YOLO: [x_center, y_center, width, height] (normalized)
            # COCO: [x, y, width, height] (absolute)
            x_center, y_center, norm_width, norm_height = annotation.bbox
            w = norm_width * image_width
            h = norm_height * image_height
            x = (x_center * image_width) - (w / 2)
            y = (y_center * image_height) - (h / 2)
            
            converted.bbox = [x, y, w, h]
            converted.area = w * h
        
        elif from_format == "pascal_voc" and to_format == "coco":
            # Pascal VOC: [xmin, ymin, xmax, ymax]
            # COCO: [x, y, width, height]
            xmin, ymin, xmax, ymax = annotation.bbox
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            
            converted.bbox = [x, y, w, h]
            converted.area = w * h
        
        elif from_format == "coco" and to_format == "pascal_voc":
            # COCO: [x, y, width, height]
            # Pascal VOC: [xmin, ymin, xmax, ymax]
            x, y, w, h = annotation.bbox
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            
            converted.bbox = [xmin, ymin, xmax, ymax]
        
        return converted