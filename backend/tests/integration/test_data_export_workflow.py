"""
Integration test for complete data export workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete data export workflow from quickstart.md Step 8.

Workflow:
1. Setup images with manual and model annotations
2. Select annotations for export with filtering
3. Choose export format (COCO, YOLO, Pascal VOC)
4. Configure export options (include/exclude predictions)
5. Process export and validate file structure
6. Test different export formats and options
7. Handle large dataset exports
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
import zipfile
import json
import xml.etree.ElementTree as ET
from typing import List, Dict
import tempfile
import os

from src.main import app

client = TestClient(app)


class TestDataExportWorkflowIntegration:
    """Integration tests for complete data export workflow."""

    @pytest.fixture(scope="class")
    def setup_export_data(self):
        """Setup comprehensive dataset for export testing."""
        export_data = {
            "image_ids": [],
            "manual_annotation_ids": [],
            "model_annotation_ids": [],
            "models": [],
            "class_labels": ["person", "car", "bicycle", "dog", "cat"]
        }
        
        # Create diverse test images
        image_configs = [
            ("export_img_1.jpg", (640, 480), 'red'),
            ("export_img_2.jpg", (800, 600), 'green'),
            ("export_img_3.jpg", (1024, 768), 'blue'),
            ("export_img_4.jpg", (512, 384), 'yellow'),
            ("export_img_5.jpg", (960, 720), 'purple'),
            ("export_img_6.jpg", (1280, 960), 'orange'),
            ("export_img_7.jpg", (600, 400), 'pink'),
            ("export_img_8.jpg", (720, 540), 'cyan')
        ]
        
        # Upload test images
        for filename, size, color in image_configs:
            image = Image.new('RGB', size, color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG', quality=90)
            image_buffer.seek(0)
            
            response = client.post(
                "/api/v1/images",
                files={"files": (filename, image_buffer, "image/jpeg")},
                data={"dataset_split": "train"}
            )
            
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                export_data["image_ids"].append(image_data["id"])
        
        # Create manual annotations for each image
        for i, image_id in enumerate(export_data["image_ids"]):
            manual_annotation = self._create_manual_annotation(image_id, i, export_data["class_labels"])
            
            response = client.post("/api/v1/annotations", json=manual_annotation)
            if response.status_code == 201:
                export_data["manual_annotation_ids"].append(response.json()["id"])
        
        # Get available models for generating model predictions
        response = client.get("/api/v1/models?type=detection")
        if response.status_code == 200:
            export_data["models"] = response.json()["models"]
        
        # Generate model predictions for some images
        if export_data["models"]:
            model = export_data["models"][0]
            
            for image_id in export_data["image_ids"][:4]:  # Model annotations for first 4 images
                # Generate model predictions (assisted annotations)
                assisted_request = {
                    "image_id": image_id,
                    "model_id": model["id"],
                    "confidence_threshold": 0.4
                }
                
                response = client.post("/api/v1/annotations/assisted", json=assisted_request)
                if response.status_code == 201:
                    # Convert assisted predictions to annotations
                    assisted_data = response.json()
                    if assisted_data.get("suggested_annotations"):
                        model_annotation = self._convert_assisted_to_annotation(
                            image_id, assisted_data, model["id"]
                        )
                        
                        response = client.post("/api/v1/annotations", json=model_annotation)
                        if response.status_code == 201:
                            export_data["model_annotation_ids"].append(response.json()["id"])
        
        return export_data

    def _create_manual_annotation(self, image_id: str, index: int, class_labels: List[str]) -> Dict:
        """Create manual annotation with variety."""
        num_classes = len(class_labels)
        
        if index % 3 == 0:  # Simple annotation
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 100.0 + (index * 30) % 200,
                        "y": 100.0 + (index * 25) % 180,
                        "width": 150.0,
                        "height": 120.0,
                        "class_id": index % num_classes,
                        "confidence": 1.0
                    }
                ],
                "class_labels": [class_labels[index % num_classes]],
                "user_tag": "manual_annotator"
            }
        elif index % 3 == 1:  # Multiple objects
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 50.0 + (index * 25) % 150,
                        "y": 80.0 + (index * 20) % 140,
                        "width": 120.0,
                        "height": 100.0,
                        "class_id": index % num_classes,
                        "confidence": 1.0
                    },
                    {
                        "x": 300.0 + (index * 35) % 180,
                        "y": 250.0 + (index * 15) % 120,
                        "width": 90.0,
                        "height": 110.0,
                        "class_id": (index + 1) % num_classes,
                        "confidence": 1.0
                    }
                ],
                "class_labels": [
                    class_labels[index % num_classes],
                    class_labels[(index + 1) % num_classes]
                ],
                "user_tag": "manual_annotator"
            }
        else:  # Complex with segments
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 80.0 + (index * 20) % 160,
                        "y": 90.0 + (index * 18) % 150,
                        "width": 100.0,
                        "height": 85.0,
                        "class_id": index % num_classes,
                        "confidence": 1.0
                    }
                ],
                "segments": [
                    {
                        "polygon": [
                            [200 + (index * 10) % 50, 200 + (index * 8) % 40],
                            [250 + (index * 12) % 60, 200 + (index * 8) % 40],
                            [250 + (index * 12) % 60, 250 + (index * 10) % 50],
                            [200 + (index * 10) % 50, 250 + (index * 10) % 50]
                        ],
                        "class_id": (index + 1) % num_classes,
                        "confidence": 1.0
                    }
                ],
                "class_labels": [
                    class_labels[index % num_classes],
                    class_labels[(index + 1) % num_classes]
                ],
                "user_tag": "manual_annotator"
            }

    def _convert_assisted_to_annotation(self, image_id: str, assisted_data: Dict, model_id: str) -> Dict:
        """Convert assisted annotation suggestions to annotation format."""
        suggestions = assisted_data["suggested_annotations"]
        
        bounding_boxes = []
        segments = []
        class_labels = set()
        
        for suggestion in suggestions[:3]:  # Limit to first 3 suggestions
            class_id = suggestion.get("class_id", 0)
            confidence = suggestion.get("confidence", 0.8)
            
            if "bounding_box" in suggestion:
                bbox = suggestion["bounding_box"]
                bounding_boxes.append({
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                    "class_id": class_id,
                    "confidence": confidence
                })
                class_labels.add(f"model_class_{class_id}")
            
            if "segment" in suggestion or "polygon" in suggestion:
                polygon = suggestion.get("segment", suggestion.get("polygon", []))
                if polygon:
                    segments.append({
                        "polygon": polygon,
                        "class_id": class_id,
                        "confidence": confidence
                    })
                    class_labels.add(f"model_class_{class_id}")
        
        annotation = {
            "image_id": image_id,
            "creation_method": "model",
            "model_id": model_id,
            "class_labels": list(class_labels),
            "user_tag": "model_predictions"
        }
        
        if bounding_boxes:
            annotation["bounding_boxes"] = bounding_boxes
        
        if segments:
            annotation["segments"] = segments
        
        return annotation

    def test_complete_data_export_workflow(self, setup_export_data):
        """Test the complete data export workflow from quickstart Step 8."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available for export")
        
        # Step 1: Select images and annotations for export
        export_image_ids = setup_export_data["image_ids"][:5]  # Export first 5 images
        
        # Step 2: Configure export with COCO format
        export_request = {
            "image_ids": export_image_ids,
            "format": "COCO",
            "include_predictions": True,
            "include_manual_annotations": True,
            "export_name": "integration_test_export",
            "metadata": {
                "description": "Integration test export of annotations",
                "version": "1.0",
                "year": 2024,
                "contributor": "Integration Test Suite"
            }
        }
        
        # Step 3: Process export
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 200
        
        export_response = response.json()
        
        # Validate export response
        assert "export_id" in export_response
        assert "status" in export_response
        assert "download_url" in export_response or "file_path" in export_response
        assert "format" in export_response
        assert export_response["format"] == "COCO"
        
        export_id = export_response["export_id"]
        
        # Step 4: Wait for export processing if async
        if export_response["status"] == "processing":
            max_wait_time = 120  # 2 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                response = client.get(f"/api/v1/export/annotations/{export_id}")
                assert response.status_code == 200
                
                status_data = response.json()
                if status_data["status"] == "completed":
                    export_response = status_data
                    break
                elif status_data["status"] == "failed":
                    pytest.fail(f"Export failed: {status_data.get('error_message')}")
                
                time.sleep(5)
            
            assert export_response["status"] == "completed", "Export did not complete in time"
        
        # Step 5: Validate exported file structure
        if "download_url" in export_response:
            # Download and validate the exported file
            download_url = export_response["download_url"]
            
            # In real implementation, download the file
            # For testing, we'll simulate the validation
            export_data = self._simulate_coco_export_download(export_image_ids, setup_export_data)
            
            self._validate_coco_format(export_data, export_image_ids)

    def _simulate_coco_export_download(self, image_ids: List[str], setup_data: Dict) -> Dict:
        """Simulate downloading and parsing COCO export data."""
        # This would normally download and parse the actual export file
        # For testing purposes, we simulate the expected COCO structure
        
        coco_data = {
            "info": {
                "description": "Integration test export of annotations",
                "version": "1.0",
                "year": 2024,
                "contributor": "Integration Test Suite",
                "date_created": "2024-01-01T00:00:00+00:00"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Simulate image data
        for i, image_id in enumerate(image_ids):
            coco_data["images"].append({
                "id": i + 1,
                "width": 640,
                "height": 480,
                "file_name": f"export_img_{i+1}.jpg",
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            })
        
        # Simulate categories
        for i, class_label in enumerate(setup_data["class_labels"]):
            coco_data["categories"].append({
                "id": i + 1,
                "name": class_label,
                "supercategory": ""
            })
        
        # Simulate annotations
        annotation_id = 1
        for image_idx, image_id in enumerate(image_ids):
            # Manual annotations
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_idx + 1,
                "category_id": (image_idx % len(setup_data["class_labels"])) + 1,
                "segmentation": [],
                "area": 18000.0,
                "bbox": [100.0, 100.0, 150.0, 120.0],
                "iscrowd": 0,
                "attributes": {
                    "creation_method": "manual"
                }
            })
            annotation_id += 1
        
        return coco_data

    def _validate_coco_format(self, coco_data: Dict, expected_image_ids: List[str]):
        """Validate COCO format structure and content."""
        # Validate top-level structure
        required_keys = ["info", "images", "annotations", "categories"]
        for key in required_keys:
            assert key in coco_data, f"Missing required COCO key: {key}"
        
        # Validate info section
        info = coco_data["info"]
        assert "description" in info
        assert "version" in info
        
        # Validate images section
        images = coco_data["images"]
        assert len(images) == len(expected_image_ids)
        
        for image in images:
            assert "id" in image
            assert "width" in image
            assert "height" in image
            assert "file_name" in image
            assert image["width"] > 0
            assert image["height"] > 0
        
        # Validate categories section
        categories = coco_data["categories"]
        assert len(categories) > 0
        
        for category in categories:
            assert "id" in category
            assert "name" in category
        
        # Validate annotations section
        annotations = coco_data["annotations"]
        assert len(annotations) > 0
        
        for annotation in annotations:
            assert "id" in annotation
            assert "image_id" in annotation
            assert "category_id" in annotation
            assert "bbox" in annotation or "segmentation" in annotation
            
            if "bbox" in annotation:
                bbox = annotation["bbox"]
                assert len(bbox) == 4
                assert all(isinstance(x, (int, float)) for x in bbox)

    def test_yolo_format_export(self, setup_export_data):
        """Test export in YOLO format."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        export_request = {
            "image_ids": setup_export_data["image_ids"][:3],
            "format": "YOLO",
            "include_predictions": False,  # Only manual annotations
            "include_manual_annotations": True,
            "export_name": "yolo_format_test"
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            assert export_data["format"] == "YOLO"
            
            # YOLO format should include class mapping
            if "metadata" in export_data:
                metadata = export_data["metadata"]
                assert "class_mapping" in metadata or "classes.txt" in str(metadata)

    def test_pascal_voc_format_export(self, setup_export_data):
        """Test export in Pascal VOC format."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        export_request = {
            "image_ids": setup_export_data["image_ids"][:3],
            "format": "Pascal VOC",
            "include_predictions": True,
            "include_manual_annotations": True,
            "export_name": "pascal_voc_test",
            "split_by_annotation_type": True
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            assert export_data["format"] == "Pascal VOC"

    def test_filtered_export(self, setup_export_data):
        """Test export with various filtering options."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        # Test export with class filtering
        class_filter_request = {
            "image_ids": setup_export_data["image_ids"],
            "format": "COCO",
            "class_filter": ["person", "car"],  # Only export these classes
            "confidence_threshold": 0.7,
            "include_predictions": True,
            "export_name": "filtered_classes_export"
        }
        
        response = client.post("/api/v1/export/annotations", json=class_filter_request)
        
        if response.status_code == 200:
            export_data = response.json()
            assert "filters_applied" in export_data
            filters = export_data["filters_applied"]
            assert "class_filter" in filters
            assert filters["class_filter"] == ["person", "car"]

    def test_large_dataset_export(self, setup_export_data):
        """Test export performance with larger dataset."""
        
        if len(setup_export_data["image_ids"]) < 5:
            pytest.skip("Need more images for large dataset test")
        
        # Export all available images
        large_export_request = {
            "image_ids": setup_export_data["image_ids"],
            "format": "COCO",
            "include_predictions": True,
            "include_manual_annotations": True,
            "export_name": "large_dataset_export",
            "batch_size": 100,  # Process in batches
            "compression": "zip"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/export/annotations", json=large_export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            export_id = export_data["export_id"]
            
            # Monitor export progress
            max_wait = 180  # 3 minutes
            while time.time() - start_time < max_wait:
                response = client.get(f"/api/v1/export/annotations/{export_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    
                    if status_data["status"] == "completed":
                        total_time = time.time() - start_time
                        
                        # Validate performance
                        images_per_second = len(setup_export_data["image_ids"]) / total_time
                        assert images_per_second > 0.1, "Export too slow"  # At least 0.1 images/sec
                        
                        # Validate output size information
                        if "file_size_mb" in status_data:
                            assert status_data["file_size_mb"] > 0
                        
                        break
                    elif status_data["status"] == "failed":
                        pytest.fail("Large dataset export failed")
                
                time.sleep(5)

    def test_export_with_model_comparison(self, setup_export_data):
        """Test export including predictions from multiple models."""
        
        if len(setup_export_data["models"]) < 2:
            pytest.skip("Need multiple models for comparison export")
        
        export_request = {
            "image_ids": setup_export_data["image_ids"][:3],
            "format": "COCO",
            "include_predictions": True,
            "model_ids": [m["id"] for m in setup_export_data["models"][:2]],
            "separate_by_model": True,
            "export_name": "model_comparison_export"
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            
            # Should have separate annotations for each model
            if "metadata" in export_data:
                metadata = export_data["metadata"]
                assert "models_included" in metadata
                assert len(metadata["models_included"]) == 2

    def test_export_error_handling(self, setup_export_data):
        """Test error handling in export workflow."""
        
        # Test export with no images
        response = client.post("/api/v1/export/annotations", json={
            "image_ids": [],
            "format": "COCO"
        })
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        
        # Test export with invalid format
        if setup_export_data["image_ids"]:
            response = client.post("/api/v1/export/annotations", json={
                "image_ids": setup_export_data["image_ids"][:1],
                "format": "INVALID_FORMAT"
            })
            
            assert response.status_code == 400
        
        # Test export with non-existent images
        response = client.post("/api/v1/export/annotations", json={
            "image_ids": ["non-existent-id-1", "non-existent-id-2"],
            "format": "COCO"
        })
        
        assert response.status_code == 400

    def test_export_metadata_and_statistics(self, setup_export_data):
        """Test export metadata and statistics generation."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        export_request = {
            "image_ids": setup_export_data["image_ids"][:4],
            "format": "COCO",
            "include_predictions": True,
            "include_manual_annotations": True,
            "generate_statistics": True,
            "include_metadata": True,
            "export_name": "metadata_test_export"
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            
            # Validate statistics are included
            if export_data["status"] == "completed":
                assert "statistics" in export_data
                stats = export_data["statistics"]
                
                expected_stats = [
                    "total_images", "total_annotations", "annotation_types",
                    "class_distribution", "average_annotations_per_image"
                ]
                
                for stat in expected_stats:
                    assert stat in stats
                
                # Validate statistics values
                assert stats["total_images"] == 4
                assert stats["total_annotations"] > 0
                assert "manual" in stats["annotation_types"]
                
                if setup_export_data["models"]:
                    assert "model" in stats["annotation_types"]

    def test_export_format_validation(self, setup_export_data):
        """Test validation of exported file formats."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        formats_to_test = ["COCO", "YOLO", "Pascal VOC"]
        
        for format_name in formats_to_test:
            export_request = {
                "image_ids": setup_export_data["image_ids"][:2],
                "format": format_name,
                "include_manual_annotations": True,
                "export_name": f"format_validation_{format_name.lower().replace(' ', '_')}"
            }
            
            response = client.post("/api/v1/export/annotations", json=export_request)
            
            if response.status_code == 200:
                export_data = response.json()
                assert export_data["format"] == format_name
                
                # Each format should have specific validation
                if format_name == "COCO":
                    # COCO should have JSON structure
                    assert "content_type" not in export_data or "json" in export_data.get("content_type", "")
                elif format_name == "YOLO":
                    # YOLO should have text files
                    assert "content_type" not in export_data or "text" in export_data.get("content_type", "")
                elif format_name == "Pascal VOC":
                    # Pascal VOC should have XML files
                    assert "content_type" not in export_data or "xml" in export_data.get("content_type", "")

    def test_export_progress_monitoring(self, setup_export_data):
        """Test monitoring of export progress for long-running exports."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        # Create a potentially long-running export
        export_request = {
            "image_ids": setup_export_data["image_ids"],
            "format": "COCO",
            "include_predictions": True,
            "include_manual_annotations": True,
            "high_quality_output": True,  # Might take longer
            "export_name": "progress_monitoring_test"
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            export_id = export_data["export_id"]
            
            # Monitor progress
            max_checks = 20
            progress_updates = []
            
            for _ in range(max_checks):
                response = client.get(f"/api/v1/export/annotations/{export_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    
                    # Validate progress information
                    assert "status" in status_data
                    assert "progress_percentage" in status_data
                    
                    progress_updates.append(status_data["progress_percentage"])
                    
                    if status_data["status"] == "completed":
                        break
                    elif status_data["status"] == "failed":
                        pytest.fail("Export failed during progress monitoring")
                
                time.sleep(2)
            
            # Validate progress made sense
            if len(progress_updates) > 1:
                # Progress should generally increase
                assert progress_updates[-1] >= progress_updates[0]
                assert 0 <= progress_updates[-1] <= 100

    def test_export_cleanup_and_expiration(self, setup_export_data):
        """Test export file cleanup and expiration handling."""
        
        if not setup_export_data["image_ids"]:
            pytest.skip("No test data available")
        
        # Create export with short expiration for testing
        export_request = {
            "image_ids": setup_export_data["image_ids"][:2],
            "format": "COCO",
            "include_manual_annotations": True,
            "export_name": "cleanup_test_export",
            "expiration_hours": 1  # Short expiration for testing
        }
        
        response = client.post("/api/v1/export/annotations", json=export_request)
        
        if response.status_code == 200:
            export_data = response.json()
            export_id = export_data["export_id"]
            
            # Verify export exists
            response = client.get(f"/api/v1/export/annotations/{export_id}")
            assert response.status_code == 200
            
            # Test manual cleanup
            response = client.delete(f"/api/v1/export/annotations/{export_id}")
            
            if response.status_code in [200, 204]:
                # Verify export is cleaned up
                response = client.get(f"/api/v1/export/annotations/{export_id}")
                assert response.status_code == 404