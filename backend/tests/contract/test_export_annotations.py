"""
Contract test for POST /api/v1/export/annotations (exportAnnotations operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the annotation export API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestExportAnnotationsContract:
    """Contract tests for annotation export endpoint."""

    def test_export_annotations_coco_format(self):
        """Test exporting annotations in COCO format."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train", "validation"]
            },
            "export_format": "coco",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        # Should return a ZIP file
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert "attachment" in response.headers["content-disposition"]

    def test_export_annotations_yolo_format(self):
        """Test exporting annotations in YOLO format."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "export_format": "yolo",
            "include_images": False
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_export_annotations_pascal_voc_format(self):
        """Test exporting annotations in Pascal VOC format."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["validation"],
                "class_labels": ["person", "vehicle"]
            },
            "export_format": "pascal_voc",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_export_annotations_missing_format(self):
        """Test exporting annotations without required format."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train"]
            },
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_export_annotations_invalid_format(self):
        """Test exporting annotations with invalid format."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train"]
            },
            "export_format": "invalid_format",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_export_annotations_invalid_dataset_split(self):
        """Test exporting annotations with invalid dataset split."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["invalid_split"]
            },
            "export_format": "coco",
            "include_images": False
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_export_annotations_empty_dataset(self):
        """Test exporting annotations with empty dataset filter."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": []
            },
            "export_format": "coco",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Should return 400 Bad Request for empty splits
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_export_annotations_with_model_filter(self):
        """Test exporting annotations filtered by model."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["validation"],
                "model_id": model_id,
                "creation_method": "model"
            },
            "export_format": "yolo",
            "include_images": False
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_export_annotations_with_confidence_filter(self):
        """Test exporting annotations with confidence threshold filter."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["test"],
                "min_confidence": 0.8
            },
            "export_format": "coco",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_export_annotations_response_headers(self):
        """Test that response headers are correct for ZIP download."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train"]
            },
            "export_format": "coco",
            "include_images": False
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Should have proper ZIP file headers
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert "filename" in response.headers["content-disposition"]

    def test_export_annotations_user_filter(self):
        """Test exporting annotations filtered by user."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train"],
                "user_tag": "annotator_1",
                "creation_method": "user"
            },
            "export_format": "pascal_voc",
            "include_images": True
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_export_annotations_large_dataset(self):
        """Test exporting annotations for large dataset."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["train", "validation", "test"]
            },
            "export_format": "coco",
            "include_images": True,
            "compression_level": 9  # Maximum compression for large files
        }

        response = client.post("/api/v1/export/annotations", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        
        # Should handle large files properly
        if "content-length" in response.headers:
            assert int(response.headers["content-length"]) > 0
