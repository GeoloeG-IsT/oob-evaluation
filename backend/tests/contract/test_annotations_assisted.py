"""
Contract test for POST /api/v1/annotations/assisted (generateAssistedAnnotation operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the assisted annotation generation API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestAnnotationsAssistedContract:
    """Contract tests for assisted annotation endpoint."""

    def test_generate_assisted_annotation_with_sam2(self):
        """Test generating assisted annotation using SAM2 model."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "prompt_points": [
                {"x": 100, "y": 150, "type": "positive"},
                {"x": 200, "y": 250, "type": "negative"}
            ]
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 201
        
        response_data = response.json()
        
        # Validate Annotation schema structure per api-spec.yaml
        required_fields = [
            "id", "image_id", "class_labels", "creation_method", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type and value validations
        assert isinstance(response_data["id"], str)
        assert response_data["image_id"] == image_id
        assert response_data["creation_method"] == "model"
        assert isinstance(response_data["created_at"], str)
        assert isinstance(response_data["class_labels"], list)

    def test_generate_assisted_annotation_with_detection_model(self):
        """Test generating assisted annotation using detection model (YOLO/RT-DETR)."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["creation_method"] == "model"
        
        # Should have bounding boxes for detection models
        if "bounding_boxes" in response_data:
            assert isinstance(response_data["bounding_boxes"], list)
            if response_data["bounding_boxes"]:
                bbox = response_data["bounding_boxes"][0]
                bbox_fields = ["x", "y", "width", "height", "class_id", "confidence"]
                for field in bbox_fields:
                    assert field in bbox

    def test_generate_assisted_annotation_missing_image_id(self):
        """Test generating assisted annotation without required image_id."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_missing_model_id(self):
        """Test generating assisted annotation without required model_id."""
        image_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_invalid_image_id(self):
        """Test generating assisted annotation with invalid UUID format for image_id."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": "not-a-valid-uuid",
            "model_id": model_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_invalid_model_id(self):
        """Test generating assisted annotation with invalid UUID format for model_id."""
        image_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": "not-a-valid-uuid"
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_with_custom_class_labels(self):
        """Test generating assisted annotation with custom class labels."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "class_labels": ["person", "vehicle", "object"],
            "confidence_threshold": 0.7
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["class_labels"] == ["person", "vehicle", "object"]

    def test_generate_assisted_annotation_invalid_confidence_threshold(self):
        """Test generating assisted annotation with invalid confidence threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": 1.5  # Invalid - should be between 0 and 1
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_invalid_nms_threshold(self):
        """Test generating assisted annotation with invalid NMS threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "nms_threshold": -0.1  # Invalid - should be between 0 and 1
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_with_bounding_box_prompts(self):
        """Test generating assisted annotation with bounding box prompts."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "prompt_boxes": [
                {"x": 50, "y": 60, "width": 100, "height": 80}
            ]
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["creation_method"] == "model"

    def test_generate_assisted_annotation_response_headers(self):
        """Test that response headers are correct."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_generate_assisted_annotation_with_metadata(self):
        """Test generating assisted annotation with metadata."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "metadata": {
                "inference_time": 0.123,
                "model_version": "v2.1",
                "parameters": {"threshold": 0.5}
            }
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        if "metadata" in response_data:
            assert isinstance(response_data["metadata"], dict)

    def test_generate_assisted_annotation_nonexistent_image(self):
        """Test generating assisted annotation for non-existent image."""
        image_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 404 Not Found for non-existent image
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_generate_assisted_annotation_nonexistent_model(self):
        """Test generating assisted annotation with non-existent model."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/annotations/assisted", json=request_data)

        # Should return 404 Not Found for non-existent model
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data