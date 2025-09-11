"""
Contract test for POST /api/v1/annotations (createAnnotation operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the annotation creation API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestAnnotationsCreateContract:
    """Contract tests for annotation creation endpoint."""

    def test_create_annotation_with_bounding_boxes(self):
        """Test creating annotation with bounding boxes."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 100.0,
                    "height": 50.0,
                    "class_id": 0,
                    "confidence": 0.95
                },
                {
                    "x": 150.0,
                    "y": 75.0,
                    "width": 80.0,
                    "height": 120.0,
                    "class_id": 1,
                    "confidence": 0.87
                }
            ],
            "class_labels": ["person", "car"],
            "user_tag": "test_user"
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

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
        assert response_data["class_labels"] == ["person", "car"]
        assert response_data["creation_method"] == "user"
        assert response_data["user_tag"] == "test_user"
        assert isinstance(response_data["created_at"], str)
        
        # Validate bounding boxes
        assert "bounding_boxes" in response_data
        assert len(response_data["bounding_boxes"]) == 2
        
        bbox1 = response_data["bounding_boxes"][0]
        assert bbox1["x"] == 10.0
        assert bbox1["y"] == 20.0
        assert bbox1["width"] == 100.0
        assert bbox1["height"] == 50.0

    def test_create_annotation_with_segments(self):
        """Test creating annotation with segmentation polygons."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "segments": [
                {
                    "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    "class_id": 0,
                    "confidence": 0.92
                }
            ],
            "class_labels": ["building"],
            "user_tag": "annotator_1"
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["class_labels"] == ["building"]
        assert response_data["creation_method"] == "user"
        
        # Validate segments
        assert "segments" in response_data
        assert len(response_data["segments"]) == 1
        
        segment = response_data["segments"][0]
        assert segment["polygon"] == [[10, 10], [20, 10], [20, 20], [10, 20]]
        assert segment["class_id"] == 0

    def test_create_annotation_minimal_required_fields(self):
        """Test creating annotation with only required fields."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "class_labels": ["object"]
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["class_labels"] == ["object"]
        assert response_data["creation_method"] == "user"

    def test_create_annotation_missing_image_id(self):
        """Test creating annotation without required image_id."""
        annotation_data = {
            "class_labels": ["object"]
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_create_annotation_missing_class_labels(self):
        """Test creating annotation without required class_labels."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_create_annotation_empty_class_labels(self):
        """Test creating annotation with empty class_labels array."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "class_labels": []
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_create_annotation_invalid_image_id_format(self):
        """Test creating annotation with invalid UUID format for image_id."""
        annotation_data = {
            "image_id": "not-a-valid-uuid",
            "class_labels": ["object"]
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_create_annotation_with_metadata(self):
        """Test creating annotation with optional metadata."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "class_labels": ["person"],
            "user_tag": "expert_annotator",
            "metadata": {
                "difficulty": "hard",
                "quality": "high",
                "notes": "Partially occluded"
            }
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert "metadata" in response_data
        assert response_data["metadata"]["difficulty"] == "hard"

    def test_create_annotation_response_headers(self):
        """Test that response headers are correct."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "class_labels": ["test"]
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_create_annotation_bounding_box_validation(self):
        """Test validation of bounding box coordinates."""
        image_id = str(uuid.uuid4())
        
        # Test with negative coordinates (should be allowed)
        annotation_data = {
            "image_id": image_id,
            "bounding_boxes": [
                {
                    "x": -10.0,  # Negative coordinates might be valid
                    "y": 5.0,
                    "width": 50.0,
                    "height": 30.0
                }
            ],
            "class_labels": ["object"]
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # This should succeed or fail based on business rules
        assert response.status_code in [201, 400]

    def test_create_annotation_confidence_scores_length_mismatch(self):
        """Test mismatched confidence_scores and class_labels lengths."""
        image_id = str(uuid.uuid4())
        
        annotation_data = {
            "image_id": image_id,
            "class_labels": ["person", "car"],
            "confidence_scores": [0.95]  # Only one score for two labels
        }

        response = client.post("/api/v1/annotations", json=annotation_data)

        # Should return 400 Bad Request for mismatched arrays
        assert response.status_code == 400