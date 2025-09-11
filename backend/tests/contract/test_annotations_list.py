"""
Contract test for GET /api/v1/annotations (listAnnotations operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the annotation listing API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestAnnotationsListContract:
    """Contract tests for annotation listing endpoint."""

    def test_list_annotations_no_filters(self):
        """Test listing all annotations without filters."""
        response = client.get("/api/v1/annotations")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        assert isinstance(response_data["annotations"], list)

    def test_list_annotations_filter_by_image_id(self):
        """Test listing annotations filtered by image_id."""
        image_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/annotations?image_id={image_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        
        # All returned annotations should have the specified image_id
        for annotation in response_data["annotations"]:
            assert annotation["image_id"] == image_id

    def test_list_annotations_filter_by_model_id(self):
        """Test listing annotations filtered by model_id."""
        model_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/annotations?model_id={model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        
        # All returned annotations should have the specified model_id
        for annotation in response_data["annotations"]:
            assert annotation.get("model_id") == model_id

    def test_list_annotations_filter_by_creation_method_user(self):
        """Test listing annotations filtered by creation_method=user."""
        response = client.get("/api/v1/annotations?creation_method=user")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        
        # All returned annotations should have creation_method=user
        for annotation in response_data["annotations"]:
            assert annotation["creation_method"] == "user"

    def test_list_annotations_filter_by_creation_method_model(self):
        """Test listing annotations filtered by creation_method=model."""
        response = client.get("/api/v1/annotations?creation_method=model")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        
        # All returned annotations should have creation_method=model
        for annotation in response_data["annotations"]:
            assert annotation["creation_method"] == "model"

    def test_list_annotations_invalid_creation_method(self):
        """Test listing annotations with invalid creation_method."""
        response = client.get("/api/v1/annotations?creation_method=invalid")

        # Should return 400 Bad Request for invalid enum value
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_list_annotations_invalid_uuid_format(self):
        """Test listing annotations with invalid UUID format."""
        response = client.get("/api/v1/annotations?image_id=not-a-valid-uuid")

        # Should return 400 Bad Request for invalid UUID
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_list_annotations_combined_filters(self):
        """Test listing annotations with multiple filters combined."""
        image_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/annotations?image_id={image_id}&creation_method=user"
        )

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        
        # All returned annotations should match both filters
        for annotation in response_data["annotations"]:
            assert annotation["image_id"] == image_id
            assert annotation["creation_method"] == "user"

    def test_list_annotations_annotation_structure(self):
        """Test that returned annotations have correct structure."""
        response = client.get("/api/v1/annotations")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["annotations"]:  # If any annotations exist
            annotation = response_data["annotations"][0]
            
            # Required fields per Annotation schema in api-spec.yaml
            required_fields = [
                "id", "image_id", "class_labels", "creation_method", "created_at"
            ]
            
            for field in required_fields:
                assert field in annotation
            
            # Type validations
            assert isinstance(annotation["id"], str)
            assert isinstance(annotation["image_id"], str)
            assert isinstance(annotation["class_labels"], list)
            assert annotation["creation_method"] in ["user", "model"]
            assert isinstance(annotation["created_at"], str)

    def test_list_annotations_empty_result(self):
        """Test listing annotations when no annotations match filters."""
        # Use a filter that might return empty results
        nonexistent_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/annotations?image_id={nonexistent_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "annotations" in response_data
        assert isinstance(response_data["annotations"], list)

    def test_list_annotations_response_headers(self):
        """Test that response headers are correct."""
        response = client.get("/api/v1/annotations")

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_list_annotations_with_bounding_boxes(self):
        """Test annotation structure when bounding boxes are present."""
        response = client.get("/api/v1/annotations")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for annotation in response_data["annotations"]:
            if "bounding_boxes" in annotation:
                assert isinstance(annotation["bounding_boxes"], list)
                if annotation["bounding_boxes"]:
                    bbox = annotation["bounding_boxes"][0]
                    bbox_fields = ["x", "y", "width", "height", "class_id"]
                    for field in bbox_fields:
                        assert field in bbox

    def test_list_annotations_with_segments(self):
        """Test annotation structure when segments are present."""
        response = client.get("/api/v1/annotations")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for annotation in response_data["annotations"]:
            if "segments" in annotation:
                assert isinstance(annotation["segments"], list)
                if annotation["segments"]:
                    segment = annotation["segments"][0]
                    segment_fields = ["polygon", "class_id"]
                    for field in segment_fields:
                        assert field in segment
                    assert isinstance(segment["polygon"], list)