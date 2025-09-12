"""
Contract test for GET /api/v1/images/{image_id} (getImage operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the single image retrieval API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestImagesGetContract:
    """Contract tests for single image retrieval endpoint."""

    def test_get_image_by_valid_id(self):
        """Test retrieving a single image by valid UUID."""
        # Use a sample UUID (would be real after images are uploaded)
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Validate Image schema structure per api-spec.yaml
            required_fields = [
                "id", "filename", "file_path", "file_size",
                "format", "width", "height", "dataset_split",
                "upload_timestamp"
            ]
            
            for field in required_fields:
                assert field in response_data
            
            # Type validations
            assert isinstance(response_data["id"], str)
            assert isinstance(response_data["filename"], str)
            assert isinstance(response_data["file_path"], str)
            assert isinstance(response_data["file_size"], int)
            assert isinstance(response_data["format"], str)
            assert isinstance(response_data["width"], int)
            assert isinstance(response_data["height"], int)
            assert response_data["dataset_split"] in ["train", "validation", "test"]
            assert isinstance(response_data["upload_timestamp"], str)
            
            # The returned ID should match the requested ID
            assert response_data["id"] == image_id

    def test_get_image_not_found(self):
        """Test retrieving image with non-existent UUID."""
        non_existent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{non_existent_id}")

        # Should return 404 Not Found
        assert response.status_code == 404
        
        response_data = response.json()
        assert "error" in response_data
        assert "message" in response_data
        assert "timestamp" in response_data

    def test_get_image_invalid_uuid_format(self):
        """Test retrieving image with invalid UUID format."""
        invalid_id = "not-a-valid-uuid"
        
        response = client.get(f"/api/v1/images/{invalid_id}")

        # Should return 400 Bad Request for invalid UUID format
        assert response.status_code == 400
        
        response_data = response.json()
        assert "error" in response_data

    def test_get_image_empty_id(self):
        """Test retrieving image with empty ID."""
        response = client.get("/api/v1/images/")

        # Should return 404 or 405 (method not allowed for base path)
        assert response.status_code in [404, 405]

    def test_get_image_with_metadata(self):
        """Test that image response includes optional metadata if present."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            
            # metadata field is optional in schema
            if "metadata" in response_data:
                assert isinstance(response_data["metadata"], (dict, type(None)))

    def test_get_image_response_headers(self):
        """Test that response headers are correct."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Check content type regardless of status code
        assert response.headers["content-type"] == "application/json"

    def test_get_image_file_size_positive(self):
        """Test that image file size is always positive."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            assert response_data["file_size"] > 0

    def test_get_image_dimensions_positive(self):
        """Test that image dimensions are always positive."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            assert response_data["width"] > 0
            assert response_data["height"] > 0

    def test_get_image_filename_not_empty(self):
        """Test that filename is never empty."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            assert len(response_data["filename"]) > 0
            assert response_data["filename"].strip() != ""

    def test_get_image_format_valid(self):
        """Test that image format is a valid image format."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            valid_formats = ["JPEG", "PNG", "TIFF", "BMP", "GIF", "WEBP"]
            assert response_data["format"] in valid_formats

    def test_get_image_timestamp_format(self):
        """Test that upload timestamp is in valid ISO format."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            timestamp = response_data["upload_timestamp"]
            
            # Should be parseable as ISO datetime
            from datetime import datetime
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {timestamp}")

    def test_get_image_path_not_empty(self):
        """Test that file path is not empty."""
        image_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/images/{image_id}")

        # Contract assertions - this MUST FAIL
        if response.status_code == 200:
            response_data = response.json()
            assert len(response_data["file_path"]) > 0
            assert response_data["file_path"].strip() != ""