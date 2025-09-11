"""
Contract test for GET /api/v1/images (listImages operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the image listing API contract per api-spec.yaml.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestImagesListContract:
    """Contract tests for image listing endpoint."""

    def test_list_images_no_filters(self):
        """Test listing all images without filters."""
        response = client.get("/api/v1/images")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        assert "images" in response_data
        assert "total_count" in response_data
        assert "limit" in response_data
        assert "offset" in response_data
        
        assert isinstance(response_data["images"], list)
        assert isinstance(response_data["total_count"], int)
        assert isinstance(response_data["limit"], int)
        assert isinstance(response_data["offset"], int)
        
        # Check default pagination
        assert response_data["limit"] == 50  # Default limit per spec
        assert response_data["offset"] == 0   # Default offset

    def test_list_images_with_dataset_split_filter(self):
        """Test listing images filtered by dataset split."""
        response = client.get("/api/v1/images?dataset_split=train")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "images" in response_data
        
        # All returned images should have train split
        for image in response_data["images"]:
            assert image["dataset_split"] == "train"

    def test_list_images_with_pagination(self):
        """Test listing images with custom pagination."""
        response = client.get("/api/v1/images?limit=10&offset=5")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["limit"] == 10
        assert response_data["offset"] == 5
        assert len(response_data["images"]) <= 10

    def test_list_images_validation_split(self):
        """Test listing images with validation split filter."""
        response = client.get("/api/v1/images?dataset_split=validation")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for image in response_data["images"]:
            assert image["dataset_split"] == "validation"

    def test_list_images_test_split(self):
        """Test listing images with test split filter."""
        response = client.get("/api/v1/images?dataset_split=test")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for image in response_data["images"]:
            assert image["dataset_split"] == "test"

    def test_list_images_invalid_dataset_split(self):
        """Test listing images with invalid dataset split."""
        response = client.get("/api/v1/images?dataset_split=invalid")

        # Should return 400 Bad Request for invalid enum value
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_list_images_large_limit(self):
        """Test listing images with limit exceeding maximum."""
        response = client.get("/api/v1/images?limit=2000")

        # Should either accept it or cap at maximum (1000 per spec)
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["limit"] <= 1000  # Max limit per spec

    def test_list_images_negative_offset(self):
        """Test listing images with negative offset."""
        response = client.get("/api/v1/images?offset=-1")

        # Should return 400 Bad Request for invalid offset
        assert response.status_code == 400

    def test_list_images_image_structure(self):
        """Test that returned images have correct structure."""
        response = client.get("/api/v1/images?limit=1")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["images"]:  # If any images exist
            image = response_data["images"][0]
            
            # Required fields per Image schema in api-spec.yaml
            required_fields = [
                "id", "filename", "file_path", "file_size", 
                "format", "width", "height", "dataset_split", 
                "upload_timestamp"
            ]
            
            for field in required_fields:
                assert field in image
            
            # Type validations
            assert isinstance(image["id"], str)
            assert isinstance(image["filename"], str)
            assert isinstance(image["file_path"], str)
            assert isinstance(image["file_size"], int)
            assert isinstance(image["format"], str)
            assert isinstance(image["width"], int)
            assert isinstance(image["height"], int)
            assert image["dataset_split"] in ["train", "validation", "test"]
            assert isinstance(image["upload_timestamp"], str)

    def test_list_images_empty_result(self):
        """Test listing images when no images match filters."""
        # Use a filter that might return empty results
        response = client.get("/api/v1/images?dataset_split=test&limit=1000")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "images" in response_data
        assert isinstance(response_data["images"], list)
        assert response_data["total_count"] >= 0

    def test_list_images_response_headers(self):
        """Test that response headers are correct."""
        response = client.get("/api/v1/images")

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_list_images_query_parameter_combinations(self):
        """Test various combinations of query parameters."""
        # Test all parameters together
        response = client.get(
            "/api/v1/images?dataset_split=train&limit=25&offset=10"
        )

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["limit"] == 25
        assert response_data["offset"] == 10
        
        for image in response_data["images"]:
            assert image["dataset_split"] == "train"