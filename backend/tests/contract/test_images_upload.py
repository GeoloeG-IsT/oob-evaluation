"""
Contract test for POST /api/v1/images (uploadImages operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the image upload API contract per api-spec.yaml.
"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import tempfile
import os

from src.main import app

client = TestClient(app)


class TestImagesUploadContract:
    """Contract tests for image upload endpoint."""

    def test_upload_single_image_success(self):
        """Test successful upload of a single image."""
        # Create a test image
        image = Image.new('RGB', (100, 100), color='red')
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        # Test the contract
        response = client.post(
            "/api/v1/images",
            files={"files": ("test.jpg", image_buffer, "image/jpeg")},
            data={"dataset_split": "train"}
        )

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 201
        
        response_data = response.json()
        assert "uploaded_images" in response_data
        assert "total_count" in response_data
        assert "success_count" in response_data
        assert "failed_count" in response_data
        
        assert response_data["total_count"] == 1
        assert response_data["success_count"] == 1
        assert response_data["failed_count"] == 0
        assert len(response_data["uploaded_images"]) == 1

        # Validate uploaded image structure
        uploaded_image = response_data["uploaded_images"][0]
        assert "id" in uploaded_image
        assert "filename" in uploaded_image
        assert "file_path" in uploaded_image
        assert "file_size" in uploaded_image
        assert "format" in uploaded_image
        assert "width" in uploaded_image
        assert "height" in uploaded_image
        assert "dataset_split" in uploaded_image
        assert "upload_timestamp" in uploaded_image

        assert uploaded_image["filename"] == "test.jpg"
        assert uploaded_image["format"] == "JPEG"
        assert uploaded_image["width"] == 100
        assert uploaded_image["height"] == 100
        assert uploaded_image["dataset_split"] == "train"

    def test_upload_multiple_images_success(self):
        """Test successful upload of multiple images."""
        # Create test images
        files = []
        for i in range(3):
            image = Image.new('RGB', (50, 50), color=['red', 'green', 'blue'][i])
            image_buffer = BytesIO()
            image.save(image_buffer, format='PNG')
            image_buffer.seek(0)
            files.append(("files", (f"test_{i}.png", image_buffer, "image/png")))

        # Test the contract
        response = client.post(
            "/api/v1/images",
            files=files,
            data={"dataset_split": "validation"}
        )

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["total_count"] == 3
        assert response_data["success_count"] == 3
        assert response_data["failed_count"] == 0
        assert len(response_data["uploaded_images"]) == 3

        # All images should have validation split
        for uploaded_image in response_data["uploaded_images"]:
            assert uploaded_image["dataset_split"] == "validation"

    def test_upload_with_default_dataset_split(self):
        """Test upload with default dataset split (train)."""
        image = Image.new('RGB', (100, 100), color='blue')
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        response = client.post(
            "/api/v1/images",
            files={"files": ("test.jpg", image_buffer, "image/jpeg")}
            # No dataset_split provided - should default to train
        )

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["uploaded_images"][0]["dataset_split"] == "train"

    def test_upload_invalid_dataset_split(self):
        """Test upload with invalid dataset split."""
        image = Image.new('RGB', (100, 100), color='green')
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        response = client.post(
            "/api/v1/images",
            files={"files": ("test.jpg", image_buffer, "image/jpeg")},
            data={"dataset_split": "invalid_split"}
        )

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_upload_no_files(self):
        """Test upload with no files provided."""
        response = client.post(
            "/api/v1/images",
            data={"dataset_split": "train"}
        )

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_upload_non_image_file(self):
        """Test upload with non-image file."""
        # Create a text file
        text_buffer = BytesIO(b"This is not an image")
        
        response = client.post(
            "/api/v1/images",
            files={"files": ("test.txt", text_buffer, "text/plain")},
            data={"dataset_split": "train"}
        )

        # Should return 400 Bad Request or handle gracefully
        assert response.status_code in [400, 413]  # Bad Request or Payload Too Large

    def test_upload_large_file(self):
        """Test upload with very large file (if limits exist)."""
        # Create a large image
        large_image = Image.new('RGB', (5000, 5000), color='red')
        image_buffer = BytesIO()
        large_image.save(image_buffer, format='JPEG', quality=95)
        image_buffer.seek(0)

        response = client.post(
            "/api/v1/images",
            files={"files": ("large_test.jpg", image_buffer, "image/jpeg")},
            data={"dataset_split": "test"}
        )

        # Should either succeed or return 413 if size limits enforced
        assert response.status_code in [201, 413]

    def test_upload_response_headers(self):
        """Test response headers are correct."""
        image = Image.new('RGB', (100, 100), color='yellow')
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        response = client.post(
            "/api/v1/images",
            files={"files": ("test.jpg", image_buffer, "image/jpeg")},
            data={"dataset_split": "train"}
        )

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_upload_concurrent_requests(self):
        """Test that concurrent uploads work correctly."""
        # This would test thread safety but for now just test basic functionality
        image = Image.new('RGB', (100, 100), color='purple')
        image_buffer = BytesIO()
        image.save(image_buffer, format='JPEG')
        image_buffer.seek(0)

        response = client.post(
            "/api/v1/images",
            files={"files": ("concurrent_test.jpg", image_buffer, "image/jpeg")},
            data={"dataset_split": "train"}
        )

        # Contract assertions - this MUST FAIL
        assert response.status_code == 201