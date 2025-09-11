"""
Integration test for complete image upload workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete image upload workflow from quickstart.md Step 1.

Workflow:
1. Upload multiple images with different formats
2. Organize images into train/validation/test splits
3. Verify image metadata and thumbnails
4. Test concurrent uploads
5. Validate error handling for unsupported formats
"""

import pytest
import tempfile
import os
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
import concurrent.futures

from src.main import app

client = TestClient(app)


class TestImageUploadWorkflowIntegration:
    """Integration tests for complete image upload workflow."""

    @pytest.fixture(scope="class")
    def test_images(self):
        """Create test images in different formats."""
        images = {}
        
        # Create JPEG image
        jpeg_image = Image.new('RGB', (640, 480), color='red')
        jpeg_buffer = BytesIO()
        jpeg_image.save(jpeg_buffer, format='JPEG', quality=90)
        jpeg_buffer.seek(0)
        images['jpeg'] = ("test_image.jpg", jpeg_buffer, "image/jpeg")
        
        # Create PNG image
        png_image = Image.new('RGBA', (512, 384), color='green')
        png_buffer = BytesIO()
        png_image.save(png_buffer, format='PNG')
        png_buffer.seek(0)
        images['png'] = ("test_image.png", png_buffer, "image/png")
        
        # Create large TIFF image
        tiff_image = Image.new('RGB', (2048, 1536), color='blue')
        tiff_buffer = BytesIO()
        tiff_image.save(tiff_buffer, format='TIFF')
        tiff_buffer.seek(0)
        images['tiff'] = ("test_image.tiff", tiff_buffer, "image/tiff")
        
        return images

    def test_complete_image_upload_workflow(self, test_images):
        """Test the complete image upload workflow from quickstart Step 1."""
        
        # Step 1: Upload training images (60% of total)
        training_files = [
            test_images['jpeg'],
            test_images['png']
        ]
        
        response = client.post(
            "/api/v1/images",
            files=[("files", file) for file in training_files],
            data={"dataset_split": "train"}
        )
        
        # This MUST FAIL since endpoints don't exist yet
        assert response.status_code == 201
        
        train_response = response.json()
        assert train_response["total_count"] == 2
        assert train_response["success_count"] == 2
        assert train_response["failed_count"] == 0
        
        # Verify all images in training split
        for uploaded_image in train_response["uploaded_images"]:
            assert uploaded_image["dataset_split"] == "train"
            assert "id" in uploaded_image
            assert "file_path" in uploaded_image
            assert uploaded_image["width"] > 0
            assert uploaded_image["height"] > 0
        
        train_image_ids = [img["id"] for img in train_response["uploaded_images"]]
        
        # Step 2: Upload validation images (20% of total)
        validation_file = [test_images['tiff']]
        
        response = client.post(
            "/api/v1/images",
            files=[("files", file) for file in validation_file],
            data={"dataset_split": "validation"}
        )
        
        assert response.status_code == 201
        val_response = response.json()
        assert val_response["total_count"] == 1
        assert val_response["uploaded_images"][0]["dataset_split"] == "validation"
        
        val_image_ids = [img["id"] for img in val_response["uploaded_images"]]
        
        # Step 3: Upload test images using default split (should default to train)
        test_file = [("test_default.jpg", test_images['jpeg'][1], "image/jpeg")]
        
        response = client.post(
            "/api/v1/images",
            files=[("files", file) for file in test_file]
            # No dataset_split specified - should default to train
        )
        
        assert response.status_code == 201
        default_response = response.json()
        assert default_response["uploaded_images"][0]["dataset_split"] == "train"
        
        # Step 4: Verify images are organized correctly by split
        # List training images
        response = client.get("/api/v1/images?dataset_split=train&limit=10")
        assert response.status_code == 200
        
        train_list = response.json()
        assert "images" in train_list
        assert len(train_list["images"]) >= 3  # 2 explicit + 1 default
        
        # Verify training images contain our uploaded ones
        listed_train_ids = [img["id"] for img in train_list["images"]]
        for train_id in train_image_ids:
            assert train_id in listed_train_ids
        
        # List validation images
        response = client.get("/api/v1/images?dataset_split=validation")
        assert response.status_code == 200
        
        val_list = response.json()
        assert len(val_list["images"]) >= 1
        assert val_list["images"][0]["id"] in val_image_ids
        
        # Step 5: Verify image metadata is correctly stored
        test_image_id = train_image_ids[0]
        response = client.get(f"/api/v1/images/{test_image_id}")
        assert response.status_code == 200
        
        image_detail = response.json()
        assert image_detail["id"] == test_image_id
        assert "filename" in image_detail
        assert "file_size" in image_detail
        assert "format" in image_detail
        assert "width" in image_detail
        assert "height" in image_detail
        assert "upload_timestamp" in image_detail
        assert image_detail["dataset_split"] == "train"

    def test_concurrent_image_uploads(self, test_images):
        """Test concurrent image uploads work correctly."""
        
        def upload_image(image_data, split):
            """Helper function for concurrent uploads."""
            return client.post(
                "/api/v1/images",
                files={"files": image_data},
                data={"dataset_split": split}
            )
        
        # Prepare concurrent uploads
        upload_tasks = [
            (test_images['jpeg'], "train"),
            (test_images['png'], "validation"),
            (test_images['tiff'], "test")
        ]
        
        # Execute concurrent uploads
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(upload_image, img_data, split)
                for img_data, split in upload_tasks
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all uploads succeeded
        for response in results:
            assert response.status_code == 201
            response_data = response.json()
            assert response_data["success_count"] == 1
            assert response_data["failed_count"] == 0

    def test_batch_upload_different_formats(self, test_images):
        """Test uploading multiple images with different formats in one request."""
        
        mixed_files = [
            ("files", test_images['jpeg']),
            ("files", test_images['png']),
            ("files", test_images['tiff'])
        ]
        
        response = client.post(
            "/api/v1/images",
            files=mixed_files,
            data={"dataset_split": "test"}
        )
        
        assert response.status_code == 201
        
        response_data = response.json()
        assert response_data["total_count"] == 3
        assert response_data["success_count"] == 3
        assert response_data["failed_count"] == 0
        
        # Verify different formats handled correctly
        formats_found = set()
        for uploaded_image in response_data["uploaded_images"]:
            formats_found.add(uploaded_image["format"])
            assert uploaded_image["dataset_split"] == "test"
        
        assert "JPEG" in formats_found
        assert "PNG" in formats_found
        assert "TIFF" in formats_found

    def test_error_handling_unsupported_files(self):
        """Test error handling for unsupported file formats."""
        
        # Create a text file disguised as image
        text_buffer = BytesIO(b"This is not an image file")
        
        response = client.post(
            "/api/v1/images",
            files={"files": ("fake_image.txt", text_buffer, "text/plain")},
            data={"dataset_split": "train"}
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 413, 422]
        
        if response.status_code == 400:
            error_data = response.json()
            assert "error" in error_data
            assert "unsupported" in error_data["error"].lower() or "invalid" in error_data["error"].lower()

    def test_error_handling_no_files(self):
        """Test error handling when no files provided."""
        
        response = client.post(
            "/api/v1/images",
            data={"dataset_split": "train"}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data

    def test_error_handling_invalid_dataset_split(self, test_images):
        """Test error handling for invalid dataset split values."""
        
        response = client.post(
            "/api/v1/images",
            files={"files": test_images['jpeg']},
            data={"dataset_split": "invalid_split"}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data

    def test_large_file_handling(self):
        """Test handling of very large image files."""
        
        # Create a large image (5000x5000 pixels)
        large_image = Image.new('RGB', (5000, 5000), color='red')
        large_buffer = BytesIO()
        large_image.save(large_buffer, format='JPEG', quality=95)
        large_buffer.seek(0)
        
        response = client.post(
            "/api/v1/images",
            files={"files": ("large_image.jpg", large_buffer, "image/jpeg")},
            data={"dataset_split": "test"}
        )
        
        # Should either succeed or return appropriate error code
        assert response.status_code in [201, 413, 422]
        
        if response.status_code == 201:
            response_data = response.json()
            assert response_data["success_count"] == 1
            large_image_data = response_data["uploaded_images"][0]
            assert large_image_data["width"] == 5000
            assert large_image_data["height"] == 5000

    def test_upload_progress_and_metadata_validation(self, test_images):
        """Test upload progress tracking and metadata validation."""
        
        response = client.post(
            "/api/v1/images",
            files={"files": test_images['jpeg']},
            data={"dataset_split": "train"}
        )
        
        assert response.status_code == 201
        response_data = response.json()
        
        uploaded_image = response_data["uploaded_images"][0]
        
        # Validate all required metadata fields present
        required_fields = [
            "id", "filename", "file_path", "file_size",
            "format", "width", "height", "dataset_split", "upload_timestamp"
        ]
        
        for field in required_fields:
            assert field in uploaded_image, f"Missing required field: {field}"
        
        # Validate metadata types and values
        assert isinstance(uploaded_image["id"], str)
        assert isinstance(uploaded_image["filename"], str)
        assert isinstance(uploaded_image["file_size"], int)
        assert uploaded_image["file_size"] > 0
        assert isinstance(uploaded_image["width"], int)
        assert uploaded_image["width"] > 0
        assert isinstance(uploaded_image["height"], int)
        assert uploaded_image["height"] > 0
        assert uploaded_image["format"] in ["JPEG", "PNG", "TIFF"]

    def test_image_listing_pagination_and_filtering(self, test_images):
        """Test image listing with pagination and filtering."""
        
        # First upload some test images
        for i, (format_name, image_data) in enumerate(test_images.items()):
            split = ["train", "validation", "test"][i % 3]
            client.post(
                "/api/v1/images",
                files={"files": image_data},
                data={"dataset_split": split}
            )
        
        # Test pagination
        response = client.get("/api/v1/images?limit=2&offset=0")
        assert response.status_code == 200
        
        page1 = response.json()
        assert "images" in page1
        assert len(page1["images"]) <= 2
        assert "total_count" in page1
        assert "has_more" in page1
        
        # Test filtering by dataset split
        response = client.get("/api/v1/images?dataset_split=train")
        assert response.status_code == 200
        
        train_images = response.json()
        for image in train_images["images"]:
            assert image["dataset_split"] == "train"

    def test_cleanup_after_workflow(self):
        """Test that system maintains clean state after workflow completion."""
        
        # This would test cleanup mechanisms, temporary file removal, etc.
        # For now, just verify the system is responsive after all operations
        response = client.get("/api/v1/images")
        assert response.status_code == 200
        
        # Verify basic health of the system
        assert "images" in response.json()