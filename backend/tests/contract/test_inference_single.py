"""
Contract test for POST /api/v1/inference/single (runSingleInference operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the single image inference API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestInferenceSingleContract:
    """Contract tests for single image inference endpoint."""

    def test_run_single_inference_basic(self):
        """Test running single image inference with basic parameters."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per InferenceResult schema
        required_fields = [
            "inference_id", "image_id", "model_id", "predictions", 
            "inference_time_ms", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["inference_id"], str)
        assert response_data["image_id"] == image_id
        assert response_data["model_id"] == model_id
        assert isinstance(response_data["predictions"], list)
        assert isinstance(response_data["inference_time_ms"], (int, float))
        assert response_data["inference_time_ms"] > 0
        assert isinstance(response_data["created_at"], str)

    def test_run_single_inference_with_confidence_threshold(self):
        """Test running single image inference with confidence threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": 0.7
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["model_id"] == model_id
        
        # All predictions should have confidence >= threshold
        for prediction in response_data["predictions"]:
            if "confidence" in prediction:
                assert prediction["confidence"] >= 0.7

    def test_run_single_inference_with_nms_threshold(self):
        """Test running single image inference with NMS threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "nms_threshold": 0.4
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["model_id"] == model_id

    def test_run_single_inference_missing_image_id(self):
        """Test running single image inference without required image_id."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_missing_model_id(self):
        """Test running single image inference without required model_id."""
        image_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_invalid_image_id(self):
        """Test running single image inference with invalid image_id format."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": "not-a-valid-uuid",
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_invalid_model_id(self):
        """Test running single image inference with invalid model_id format."""
        image_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": "not-a-valid-uuid"
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_invalid_confidence_threshold(self):
        """Test running single image inference with invalid confidence threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": 1.5  # Invalid - should be between 0 and 1
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_invalid_nms_threshold(self):
        """Test running single image inference with invalid NMS threshold."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "nms_threshold": -0.1  # Invalid - should be between 0 and 1
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_nonexistent_image(self):
        """Test running single image inference with non-existent image."""
        image_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 404 Not Found
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_nonexistent_model(self):
        """Test running single image inference with non-existent model."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Should return 404 Not Found
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_run_single_inference_detection_predictions(self):
        """Test detection model predictions structure."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for prediction in response_data["predictions"]:
            if "bounding_box" in prediction:
                bbox = prediction["bounding_box"]
                bbox_fields = ["x", "y", "width", "height"]
                for field in bbox_fields:
                    assert field in bbox
                    assert isinstance(bbox[field], (int, float))
            
            if "class_id" in prediction:
                assert isinstance(prediction["class_id"], int)
                assert prediction["class_id"] >= 0
            
            if "confidence" in prediction:
                assert isinstance(prediction["confidence"], (int, float))
                assert 0 <= prediction["confidence"] <= 1

    def test_run_single_inference_segmentation_predictions(self):
        """Test segmentation model predictions structure."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for prediction in response_data["predictions"]:
            if "mask" in prediction:
                mask = prediction["mask"]
                assert isinstance(mask, list)  # Could be polygon or mask data
            
            if "polygon" in prediction:
                polygon = prediction["polygon"]
                assert isinstance(polygon, list)
                for point in polygon:
                    assert isinstance(point, list)
                    assert len(point) == 2  # [x, y] coordinates

    def test_run_single_inference_with_custom_parameters(self):
        """Test running single image inference with custom parameters."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id,
            "confidence_threshold": 0.6,
            "nms_threshold": 0.45,
            "max_detections": 100,
            "class_filter": [0, 1, 2]  # Only detect specific classes
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["image_id"] == image_id
        assert response_data["model_id"] == model_id
        
        # Validate that max_detections is respected
        if "max_detections" in request_data:
            assert len(response_data["predictions"]) <= request_data["max_detections"]

    def test_run_single_inference_response_headers(self):
        """Test that response headers are correct."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_run_single_inference_performance_metrics(self):
        """Test inference performance metrics in response."""
        image_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        
        request_data = {
            "image_id": image_id,
            "model_id": model_id
        }

        response = client.post("/api/v1/inference/single", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Should include timing information
        assert "inference_time_ms" in response_data
        assert isinstance(response_data["inference_time_ms"], (int, float))
        assert response_data["inference_time_ms"] > 0
        
        # Optional performance metrics
        if "preprocessing_time_ms" in response_data:
            assert isinstance(response_data["preprocessing_time_ms"], (int, float))
            assert response_data["preprocessing_time_ms"] >= 0
            
        if "postprocessing_time_ms" in response_data:
            assert isinstance(response_data["postprocessing_time_ms"], (int, float))
            assert response_data["postprocessing_time_ms"] >= 0