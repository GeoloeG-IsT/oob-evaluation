"""
Contract test for GET /api/v1/models/{model_id} (getModel operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the model details API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestModelsGetContract:
    """Contract tests for model details endpoint."""

    def test_get_model_valid_id(self):
        """Test retrieving model details with valid model ID."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per Model schema in api-spec.yaml
        required_fields = [
            "id", "name", "framework", "type", "variant", 
            "description", "version", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["id"], str)
        assert isinstance(response_data["name"], str)
        assert response_data["framework"] in ["YOLO11", "YOLO12", "RT-DETR", "SAM2"]
        assert response_data["type"] in ["detection", "segmentation"]
        assert isinstance(response_data["variant"], str)
        assert isinstance(response_data["description"], str)
        assert isinstance(response_data["version"], str)
        assert isinstance(response_data["created_at"], str)
        
        # The returned model ID should match the requested ID
        assert response_data["id"] == model_id

    def test_get_model_invalid_uuid_format(self):
        """Test retrieving model with invalid UUID format."""
        response = client.get("/api/v1/models/not-a-valid-uuid")

        # Should return 400 Bad Request for invalid UUID format
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_get_model_nonexistent_id(self):
        """Test retrieving model with non-existent ID."""
        nonexistent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{nonexistent_id}")

        # Should return 404 Not Found for non-existent model
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_get_model_yolo11_structure(self):
        """Test YOLO11 model structure and variants."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["framework"] == "YOLO11":
            valid_yolo11_variants = ["nano", "small", "medium", "large", "extra-large"]
            assert response_data["variant"] in valid_yolo11_variants
            assert response_data["type"] == "detection"

    def test_get_model_yolo12_structure(self):
        """Test YOLO12 model structure and variants."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["framework"] == "YOLO12":
            valid_yolo12_variants = ["nano", "small", "medium", "large", "extra-large"]
            assert response_data["variant"] in valid_yolo12_variants
            assert response_data["type"] == "detection"

    def test_get_model_rt_detr_structure(self):
        """Test RT-DETR model structure and variants."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["framework"] == "RT-DETR":
            valid_rt_detr_variants = [
                "R18", "R34", "R50", "R101", 
                "RF-DETR-nano", "RF-DETR-small", "RF-DETR-medium"
            ]
            assert response_data["variant"] in valid_rt_detr_variants
            assert response_data["type"] == "detection"

    def test_get_model_sam2_structure(self):
        """Test SAM2 model structure and variants."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["framework"] == "SAM2":
            valid_sam2_variants = ["tiny", "small", "base+", "large"]
            assert response_data["variant"] in valid_sam2_variants
            assert response_data["type"] == "segmentation"

    def test_get_model_performance_metrics(self):
        """Test model performance metrics if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "performance_metrics" in response_data:
            metrics = response_data["performance_metrics"]
            assert isinstance(metrics, dict)
            
            # Validate common performance metrics
            if "inference_time_ms" in metrics:
                assert isinstance(metrics["inference_time_ms"], (int, float))
                assert metrics["inference_time_ms"] > 0
            
            if "model_size_mb" in metrics:
                assert isinstance(metrics["model_size_mb"], (int, float))
                assert metrics["model_size_mb"] > 0
            
            if "memory_usage_mb" in metrics:
                assert isinstance(metrics["memory_usage_mb"], (int, float))
                assert metrics["memory_usage_mb"] > 0

    def test_get_model_training_info(self):
        """Test model training information if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "training_info" in response_data:
            training_info = response_data["training_info"]
            assert isinstance(training_info, dict)
            
            if "dataset_size" in training_info:
                assert isinstance(training_info["dataset_size"], int)
                assert training_info["dataset_size"] >= 0
            
            if "epochs" in training_info:
                assert isinstance(training_info["epochs"], int)
                assert training_info["epochs"] > 0
            
            if "batch_size" in training_info:
                assert isinstance(training_info["batch_size"], int)
                assert training_info["batch_size"] > 0

    def test_get_model_supported_formats(self):
        """Test model supported formats if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "supported_formats" in response_data:
            supported_formats = response_data["supported_formats"]
            assert isinstance(supported_formats, list)
            
            # Common image formats that should be supported
            common_formats = ["JPEG", "PNG", "TIFF", "BMP", "WEBP"]
            for format_item in supported_formats:
                assert isinstance(format_item, str)
                assert format_item.upper() in common_formats

    def test_get_model_class_labels(self):
        """Test model class labels if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "class_labels" in response_data:
            class_labels = response_data["class_labels"]
            assert isinstance(class_labels, list)
            
            for label in class_labels:
                assert isinstance(label, str)
                assert len(label) > 0

    def test_get_model_metadata(self):
        """Test model metadata if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "metadata" in response_data:
            metadata = response_data["metadata"]
            assert isinstance(metadata, dict)

    def test_get_model_is_pretrained_flag(self):
        """Test model is_pretrained flag if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "is_pretrained" in response_data:
            assert isinstance(response_data["is_pretrained"], bool)

    def test_get_model_response_headers(self):
        """Test that response headers are correct."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_get_model_file_path(self):
        """Test model file path if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "model_path" in response_data:
            assert isinstance(response_data["model_path"], str)
            assert len(response_data["model_path"]) > 0

    def test_get_model_config_parameters(self):
        """Test model configuration parameters if present."""
        model_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/models/{model_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "config" in response_data:
            config = response_data["config"]
            assert isinstance(config, dict)
            
            # Common config parameters
            if "input_size" in config:
                input_size = config["input_size"]
                assert isinstance(input_size, list)
                assert len(input_size) in [2, 3]  # [width, height] or [width, height, channels]
                
            if "confidence_threshold" in config:
                assert isinstance(config["confidence_threshold"], (int, float))
                assert 0 <= config["confidence_threshold"] <= 1
                
            if "nms_threshold" in config:
                assert isinstance(config["nms_threshold"], (int, float))
                assert 0 <= config["nms_threshold"] <= 1