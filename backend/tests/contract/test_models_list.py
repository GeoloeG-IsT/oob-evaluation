"""
Contract test for GET /api/v1/models (listModels operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the model listing API contract per api-spec.yaml.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestModelsListContract:
    """Contract tests for model listing endpoint."""

    def test_list_models_no_filters(self):
        """Test listing all models without filters."""
        response = client.get("/api/v1/models")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        assert isinstance(response_data["models"], list)

    def test_list_models_filter_by_type_detection(self):
        """Test listing models filtered by type=detection."""
        response = client.get("/api/v1/models?type=detection")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should be detection models
        for model in response_data["models"]:
            assert model["type"] == "detection"

    def test_list_models_filter_by_type_segmentation(self):
        """Test listing models filtered by type=segmentation."""
        response = client.get("/api/v1/models?type=segmentation")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should be segmentation models
        for model in response_data["models"]:
            assert model["type"] == "segmentation"

    def test_list_models_filter_by_framework_yolo11(self):
        """Test listing models filtered by framework=YOLO11."""
        response = client.get("/api/v1/models?framework=YOLO11")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should use YOLO11 framework
        for model in response_data["models"]:
            assert model["framework"] == "YOLO11"

    def test_list_models_filter_by_framework_yolo12(self):
        """Test listing models filtered by framework=YOLO12."""
        response = client.get("/api/v1/models?framework=YOLO12")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should use YOLO12 framework
        for model in response_data["models"]:
            assert model["framework"] == "YOLO12"

    def test_list_models_filter_by_framework_rt_detr(self):
        """Test listing models filtered by framework=RT-DETR."""
        response = client.get("/api/v1/models?framework=RT-DETR")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should use RT-DETR framework
        for model in response_data["models"]:
            assert model["framework"] == "RT-DETR"

    def test_list_models_filter_by_framework_sam2(self):
        """Test listing models filtered by framework=SAM2."""
        response = client.get("/api/v1/models?framework=SAM2")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should use SAM2 framework
        for model in response_data["models"]:
            assert model["framework"] == "SAM2"

    def test_list_models_invalid_type_filter(self):
        """Test listing models with invalid type filter."""
        response = client.get("/api/v1/models?type=invalid")

        # Should return 400 Bad Request for invalid enum value
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_list_models_invalid_framework_filter(self):
        """Test listing models with invalid framework filter."""
        response = client.get("/api/v1/models?framework=invalid")

        # Should return 400 Bad Request for invalid enum value
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_list_models_combined_filters(self):
        """Test listing models with multiple filters combined."""
        response = client.get("/api/v1/models?type=detection&framework=YOLO11")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        
        # All returned models should match both filters
        for model in response_data["models"]:
            assert model["type"] == "detection"
            assert model["framework"] == "YOLO11"

    def test_list_models_model_structure(self):
        """Test that returned models have correct structure."""
        response = client.get("/api/v1/models")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["models"]:  # If any models exist
            model = response_data["models"][0]
            
            # Required fields per Model schema in api-spec.yaml
            required_fields = [
                "id", "name", "framework", "type", "variant", 
                "description", "version", "created_at"
            ]
            
            for field in required_fields:
                assert field in model
            
            # Type validations
            assert isinstance(model["id"], str)
            assert isinstance(model["name"], str)
            assert model["framework"] in ["YOLO11", "YOLO12", "RT-DETR", "SAM2"]
            assert model["type"] in ["detection", "segmentation"]
            assert isinstance(model["variant"], str)
            assert isinstance(model["description"], str)
            assert isinstance(model["version"], str)
            assert isinstance(model["created_at"], str)

    def test_list_models_yolo11_variants(self):
        """Test that YOLO11 models have correct variants."""
        response = client.get("/api/v1/models?framework=YOLO11")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        valid_yolo11_variants = ["nano", "small", "medium", "large", "extra-large"]
        
        for model in response_data["models"]:
            if model["framework"] == "YOLO11":
                assert model["variant"] in valid_yolo11_variants

    def test_list_models_yolo12_variants(self):
        """Test that YOLO12 models have correct variants."""
        response = client.get("/api/v1/models?framework=YOLO12")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        valid_yolo12_variants = ["nano", "small", "medium", "large", "extra-large"]
        
        for model in response_data["models"]:
            if model["framework"] == "YOLO12":
                assert model["variant"] in valid_yolo12_variants

    def test_list_models_rt_detr_variants(self):
        """Test that RT-DETR models have correct variants."""
        response = client.get("/api/v1/models?framework=RT-DETR")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        valid_rt_detr_variants = [
            "R18", "R34", "R50", "R101", 
            "RF-DETR-nano", "RF-DETR-small", "RF-DETR-medium"
        ]
        
        for model in response_data["models"]:
            if model["framework"] == "RT-DETR":
                assert model["variant"] in valid_rt_detr_variants

    def test_list_models_sam2_variants(self):
        """Test that SAM2 models have correct variants."""
        response = client.get("/api/v1/models?framework=SAM2")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        valid_sam2_variants = ["tiny", "small", "base+", "large"]
        
        for model in response_data["models"]:
            if model["framework"] == "SAM2":
                assert model["variant"] in valid_sam2_variants

    def test_list_models_response_headers(self):
        """Test that response headers are correct."""
        response = client.get("/api/v1/models")

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_list_models_performance_metrics_structure(self):
        """Test model performance metrics structure if present."""
        response = client.get("/api/v1/models")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for model in response_data["models"]:
            if "performance_metrics" in model:
                metrics = model["performance_metrics"]
                assert isinstance(metrics, dict)
                
                # Common metrics that might be present
                if "inference_time_ms" in metrics:
                    assert isinstance(metrics["inference_time_ms"], (int, float))
                if "model_size_mb" in metrics:
                    assert isinstance(metrics["model_size_mb"], (int, float))

    def test_list_models_empty_result(self):
        """Test listing models when no models match filters."""
        # Use filters that might return empty results
        response = client.get("/api/v1/models?type=detection&framework=SAM2")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert "models" in response_data
        assert isinstance(response_data["models"], list)

    def test_list_models_pretrained_flag(self):
        """Test model pretrained flag if present."""
        response = client.get("/api/v1/models")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for model in response_data["models"]:
            if "is_pretrained" in model:
                assert isinstance(model["is_pretrained"], bool)

    def test_list_models_supported_formats(self):
        """Test model supported formats if present."""
        response = client.get("/api/v1/models")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for model in response_data["models"]:
            if "supported_formats" in model:
                assert isinstance(model["supported_formats"], list)
                for format_item in model["supported_formats"]:
                    assert isinstance(format_item, str)