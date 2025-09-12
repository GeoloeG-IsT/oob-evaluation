"""
Contract test for POST /api/v1/evaluation/metrics (calculateMetrics operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the performance metrics calculation API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestEvaluationMetricsContract:
    """Contract tests for performance metrics calculation endpoint."""

    def test_calculate_metrics_basic(self):
        """Test calculating metrics with basic parameters."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map", "precision", "recall"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per MetricsResponse schema
        required_fields = [
            "model_id", "dataset_info", "metrics", "calculated_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert response_data["model_id"] == model_id
        assert isinstance(response_data["dataset_info"], dict)
        assert isinstance(response_data["metrics"], dict)
        assert isinstance(response_data["calculated_at"], str)

    def test_calculate_metrics_all_types(self):
        """Test calculating all available metric types."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["validation", "test"]
            },
            "metrics_to_calculate": [
                "map", "map_50", "map_75", "precision", "recall", 
                "f1_score", "iou", "execution_time"
            ]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        metrics = response_data["metrics"]
        
        # Check that requested metrics are present
        for metric_name in request_data["metrics_to_calculate"]:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], (int, float))

    def test_calculate_metrics_per_class(self):
        """Test calculating metrics with per-class breakdown."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"],
                "class_labels": ["person", "vehicle", "object"]
            },
            "metrics_to_calculate": ["precision", "recall", "f1_score"],
            "per_class_breakdown": True
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "per_class_metrics" in response_data:
            per_class_metrics = response_data["per_class_metrics"]
            assert isinstance(per_class_metrics, dict)
            
            for class_name in request_data["dataset_filter"]["class_labels"]:
                if class_name in per_class_metrics:
                    class_metrics = per_class_metrics[class_name]
                    assert isinstance(class_metrics, dict)

    def test_calculate_metrics_missing_model_id(self):
        """Test calculating metrics without required model_id."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_calculate_metrics_invalid_model_id(self):
        """Test calculating metrics with invalid model_id format."""
        request_data = {
            "model_id": "not-a-valid-uuid",
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_calculate_metrics_nonexistent_model(self):
        """Test calculating metrics with non-existent model."""
        model_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Should return 404 Not Found
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_calculate_metrics_invalid_dataset_split(self):
        """Test calculating metrics with invalid dataset split."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["invalid_split"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_calculate_metrics_invalid_metric_type(self):
        """Test calculating metrics with invalid metric type."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["invalid_metric"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_calculate_metrics_confidence_threshold(self):
        """Test calculating metrics with specific confidence threshold."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["precision", "recall"],
            "confidence_threshold": 0.7
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "evaluation_config" in response_data:
            config = response_data["evaluation_config"]
            assert config.get("confidence_threshold") == 0.7

    def test_calculate_metrics_iou_threshold(self):
        """Test calculating metrics with specific IoU threshold."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map", "iou"],
            "iou_threshold": 0.6
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "evaluation_config" in response_data:
            config = response_data["evaluation_config"]
            assert config.get("iou_threshold") == 0.6

    def test_calculate_metrics_dataset_info(self):
        """Test that dataset information is included in response."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        dataset_info = response_data["dataset_info"]
        
        # Check dataset info structure
        if "total_images" in dataset_info:
            assert isinstance(dataset_info["total_images"], int)
            assert dataset_info["total_images"] >= 0
        
        if "total_annotations" in dataset_info:
            assert isinstance(dataset_info["total_annotations"], int)
            assert dataset_info["total_annotations"] >= 0
        
        if "class_distribution" in dataset_info:
            assert isinstance(dataset_info["class_distribution"], dict)

    def test_calculate_metrics_execution_time(self):
        """Test that execution time metrics are properly calculated."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["execution_time"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        metrics = response_data["metrics"]
        
        if "execution_time" in metrics:
            exec_time = metrics["execution_time"]
            assert isinstance(exec_time, (int, float))
            assert exec_time > 0  # Should be positive

    def test_calculate_metrics_with_metadata(self):
        """Test calculating metrics with metadata."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"],
            "metadata": {
                "evaluation_name": "test_evaluation_001",
                "description": "Testing metrics calculation"
            }
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "metadata" in response_data:
            assert isinstance(response_data["metadata"], dict)

    def test_calculate_metrics_response_headers(self):
        """Test that response headers are correct."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["map"]
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_calculate_metrics_confusion_matrix(self):
        """Test calculating metrics with confusion matrix."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_calculate": ["precision", "recall"],
            "include_confusion_matrix": True
        }

        response = client.post("/api/v1/evaluation/metrics", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "confusion_matrix" in response_data:
            confusion_matrix = response_data["confusion_matrix"]
            assert isinstance(confusion_matrix, list)
            # Should be a 2D array/matrix
            for row in confusion_matrix:
                assert isinstance(row, list)
