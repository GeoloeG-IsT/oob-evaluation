"""
Contract test for POST /api/v1/evaluation/compare (compareModels operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the model comparison API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestEvaluationCompareContract:
    """Contract tests for model comparison endpoint."""

    def test_compare_models_basic(self):
        """Test comparing models with basic parameters."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map", "precision", "recall"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per ModelComparisonResponse schema
        required_fields = [
            "model_comparisons", "dataset_info", "comparison_summary", "compared_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["model_comparisons"], list)
        assert isinstance(response_data["dataset_info"], dict)
        assert isinstance(response_data["comparison_summary"], dict)
        assert isinstance(response_data["compared_at"], str)

    def test_compare_models_multiple(self):
        """Test comparing multiple models."""
        model_ids = [str(uuid.uuid4()) for _ in range(4)]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["validation", "test"]
            },
            "metrics_to_compare": ["map", "f1_score", "execution_time"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        assert len(response_data["model_comparisons"]) == len(model_ids)
        
        for comparison in response_data["model_comparisons"]:
            assert "model_id" in comparison
            assert "metrics" in comparison
            assert comparison["model_id"] in model_ids

    def test_compare_models_missing_model_ids(self):
        """Test comparing models without required model_ids."""
        request_data = {
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_compare_models_empty_model_ids(self):
        """Test comparing models with empty model_ids array."""
        request_data = {
            "model_ids": [],
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_compare_models_single_model(self):
        """Test comparing models with only one model (should fail or handle gracefully)."""
        model_ids = [str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Should require at least 2 models for comparison
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_compare_models_invalid_model_id(self):
        """Test comparing models with invalid model_id format."""
        request_data = {
            "model_ids": [str(uuid.uuid4()), "not-a-valid-uuid"],
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_compare_models_nonexistent_model(self):
        """Test comparing models with non-existent model."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]  # Random UUIDs
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Should return 404 Not Found for non-existent models
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_compare_models_with_ranking(self):
        """Test model comparison with ranking."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map", "precision"],
            "ranking_metric": "map"
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "ranking" in response_data:
            ranking = response_data["ranking"]
            assert isinstance(ranking, list)
            assert len(ranking) == len(model_ids)
            
            for rank_entry in ranking:
                assert "model_id" in rank_entry
                assert "rank" in rank_entry
                assert "score" in rank_entry

    def test_compare_models_statistical_significance(self):
        """Test model comparison with statistical significance testing."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["precision", "recall"],
            "include_statistical_tests": True
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "statistical_tests" in response_data:
            stats = response_data["statistical_tests"]
            assert isinstance(stats, dict)
            
            for test_name, test_result in stats.items():
                if "p_value" in test_result:
                    assert isinstance(test_result["p_value"], (int, float))
                    assert 0 <= test_result["p_value"] <= 1

    def test_compare_models_confidence_intervals(self):
        """Test model comparison with confidence intervals."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"],
            "confidence_level": 0.95
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for comparison in response_data["model_comparisons"]:
            metrics = comparison["metrics"]
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "confidence_interval" in metric_data:
                    ci = metric_data["confidence_interval"]
                    assert "lower" in ci and "upper" in ci
                    assert ci["lower"] <= ci["upper"]

    def test_compare_models_summary_statistics(self):
        """Test model comparison summary statistics."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map", "precision"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        summary = response_data["comparison_summary"]
        
        if "best_model" in summary:
            best_model = summary["best_model"]
            assert "model_id" in best_model
            assert "metric" in best_model
            assert "score" in best_model
        
        if "metric_statistics" in summary:
            stats = summary["metric_statistics"]
            assert isinstance(stats, dict)

    def test_compare_models_response_headers(self):
        """Test that response headers are correct."""
        model_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        
        request_data = {
            "model_ids": model_ids,
            "dataset_filter": {
                "dataset_splits": ["test"]
            },
            "metrics_to_compare": ["map"]
        }

        response = client.post("/api/v1/evaluation/compare", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"
