"""
Contract test for GET /api/v1/inference/jobs/{job_id} (getInferenceJob operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the inference job status API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestInferenceJobsContract:
    """Contract tests for inference job status endpoint."""

    def test_get_inference_job_valid_id(self):
        """Test retrieving inference job with valid job ID."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per InferenceJob schema
        required_fields = [
            "job_id", "status", "image_ids", "model_id", 
            "total_images", "processed_images", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["job_id"], str)
        assert response_data["status"] in ["pending", "running", "completed", "failed"]
        assert isinstance(response_data["image_ids"], list)
        assert isinstance(response_data["model_id"], str)
        assert isinstance(response_data["total_images"], int)
        assert isinstance(response_data["processed_images"], int)
        assert isinstance(response_data["created_at"], str)
        
        # The returned job ID should match the requested ID
        assert response_data["job_id"] == job_id

    def test_get_inference_job_invalid_uuid_format(self):
        """Test retrieving inference job with invalid UUID format."""
        response = client.get("/api/v1/inference/jobs/not-a-valid-uuid")

        # Should return 400 Bad Request for invalid UUID format
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_get_inference_job_nonexistent_id(self):
        """Test retrieving inference job with non-existent ID."""
        nonexistent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{nonexistent_id}")

        # Should return 404 Not Found for non-existent job
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_get_inference_job_pending_status(self):
        """Test inference job with pending status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "pending":
            assert response_data["processed_images"] == 0
            assert "started_at" not in response_data or response_data["started_at"] is None

    def test_get_inference_job_running_status(self):
        """Test inference job with running status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "running":
            assert response_data["processed_images"] >= 0
            assert response_data["processed_images"] <= response_data["total_images"]
            if "started_at" in response_data:
                assert isinstance(response_data["started_at"], str)

    def test_get_inference_job_completed_status(self):
        """Test inference job with completed status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "completed":
            assert response_data["processed_images"] == response_data["total_images"]
            if "completed_at" in response_data:
                assert isinstance(response_data["completed_at"], str)
            if "results" in response_data:
                assert isinstance(response_data["results"], list)

    def test_get_inference_job_failed_status(self):
        """Test inference job with failed status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "failed":
            if "error_message" in response_data:
                assert isinstance(response_data["error_message"], str)
                assert len(response_data["error_message"]) > 0
            if "failed_at" in response_data:
                assert isinstance(response_data["failed_at"], str)

    def test_get_inference_job_progress_tracking(self):
        """Test inference job progress tracking fields."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Progress should be consistent
        assert response_data["processed_images"] >= 0
        assert response_data["processed_images"] <= response_data["total_images"]
        assert response_data["total_images"] == len(response_data["image_ids"])
        
        # Optional progress fields
        if "progress_percentage" in response_data:
            progress = response_data["progress_percentage"]
            assert isinstance(progress, (int, float))
            assert 0 <= progress <= 100

    def test_get_inference_job_timing_information(self):
        """Test inference job timing information."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Check timing fields if present
        if "estimated_completion_time" in response_data:
            assert isinstance(response_data["estimated_completion_time"], str)
        
        if "elapsed_time_ms" in response_data:
            assert isinstance(response_data["elapsed_time_ms"], (int, float))
            assert response_data["elapsed_time_ms"] >= 0
        
        if "remaining_time_ms" in response_data:
            assert isinstance(response_data["remaining_time_ms"], (int, float))
            assert response_data["remaining_time_ms"] >= 0

    def test_get_inference_job_parameters(self):
        """Test inference job parameters preservation."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Optional inference parameters
        if "confidence_threshold" in response_data:
            assert isinstance(response_data["confidence_threshold"], (int, float))
            assert 0 <= response_data["confidence_threshold"] <= 1
        
        if "nms_threshold" in response_data:
            assert isinstance(response_data["nms_threshold"], (int, float))
            assert 0 <= response_data["nms_threshold"] <= 1

    def test_get_inference_job_results_structure(self):
        """Test inference job results structure when available."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "results" in response_data and response_data["results"]:
            results = response_data["results"]
            assert isinstance(results, list)
            
            for result in results:
                assert "image_id" in result
                assert "predictions" in result
                assert isinstance(result["predictions"], list)

    def test_get_inference_job_metadata(self):
        """Test inference job metadata if present."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "metadata" in response_data:
            metadata = response_data["metadata"]
            assert isinstance(metadata, dict)

    def test_get_inference_job_response_headers(self):
        """Test that response headers are correct."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_get_inference_job_performance_metrics(self):
        """Test inference job performance metrics if present."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "performance_metrics" in response_data:
            metrics = response_data["performance_metrics"]
            assert isinstance(metrics, dict)
            
            if "average_inference_time_ms" in metrics:
                assert isinstance(metrics["average_inference_time_ms"], (int, float))
                assert metrics["average_inference_time_ms"] > 0
            
            if "total_inference_time_ms" in metrics:
                assert isinstance(metrics["total_inference_time_ms"], (int, float))
                assert metrics["total_inference_time_ms"] >= 0

    def test_get_inference_job_error_handling(self):
        """Test inference job error information when job failed."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if response_data["status"] == "failed":
            # Should have error information
            if "failed_images" in response_data:
                assert isinstance(response_data["failed_images"], list)
            
            if "error_details" in response_data:
                assert isinstance(response_data["error_details"], dict)

    def test_get_inference_job_partial_results(self):
        """Test inference job with partial results (some images processed)."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/inference/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if response_data["status"] == "running" and "partial_results" in response_data:
            partial_results = response_data["partial_results"]
            assert isinstance(partial_results, list)
            assert len(partial_results) == response_data["processed_images"]
