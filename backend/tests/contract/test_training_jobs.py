"""
Contract test for GET /api/v1/training/jobs/{job_id} (getTrainingJob operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the training job monitoring API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestTrainingJobsContract:
    """Contract tests for training job monitoring endpoint."""

    def test_get_training_job_valid_id(self):
        """Test retrieving training job with valid job ID."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Required fields per TrainingJob schema
        required_fields = [
            "job_id", "model_id", "status", "config", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["job_id"], str)
        assert isinstance(response_data["model_id"], str)
        assert response_data["status"] in ["pending", "running", "completed", "failed"]
        assert isinstance(response_data["config"], dict)
        assert isinstance(response_data["created_at"], str)
        
        # The returned job ID should match the requested ID
        assert response_data["job_id"] == job_id

    def test_get_training_job_invalid_uuid_format(self):
        """Test retrieving training job with invalid UUID format."""
        response = client.get("/api/v1/training/jobs/not-a-valid-uuid")

        # Should return 400 Bad Request for invalid UUID format
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_get_training_job_nonexistent_id(self):
        """Test retrieving training job with non-existent ID."""
        nonexistent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{nonexistent_id}")

        # Should return 404 Not Found for non-existent job
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_get_training_job_pending_status(self):
        """Test training job with pending status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "pending":
            assert "started_at" not in response_data or response_data["started_at"] is None
            if "progress" in response_data:
                assert response_data["progress"] == 0

    def test_get_training_job_running_status(self):
        """Test training job with running status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "running":
            if "started_at" in response_data:
                assert isinstance(response_data["started_at"], str)
            if "current_epoch" in response_data:
                assert isinstance(response_data["current_epoch"], int)
                assert response_data["current_epoch"] >= 0
            if "progress" in response_data:
                assert isinstance(response_data["progress"], (int, float))
                assert 0 <= response_data["progress"] <= 100

    def test_get_training_job_completed_status(self):
        """Test training job with completed status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "completed":
            if "completed_at" in response_data:
                assert isinstance(response_data["completed_at"], str)
            if "final_metrics" in response_data:
                assert isinstance(response_data["final_metrics"], dict)
            if "model_path" in response_data:
                assert isinstance(response_data["model_path"], str)

    def test_get_training_job_failed_status(self):
        """Test training job with failed status."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["status"] == "failed":
            if "error_message" in response_data:
                assert isinstance(response_data["error_message"], str)
                assert len(response_data["error_message"]) > 0
            if "failed_at" in response_data:
                assert isinstance(response_data["failed_at"], str)

    def test_get_training_job_progress_tracking(self):
        """Test training job progress tracking fields."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Check progress fields
        if "current_epoch" in response_data:
            assert isinstance(response_data["current_epoch"], int)
            assert response_data["current_epoch"] >= 0
        
        if "total_epochs" in response_data:
            assert isinstance(response_data["total_epochs"], int)
            assert response_data["total_epochs"] > 0
        
        if "progress" in response_data:
            progress = response_data["progress"]
            assert isinstance(progress, (int, float))
            assert 0 <= progress <= 100

    def test_get_training_job_metrics_history(self):
        """Test training job metrics history."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "metrics_history" in response_data:
            metrics_history = response_data["metrics_history"]
            assert isinstance(metrics_history, list)
            
            for metric_entry in metrics_history:
                assert isinstance(metric_entry, dict)
                if "epoch" in metric_entry:
                    assert isinstance(metric_entry["epoch"], int)
                if "loss" in metric_entry:
                    assert isinstance(metric_entry["loss"], (int, float))

    def test_get_training_job_config_preservation(self):
        """Test training job configuration preservation."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        config = response_data["config"]
        assert isinstance(config, dict)
        
        # Common config fields
        if "epochs" in config:
            assert isinstance(config["epochs"], int)
            assert config["epochs"] > 0
        
        if "batch_size" in config:
            assert isinstance(config["batch_size"], int)
            assert config["batch_size"] > 0
        
        if "learning_rate" in config:
            assert isinstance(config["learning_rate"], (int, float))
            assert config["learning_rate"] > 0

    def test_get_training_job_logs(self):
        """Test training job logs if available."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "logs" in response_data:
            logs = response_data["logs"]
            assert isinstance(logs, list)
            
            for log_entry in logs:
                assert isinstance(log_entry, dict)
                if "timestamp" in log_entry:
                    assert isinstance(log_entry["timestamp"], str)
                if "message" in log_entry:
                    assert isinstance(log_entry["message"], str)
                if "level" in log_entry:
                    assert log_entry["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_get_training_job_checkpoints(self):
        """Test training job checkpoints information."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "checkpoints" in response_data:
            checkpoints = response_data["checkpoints"]
            assert isinstance(checkpoints, list)
            
            for checkpoint in checkpoints:
                assert isinstance(checkpoint, dict)
                if "epoch" in checkpoint:
                    assert isinstance(checkpoint["epoch"], int)
                if "path" in checkpoint:
                    assert isinstance(checkpoint["path"], str)
                if "metrics" in checkpoint:
                    assert isinstance(checkpoint["metrics"], dict)

    def test_get_training_job_resource_usage(self):
        """Test training job resource usage information."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        if "resource_usage" in response_data:
            resource_usage = response_data["resource_usage"]
            assert isinstance(resource_usage, dict)
            
            if "gpu_utilization" in resource_usage:
                assert isinstance(resource_usage["gpu_utilization"], (int, float))
                assert 0 <= resource_usage["gpu_utilization"] <= 100
            
            if "memory_usage_mb" in resource_usage:
                assert isinstance(resource_usage["memory_usage_mb"], (int, float))
                assert resource_usage["memory_usage_mb"] >= 0

    def test_get_training_job_timing_information(self):
        """Test training job timing information."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Check timing fields
        if "estimated_completion_time" in response_data:
            assert isinstance(response_data["estimated_completion_time"], str)
        
        if "elapsed_time_ms" in response_data:
            assert isinstance(response_data["elapsed_time_ms"], (int, float))
            assert response_data["elapsed_time_ms"] >= 0
        
        if "remaining_time_ms" in response_data:
            assert isinstance(response_data["remaining_time_ms"], (int, float))
            assert response_data["remaining_time_ms"] >= 0

    def test_get_training_job_response_headers(self):
        """Test that response headers are correct."""
        job_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/training/jobs/{job_id}")

        # Check content type
        assert response.headers["content-type"] == "application/json"
