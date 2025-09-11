"""
Contract test for POST /api/v1/training/jobs (startTraining operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the model training initiation API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestTrainingStartContract:
    """Contract tests for model training initiation endpoint."""

    def test_start_training_basic(self):
        """Test starting training with basic parameters."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["train", "validation"]
            },
            "training_config": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.001
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 202  # Accepted - async job created
        
        response_data = response.json()
        
        # Required fields per TrainingJob schema
        required_fields = [
            "job_id", "model_id", "status", "config", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["job_id"], str)
        assert response_data["model_id"] == model_id
        assert response_data["status"] in ["pending", "running", "completed", "failed"]
        assert isinstance(response_data["config"], dict)
        assert isinstance(response_data["created_at"], str)

    def test_start_training_with_custom_config(self):
        """Test starting training with custom configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["train"],
                "class_labels": ["person", "vehicle", "object"]
            },
            "training_config": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.01,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "augmentation": {
                    "horizontal_flip": True,
                    "rotation": 15,
                    "color_jitter": 0.2
                }
            },
            "validation_split": 0.2
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        assert response_data["model_id"] == model_id
        assert isinstance(response_data["config"], dict)

    def test_start_training_missing_model_id(self):
        """Test starting training without required model_id."""
        request_data = {
            "training_config": {
                "epochs": 100,
                "batch_size": 16
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_invalid_model_id(self):
        """Test starting training with invalid model_id format."""
        request_data = {
            "model_id": "not-a-valid-uuid",
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_nonexistent_model(self):
        """Test starting training with non-existent model."""
        model_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 404 Not Found
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_invalid_epochs(self):
        """Test starting training with invalid epochs value."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": -1  # Invalid - should be positive
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_invalid_batch_size(self):
        """Test starting training with invalid batch_size value."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100,
                "batch_size": 0  # Invalid - should be positive
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_invalid_learning_rate(self):
        """Test starting training with invalid learning_rate value."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100,
                "learning_rate": -0.001  # Invalid - should be positive
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_with_early_stopping(self):
        """Test starting training with early stopping configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 200,
                "batch_size": 16,
                "early_stopping": {
                    "patience": 10,
                    "min_delta": 0.001,
                    "monitor": "val_loss"
                }
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        assert response_data["model_id"] == model_id

    def test_start_training_with_checkpointing(self):
        """Test starting training with checkpointing configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100,
                "checkpointing": {
                    "save_every": 10,
                    "save_best_only": True,
                    "monitor": "val_map"
                }
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        assert response_data["model_id"] == model_id

    def test_start_training_dataset_filter_validation(self):
        """Test dataset filter validation."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["invalid_split"]  # Invalid split
            },
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Should return 400 Bad Request for invalid dataset split
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_start_training_with_validation_config(self):
        """Test starting training with validation configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "dataset_filter": {
                "dataset_splits": ["train", "validation"]
            },
            "training_config": {
                "epochs": 100,
                "validation_frequency": 5,
                "validation_metrics": ["map", "precision", "recall"]
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        assert response_data["model_id"] == model_id

    def test_start_training_with_metadata(self):
        """Test starting training with metadata."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            },
            "metadata": {
                "experiment_name": "test_experiment_001",
                "description": "Testing model training",
                "tags": ["test", "yolo", "detection"]
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        if "metadata" in response_data:
            assert isinstance(response_data["metadata"], dict)

    def test_start_training_priority_setting(self):
        """Test starting training with priority setting."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            },
            "priority": "high"
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        if "priority" in response_data:
            assert response_data["priority"] in ["low", "normal", "high"]

    def test_start_training_response_headers(self):
        """Test that response headers are correct."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_start_training_job_id_format(self):
        """Test that job_id is properly formatted UUID."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        
        # Validate job_id is a valid UUID
        try:
            uuid.UUID(response_data["job_id"])
        except ValueError:
            pytest.fail(f"job_id '{response_data['job_id']}' is not a valid UUID")

    def test_start_training_initial_status(self):
        """Test that initial job status is appropriate."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        
        # Initial status should be pending or running
        assert response_data["status"] in ["pending", "running"]

    def test_start_training_estimated_duration(self):
        """Test training job estimated duration if provided."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "training_config": {
                "epochs": 100
            }
        }

        response = client.post("/api/v1/training/jobs", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        
        if "estimated_duration_ms" in response_data:
            assert isinstance(response_data["estimated_duration_ms"], (int, float))
            assert response_data["estimated_duration_ms"] > 0