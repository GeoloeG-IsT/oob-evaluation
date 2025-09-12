"""
Integration test for complete model training workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete model training workflow from quickstart.md Step 6.

Workflow:
1. Ensure sufficient annotations exist for training
2. Select base model and configure hyperparameters
3. Start training job with monitoring
4. Monitor real-time progress and logs
5. Handle training completion and model artifacts
6. Validate trained model performance
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
from typing import List, Dict
import json

from src.main import app

client = TestClient(app)


class TestModelTrainingWorkflowIntegration:
    """Integration tests for complete model training workflow."""

    @pytest.fixture(scope="class")
    def setup_training_data(self):
        """Setup comprehensive training dataset with annotations."""
        training_data = {
            "image_ids": [],
            "annotation_ids": [],
            "base_models": [],
            "dataset_id": None
        }
        
        # Create training images with variety
        image_configs = [
            ("train_img_1.jpg", (640, 480), 'red'),
            ("train_img_2.jpg", (800, 600), 'green'),
            ("train_img_3.jpg", (1024, 768), 'blue'),
            ("train_img_4.jpg", (512, 384), 'yellow'),
            ("train_img_5.jpg", (960, 720), 'purple'),
            ("train_img_6.jpg", (720, 540), 'orange'),
            ("train_img_7.jpg", (1280, 960), 'pink'),
            ("train_img_8.jpg", (600, 400), 'cyan'),
            ("train_img_9.jpg", (1100, 800), 'magenta'),
            ("train_img_10.jpg", (850, 650), 'lime'),
            ("train_img_11.jpg", (750, 550), 'brown'),
            ("train_img_12.jpg", (900, 700), 'navy')
        ]
        
        # Upload training images
        for filename, size, color in image_configs:
            image = Image.new('RGB', size, color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG', quality=90)
            image_buffer.seek(0)
            
            response = client.post(
                "/api/v1/images",
                files={"files": (filename, image_buffer, "image/jpeg")},
                data={"dataset_split": "train"}
            )
            
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                training_data["image_ids"].append(image_data["id"])
        
        # Create annotations for training (need minimum 10+ annotated images)
        for i, image_id in enumerate(training_data["image_ids"]):
            annotation_data = self._create_training_annotation(image_id, i)
            
            response = client.post("/api/v1/annotations", json=annotation_data)
            if response.status_code == 201:
                training_data["annotation_ids"].append(response.json()["id"])
        
        # Get available base models for fine-tuning
        response = client.get("/api/v1/models?type=detection&pretrained=true")
        if response.status_code == 200:
            models = response.json()["models"]
            training_data["base_models"] = [m for m in models if "yolo" in m["name"].lower()]
        
        # Create training dataset
        if training_data["image_ids"]:
            dataset_request = {
                "name": "model_training_dataset",
                "description": "Dataset for model training integration test",
                "image_ids": training_data["image_ids"],
                "split": "train"
            }
            
            response = client.post("/api/v1/datasets", json=dataset_request)
            if response.status_code == 201:
                training_data["dataset_id"] = response.json()["id"]
        
        return training_data

    def _create_training_annotation(self, image_id: str, index: int) -> Dict:
        """Create diverse training annotations."""
        class_labels = ["person", "car", "bicycle", "dog", "cat"]
        
        # Vary annotation complexity based on index
        if index % 3 == 0:  # Simple case
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 100.0 + (index * 20) % 200,
                        "y": 100.0 + (index * 15) % 150,
                        "width": 150.0,
                        "height": 120.0,
                        "class_id": index % len(class_labels),
                        "confidence": 1.0
                    }
                ],
                "class_labels": [class_labels[index % len(class_labels)]],
                "user_tag": "training_annotator"
            }
        elif index % 3 == 1:  # Medium complexity
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 50.0 + (index * 30) % 250,
                        "y": 80.0 + (index * 25) % 200,
                        "width": 120.0,
                        "height": 100.0,
                        "class_id": index % len(class_labels),
                        "confidence": 1.0
                    },
                    {
                        "x": 300.0 + (index * 20) % 150,
                        "y": 250.0 + (index * 10) % 100,
                        "width": 80.0,
                        "height": 90.0,
                        "class_id": (index + 1) % len(class_labels),
                        "confidence": 1.0
                    }
                ],
                "class_labels": [
                    class_labels[index % len(class_labels)],
                    class_labels[(index + 1) % len(class_labels)]
                ],
                "user_tag": "training_annotator"
            }
        else:  # Complex case
            return {
                "image_id": image_id,
                "bounding_boxes": [
                    {
                        "x": 30.0 + (index * 25) % 200,
                        "y": 40.0 + (index * 20) % 180,
                        "width": 100.0,
                        "height": 80.0,
                        "class_id": index % len(class_labels),
                        "confidence": 1.0
                    },
                    {
                        "x": 200.0 + (index * 35) % 180,
                        "y": 150.0 + (index * 15) % 120,
                        "width": 90.0,
                        "height": 110.0,
                        "class_id": (index + 1) % len(class_labels),
                        "confidence": 1.0
                    },
                    {
                        "x": 400.0 + (index * 15) % 100,
                        "y": 300.0 + (index * 10) % 80,
                        "width": 70.0,
                        "height": 60.0,
                        "class_id": (index + 2) % len(class_labels),
                        "confidence": 1.0
                    }
                ],
                "class_labels": [
                    class_labels[index % len(class_labels)],
                    class_labels[(index + 1) % len(class_labels)],
                    class_labels[(index + 2) % len(class_labels)]
                ],
                "user_tag": "training_annotator"
            }

    def test_complete_model_training_workflow(self, setup_training_data):
        """Test the complete model training workflow from quickstart Step 6."""
        
        if len(setup_training_data["image_ids"]) < 10:
            pytest.skip("Insufficient training data - need at least 10 annotated images")
        
        if not setup_training_data["base_models"]:
            pytest.skip("No base models available for fine-tuning")
        
        # Step 1: Verify sufficient annotations exist
        annotation_count = len(setup_training_data["annotation_ids"])
        assert annotation_count >= 10, f"Need at least 10 annotations, got {annotation_count}"
        
        # Step 2: Select base model and configure hyperparameters
        base_model = setup_training_data["base_models"][0]  # Use first available YOLO model
        dataset_id = setup_training_data["dataset_id"]
        
        training_config = {
            "base_model_id": base_model["id"],
            "dataset_id": dataset_id,
            "hyperparameters": {
                "epochs": 5,  # Short training for testing
                "batch_size": 4,  # Small batch for testing
                "learning_rate": 0.001,
                "patience": 3,
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "optimizer": "SGD",
                "scheduler": "cosine",
                "augmentation": True
            },
            "training_name": "integration_test_training",
            "description": "Integration test model training",
            "validation_split": 0.2,  # Use 20% for validation
            "save_checkpoints": True,
            "log_frequency": 1  # Log every epoch for testing
        }
        
        # Step 3: Start training job
        response = client.post("/api/v1/training/jobs", json=training_config)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 201
        
        training_response = response.json()
        
        # Validate training job creation
        assert "job_id" in training_response
        assert "status" in training_response
        assert training_response["status"] == "queued"
        assert "estimated_duration_minutes" in training_response
        assert "training_config" in training_response
        
        job_id = training_response["job_id"]
        
        # Verify training config is properly stored
        stored_config = training_response["training_config"]
        assert stored_config["base_model_id"] == base_model["id"]
        assert stored_config["dataset_id"] == dataset_id
        assert stored_config["hyperparameters"]["epochs"] == 5

    def test_real_time_training_monitoring(self, setup_training_data):
        """Test real-time monitoring of training progress and logs."""
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        # Start a training job first
        job_id = self._start_test_training_job(setup_training_data)
        if not job_id:
            pytest.skip("Could not start training job")
        
        # Step 4: Monitor training progress in real-time
        max_monitor_time = 300  # 5 minutes max monitoring
        start_time = time.time()
        
        previous_epoch = -1
        status_history = []
        
        while time.time() - start_time < max_monitor_time:
            response = client.get(f"/api/v1/training/jobs/{job_id}")
            assert response.status_code == 200
            
            job_status = response.json()
            status_history.append(job_status["status"])
            
            # Validate progress structure
            assert job_status["id"] == job_id
            assert "status" in job_status
            assert "progress" in job_status
            assert "current_epoch" in job_status
            assert "total_epochs" in job_status
            assert "training_logs" in job_status
            assert "metrics" in job_status
            
            current_status = job_status["status"]
            current_epoch = job_status["current_epoch"]
            
            # Validate progress metrics
            progress = job_status["progress"]
            assert "percentage" in progress
            assert "estimated_remaining_minutes" in progress
            assert 0 <= progress["percentage"] <= 100
            
            # Check training logs
            logs = job_status["training_logs"]
            if logs:
                latest_log = logs[-1]
                assert "epoch" in latest_log
                assert "timestamp" in latest_log
                assert "message" in latest_log or "metrics" in latest_log
            
            # Monitor epoch progression
            if current_epoch > previous_epoch:
                previous_epoch = current_epoch
                
                # Validate epoch metrics
                metrics = job_status["metrics"]
                if metrics:
                    assert "train_loss" in metrics
                    assert "learning_rate" in metrics
                    
                    # If validation is configured, check validation metrics
                    if "val_loss" in metrics:
                        assert "val_mAP" in metrics or "val_accuracy" in metrics
            
            # Check if training completed or failed
            if current_status in ["completed", "failed"]:
                break
            elif current_status == "running":
                # Verify training is actually progressing
                assert current_epoch >= 0, "Training should have started"
            
            time.sleep(5)  # Check every 5 seconds
        
        # Verify status progression
        assert "queued" in status_history or "running" in status_history
        final_status = status_history[-1]
        
        if final_status == "completed":
            # Validate completion
            final_response = client.get(f"/api/v1/training/jobs/{job_id}")
            final_job = final_response.json()
            
            assert "trained_model_id" in final_job
            assert "final_metrics" in final_job
            assert "training_duration_minutes" in final_job
            
            # Check final metrics
            final_metrics = final_job["final_metrics"]
            assert "final_train_loss" in final_metrics
            assert "best_epoch" in final_metrics
            assert "model_size_mb" in final_metrics

    def _has_sufficient_training_data(self, training_data) -> bool:
        """Check if there's sufficient data for training."""
        return (len(training_data["image_ids"]) >= 10 and 
                len(training_data["annotation_ids"]) >= 10 and 
                len(training_data["base_models"]) > 0)

    def _start_test_training_job(self, training_data) -> str:
        """Helper to start a test training job."""
        if not self._has_sufficient_training_data(training_data):
            return None
        
        config = {
            "base_model_id": training_data["base_models"][0]["id"],
            "dataset_id": training_data["dataset_id"],
            "hyperparameters": {
                "epochs": 3,  # Very short for testing
                "batch_size": 2,
                "learning_rate": 0.001
            },
            "training_name": "test_monitoring"
        }
        
        response = client.post("/api/v1/training/jobs", json=config)
        if response.status_code == 201:
            return response.json()["job_id"]
        return None

    def test_hyperparameter_configurations(self, setup_training_data):
        """Test different hyperparameter configurations."""
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        base_model = setup_training_data["base_models"][0]
        dataset_id = setup_training_data["dataset_id"]
        
        # Test different hyperparameter sets
        hyperparameter_configs = [
            {
                "name": "fast_training",
                "config": {
                    "epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 0.01,
                    "optimizer": "Adam"
                }
            },
            {
                "name": "precise_training",
                "config": {
                    "epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "optimizer": "SGD",
                    "momentum": 0.9
                }
            },
            {
                "name": "regularized_training",
                "config": {
                    "epochs": 2,
                    "batch_size": 6,
                    "learning_rate": 0.005,
                    "weight_decay": 0.001,
                    "dropout": 0.1
                }
            }
        ]
        
        for hp_config in hyperparameter_configs:
            training_request = {
                "base_model_id": base_model["id"],
                "dataset_id": dataset_id,
                "hyperparameters": hp_config["config"],
                "training_name": f"test_{hp_config['name']}"
            }
            
            response = client.post("/api/v1/training/jobs", json=training_request)
            
            if response.status_code == 201:
                job_data = response.json()
                assert job_data["status"] == "queued"
                
                # Verify hyperparameters are stored correctly
                stored_hp = job_data["training_config"]["hyperparameters"]
                for key, value in hp_config["config"].items():
                    assert stored_hp[key] == value

    def test_training_validation_and_checkpoints(self, setup_training_data):
        """Test training with validation split and checkpoint saving."""
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        training_config = {
            "base_model_id": setup_training_data["base_models"][0]["id"],
            "dataset_id": setup_training_data["dataset_id"],
            "hyperparameters": {
                "epochs": 4,
                "batch_size": 4,
                "learning_rate": 0.001
            },
            "validation_split": 0.3,  # Use 30% for validation
            "save_checkpoints": True,
            "checkpoint_frequency": 2,  # Save every 2 epochs
            "early_stopping": True,
            "patience": 2,
            "training_name": "checkpoint_test"
        }
        
        response = client.post("/api/v1/training/jobs", json=training_config)
        
        if response.status_code == 201:
            job_id = response.json()["job_id"]
            
            # Monitor for validation metrics and checkpoints
            max_wait = 180  # 3 minutes
            start_time = time.time()
            
            checkpoints_found = []
            validation_metrics_found = False
            
            while time.time() - start_time < max_wait:
                response = client.get(f"/api/v1/training/jobs/{job_id}")
                if response.status_code == 200:
                    job_status = response.json()
                    
                    # Check for validation metrics
                    metrics = job_status.get("metrics", {})
                    if "val_loss" in metrics:
                        validation_metrics_found = True
                        assert "val_mAP" in metrics or "val_accuracy" in metrics
                    
                    # Check for checkpoints
                    if "checkpoints" in job_status:
                        checkpoints = job_status["checkpoints"]
                        for checkpoint in checkpoints:
                            if checkpoint not in checkpoints_found:
                                checkpoints_found.append(checkpoint)
                                assert "epoch" in checkpoint
                                assert "model_path" in checkpoint
                                assert "metrics" in checkpoint
                    
                    if job_status["status"] in ["completed", "failed"]:
                        break
                
                time.sleep(5)
            
            # Verify validation and checkpoints were used
            if job_status["status"] == "completed":
                assert validation_metrics_found, "Validation metrics should be present"
                assert len(checkpoints_found) > 0, "Checkpoints should be saved"

    def test_training_error_handling(self, setup_training_data):
        """Test error handling in training workflow."""
        
        # Test invalid base model
        response = client.post("/api/v1/training/jobs", json={
            "base_model_id": "invalid-model-id",
            "dataset_id": "test-dataset",
            "hyperparameters": {"epochs": 1, "batch_size": 1}
        })
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        
        # Test invalid dataset
        if setup_training_data["base_models"]:
            response = client.post("/api/v1/training/jobs", json={
                "base_model_id": setup_training_data["base_models"][0]["id"],
                "dataset_id": "invalid-dataset-id",
                "hyperparameters": {"epochs": 1, "batch_size": 1}
            })
            
            assert response.status_code == 400
        
        # Test invalid hyperparameters
        if setup_training_data["base_models"] and setup_training_data["dataset_id"]:
            response = client.post("/api/v1/training/jobs", json={
                "base_model_id": setup_training_data["base_models"][0]["id"],
                "dataset_id": setup_training_data["dataset_id"],
                "hyperparameters": {
                    "epochs": -1,  # Invalid
                    "batch_size": 0,  # Invalid
                    "learning_rate": 2.0  # Too high
                }
            })
            
            assert response.status_code == 400

    def test_training_with_insufficient_data(self):
        """Test training behavior with insufficient training data."""
        
        # Create minimal dataset with too few annotations
        minimal_image = Image.new('RGB', (640, 480), color='red')
        buffer = BytesIO()
        minimal_image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        response = client.post(
            "/api/v1/images",
            files={"files": ("minimal.jpg", buffer, "image/jpeg")},
            data={"dataset_split": "train"}
        )
        
        if response.status_code == 201:
            image_id = response.json()["uploaded_images"][0]["id"]
            
            # Create single annotation (insufficient for training)
            annotation = {
                "image_id": image_id,
                "bounding_boxes": [
                    {"x": 100, "y": 100, "width": 100, "height": 100, "class_id": 0, "confidence": 1.0}
                ],
                "class_labels": ["test"],
                "user_tag": "minimal_test"
            }
            
            client.post("/api/v1/annotations", json=annotation)
            
            # Try to start training with insufficient data
            response = client.get("/api/v1/models?type=detection&pretrained=true")
            if response.status_code == 200:
                models = response.json()["models"]
                if models:
                    training_request = {
                        "base_model_id": models[0]["id"],
                        "image_ids": [image_id],
                        "hyperparameters": {"epochs": 1, "batch_size": 1}
                    }
                    
                    response = client.post("/api/v1/training/jobs", json=training_request)
                    
                    # Should reject due to insufficient data
                    assert response.status_code == 400
                    error_data = response.json()
                    assert "insufficient" in error_data["error"].lower()

    def test_training_cancellation(self, setup_training_data):
        """Test cancelling a running training job."""
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        # Start a training job
        job_id = self._start_test_training_job(setup_training_data)
        if not job_id:
            pytest.skip("Could not start training job")
        
        # Wait a moment for it to potentially start
        time.sleep(5)
        
        # Cancel the training job
        response = client.delete(f"/api/v1/training/jobs/{job_id}")
        
        # Should work once cancellation is implemented
        if response.status_code == 200:
            cancel_response = response.json()
            assert "status" in cancel_response
            assert cancel_response["status"] in ["cancelled", "cancelling"]
            
            # Verify cancellation took effect
            time.sleep(2)
            response = client.get(f"/api/v1/training/jobs/{job_id}")
            if response.status_code == 200:
                final_status = response.json()["status"]
                assert final_status in ["cancelled", "failed"], "Job should be cancelled"

    def test_trained_model_validation(self, setup_training_data):
        """Test validation of successfully trained models."""
        
        # This test would need a completed training job
        # For integration testing, we could mock or use a pre-trained model
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        # Start training (in real scenario, wait for completion)
        job_id = self._start_test_training_job(setup_training_data)
        if not job_id:
            pytest.skip("Could not start training job")
        
        # In a real test, we'd wait for completion, but for now test the expected structure
        # when we get a completed training job
        
        # Simulate checking a completed job
        response = client.get(f"/api/v1/training/jobs/{job_id}")
        if response.status_code == 200:
            job_data = response.json()
            
            # If job is completed, validate trained model
            if job_data.get("status") == "completed":
                assert "trained_model_id" in job_data
                trained_model_id = job_data["trained_model_id"]
                
                # Get the trained model details
                response = client.get(f"/api/v1/models/{trained_model_id}")
                if response.status_code == 200:
                    model_data = response.json()
                    
                    # Validate trained model properties
                    assert model_data["type"] == "detection"
                    assert "training_job_id" in model_data
                    assert model_data["training_job_id"] == job_id
                    assert "model_path" in model_data
                    assert "training_metrics" in model_data
                    
                    # Test inference with trained model
                    if setup_training_data["image_ids"]:
                        test_image_id = setup_training_data["image_ids"][0]
                        
                        inference_request = {
                            "image_id": test_image_id,
                            "model_id": trained_model_id,
                            "confidence_threshold": 0.5
                        }
                        
                        response = client.post("/api/v1/inference/single", json=inference_request)
                        if response.status_code == 200:
                            result = response.json()
                            assert result["model_id"] == trained_model_id
                            assert "predictions" in result

    def test_training_resource_management(self, setup_training_data):
        """Test resource management during training."""
        
        if not self._has_sufficient_training_data(setup_training_data):
            pytest.skip("Insufficient training data")
        
        # Start training with resource constraints
        training_config = {
            "base_model_id": setup_training_data["base_models"][0]["id"],
            "dataset_id": setup_training_data["dataset_id"],
            "hyperparameters": {
                "epochs": 2,
                "batch_size": 2
            },
            "resource_limits": {
                "max_memory_mb": 2048,
                "max_gpu_memory_mb": 1024,
                "cpu_cores": 2
            },
            "training_name": "resource_test"
        }
        
        response = client.post("/api/v1/training/jobs", json=training_config)
        
        if response.status_code == 201:
            job_id = response.json()["job_id"]
            
            # Monitor resource usage
            max_monitor_time = 60  # 1 minute
            start_time = time.time()
            
            max_memory_seen = 0
            
            while time.time() - start_time < max_monitor_time:
                response = client.get(f"/api/v1/training/jobs/{job_id}")
                if response.status_code == 200:
                    job_status = response.json()
                    
                    if "resource_usage" in job_status:
                        usage = job_status["resource_usage"]
                        
                        if "memory_mb" in usage:
                            max_memory_seen = max(max_memory_seen, usage["memory_mb"])
                            
                            # Memory shouldn't exceed limits (with some tolerance)
                            assert usage["memory_mb"] < 2500, "Memory usage exceeds limit"
                    
                    if job_status["status"] in ["completed", "failed"]:
                        break
                
                time.sleep(5)