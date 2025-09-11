"""
Integration test for complete model deployment workflow

This test MUST FAIL initially since the endpoints are not implemented.
Tests the complete model deployment workflow from quickstart.md Step 7.

Workflow:
1. Select trained/pre-trained model for deployment
2. Configure deployment settings (replicas, resources)
3. Deploy model as REST API endpoint
4. Test deployed endpoint with sample images
5. Monitor deployment performance and health
6. Scale and update deployments
7. Handle deployment failures and rollback
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import uuid
import time
import requests
from typing import List, Dict
import json

from src.main import app

client = TestClient(app)


class TestModelDeploymentWorkflowIntegration:
    """Integration tests for complete model deployment workflow."""

    @pytest.fixture(scope="class")
    def setup_deployment_data(self):
        """Setup models and test data for deployment testing."""
        deployment_data = {
            "deployable_models": [],
            "test_images": [],
            "deployment_ids": []
        }
        
        # Get available models for deployment
        response = client.get("/api/v1/models?deployable=true")
        if response.status_code == 200:
            models = response.json()["models"]
            deployment_data["deployable_models"] = models
        
        # Create test images for deployment testing
        test_image_configs = [
            ("deploy_test_1.jpg", (640, 480), 'red'),
            ("deploy_test_2.jpg", (1024, 768), 'green'),
            ("deploy_test_3.jpg", (800, 600), 'blue')
        ]
        
        for filename, size, color in test_image_configs:
            image = Image.new('RGB', size, color=color)
            image_buffer = BytesIO()
            image.save(image_buffer, format='JPEG', quality=90)
            image_buffer.seek(0)
            
            response = client.post(
                "/api/v1/images",
                files={"files": (filename, image_buffer, "image/jpeg")},
                data={"dataset_split": "test"}
            )
            
            if response.status_code == 201:
                image_data = response.json()["uploaded_images"][0]
                deployment_data["test_images"].append({
                    "id": image_data["id"],
                    "filename": filename,
                    "buffer": BytesIO(image_buffer.getvalue())
                })
        
        return deployment_data

    def test_complete_model_deployment_workflow(self, setup_deployment_data):
        """Test the complete model deployment workflow from quickstart Step 7."""
        
        if not setup_deployment_data["deployable_models"]:
            pytest.skip("No deployable models available")
        
        # Step 1: Select trained model for deployment
        model_to_deploy = setup_deployment_data["deployable_models"][0]
        
        # Step 2: Configure deployment settings
        deployment_config = {
            "model_id": model_to_deploy["id"],
            "version": "1.0.0",
            "name": f"test_deployment_{int(time.time())}",
            "description": "Integration test model deployment",
            "configuration": {
                "replicas": 1,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "gpu_required": False,
                "auto_scaling": {
                    "enabled": True,
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "target_cpu_percentage": 70
                }
            },
            "environment": {
                "confidence_threshold": 0.5,
                "max_detections": 100,
                "batch_size": 1
            },
            "health_check": {
                "enabled": True,
                "path": "/health",
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "failure_threshold": 3
            }
        }
        
        # Step 3: Deploy model as API endpoint
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        # This MUST FAIL since endpoint doesn't exist yet
        assert response.status_code == 201
        
        deployment_response = response.json()
        
        # Validate deployment creation
        assert "deployment_id" in deployment_response
        assert "status" in deployment_response
        assert deployment_response["status"] == "deploying"
        assert "endpoint_url" in deployment_response
        assert "estimated_ready_time" in deployment_response
        
        deployment_id = deployment_response["deployment_id"]
        endpoint_url = deployment_response["endpoint_url"]
        
        # Store for cleanup
        setup_deployment_data["deployment_ids"].append(deployment_id)
        
        # Wait for deployment to become ready
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        deployment_ready = False
        
        while time.time() - start_time < max_wait_time:
            response = client.get(f"/api/v1/deployments/{deployment_id}")
            assert response.status_code == 200
            
            deployment_status = response.json()
            current_status = deployment_status["status"]
            
            # Validate deployment status structure
            assert "deployment_id" in deployment_status
            assert "model_id" in deployment_status
            assert "status" in deployment_status
            assert "endpoint_url" in deployment_status
            assert "replicas" in deployment_status
            assert "health" in deployment_status
            
            if current_status == "running":
                deployment_ready = True
                
                # Validate running deployment details
                assert deployment_status["replicas"]["available"] > 0
                assert deployment_status["health"]["status"] == "healthy"
                break
                
            elif current_status == "failed":
                pytest.fail(f"Deployment failed: {deployment_status.get('error_message')}")
            
            time.sleep(10)  # Check every 10 seconds
        
        assert deployment_ready, "Deployment did not become ready within timeout"
        
        return deployment_id, endpoint_url

    def test_deployed_endpoint_functionality(self, setup_deployment_data):
        """Test the functionality of deployed model endpoints."""
        
        # First deploy a model
        deployment_result = self.test_complete_model_deployment_workflow(setup_deployment_data)
        if not deployment_result:
            pytest.skip("No deployment available for testing")
        
        deployment_id, endpoint_url = deployment_result
        
        if not setup_deployment_data["test_images"]:
            pytest.skip("No test images available")
        
        # Step 4: Test deployed endpoint with sample images
        test_image = setup_deployment_data["test_images"][0]
        
        # Test single image prediction
        test_image["buffer"].seek(0)
        files = {"file": (test_image["filename"], test_image["buffer"], "image/jpeg")}
        
        # Use requests to call the actual deployed endpoint
        prediction_response = requests.post(f"{endpoint_url}/predict", files=files, timeout=30)
        
        assert prediction_response.status_code == 200, f"Endpoint prediction failed: {prediction_response.text}"
        
        prediction_result = prediction_response.json()
        
        # Validate prediction response structure
        assert "predictions" in prediction_result
        assert "model_id" in prediction_result
        assert "version" in prediction_result
        assert "inference_time_ms" in prediction_result
        assert "image_metadata" in prediction_result
        
        predictions = prediction_result["predictions"]
        for prediction in predictions:
            assert "class_id" in prediction
            assert "confidence" in prediction
            assert "bounding_box" in prediction or "segment" in prediction
            assert 0.0 <= prediction["confidence"] <= 1.0

    def test_deployment_health_monitoring(self, setup_deployment_data):
        """Test deployment health monitoring and metrics."""
        
        if not setup_deployment_data["deployable_models"]:
            pytest.skip("No deployable models available")
        
        # Create deployment with health monitoring
        model = setup_deployment_data["deployable_models"][0]
        
        deployment_config = {
            "model_id": model["id"],
            "version": "1.0.1",
            "name": f"health_test_deployment_{int(time.time())}",
            "configuration": {
                "replicas": 2,  # Multiple replicas for testing
                "cpu_limit": "500m",
                "memory_limit": "1Gi"
            },
            "health_check": {
                "enabled": True,
                "path": "/health",
                "interval_seconds": 15,
                "timeout_seconds": 5,
                "failure_threshold": 2
            },
            "monitoring": {
                "enabled": True,
                "metrics_collection": True,
                "logging_level": "INFO"
            }
        }
        
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for deployment to be ready
            time.sleep(30)
            
            # Step 5: Monitor deployment performance and health
            response = client.get(f"/api/v1/deployments/{deployment_id}")
            if response.status_code == 200:
                deployment_status = response.json()
                
                # Validate health monitoring data
                health = deployment_status["health"]
                assert "status" in health
                assert "last_check" in health
                assert "failure_count" in health
                
                # Check metrics if available
                if "metrics" in deployment_status:
                    metrics = deployment_status["metrics"]
                    assert "request_count" in metrics
                    assert "average_response_time_ms" in metrics
                    assert "error_rate" in metrics
                    assert "cpu_usage_percentage" in metrics
                    assert "memory_usage_mb" in metrics

    def test_deployment_scaling(self, setup_deployment_data):
        """Test deployment scaling operations."""
        
        if not setup_deployment_data["deployable_models"]:
            pytest.skip("No deployable models available")
        
        model = setup_deployment_data["deployable_models"][0]
        
        # Create deployment with auto-scaling
        deployment_config = {
            "model_id": model["id"],
            "version": "1.0.2",
            "name": f"scaling_test_{int(time.time())}",
            "configuration": {
                "replicas": 1,
                "cpu_limit": "500m",
                "memory_limit": "1Gi",
                "auto_scaling": {
                    "enabled": True,
                    "min_replicas": 1,
                    "max_replicas": 4,
                    "target_cpu_percentage": 60
                }
            }
        }
        
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for initial deployment
            time.sleep(20)
            
            # Manual scaling test
            scale_request = {
                "replicas": 3,
                "reason": "Load testing"
            }
            
            response = client.patch(f"/api/v1/deployments/{deployment_id}/scale", json=scale_request)
            
            if response.status_code == 200:
                scale_response = response.json()
                assert "status" in scale_response
                assert scale_response["status"] == "scaling"
                
                # Wait for scaling to complete
                max_wait = 120  # 2 minutes
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    response = client.get(f"/api/v1/deployments/{deployment_id}")
                    if response.status_code == 200:
                        status = response.json()
                        replicas = status["replicas"]
                        
                        if replicas["desired"] == 3 and replicas["available"] == 3:
                            break
                    
                    time.sleep(5)
                
                # Verify scaling completed
                final_response = client.get(f"/api/v1/deployments/{deployment_id}")
                final_status = final_response.json()
                assert final_status["replicas"]["available"] >= 2  # Allow for some variance

    def test_deployment_update_and_rollback(self, setup_deployment_data):
        """Test deployment updates and rollback functionality."""
        
        if len(setup_deployment_data["deployable_models"]) < 2:
            pytest.skip("Need at least 2 models for update testing")
        
        original_model = setup_deployment_data["deployable_models"][0]
        updated_model = setup_deployment_data["deployable_models"][1]
        
        # Create initial deployment
        initial_config = {
            "model_id": original_model["id"],
            "version": "1.0.0",
            "name": f"update_test_{int(time.time())}",
            "configuration": {
                "replicas": 2,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi"
            }
        }
        
        response = client.post("/api/v1/deployments", json=initial_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for initial deployment to be ready
            time.sleep(30)
            
            # Update deployment with new model
            update_config = {
                "model_id": updated_model["id"],
                "version": "1.1.0",
                "update_strategy": "rolling",
                "rollback_on_failure": True
            }
            
            response = client.patch(f"/api/v1/deployments/{deployment_id}", json=update_config)
            
            if response.status_code == 200:
                update_response = response.json()
                assert "status" in update_response
                assert update_response["status"] == "updating"
                
                # Monitor update progress
                max_wait = 180  # 3 minutes
                start_time = time.time()
                update_completed = False
                
                while time.time() - start_time < max_wait:
                    response = client.get(f"/api/v1/deployments/{deployment_id}")
                    if response.status_code == 200:
                        status = response.json()
                        
                        if status["status"] == "running":
                            # Verify model was updated
                            assert status["model_id"] == updated_model["id"]
                            assert status["version"] == "1.1.0"
                            update_completed = True
                            break
                        elif status["status"] == "failed":
                            # Check if rollback occurred
                            if status["model_id"] == original_model["id"]:
                                # Rollback successful
                                update_completed = True
                                break
                    
                    time.sleep(10)
                
                assert update_completed, "Deployment update did not complete"

    def test_deployment_error_handling(self, setup_deployment_data):
        """Test error handling in deployment workflow."""
        
        # Test deployment with invalid model
        response = client.post("/api/v1/deployments", json={
            "model_id": "invalid-model-id",
            "version": "1.0.0",
            "name": "invalid_test",
            "configuration": {"replicas": 1}
        })
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        
        # Test deployment with invalid configuration
        if setup_deployment_data["deployable_models"]:
            model = setup_deployment_data["deployable_models"][0]
            
            response = client.post("/api/v1/deployments", json={
                "model_id": model["id"],
                "version": "1.0.0",
                "name": "invalid_config_test",
                "configuration": {
                    "replicas": -1,  # Invalid
                    "cpu_limit": "invalid",  # Invalid format
                    "memory_limit": "0"  # Invalid
                }
            })
            
            assert response.status_code == 400
        
        # Test accessing non-existent deployment
        response = client.get("/api/v1/deployments/non-existent-deployment-id")
        assert response.status_code == 404

    def test_deployment_performance_under_load(self, setup_deployment_data):
        """Test deployment performance under simulated load."""
        
        if not setup_deployment_data["deployable_models"] or not setup_deployment_data["test_images"]:
            pytest.skip("Insufficient test data for load testing")
        
        # Create high-performance deployment
        model = setup_deployment_data["deployable_models"][0]
        
        deployment_config = {
            "model_id": model["id"],
            "version": "1.0.0",
            "name": f"load_test_{int(time.time())}",
            "configuration": {
                "replicas": 3,  # Multiple replicas for load handling
                "cpu_limit": "1000m",
                "memory_limit": "2Gi",
                "auto_scaling": {
                    "enabled": True,
                    "min_replicas": 2,
                    "max_replicas": 5,
                    "target_cpu_percentage": 70
                }
            },
            "performance": {
                "timeout_seconds": 30,
                "max_concurrent_requests": 10
            }
        }
        
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            endpoint_url = response.json()["endpoint_url"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for deployment to be ready
            time.sleep(45)
            
            # Simulate load testing with multiple concurrent requests
            test_image = setup_deployment_data["test_images"][0]
            
            def make_prediction_request():
                test_image["buffer"].seek(0)
                files = {"file": (test_image["filename"], test_image["buffer"], "image/jpeg")}
                try:
                    response = requests.post(f"{endpoint_url}/predict", files=files, timeout=30)
                    return response.status_code, response.elapsed.total_seconds() * 1000
                except Exception as e:
                    return 500, float('inf')
            
            # Make concurrent requests
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_prediction_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Analyze results
            successful_requests = [r for r in results if r[0] == 200]
            response_times = [r[1] for r in successful_requests]
            
            assert len(successful_requests) >= 8, f"Too many failed requests: {len(successful_requests)}/10"
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                assert avg_response_time < 5000, f"Average response time too high: {avg_response_time}ms"

    def test_deployment_resource_monitoring(self, setup_deployment_data):
        """Test monitoring of deployment resource usage."""
        
        if not setup_deployment_data["deployable_models"]:
            pytest.skip("No deployable models available")
        
        model = setup_deployment_data["deployable_models"][0]
        
        deployment_config = {
            "model_id": model["id"],
            "version": "1.0.0",
            "name": f"resource_monitor_test_{int(time.time())}",
            "configuration": {
                "replicas": 1,
                "cpu_limit": "1000m",
                "memory_limit": "2Gi"
            },
            "monitoring": {
                "enabled": True,
                "resource_tracking": True,
                "metrics_interval_seconds": 30
            }
        }
        
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for deployment and initial metrics collection
            time.sleep(60)
            
            # Get resource metrics
            response = client.get(f"/api/v1/deployments/{deployment_id}/metrics")
            
            if response.status_code == 200:
                metrics_data = response.json()
                
                # Validate resource metrics
                assert "resource_usage" in metrics_data
                assert "performance_metrics" in metrics_data
                assert "time_series" in metrics_data
                
                resource_usage = metrics_data["resource_usage"]
                assert "cpu_usage_cores" in resource_usage
                assert "memory_usage_mb" in resource_usage
                assert "network_io_mb" in resource_usage
                
                # Validate metrics are reasonable
                assert 0 <= resource_usage["cpu_usage_cores"] <= 1.2  # Allow slight overhead
                assert 0 <= resource_usage["memory_usage_mb"] <= 2500  # Allow overhead

    def test_deployment_cleanup(self, setup_deployment_data):
        """Test cleanup of deployment resources."""
        
        # This test runs last to clean up created deployments
        for deployment_id in setup_deployment_data["deployment_ids"]:
            # Delete deployment
            response = client.delete(f"/api/v1/deployments/{deployment_id}")
            
            if response.status_code in [200, 204]:
                # Wait for deletion to complete
                max_wait = 60
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    response = client.get(f"/api/v1/deployments/{deployment_id}")
                    
                    if response.status_code == 404:
                        break  # Successfully deleted
                    elif response.status_code == 200:
                        status = response.json()["status"]
                        if status == "deleted":
                            break
                    
                    time.sleep(5)

    def test_deployment_logging_and_debugging(self, setup_deployment_data):
        """Test deployment logging and debugging capabilities."""
        
        if not setup_deployment_data["deployable_models"]:
            pytest.skip("No deployable models available")
        
        model = setup_deployment_data["deployable_models"][0]
        
        deployment_config = {
            "model_id": model["id"],
            "version": "1.0.0",
            "name": f"logging_test_{int(time.time())}",
            "configuration": {
                "replicas": 1,
                "cpu_limit": "500m",
                "memory_limit": "1Gi"
            },
            "logging": {
                "level": "DEBUG",
                "include_request_response": True,
                "max_log_size_mb": 100
            }
        }
        
        response = client.post("/api/v1/deployments", json=deployment_config)
        
        if response.status_code == 201:
            deployment_id = response.json()["deployment_id"]
            setup_deployment_data["deployment_ids"].append(deployment_id)
            
            # Wait for deployment to be ready
            time.sleep(30)
            
            # Make some requests to generate logs
            if setup_deployment_data["test_images"]:
                endpoint_url = response.json()["endpoint_url"]
                test_image = setup_deployment_data["test_images"][0]
                
                # Make a few prediction requests
                for _ in range(3):
                    test_image["buffer"].seek(0)
                    files = {"file": (test_image["filename"], test_image["buffer"], "image/jpeg")}
                    requests.post(f"{endpoint_url}/predict", files=files, timeout=30)
                    time.sleep(5)
                
                # Get deployment logs
                response = client.get(f"/api/v1/deployments/{deployment_id}/logs?limit=50")
                
                if response.status_code == 200:
                    logs_data = response.json()
                    
                    assert "logs" in logs_data
                    assert len(logs_data["logs"]) > 0
                    
                    # Validate log entry structure
                    for log_entry in logs_data["logs"][:5]:  # Check first 5
                        assert "timestamp" in log_entry
                        assert "level" in log_entry
                        assert "message" in log_entry
                        assert log_entry["level"] in ["DEBUG", "INFO", "WARNING", "ERROR"]