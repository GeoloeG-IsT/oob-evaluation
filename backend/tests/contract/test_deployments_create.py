"""
Contract test for POST /api/v1/deployments (deployModel operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the model deployment API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestDeploymentsCreateContract:
    """Contract tests for model deployment endpoint."""

    def test_deploy_model_basic(self):
        """Test deploying model with basic parameters."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "test_deployment_001",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 202  # Accepted - async deployment
        
        response_data = response.json()
        
        # Required fields per Deployment schema
        required_fields = [
            "deployment_id", "model_id", "deployment_name", "status", 
            "environment", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        # Type validations
        assert isinstance(response_data["deployment_id"], str)
        assert response_data["model_id"] == model_id
        assert response_data["deployment_name"] == "test_deployment_001"
        assert response_data["status"] in ["pending", "deploying", "active", "failed"]
        assert response_data["environment"] == "staging"
        assert isinstance(response_data["created_at"], str)

    def test_deploy_model_production(self):
        """Test deploying model to production environment."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "production_model_v1",
            "environment": "production",
            "scaling_config": {
                "min_replicas": 2,
                "max_replicas": 10,
                "cpu_limit": "2",
                "memory_limit": "4Gi"
            }
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        assert response_data["environment"] == "production"
        if "scaling_config" in response_data:
            assert isinstance(response_data["scaling_config"], dict)

    def test_deploy_model_missing_model_id(self):
        """Test deploying model without required model_id."""
        request_data = {
            "deployment_name": "test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_missing_deployment_name(self):
        """Test deploying model without required deployment_name."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_invalid_model_id(self):
        """Test deploying model with invalid model_id format."""
        request_data = {
            "model_id": "not-a-valid-uuid",
            "deployment_name": "test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 400 Bad Request
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_nonexistent_model(self):
        """Test deploying non-existent model."""
        model_id = str(uuid.uuid4())  # Random UUID that doesn't exist
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 404 Not Found
        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_invalid_environment(self):
        """Test deploying model with invalid environment."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "test_deployment",
            "environment": "invalid_env"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 400 Bad Request for invalid environment
        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_with_custom_config(self):
        """Test deploying model with custom configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "custom_deployment",
            "environment": "staging",
            "inference_config": {
                "confidence_threshold": 0.7,
                "nms_threshold": 0.4,
                "max_detections": 100
            },
            "endpoint_config": {
                "timeout_ms": 5000,
                "max_batch_size": 32
            }
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        if "inference_config" in response_data:
            assert isinstance(response_data["inference_config"], dict)
        if "endpoint_config" in response_data:
            assert isinstance(response_data["endpoint_config"], dict)

    def test_deploy_model_with_version(self):
        """Test deploying model with version specification."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "versioned_deployment",
            "environment": "production",
            "version": "v1.2.0",
            "description": "Production deployment of YOLO model v1.2.0"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        if "version" in response_data:
            assert response_data["version"] == "v1.2.0"
        if "description" in response_data:
            assert isinstance(response_data["description"], str)

    def test_deploy_model_duplicate_name(self):
        """Test deploying model with duplicate deployment name."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "existing_deployment",  # Assume this already exists
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Should return 409 Conflict for duplicate name
        assert response.status_code in [409, 400]
        response_data = response.json()
        assert "error" in response_data

    def test_deploy_model_with_health_check(self):
        """Test deploying model with health check configuration."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "health_checked_deployment",
            "environment": "production",
            "health_check": {
                "endpoint": "/health",
                "interval_seconds": 30,
                "timeout_seconds": 5,
                "retries": 3
            }
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        if "health_check" in response_data:
            health_check = response_data["health_check"]
            assert isinstance(health_check, dict)
            if "interval_seconds" in health_check:
                assert isinstance(health_check["interval_seconds"], int)

    def test_deploy_model_response_headers(self):
        """Test that response headers are correct."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Check content type
        assert response.headers["content-type"] == "application/json"

    def test_deploy_model_deployment_id_format(self):
        """Test that deployment_id is properly formatted UUID."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "uuid_test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        
        # Validate deployment_id is a valid UUID
        try:
            uuid.UUID(response_data["deployment_id"])
        except ValueError:
            pytest.fail(f"deployment_id '{response_data['deployment_id']}' is not a valid UUID")

    def test_deploy_model_initial_status(self):
        """Test that initial deployment status is appropriate."""
        model_id = str(uuid.uuid4())
        
        request_data = {
            "model_id": model_id,
            "deployment_name": "status_test_deployment",
            "environment": "staging"
        }

        response = client.post("/api/v1/deployments", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 202
        
        response_data = response.json()
        
        # Initial status should be pending or deploying
        assert response_data["status"] in ["pending", "deploying"]
