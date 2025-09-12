"""
Contract test for GET /api/v1/deployments/{deployment_id} (getDeployment operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the deployment details API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestDeploymentsGetContract:
    """Contract tests for deployment details endpoint."""

    def test_get_deployment_valid_id(self):
        """Test retrieving deployment with valid deployment ID."""
        deployment_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/deployments/{deployment_id}")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        
        required_fields = [
            "deployment_id", "model_id", "deployment_name", "status", 
            "environment", "created_at"
        ]
        
        for field in required_fields:
            assert field in response_data
        
        assert response_data["deployment_id"] == deployment_id
        assert response_data["status"] in ["pending", "deploying", "active", "failed"]

    def test_get_deployment_invalid_uuid(self):
        """Test retrieving deployment with invalid UUID format."""
        response = client.get("/api/v1/deployments/not-a-valid-uuid")

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_get_deployment_nonexistent_id(self):
        """Test retrieving deployment with non-existent ID."""
        nonexistent_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/deployments/{nonexistent_id}")

        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_get_deployment_with_metrics(self):
        """Test deployment with performance metrics."""
        deployment_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/deployments/{deployment_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "metrics" in response_data:
            metrics = response_data["metrics"]
            assert isinstance(metrics, dict)
            
            if "request_count" in metrics:
                assert isinstance(metrics["request_count"], int)
            if "avg_response_time_ms" in metrics:
                assert isinstance(metrics["avg_response_time_ms"], (int, float))

    def test_get_deployment_endpoint_info(self):
        """Test deployment endpoint information."""
        deployment_id = str(uuid.uuid4())
        
        response = client.get(f"/api/v1/deployments/{deployment_id}")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "endpoint_url" in response_data:
            assert isinstance(response_data["endpoint_url"], str)
            assert response_data["endpoint_url"].startswith(("http://", "https://"))
