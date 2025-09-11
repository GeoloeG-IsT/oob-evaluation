"""
Contract test for PATCH /api/v1/deployments/{deployment_id} (updateDeployment operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the deployment update API contract per api-spec.yaml.
"""

import pytest
import uuid
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestDeploymentsUpdateContract:
    """Contract tests for deployment update endpoint."""

    def test_update_deployment_basic(self):
        """Test updating deployment with basic parameters."""
        deployment_id = str(uuid.uuid4())
        
        request_data = {
            "status": "active",
            "description": "Updated deployment description"
        }

        response = client.patch(f"/api/v1/deployments/{deployment_id}", json=request_data)

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["deployment_id"] == deployment_id
        if "description" in response_data:
            assert response_data["description"] == "Updated deployment description"

    def test_update_deployment_scaling_config(self):
        """Test updating deployment scaling configuration."""
        deployment_id = str(uuid.uuid4())
        
        request_data = {
            "scaling_config": {
                "min_replicas": 3,
                "max_replicas": 15,
                "cpu_limit": "4",
                "memory_limit": "8Gi"
            }
        }

        response = client.patch(f"/api/v1/deployments/{deployment_id}", json=request_data)

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if "scaling_config" in response_data:
            scaling = response_data["scaling_config"]
            assert scaling["min_replicas"] == 3
            assert scaling["max_replicas"] == 15

    def test_update_deployment_invalid_uuid(self):
        """Test updating deployment with invalid UUID format."""
        request_data = {"status": "active"}

        response = client.patch("/api/v1/deployments/not-a-valid-uuid", json=request_data)

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data

    def test_update_deployment_nonexistent_id(self):
        """Test updating deployment with non-existent ID."""
        deployment_id = str(uuid.uuid4())
        request_data = {"status": "active"}
        
        response = client.patch(f"/api/v1/deployments/{deployment_id}", json=request_data)

        assert response.status_code == 404
        response_data = response.json()
        assert "error" in response_data

    def test_update_deployment_invalid_status(self):
        """Test updating deployment with invalid status."""
        deployment_id = str(uuid.uuid4())
        
        request_data = {
            "status": "invalid_status"
        }

        response = client.patch(f"/api/v1/deployments/{deployment_id}", json=request_data)

        assert response.status_code == 400
        response_data = response.json()
        assert "error" in response_data
