"""
Contract test for GET /api/v1/deployments (listDeployments operation)

This test MUST FAIL initially since the endpoint is not implemented.
Tests the deployment listing API contract per api-spec.yaml.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestDeploymentsListContract:
    """Contract tests for deployment listing endpoint."""

    def test_list_deployments_no_filters(self):
        """Test listing all deployments without filters."""
        response = client.get("/api/v1/deployments")

        # Contract assertions - this MUST FAIL since endpoint doesn't exist
        assert response.status_code == 200
        
        response_data = response.json()
        assert "deployments" in response_data
        assert isinstance(response_data["deployments"], list)

    def test_list_deployments_structure(self):
        """Test that returned deployments have correct structure."""
        response = client.get("/api/v1/deployments")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        if response_data["deployments"]:
            deployment = response_data["deployments"][0]
            
            required_fields = [
                "deployment_id", "model_id", "deployment_name", "status", 
                "environment", "created_at"
            ]
            
            for field in required_fields:
                assert field in deployment
            
            assert deployment["status"] in ["pending", "deploying", "active", "failed"]
            assert deployment["environment"] in ["development", "staging", "production"]

    def test_list_deployments_filter_by_status(self):
        """Test listing deployments filtered by status."""
        response = client.get("/api/v1/deployments?status=active")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for deployment in response_data["deployments"]:
            assert deployment["status"] == "active"

    def test_list_deployments_filter_by_environment(self):
        """Test listing deployments filtered by environment."""
        response = client.get("/api/v1/deployments?environment=production")

        # Contract assertions - this MUST FAIL
        assert response.status_code == 200
        
        response_data = response.json()
        for deployment in response_data["deployments"]:
            assert deployment["environment"] == "production"

    def test_list_deployments_response_headers(self):
        """Test that response headers are correct."""
        response = client.get("/api/v1/deployments")

        assert response.headers["content-type"] == "application/json"
