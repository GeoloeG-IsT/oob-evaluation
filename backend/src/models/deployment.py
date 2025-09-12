"""
In-memory storage for deployments (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class DeploymentModel:
    """Temporary in-memory deployment model for TDD GREEN phase."""
    
    def __init__(self, model_id: str, endpoint_url: str, version: str, 
                 configuration: Dict[str, Any], status: str = "deploying",
                 performance_monitoring: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.model_id = model_id
        self.endpoint_url = endpoint_url
        self.version = version
        self.status = status  # deploying, active, inactive, failed
        self.configuration = configuration
        self.performance_monitoring = performance_monitoring or {}
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class DeploymentStorage:
    """Temporary in-memory storage for deployments."""
    
    def __init__(self):
        self._deployments: Dict[str, DeploymentModel] = {}
    
    def save(self, deployment: DeploymentModel) -> DeploymentModel:
        """Save a deployment to storage."""
        deployment.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._deployments[deployment.id] = deployment
        return deployment
    
    def get_by_id(self, deployment_id: str) -> Optional[DeploymentModel]:
        """Get a deployment by ID."""
        return self._deployments.get(deployment_id)
    
    def get_by_endpoint_url(self, endpoint_url: str) -> Optional[DeploymentModel]:
        """Get a deployment by endpoint URL."""
        for deployment in self._deployments.values():
            if deployment.endpoint_url == endpoint_url:
                return deployment
        return None
    
    def list_deployments(self, model_id: Optional[str] = None, status: Optional[str] = None,
                        limit: int = 50, offset: int = 0) -> tuple[List[DeploymentModel], int]:
        """List deployments with optional filtering and pagination."""
        deployments = list(self._deployments.values())
        
        # Apply filters
        if model_id:
            deployments = [d for d in deployments if d.model_id == model_id]
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        # Sort by created_at descending
        deployments.sort(key=lambda x: x.created_at, reverse=True)
        
        total_count = len(deployments)
        
        # Apply pagination
        paginated_deployments = deployments[offset:offset + limit]
        
        return paginated_deployments, total_count
    
    def update_status(self, deployment_id: str, status: str, 
                     performance_monitoring: Optional[Dict[str, Any]] = None) -> bool:
        """Update deployment status and performance monitoring."""
        deployment = self.get_by_id(deployment_id)
        if not deployment:
            return False
        
        deployment.status = status
        deployment.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        if performance_monitoring is not None:
            deployment.performance_monitoring = performance_monitoring
        
        return True
    
    def get_deployments_by_model(self, model_id: str, status: Optional[str] = None) -> List[DeploymentModel]:
        """Get all deployments for a specific model."""
        deployments = [d for d in self._deployments.values() if d.model_id == model_id]
        
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        return deployments
    
    def get_active_deployments(self) -> List[DeploymentModel]:
        """Get all active deployments."""
        return [d for d in self._deployments.values() if d.status == "active"]
    
    def update_performance_monitoring(self, deployment_id: str, 
                                    performance_data: Dict[str, Any]) -> bool:
        """Update performance monitoring data for a deployment."""
        deployment = self.get_by_id(deployment_id)
        if not deployment:
            return False
        
        # Merge new performance data with existing data
        if not deployment.performance_monitoring:
            deployment.performance_monitoring = {}
        
        deployment.performance_monitoring.update(performance_data)
        deployment.updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        return True
    
    def get_deployment_by_model_and_version(self, model_id: str, version: str) -> Optional[DeploymentModel]:
        """Get a deployment by model ID and version."""
        for deployment in self._deployments.values():
            if deployment.model_id == model_id and deployment.version == version:
                return deployment
        return None


# Global storage instance (temporary for TDD)
deployment_storage = DeploymentStorage()