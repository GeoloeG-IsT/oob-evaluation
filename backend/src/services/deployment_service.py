"""
Deployment service for model endpoints.
"""
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import random
import string
from datetime import datetime, timezone
from urllib.parse import urljoin

from ..models.deployment import DeploymentModel, deployment_storage
from ..models.model import model_storage
from ..lib.ml_models import get_model_registry


class DeploymentService:
    """Service for handling deployment operations."""
    
    def __init__(self, base_endpoint_url: str = "https://api.mlplatform.com"):
        self.storage = deployment_storage
        self.model_storage = model_storage
        self.registry = get_model_registry()
        self.base_endpoint_url = base_endpoint_url.rstrip('/')
    
    def create_deployment(self, model_id: str, version: Optional[str] = None,
                         configuration: Optional[Dict[str, Any]] = None,
                         deployment_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a new model deployment."""
        # Validate required fields
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        model = self.model_storage.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Default version to model version if not specified
        if not version:
            version = f"v{model.version}" if model.version else "v1.0.0"
        
        # Default configuration
        default_config = {
            "auto_scaling": True,
            "min_instances": 1,
            "max_instances": 10,
            "cpu_request": "500m",
            "memory_request": "1Gi",
            "gpu_enabled": True,
            "timeout_seconds": 60,
            "health_check_path": "/health",
            "metrics_enabled": True
        }
        
        config = {**default_config, **(configuration or {})}
        
        # Generate unique endpoint URL
        endpoint_suffix = self._generate_endpoint_suffix(model, version)
        endpoint_url = f"{self.base_endpoint_url}/v1/models/{endpoint_suffix}/predict"
        
        # Check if deployment already exists for this model and version
        existing_deployment = self.storage.get_deployment_by_model_and_version(model_id, version)
        if existing_deployment:
            raise ValueError(f"Deployment already exists for model {model_id} version {version}")
        
        # Create deployment model
        deployment = DeploymentModel(
            model_id=model_id,
            endpoint_url=endpoint_url,
            version=version,
            configuration=config,
            status="deploying",
            metadata={
                "deployment_name": deployment_name or f"{model.name}-{version}",
                "model_name": model.name,
                "model_framework": model.framework,
                "model_type": model.type,
                "created_by": "deployment_service"
            }
        )
        
        # Save deployment
        saved_deployment = self.storage.save(deployment)
        
        # Start deployment process asynchronously
        asyncio.create_task(self._deploy_model_async(saved_deployment.id))
        
        return {
            "deployment_id": saved_deployment.id,
            "model_id": model_id,
            "endpoint_url": endpoint_url,
            "version": version,
            "status": "deploying",
            "configuration": config,
            "created_at": saved_deployment.created_at,
            "deployment_name": deployment.metadata.get("deployment_name")
        }
    
    def _generate_endpoint_suffix(self, model, version: str) -> str:
        """Generate unique endpoint suffix."""
        # Create readable endpoint from model name and version
        model_name = model.name.lower().replace(" ", "-").replace("_", "-")
        # Remove special characters
        model_name = ''.join(c for c in model_name if c.isalnum() or c == '-')
        version_clean = version.lower().replace("v", "").replace(".", "-")
        
        # Add random suffix for uniqueness
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        
        return f"{model_name}-{version_clean}-{random_suffix}"
    
    async def _deploy_model_async(self, deployment_id: str) -> None:
        """Deploy model asynchronously (simulation)."""
        try:
            # Simulate deployment steps
            await asyncio.sleep(2)  # Preparing
            self.storage.update_status(deployment_id, "deploying")
            
            await asyncio.sleep(3)  # Building container
            
            await asyncio.sleep(5)  # Starting service
            
            # Simulate deployment success/failure
            success_rate = 0.9  # 90% success rate
            if random.random() < success_rate:
                # Successful deployment
                performance_monitoring = {
                    "status": "healthy",
                    "instances": 1,
                    "cpu_usage_percent": random.uniform(10, 30),
                    "memory_usage_mb": random.uniform(512, 1024),
                    "request_count_24h": 0,
                    "average_response_time_ms": random.uniform(50, 200),
                    "error_rate_percent": 0.0,
                    "uptime_percent": 100.0,
                    "last_health_check": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                }
                
                self.storage.update_status(
                    deployment_id, 
                    "active", 
                    performance_monitoring=performance_monitoring
                )
            else:
                # Failed deployment
                self.storage.update_status(deployment_id, "failed")
                
        except Exception as e:
            print(f"Deployment failed for {deployment_id}: {str(e)}")
            self.storage.update_status(deployment_id, "failed")
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment by ID."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return None
        
        model = self.model_storage.get_by_id(deployment.model_id)
        
        return {
            "deployment_id": deployment.id,
            "model_id": deployment.model_id,
            "model_name": model.name if model else None,
            "endpoint_url": deployment.endpoint_url,
            "version": deployment.version,
            "status": deployment.status,
            "configuration": deployment.configuration,
            "performance_monitoring": deployment.performance_monitoring,
            "created_at": deployment.created_at,
            "updated_at": deployment.updated_at,
            "metadata": deployment.metadata
        }
    
    def list_deployments(self, model_id: Optional[str] = None, 
                        status: Optional[str] = None,
                        limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """List deployments with optional filtering."""
        deployments, total_count = self.storage.list_deployments(
            model_id=model_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        deployment_list = []
        for deployment in deployments:
            model = self.model_storage.get_by_id(deployment.model_id)
            
            deployment_dict = {
                "deployment_id": deployment.id,
                "model_id": deployment.model_id,
                "model_name": model.name if model else None,
                "endpoint_url": deployment.endpoint_url,
                "version": deployment.version,
                "status": deployment.status,
                "created_at": deployment.created_at,
                "updated_at": deployment.updated_at,
                "deployment_name": deployment.metadata.get("deployment_name")
            }
            
            # Add basic performance info for active deployments
            if deployment.status == "active" and deployment.performance_monitoring:
                deployment_dict["health_status"] = deployment.performance_monitoring.get("status", "unknown")
                deployment_dict["instances"] = deployment.performance_monitoring.get("instances", 0)
                deployment_dict["uptime_percent"] = deployment.performance_monitoring.get("uptime_percent", 0)
            
            deployment_list.append(deployment_dict)
        
        return deployment_list, total_count
    
    def update_deployment(self, deployment_id: str, 
                         configuration: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Optional[Dict[str, Any]]:
        """Update deployment configuration."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return None
        
        if deployment.status not in ["active", "inactive"]:
            raise ValueError(f"Cannot update deployment in status {deployment.status}")
        
        # Update configuration
        if configuration:
            updated_config = {**deployment.configuration, **configuration}
            deployment.configuration = updated_config
        
        # Update other allowed fields
        updateable_fields = ["metadata"]
        for field, value in kwargs.items():
            if field in updateable_fields and hasattr(deployment, field):
                setattr(deployment, field, value)
        
        # Save changes
        saved_deployment = self.storage.save(deployment)
        
        return self.get_deployment(saved_deployment.id)
    
    def activate_deployment(self, deployment_id: str) -> bool:
        """Activate an inactive deployment."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return False
        
        if deployment.status != "inactive":
            return False
        
        # Start redeployment
        self.storage.update_status(deployment_id, "deploying")
        asyncio.create_task(self._deploy_model_async(deployment_id))
        
        return True
    
    def deactivate_deployment(self, deployment_id: str) -> bool:
        """Deactivate an active deployment."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return False
        
        if deployment.status != "active":
            return False
        
        # Update status to inactive
        self.storage.update_status(deployment_id, "inactive")
        
        return True
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Delete a deployment."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return False
        
        # For TDD GREEN phase, we don't actually delete from storage
        # In real implementation, this would remove the deployment
        self.storage.update_status(deployment_id, "deleted")
        
        return True
    
    def get_deployment_by_endpoint(self, endpoint_url: str) -> Optional[Dict[str, Any]]:
        """Get deployment by endpoint URL."""
        deployment = self.storage.get_by_endpoint_url(endpoint_url)
        if not deployment:
            return None
        
        return self.get_deployment(deployment.id)
    
    def get_deployment_health(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed health information for a deployment."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return None
        
        if deployment.status != "active":
            return {
                "deployment_id": deployment_id,
                "status": deployment.status,
                "healthy": False,
                "message": f"Deployment is not active (status: {deployment.status})"
            }
        
        performance = deployment.performance_monitoring or {}
        
        # Simulate health check updates
        if random.random() < 0.1:  # 10% chance to update metrics
            asyncio.create_task(self._update_performance_metrics(deployment_id))
        
        health_info = {
            "deployment_id": deployment_id,
            "status": deployment.status,
            "healthy": performance.get("status") == "healthy",
            "instances": {
                "total": performance.get("instances", 0),
                "healthy": performance.get("instances", 0) if performance.get("status") == "healthy" else 0,
                "unhealthy": 0
            },
            "resource_usage": {
                "cpu_percent": performance.get("cpu_usage_percent", 0),
                "memory_mb": performance.get("memory_usage_mb", 0),
                "gpu_percent": performance.get("gpu_usage_percent", 0)
            },
            "performance_metrics": {
                "request_count_24h": performance.get("request_count_24h", 0),
                "average_response_time_ms": performance.get("average_response_time_ms", 0),
                "error_rate_percent": performance.get("error_rate_percent", 0),
                "uptime_percent": performance.get("uptime_percent", 0)
            },
            "last_health_check": performance.get("last_health_check"),
            "endpoint_url": deployment.endpoint_url
        }
        
        return health_info
    
    async def _update_performance_metrics(self, deployment_id: str) -> None:
        """Update performance metrics for a deployment."""
        try:
            # Simulate realistic metrics updates
            updated_metrics = {
                "cpu_usage_percent": random.uniform(15, 45),
                "memory_usage_mb": random.uniform(600, 1200),
                "gpu_usage_percent": random.uniform(20, 80),
                "request_count_24h": random.randint(50, 1000),
                "average_response_time_ms": random.uniform(80, 300),
                "error_rate_percent": random.uniform(0, 2),
                "uptime_percent": random.uniform(99.5, 100),
                "last_health_check": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
            
            self.storage.update_performance_monitoring(deployment_id, updated_metrics)
            
        except Exception as e:
            print(f"Failed to update performance metrics for {deployment_id}: {str(e)}")
    
    def get_deployment_logs(self, deployment_id: str, 
                          lines: int = 100, level: str = "info") -> Dict[str, Any]:
        """Get deployment logs (simulated)."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return {"error": "Deployment not found"}
        
        # Simulate log entries
        log_entries = []
        log_levels = ["info", "warning", "error"] if level == "all" else [level]
        
        for i in range(min(lines, 100)):
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            log_level = random.choice(log_levels)
            
            if log_level == "info":
                messages = [
                    "Model prediction completed successfully",
                    "Health check passed",
                    f"Processing request {random.randint(1000, 9999)}",
                    "Resource usage within normal limits"
                ]
            elif log_level == "warning":
                messages = [
                    "High memory usage detected",
                    "Slow response time observed",
                    "Rate limit approaching"
                ]
            else:  # error
                messages = [
                    "Model prediction failed",
                    "Connection timeout",
                    "Invalid input format"
                ]
            
            log_entries.append({
                "timestamp": timestamp,
                "level": log_level.upper(),
                "message": random.choice(messages),
                "deployment_id": deployment_id
            })
        
        return {
            "deployment_id": deployment_id,
            "log_entries": log_entries,
            "total_lines": len(log_entries),
            "level_filter": level
        }
    
    def get_deployment_metrics(self, deployment_id: str, 
                             hours: int = 24) -> Dict[str, Any]:
        """Get deployment metrics over time period."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            return {"error": "Deployment not found"}
        
        if deployment.status != "active":
            return {
                "deployment_id": deployment_id,
                "status": deployment.status,
                "message": "Metrics only available for active deployments"
            }
        
        # Simulate time series metrics
        current_time = datetime.now(timezone.utc)
        metrics_data = {
            "deployment_id": deployment_id,
            "period_hours": hours,
            "metrics": {
                "request_count": [],
                "response_time": [],
                "error_rate": [],
                "cpu_usage": [],
                "memory_usage": []
            }
        }
        
        # Generate hourly data points
        for i in range(hours):
            timestamp = current_time.timestamp() - (i * 3600)
            iso_timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            
            metrics_data["metrics"]["request_count"].append({
                "timestamp": iso_timestamp,
                "value": random.randint(10, 100)
            })
            
            metrics_data["metrics"]["response_time"].append({
                "timestamp": iso_timestamp,
                "value": random.uniform(80, 250)
            })
            
            metrics_data["metrics"]["error_rate"].append({
                "timestamp": iso_timestamp,
                "value": random.uniform(0, 3)
            })
            
            metrics_data["metrics"]["cpu_usage"].append({
                "timestamp": iso_timestamp,
                "value": random.uniform(20, 60)
            })
            
            metrics_data["metrics"]["memory_usage"].append({
                "timestamp": iso_timestamp,
                "value": random.uniform(700, 1300)
            })
        
        return metrics_data
    
    def scale_deployment(self, deployment_id: str, instances: int) -> Dict[str, Any]:
        """Scale deployment to specified number of instances."""
        deployment = self.storage.get_by_id(deployment_id)
        if not deployment:
            raise ValueError("Deployment not found")
        
        if deployment.status != "active":
            raise ValueError(f"Cannot scale deployment in status {deployment.status}")
        
        if instances < 1:
            raise ValueError("Instance count must be at least 1")
        
        max_instances = deployment.configuration.get("max_instances", 10)
        if instances > max_instances:
            raise ValueError(f"Cannot scale beyond max instances ({max_instances})")
        
        # Update configuration
        deployment.configuration["min_instances"] = instances
        current_instances = deployment.performance_monitoring.get("instances", 1)
        
        # Update performance monitoring
        updated_performance = {
            "instances": instances,
            "scaling_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "previous_instances": current_instances
        }
        
        self.storage.save(deployment)
        self.storage.update_performance_monitoring(deployment_id, updated_performance)
        
        return {
            "deployment_id": deployment_id,
            "scaling_action": "completed",
            "previous_instances": current_instances,
            "current_instances": instances,
            "timestamp": updated_performance["scaling_timestamp"]
        }
    
    def get_deployment_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get deployment statistics for the last N days."""
        deployments, _ = self.storage.list_deployments(limit=1000)
        
        # Filter deployments from last N days
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
        recent_deployments = []
        
        for deployment in deployments:
            try:
                deployment_time = datetime.fromisoformat(deployment.created_at.replace("Z", "+00:00")).timestamp()
                if deployment_time >= cutoff_date:
                    recent_deployments.append(deployment)
            except (ValueError, AttributeError):
                continue
        
        if not recent_deployments:
            return {
                "period_days": days,
                "total_deployments": 0
            }
        
        # Calculate statistics
        status_counts = {}
        for deployment in recent_deployments:
            status_counts[deployment.status] = status_counts.get(deployment.status, 0) + 1
        
        active_deployments = [d for d in recent_deployments if d.status == "active"]
        failed_deployments = [d for d in recent_deployments if d.status == "failed"]
        
        stats = {
            "period_days": days,
            "total_deployments": len(recent_deployments),
            "status_breakdown": status_counts,
            "success_rate": ((len(recent_deployments) - len(failed_deployments)) / len(recent_deployments) * 100) if recent_deployments else 0,
            "active_deployments": len(active_deployments)
        }
        
        # Calculate average uptime for active deployments
        if active_deployments:
            uptimes = []
            for deployment in active_deployments:
                if deployment.performance_monitoring and "uptime_percent" in deployment.performance_monitoring:
                    uptimes.append(deployment.performance_monitoring["uptime_percent"])
            
            if uptimes:
                stats["average_uptime_percent"] = sum(uptimes) / len(uptimes)
        
        return stats
    
    def create_model_version_deployment(self, model_id: str, 
                                      source_deployment_id: str,
                                      new_version: str) -> Dict[str, Any]:
        """Create a new deployment for a model version based on existing deployment."""
        # Get source deployment
        source_deployment = self.storage.get_by_id(source_deployment_id)
        if not source_deployment:
            raise ValueError("Source deployment not found")
        
        if source_deployment.model_id != model_id:
            raise ValueError("Source deployment model ID does not match")
        
        # Create new deployment with same configuration
        return self.create_deployment(
            model_id=model_id,
            version=new_version,
            configuration=source_deployment.configuration.copy(),
            deployment_name=f"{source_deployment.metadata.get('deployment_name', 'deployment')}-{new_version}"
        )