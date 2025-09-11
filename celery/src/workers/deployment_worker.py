"""
Deployment worker for model endpoint tasks.

Handles async deployment jobs using Celery for distributed processing.
Integrates with backend deployment service and manages model endpoint lifecycle.
"""
import asyncio
from typing import Dict, Any, Optional, List
import traceback
import logging
import time
from datetime import datetime, timezone

from celery import current_task
from celery.exceptions import Ignore
from ..celery_app import celery_app

# Import backend services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'src'))

from services.deployment_service import DeploymentService
from lib.ml_models import get_model_registry

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='deployment_worker.create_deployment')
def create_deployment_task(self, model_id: str, version: Optional[str] = None,
                          configuration: Optional[Dict[str, Any]] = None,
                          deployment_name: Optional[str] = None):
    """
    Create a model deployment task.
    
    Args:
        model_id: ID of the model to deploy
        version: Version of the deployment (optional)
        configuration: Deployment configuration options
        deployment_name: Custom deployment name
        
    Returns:
        Dict with deployment details
    """
    task_id = self.request.id
    logger.info(f"Starting deployment task {task_id} for model {model_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing deployment...'
            }
        )
        
        # Initialize services
        deployment_service = DeploymentService()
        registry = get_model_registry()
        
        # Validate inputs
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        model_config = registry.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        logger.info(f"Deploying model: {model_config}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'preparing',
                'progress': 10,
                'message': 'Preparing deployment configuration...'
            }
        )
        
        # Create deployment
        deployment_info = deployment_service.create_deployment(
            model_id=model_id,
            version=version,
            configuration=configuration,
            deployment_name=deployment_name
        )
        
        deployment_id = deployment_info['deployment_id']
        endpoint_url = deployment_info['endpoint_url']
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'provisioning',
                'progress': 30,
                'message': 'Provisioning infrastructure...',
                'deployment_id': deployment_id,
                'endpoint_url': endpoint_url
            }
        )
        
        # Simulate infrastructure provisioning
        await asyncio.sleep(2)
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'loading_model',
                'progress': 50,
                'message': 'Loading model into serving container...',
                'deployment_id': deployment_id,
                'endpoint_url': endpoint_url
            }
        )
        
        # Simulate model loading
        await asyncio.sleep(3)
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'health_check',
                'progress': 70,
                'message': 'Running health checks...',
                'deployment_id': deployment_id,
                'endpoint_url': endpoint_url
            }
        )
        
        # Simulate health checks
        await asyncio.sleep(2)
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'finalizing',
                'progress': 90,
                'message': 'Finalizing deployment...',
                'deployment_id': deployment_id,
                'endpoint_url': endpoint_url
            }
        )
        
        # Update deployment status to active
        deployment_service.update_deployment_status(deployment_id, "active")
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'active',
                'progress': 100,
                'message': 'Deployment completed successfully',
                'deployment_id': deployment_id,
                'endpoint_url': endpoint_url,
                'model_id': model_id,
                'version': deployment_info['version']
            }
        )
        
        logger.info(f"Deployment completed successfully: {endpoint_url}")
        
        return {
            'task_id': task_id,
            'deployment_id': deployment_id,
            'model_id': model_id,
            'endpoint_url': endpoint_url,
            'version': deployment_info['version'],
            'status': 'active',
            'configuration': deployment_info['configuration'],
            'deployment_name': deployment_info.get('deployment_name'),
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Deployment task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Deployment failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='deployment_worker.update_deployment')
def update_deployment_task(self, deployment_id: str, 
                          configuration: Optional[Dict[str, Any]] = None,
                          target_version: Optional[str] = None):
    """
    Update an existing deployment with new configuration or version.
    
    Args:
        deployment_id: ID of the deployment to update
        configuration: New configuration options
        target_version: New version to deploy
        
    Returns:
        Dict with update status
    """
    task_id = self.request.id
    logger.info(f"Starting deployment update task {task_id} for deployment {deployment_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing deployment update...'
            }
        )
        
        # Initialize services
        deployment_service = DeploymentService()
        
        # Get current deployment
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        logger.info(f"Updating deployment: {deployment['endpoint_url']}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'preparing_update',
                'progress': 20,
                'message': 'Preparing deployment update...'
            }
        )
        
        # Perform rolling update
        if configuration or target_version:
            # Update progress
            self.update_state(
                state='RUNNING',
                meta={
                    'status': 'updating',
                    'progress': 40,
                    'message': 'Performing rolling update...'
                }
            )
            
            # Simulate rolling update
            await asyncio.sleep(3)
            
            # Update deployment
            updated_deployment = deployment_service.update_deployment(
                deployment_id=deployment_id,
                configuration=configuration,
                version=target_version
            )
            
            # Update progress
            self.update_state(
                state='RUNNING',
                meta={
                    'status': 'health_check',
                    'progress': 80,
                    'message': 'Verifying updated deployment...'
                }
            )
            
            # Simulate health check
            await asyncio.sleep(2)
            
            # Update final status
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'active',
                    'progress': 100,
                    'message': 'Deployment updated successfully',
                    'deployment_id': deployment_id
                }
            )
            
            logger.info(f"Deployment {deployment_id} updated successfully")
            
            return {
                'task_id': task_id,
                'deployment_id': deployment_id,
                'status': 'updated',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                **updated_deployment
            }
        else:
            # No changes needed
            return {
                'task_id': task_id,
                'deployment_id': deployment_id,
                'status': 'no_changes',
                'message': 'No configuration or version changes specified'
            }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Deployment update task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'update_failed',
                'progress': 0,
                'message': f'Deployment update failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='deployment_worker.scale_deployment')
def scale_deployment_task(self, deployment_id: str, target_instances: int):
    """
    Scale deployment to target number of instances.
    
    Args:
        deployment_id: ID of the deployment to scale
        target_instances: Target number of instances
        
    Returns:
        Dict with scaling status
    """
    task_id = self.request.id
    logger.info(f"Starting scaling task {task_id} for deployment {deployment_id} to {target_instances} instances")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': f'Initializing scaling to {target_instances} instances...'
            }
        )
        
        # Initialize services
        deployment_service = DeploymentService()
        
        # Get current deployment
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        current_instances = deployment.get('current_instances', 1)
        logger.info(f"Scaling from {current_instances} to {target_instances} instances")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'scaling',
                'progress': 30,
                'message': f'Scaling from {current_instances} to {target_instances} instances...',
                'current_instances': current_instances,
                'target_instances': target_instances
            }
        )
        
        # Simulate scaling process
        if target_instances > current_instances:
            # Scaling up
            for i in range(current_instances + 1, target_instances + 1):
                progress = 30 + ((i - current_instances) / (target_instances - current_instances)) * 60
                self.update_state(
                    state='RUNNING',
                    meta={
                        'status': 'scaling_up',
                        'progress': progress,
                        'message': f'Starting instance {i}/{target_instances}...',
                        'current_instances': i - 1,
                        'target_instances': target_instances
                    }
                )
                await asyncio.sleep(1)
        elif target_instances < current_instances:
            # Scaling down
            for i in range(current_instances - 1, target_instances - 1, -1):
                progress = 30 + ((current_instances - i) / (current_instances - target_instances)) * 60
                self.update_state(
                    state='RUNNING',
                    meta={
                        'status': 'scaling_down',
                        'progress': progress,
                        'message': f'Stopping instance {i + 1}...',
                        'current_instances': i + 1,
                        'target_instances': target_instances
                    }
                )
                await asyncio.sleep(1)
        
        # Update deployment with new instance count
        deployment_service.update_deployment_scaling(deployment_id, target_instances)
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'scaled',
                'progress': 100,
                'message': f'Successfully scaled to {target_instances} instances',
                'current_instances': target_instances,
                'target_instances': target_instances
            }
        )
        
        logger.info(f"Deployment {deployment_id} scaled successfully to {target_instances} instances")
        
        return {
            'task_id': task_id,
            'deployment_id': deployment_id,
            'status': 'scaled',
            'previous_instances': current_instances,
            'current_instances': target_instances,
            'scaled_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Scaling task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'scaling_failed',
                'progress': 0,
                'message': f'Scaling failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='deployment_worker.terminate_deployment')
def terminate_deployment_task(self, deployment_id: str, force: bool = False):
    """
    Terminate a deployment and clean up resources.
    
    Args:
        deployment_id: ID of the deployment to terminate
        force: Whether to force termination even if there are active requests
        
    Returns:
        Dict with termination status
    """
    task_id = self.request.id
    logger.info(f"Starting termination task {task_id} for deployment {deployment_id} (force={force})")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing deployment termination...'
            }
        )
        
        # Initialize services
        deployment_service = DeploymentService()
        
        # Get current deployment
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment['status'] == 'terminated':
            logger.info(f"Deployment {deployment_id} is already terminated")
            return {
                'task_id': task_id,
                'deployment_id': deployment_id,
                'status': 'already_terminated',
                'message': 'Deployment was already terminated'
            }
        
        logger.info(f"Terminating deployment: {deployment['endpoint_url']}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'draining_traffic',
                'progress': 20,
                'message': 'Draining traffic from deployment...'
            }
        )
        
        # Simulate traffic draining (unless forced)
        if not force:
            await asyncio.sleep(2)
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'stopping_instances',
                'progress': 50,
                'message': 'Stopping deployment instances...'
            }
        )
        
        # Simulate stopping instances
        await asyncio.sleep(2)
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'cleaning_resources',
                'progress': 80,
                'message': 'Cleaning up resources...'
            }
        )
        
        # Simulate resource cleanup
        await asyncio.sleep(1)
        
        # Update deployment status
        deployment_service.update_deployment_status(deployment_id, "terminated")
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'terminated',
                'progress': 100,
                'message': 'Deployment terminated successfully',
                'deployment_id': deployment_id
            }
        )
        
        logger.info(f"Deployment {deployment_id} terminated successfully")
        
        return {
            'task_id': task_id,
            'deployment_id': deployment_id,
            'status': 'terminated',
            'endpoint_url': deployment['endpoint_url'],
            'terminated_at': datetime.now(timezone.utc).isoformat(),
            'force_terminated': force
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Termination task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'termination_failed',
                'progress': 0,
                'message': f'Termination failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='deployment_worker.health_check')
def health_check_task(self, deployment_id: str):
    """
    Perform health check on a deployment.
    
    Args:
        deployment_id: ID of the deployment to check
        
    Returns:
        Dict with health status
    """
    logger.info(f"Running health check for deployment {deployment_id}")
    
    try:
        # Initialize services
        deployment_service = DeploymentService()
        
        # Get deployment
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Simulate health check
        health_status = deployment_service.check_deployment_health(deployment_id)
        
        logger.info(f"Health check completed for deployment {deployment_id}: {health_status['status']}")
        
        return {
            'deployment_id': deployment_id,
            'endpoint_url': deployment['endpoint_url'],
            'health_status': health_status,
            'checked_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Health check failed for deployment {deployment_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='deployment_worker.get_deployment_status')
def get_deployment_status_task(self, deployment_id: str):
    """
    Get deployment status and metrics.
    
    Args:
        deployment_id: ID of the deployment
        
    Returns:
        Dict with deployment status
    """
    try:
        deployment_service = DeploymentService()
        
        deployment = deployment_service.get_deployment(deployment_id)
        
        if deployment:
            return deployment
        else:
            raise Exception(f"Deployment {deployment_id} not found")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get deployment status {deployment_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='deployment_worker.collect_metrics')
def collect_deployment_metrics_task(self, deployment_id: str, time_range_hours: int = 1):
    """
    Collect deployment metrics for monitoring.
    
    Args:
        deployment_id: ID of the deployment
        time_range_hours: Time range in hours for metrics collection
        
    Returns:
        Dict with deployment metrics
    """
    logger.info(f"Collecting metrics for deployment {deployment_id} (last {time_range_hours} hours)")
    
    try:
        deployment_service = DeploymentService()
        
        # Get deployment
        deployment = deployment_service.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Collect metrics (this would integrate with monitoring systems)
        metrics = deployment_service.get_deployment_metrics(deployment_id, time_range_hours)
        
        logger.info(f"Metrics collected for deployment {deployment_id}")
        
        return {
            'deployment_id': deployment_id,
            'time_range_hours': time_range_hours,
            'metrics': metrics,
            'collected_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to collect metrics for deployment {deployment_id}: {error_msg}")
        raise Exception(error_msg)