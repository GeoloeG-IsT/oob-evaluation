"""
Training worker for model fine-tuning tasks.

Handles async training jobs using Celery for distributed processing.
Integrates with backend training service and training pipeline library.
"""
import asyncio
from typing import Dict, Any, Optional
import traceback
import logging
from datetime import datetime, timezone

from celery import current_task
from celery.exceptions import Ignore
from ..celery_app import celery_app

# Import backend services
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'src'))

from services.training_service import TrainingService
from lib.training_pipeline import (
    TrainingPipeline, TrainingConfig, HyperParameters,
    TrainingStatus, TrainingJob, JobScheduler, TrainingMetrics
)
from lib.ml_models import get_model_registry

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='training_worker.start_training')
def start_training_task(self, base_model_id: str, dataset_id: str,
                       hyperparameters: Dict[str, Any],
                       experiment_name: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
    """
    Start a model training task.
    
    Args:
        base_model_id: ID of the base model to fine-tune
        dataset_id: ID of the training dataset
        hyperparameters: Training hyperparameters
        experiment_name: Optional experiment name
        metadata: Optional metadata
        
    Returns:
        Dict with training job details
    """
    job_id = self.request.id
    logger.info(f"Starting training task {job_id} for model {base_model_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing training job...'
            }
        )
        
        # Initialize services
        training_service = TrainingService()
        pipeline = TrainingPipeline()
        registry = get_model_registry()
        
        # Validate inputs
        if not base_model_id:
            raise ValueError("base_model_id is required")
        if not dataset_id:
            raise ValueError("dataset_id is required")
        if not hyperparameters:
            raise ValueError("hyperparameters are required")
        
        # Validate model exists
        model_config = registry.get_model_config(base_model_id)
        if not model_config:
            raise ValueError(f"Model {base_model_id} not found in registry")
        
        logger.info(f"Using model: {model_config}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'validating',
                'progress': 10,
                'message': 'Validating training configuration...'
            }
        )
        
        # Create hyperparameters object
        hyper_params = HyperParameters.from_dict(hyperparameters)
        logger.info(f"Training hyperparameters: {hyper_params.to_dict()}")
        
        # Create training configuration
        training_config = TrainingConfig(
            base_model_id=base_model_id,
            dataset_path=f"/storage/datasets/{dataset_id}",
            output_dir=f"/storage/models/training/{job_id}",
            hyperparameters=hyper_params,
            model_type=model_config.get('type', 'detection'),
            num_classes=model_config.get('num_classes', 80)
        )
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'preparing',
                'progress': 20,
                'message': 'Preparing training environment...'
            }
        )
        
        # Create training job
        training_job = TrainingJob(
            job_id=job_id,
            config=training_config
        )
        
        # Set up progress callback
        def progress_callback(progress_data: Dict[str, Any]):
            """Update Celery task progress."""
            try:
                current_epoch = progress_data.get('current_epoch', 0)
                total_epochs = progress_data.get('total_epochs', 1)
                train_loss = progress_data.get('train_loss', 0.0)
                val_loss = progress_data.get('val_loss', 0.0)
                map50 = progress_data.get('map50', 0.0)
                
                progress_pct = min(20 + (current_epoch / total_epochs * 70), 90)
                
                self.update_state(
                    state='RUNNING',
                    meta={
                        'status': 'training',
                        'progress': progress_pct,
                        'current_epoch': current_epoch,
                        'total_epochs': total_epochs,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'map50': map50,
                        'message': f'Training epoch {current_epoch}/{total_epochs} - mAP@0.5: {map50:.4f}'
                    }
                )
                logger.info(f"Training progress: epoch {current_epoch}/{total_epochs}, mAP@0.5: {map50:.4f}")
            except Exception as e:
                logger.warning(f"Error updating progress: {str(e)}")
        
        # Run training with progress monitoring
        logger.info("Starting training pipeline...")
        result = pipeline.run_training(training_job, progress_callback=progress_callback)
        
        # Update final status
        if result.status == TrainingStatus.COMPLETED:
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Training completed successfully',
                    'final_metrics': {
                        'best_map50': result.metrics.best_map50 if result.metrics else 0.0,
                        'final_train_loss': result.metrics.final_train_loss if result.metrics else 0.0,
                        'final_val_loss': result.metrics.final_val_loss if result.metrics else 0.0,
                        'training_time_hours': result.training_time_seconds / 3600 if result.training_time_seconds else 0.0
                    },
                    'model_path': result.model_path,
                    'checkpoint_path': result.checkpoint_path
                }
            )
            
            logger.info(f"Training completed successfully: {result.model_path}")
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'model_path': result.model_path,
                'checkpoint_path': result.checkpoint_path,
                'metrics': result.metrics.__dict__ if result.metrics else {},
                'training_time_seconds': result.training_time_seconds,
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            error_msg = result.error_message or "Training failed with unknown error"
            logger.error(f"Training failed: {error_msg}")
            
            self.update_state(
                state='FAILURE',
                meta={
                    'status': 'failed',
                    'progress': 0,
                    'message': f'Training failed: {error_msg}',
                    'error': error_msg
                }
            )
            
            raise Exception(error_msg)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Training task {job_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Training failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='training_worker.cancel_training')
def cancel_training_task(self, job_id: str):
    """
    Cancel a running training task.
    
    Args:
        job_id: ID of the training job to cancel
        
    Returns:
        Dict with cancellation status
    """
    logger.info(f"Cancelling training task {job_id}")
    
    try:
        # Initialize training service
        training_service = TrainingService()
        
        # Cancel the job
        result = training_service.cancel_training_job(job_id)
        
        if result:
            logger.info(f"Training job {job_id} cancelled successfully")
            return {
                'job_id': job_id,
                'status': 'cancelled',
                'cancelled_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            logger.warning(f"Could not cancel training job {job_id}")
            raise Exception(f"Could not cancel training job {job_id}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to cancel training job {job_id}: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'cancellation_failed',
                'message': f'Cancellation failed: {error_msg}',
                'error': error_msg
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='training_worker.pause_training')
def pause_training_task(self, job_id: str):
    """
    Pause a running training task.
    
    Args:
        job_id: ID of the training job to pause
        
    Returns:
        Dict with pause status
    """
    logger.info(f"Pausing training task {job_id}")
    
    try:
        training_service = TrainingService()
        
        success = training_service.pause_training_job(job_id)
        
        if success:
            logger.info(f"Training job {job_id} paused successfully")
            return {
                'job_id': job_id,
                'status': 'paused',
                'paused_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise Exception(f"Could not pause training job {job_id}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to pause training job {job_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='training_worker.resume_training')
def resume_training_task(self, job_id: str):
    """
    Resume a paused training task.
    
    Args:
        job_id: ID of the training job to resume
        
    Returns:
        Dict with resume status
    """
    logger.info(f"Resuming training task {job_id}")
    
    try:
        training_service = TrainingService()
        
        success = training_service.resume_training_job(job_id)
        
        if success:
            logger.info(f"Training job {job_id} resumed successfully")
            return {
                'job_id': job_id,
                'status': 'running',
                'resumed_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise Exception(f"Could not resume training job {job_id}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to resume training job {job_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='training_worker.get_training_status')
def get_training_status_task(self, job_id: str):
    """
    Get training job status and progress.
    
    Args:
        job_id: ID of the training job
        
    Returns:
        Dict with training job status
    """
    try:
        training_service = TrainingService()
        
        job_info = training_service.get_training_job(job_id)
        
        if job_info:
            return job_info
        else:
            raise Exception(f"Training job {job_id} not found")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get training job status {job_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='training_worker.cleanup_training_job')
def cleanup_training_job_task(self, job_id: str, remove_checkpoints: bool = False):
    """
    Clean up training job artifacts.
    
    Args:
        job_id: ID of the training job to clean up
        remove_checkpoints: Whether to remove checkpoint files
        
    Returns:
        Dict with cleanup status
    """
    logger.info(f"Cleaning up training job {job_id}")
    
    try:
        import shutil
        import os
        
        # Clean up training directory
        training_dir = f"/storage/models/training/{job_id}"
        if os.path.exists(training_dir):
            if remove_checkpoints:
                shutil.rmtree(training_dir)
                logger.info(f"Removed training directory: {training_dir}")
            else:
                # Only remove temporary files, keep final model and checkpoints
                temp_files = ['temp', 'cache', 'logs']
                for temp_file in temp_files:
                    temp_path = os.path.join(training_dir, temp_file)
                    if os.path.exists(temp_path):
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                        else:
                            shutil.rmtree(temp_path)
                        logger.info(f"Removed temp file/dir: {temp_path}")
        
        return {
            'job_id': job_id,
            'cleanup_completed': True,
            'checkpoints_removed': remove_checkpoints,
            'cleaned_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to cleanup training job {job_id}: {error_msg}")
        raise Exception(error_msg)