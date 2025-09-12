"""
Inference worker for batch processing tasks.

Handles async inference jobs using Celery for distributed processing.
Integrates with backend inference service and inference engine library.
"""
import asyncio
from typing import List, Dict, Any, Optional
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

from services.inference_service import InferenceService
from lib.inference_engine import (
    InferenceEngine, InferenceRequest, InferenceResult,
    BatchInferenceJob, InferenceStatus, InferencePerformanceMetrics
)
from lib.ml_models import get_model_registry

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='inference_worker.single_inference')
def single_inference_task(self, image_id: str, model_id: str,
                         parameters: Optional[Dict[str, Any]] = None):
    """
    Perform single image inference task.
    
    Args:
        image_id: ID of the image to process
        model_id: ID of the model to use
        parameters: Optional inference parameters
        
    Returns:
        Dict with inference results
    """
    task_id = self.request.id
    logger.info(f"Starting single inference task {task_id} for image {image_id} with model {model_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing inference...'
            }
        )
        
        # Initialize services
        inference_service = InferenceService()
        engine = InferenceEngine()
        registry = get_model_registry()
        
        # Validate inputs
        if not image_id:
            raise ValueError("image_id is required")
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        model_config = registry.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        logger.info(f"Using model: {model_config}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'loading_model',
                'progress': 20,
                'message': 'Loading model...'
            }
        )
        
        # Create inference request
        request = InferenceRequest(
            model_id=model_id,
            image_path=f"/storage/images/{image_id}",
            parameters=parameters or {}
        )
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'processing',
                'progress': 50,
                'message': 'Processing image...'
            }
        )
        
        # Perform inference
        result = await engine.single_inference(request)
        
        # Update final status
        if result.status == InferenceStatus.COMPLETED:
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Inference completed successfully',
                    'predictions': result.predictions,
                    'performance_metrics': result.performance_metrics.__dict__ if result.performance_metrics else {}
                }
            )
            
            logger.info(f"Single inference completed successfully for image {image_id}")
            
            return {
                'task_id': task_id,
                'image_id': image_id,
                'model_id': model_id,
                'status': 'completed',
                'request_id': result.request_id,
                'predictions': result.predictions,
                'performance_metrics': result.performance_metrics.__dict__ if result.performance_metrics else {},
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            error_msg = result.error_message or "Inference failed with unknown error"
            logger.error(f"Inference failed: {error_msg}")
            
            self.update_state(
                state='FAILURE',
                meta={
                    'status': 'failed',
                    'progress': 0,
                    'message': f'Inference failed: {error_msg}',
                    'error': error_msg
                }
            )
            
            raise Exception(error_msg)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Single inference task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Inference failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='inference_worker.batch_inference')
def batch_inference_task(self, image_ids: List[str], model_id: str,
                        parameters: Optional[Dict[str, Any]] = None,
                        job_name: Optional[str] = None):
    """
    Perform batch inference task.
    
    Args:
        image_ids: List of image IDs to process
        model_id: ID of the model to use
        parameters: Optional inference parameters
        job_name: Optional job name
        
    Returns:
        Dict with batch inference results
    """
    task_id = self.request.id
    logger.info(f"Starting batch inference task {task_id} for {len(image_ids)} images with model {model_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing batch inference...',
                'total_images': len(image_ids),
                'processed_images': 0
            }
        )
        
        # Initialize services
        inference_service = InferenceService()
        engine = InferenceEngine()
        registry = get_model_registry()
        
        # Validate inputs
        if not image_ids:
            raise ValueError("image_ids list cannot be empty")
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        model_config = registry.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        logger.info(f"Processing {len(image_ids)} images with model: {model_config}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'loading_model',
                'progress': 5,
                'message': 'Loading model...',
                'total_images': len(image_ids),
                'processed_images': 0
            }
        )
        
        # Create image paths
        image_paths = [f"/storage/images/{image_id}" for image_id in image_ids]
        
        # Create batch inference job
        batch_job = BatchInferenceJob(
            model_id=model_id,
            image_paths=image_paths,
            parameters=parameters or {},
            job_name=job_name
        )
        
        # Set up progress callback
        results = []
        def progress_callback(completed_count: int, total_count: int, current_result: Optional[InferenceResult] = None):
            """Update Celery task progress."""
            try:
                progress_pct = min(5 + (completed_count / total_count * 85), 90)
                
                self.update_state(
                    state='RUNNING',
                    meta={
                        'status': 'processing',
                        'progress': progress_pct,
                        'total_images': total_count,
                        'processed_images': completed_count,
                        'message': f'Processed {completed_count}/{total_count} images'
                    }
                )
                
                # Store result
                if current_result:
                    results.append(current_result)
                
                logger.info(f"Batch inference progress: {completed_count}/{total_count} images processed")
            except Exception as e:
                logger.warning(f"Error updating progress: {str(e)}")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'processing',
                'progress': 10,
                'message': 'Starting batch processing...',
                'total_images': len(image_ids),
                'processed_images': 0
            }
        )
        
        # Run batch inference with progress monitoring
        logger.info("Starting batch inference...")
        batch_results = await engine.batch_inference(batch_job, progress_callback=progress_callback)
        
        # Calculate summary metrics
        successful_results = [r for r in batch_results if r.status == InferenceStatus.COMPLETED]
        failed_results = [r for r in batch_results if r.status == InferenceStatus.FAILED]
        
        # Calculate average performance metrics
        avg_inference_time = 0.0
        avg_confidence = 0.0
        total_detections = 0
        
        if successful_results:
            inference_times = [r.performance_metrics.inference_time_ms for r in successful_results if r.performance_metrics]
            if inference_times:
                avg_inference_time = sum(inference_times) / len(inference_times)
            
            # Count total detections and calculate average confidence
            confidences = []
            for result in successful_results:
                if result.predictions:
                    for pred in result.predictions:
                        if isinstance(pred, dict) and 'confidence' in pred:
                            confidences.append(pred['confidence'])
                            total_detections += 1
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        # Update final status
        if len(successful_results) == len(image_ids):
            # All successful
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Batch inference completed successfully',
                    'total_images': len(image_ids),
                    'processed_images': len(successful_results),
                    'successful_count': len(successful_results),
                    'failed_count': len(failed_results),
                    'summary_metrics': {
                        'avg_inference_time_ms': avg_inference_time,
                        'avg_confidence': avg_confidence,
                        'total_detections': total_detections
                    }
                }
            )
            
            logger.info(f"Batch inference completed successfully: {len(successful_results)}/{len(image_ids)} images")
            
        elif successful_results:
            # Partial success
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'partial_success',
                    'progress': 100,
                    'message': f'Batch inference completed with {len(failed_results)} failures',
                    'total_images': len(image_ids),
                    'processed_images': len(successful_results),
                    'successful_count': len(successful_results),
                    'failed_count': len(failed_results),
                    'summary_metrics': {
                        'avg_inference_time_ms': avg_inference_time,
                        'avg_confidence': avg_confidence,
                        'total_detections': total_detections
                    }
                }
            )
            
            logger.warning(f"Batch inference completed with failures: {len(successful_results)}/{len(image_ids)} successful")
            
        else:
            # All failed
            error_msg = "All batch inference requests failed"
            logger.error(error_msg)
            
            self.update_state(
                state='FAILURE',
                meta={
                    'status': 'failed',
                    'progress': 0,
                    'message': error_msg,
                    'total_images': len(image_ids),
                    'successful_count': 0,
                    'failed_count': len(failed_results),
                    'error': error_msg
                }
            )
            
            raise Exception(error_msg)
        
        # Prepare detailed results
        detailed_results = []
        for i, result in enumerate(batch_results):
            detailed_results.append({
                'image_id': image_ids[i],
                'status': result.status.value,
                'predictions': result.predictions,
                'performance_metrics': result.performance_metrics.__dict__ if result.performance_metrics else {},
                'error_message': result.error_message,
                'request_id': result.request_id
            })
        
        return {
            'task_id': task_id,
            'model_id': model_id,
            'job_name': job_name,
            'status': 'completed' if len(failed_results) == 0 else 'partial_success',
            'total_images': len(image_ids),
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'summary_metrics': {
                'avg_inference_time_ms': avg_inference_time,
                'avg_confidence': avg_confidence,
                'total_detections': total_detections
            },
            'results': detailed_results,
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Batch inference task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Batch inference failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='inference_worker.cancel_batch_inference')
def cancel_batch_inference_task(self, job_id: str):
    """
    Cancel a running batch inference task.
    
    Args:
        job_id: ID of the batch inference job to cancel
        
    Returns:
        Dict with cancellation status
    """
    logger.info(f"Cancelling batch inference task {job_id}")
    
    try:
        # Initialize inference service
        inference_service = InferenceService()
        
        # Cancel the job (assuming this method exists)
        result = inference_service.cancel_inference_job(job_id)
        
        if result:
            logger.info(f"Batch inference job {job_id} cancelled successfully")
            return {
                'job_id': job_id,
                'status': 'cancelled',
                'cancelled_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            logger.warning(f"Could not cancel batch inference job {job_id}")
            raise Exception(f"Could not cancel batch inference job {job_id}")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to cancel batch inference job {job_id}: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'cancellation_failed',
                'message': f'Cancellation failed: {error_msg}',
                'error': error_msg
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='inference_worker.get_inference_status')
def get_inference_status_task(self, job_id: str):
    """
    Get inference job status and progress.
    
    Args:
        job_id: ID of the inference job
        
    Returns:
        Dict with inference job status
    """
    try:
        inference_service = InferenceService()
        
        # Get job info (assuming this method exists)
        job_info = inference_service.get_inference_job(job_id)
        
        if job_info:
            return job_info
        else:
            raise Exception(f"Inference job {job_id} not found")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get inference job status {job_id}: {error_msg}")
        raise Exception(error_msg)


@celery_app.task(bind=True, name='inference_worker.warm_up_model')
def warm_up_model_task(self, model_id: str):
    """
    Warm up model by loading it and running a test inference.
    
    Args:
        model_id: ID of the model to warm up
        
    Returns:
        Dict with warm-up status
    """
    logger.info(f"Warming up model {model_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'warming_up',
                'progress': 0,
                'message': 'Warming up model...'
            }
        )
        
        # Initialize services
        engine = InferenceEngine()
        registry = get_model_registry()
        
        # Validate model exists
        model_config = registry.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'loading',
                'progress': 50,
                'message': 'Loading model...'
            }
        )
        
        # Load model (this should cache it)
        success = await engine.warm_up_model(model_id)
        
        if success:
            self.update_state(
                state='SUCCESS',
                meta={
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Model warmed up successfully'
                }
            )
            
            logger.info(f"Model {model_id} warmed up successfully")
            
            return {
                'model_id': model_id,
                'status': 'warmed_up',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
        else:
            raise Exception("Model warm-up failed")
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Model warm-up task for {model_id} failed: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Model warm-up failed: {error_msg}',
                'error': error_msg
            }
        )
        
        raise Exception(error_msg)