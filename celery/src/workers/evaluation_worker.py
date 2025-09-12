"""
Evaluation worker for performance metrics calculation tasks.

Handles async evaluation jobs using Celery for distributed processing.
Integrates with backend evaluation service and ML evaluation libraries.
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

from services.evaluation_service import EvaluationService
from lib.ml_models import get_model_registry

# Configure logging
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='evaluation_worker.calculate_model_metrics')
def calculate_model_metrics_task(self, model_id: str, dataset_id: str,
                               iou_thresholds: Optional[List[float]] = None,
                               confidence_threshold: float = 0.5):
    """
    Calculate comprehensive performance metrics for a model on a dataset.
    
    Args:
        model_id: ID of the model to evaluate
        dataset_id: ID of the dataset to evaluate against
        iou_thresholds: List of IoU thresholds for mAP calculation
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Dict with evaluation results
    """
    task_id = self.request.id
    logger.info(f"Starting evaluation task {task_id} for model {model_id} on dataset {dataset_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing evaluation...'
            }
        )
        
        # Initialize services
        evaluation_service = EvaluationService()
        registry = get_model_registry()
        
        # Validate inputs
        if not model_id:
            raise ValueError("model_id is required")
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        # Validate model exists
        model_config = registry.get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model {model_id} not found in registry")
        
        logger.info(f"Evaluating model: {model_config}")
        
        # Default IoU thresholds if not provided
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'loading_model',
                'progress': 10,
                'message': 'Loading model and dataset...'
            }
        )
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'running_inference',
                'progress': 30,
                'message': 'Running inference on dataset...'
            }
        )
        
        # Calculate metrics (this will run inference internally)
        results = evaluation_service.calculate_model_metrics(
            model_id=model_id,
            dataset_id=dataset_id,
            iou_thresholds=iou_thresholds,
            confidence_threshold=confidence_threshold
        )
        
        # Update progress
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'calculating_metrics',
                'progress': 70,
                'message': 'Calculating performance metrics...'
            }
        )
        
        # Add detailed metrics breakdown
        detailed_metrics = {
            'mAP@0.5': results['metrics'].get('mAP@0.5', 0.0),
            'mAP@0.5:0.95': results['metrics'].get('mAP@0.5:0.95', 0.0),
            'precision': results['metrics'].get('precision', 0.0),
            'recall': results['metrics'].get('recall', 0.0),
            'f1_score': results['metrics'].get('f1_score', 0.0),
            'execution_time_ms': results.get('execution_time_ms', 0.0),
            'fps': results.get('fps', 0.0)
        }
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'completed',
                'progress': 100,
                'message': 'Evaluation completed successfully',
                'metrics': detailed_metrics,
                'iou_thresholds': iou_thresholds,
                'confidence_threshold': confidence_threshold
            }
        )
        
        logger.info(f"Evaluation completed successfully for model {model_id}")
        logger.info(f"Results: mAP@0.5={detailed_metrics['mAP@0.5']:.4f}, mAP@0.5:0.95={detailed_metrics['mAP@0.5:0.95']:.4f}")
        
        return {
            'task_id': task_id,
            'model_id': model_id,
            'dataset_id': dataset_id,
            'status': 'completed',
            'metrics': detailed_metrics,
            'evaluation_config': {
                'iou_thresholds': iou_thresholds,
                'confidence_threshold': confidence_threshold
            },
            'completed_at': datetime.now(timezone.utc).isoformat(),
            **results
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Evaluation task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Evaluation failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='evaluation_worker.compare_models')
def compare_models_task(self, model_ids: List[str], dataset_id: str,
                       iou_thresholds: Optional[List[float]] = None,
                       confidence_threshold: float = 0.5):
    """
    Compare performance of multiple models on a dataset.
    
    Args:
        model_ids: List of model IDs to compare
        dataset_id: ID of the dataset to evaluate against
        iou_thresholds: List of IoU thresholds for mAP calculation
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Dict with comparison results
    """
    task_id = self.request.id
    logger.info(f"Starting model comparison task {task_id} for {len(model_ids)} models on dataset {dataset_id}")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing model comparison...',
                'total_models': len(model_ids),
                'evaluated_models': 0
            }
        )
        
        # Initialize services
        evaluation_service = EvaluationService()
        registry = get_model_registry()
        
        # Validate inputs
        if not model_ids:
            raise ValueError("model_ids list cannot be empty")
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        # Default IoU thresholds if not provided
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        comparison_results = {
            'task_id': task_id,
            'dataset_id': dataset_id,
            'evaluation_config': {
                'iou_thresholds': iou_thresholds,
                'confidence_threshold': confidence_threshold
            },
            'model_results': {},
            'comparison_summary': {},
            'started_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Evaluate each model
        for i, model_id in enumerate(model_ids):
            try:
                logger.info(f"Evaluating model {i+1}/{len(model_ids)}: {model_id}")
                
                # Update progress
                progress = 10 + (i / len(model_ids) * 80)
                self.update_state(
                    state='RUNNING',
                    meta={
                        'status': 'evaluating',
                        'progress': progress,
                        'message': f'Evaluating model {i+1}/{len(model_ids)}: {model_id}',
                        'total_models': len(model_ids),
                        'evaluated_models': i,
                        'current_model': model_id
                    }
                )
                
                # Validate model exists
                model_config = registry.get_model_config(model_id)
                if not model_config:
                    logger.warning(f"Model {model_id} not found in registry, skipping")
                    comparison_results['model_results'][model_id] = {
                        'status': 'failed',
                        'error': f"Model {model_id} not found"
                    }
                    continue
                
                # Calculate metrics for this model
                results = evaluation_service.calculate_model_metrics(
                    model_id=model_id,
                    dataset_id=dataset_id,
                    iou_thresholds=iou_thresholds,
                    confidence_threshold=confidence_threshold
                )
                
                comparison_results['model_results'][model_id] = {
                    'status': 'completed',
                    'model_name': model_config.get('name', model_id),
                    'model_type': model_config.get('type', 'unknown'),
                    'framework': model_config.get('framework', 'unknown'),
                    'metrics': results['metrics'],
                    'evaluation_timestamp': results['evaluation_timestamp']
                }
                
                logger.info(f"Model {model_id} evaluation completed: mAP@0.5={results['metrics'].get('mAP@0.5', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_id}: {str(e)}")
                comparison_results['model_results'][model_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate comparison summary
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'summarizing',
                'progress': 95,
                'message': 'Generating comparison summary...',
                'total_models': len(model_ids),
                'evaluated_models': len(model_ids)
            }
        )
        
        successful_results = {k: v for k, v in comparison_results['model_results'].items() if v['status'] == 'completed'}
        
        if successful_results:
            # Find best performing models
            best_map_50 = max(successful_results.items(), key=lambda x: x[1]['metrics'].get('mAP@0.5', 0.0))
            best_map_50_95 = max(successful_results.items(), key=lambda x: x[1]['metrics'].get('mAP@0.5:0.95', 0.0))
            fastest_model = min(successful_results.items(), key=lambda x: x[1]['metrics'].get('execution_time_ms', float('inf')))
            
            comparison_results['comparison_summary'] = {
                'total_models_evaluated': len(successful_results),
                'failed_evaluations': len(model_ids) - len(successful_results),
                'best_map_50': {
                    'model_id': best_map_50[0],
                    'model_name': best_map_50[1]['model_name'],
                    'value': best_map_50[1]['metrics'].get('mAP@0.5', 0.0)
                },
                'best_map_50_95': {
                    'model_id': best_map_50_95[0],
                    'model_name': best_map_50_95[1]['model_name'],
                    'value': best_map_50_95[1]['metrics'].get('mAP@0.5:0.95', 0.0)
                },
                'fastest_model': {
                    'model_id': fastest_model[0],
                    'model_name': fastest_model[1]['model_name'],
                    'execution_time_ms': fastest_model[1]['metrics'].get('execution_time_ms', 0.0)
                },
                'avg_map_50': sum(r['metrics'].get('mAP@0.5', 0.0) for r in successful_results.values()) / len(successful_results),
                'avg_map_50_95': sum(r['metrics'].get('mAP@0.5:0.95', 0.0) for r in successful_results.values()) / len(successful_results)
            }
        
        comparison_results['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'completed',
                'progress': 100,
                'message': 'Model comparison completed successfully',
                'total_models': len(model_ids),
                'evaluated_models': len(successful_results),
                'failed_models': len(model_ids) - len(successful_results),
                'summary': comparison_results['comparison_summary']
            }
        )
        
        logger.info(f"Model comparison completed: {len(successful_results)}/{len(model_ids)} models evaluated successfully")
        if successful_results:
            logger.info(f"Best mAP@0.5: {comparison_results['comparison_summary']['best_map_50']['model_name']} ({comparison_results['comparison_summary']['best_map_50']['value']:.4f})")
        
        return comparison_results
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Model comparison task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Model comparison failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='evaluation_worker.benchmark_models')
def benchmark_models_task(self, model_ids: List[str], benchmark_datasets: List[str],
                         iou_thresholds: Optional[List[float]] = None):
    """
    Run comprehensive benchmark evaluation across multiple models and datasets.
    
    Args:
        model_ids: List of model IDs to benchmark
        benchmark_datasets: List of dataset IDs for benchmarking
        iou_thresholds: List of IoU thresholds for mAP calculation
        
    Returns:
        Dict with comprehensive benchmark results
    """
    task_id = self.request.id
    logger.info(f"Starting benchmark task {task_id} for {len(model_ids)} models on {len(benchmark_datasets)} datasets")
    
    try:
        # Update task status
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'initializing',
                'progress': 0,
                'message': 'Initializing comprehensive benchmark...',
                'total_evaluations': len(model_ids) * len(benchmark_datasets),
                'completed_evaluations': 0
            }
        )
        
        # Initialize services
        evaluation_service = EvaluationService()
        
        # Default IoU thresholds if not provided
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        benchmark_results = {
            'task_id': task_id,
            'model_ids': model_ids,
            'benchmark_datasets': benchmark_datasets,
            'evaluation_config': {
                'iou_thresholds': iou_thresholds,
                'confidence_threshold': 0.5
            },
            'results_matrix': {},
            'model_summaries': {},
            'dataset_summaries': {},
            'overall_summary': {},
            'started_at': datetime.now(timezone.utc).isoformat()
        }
        
        total_evaluations = len(model_ids) * len(benchmark_datasets)
        completed_evaluations = 0
        
        # Run evaluation for each model-dataset combination
        for model_id in model_ids:
            benchmark_results['results_matrix'][model_id] = {}
            
            for dataset_id in benchmark_datasets:
                try:
                    logger.info(f"Evaluating model {model_id} on dataset {dataset_id}")
                    
                    # Update progress
                    progress = (completed_evaluations / total_evaluations) * 90
                    self.update_state(
                        state='RUNNING',
                        meta={
                            'status': 'evaluating',
                            'progress': progress,
                            'message': f'Evaluating {model_id} on {dataset_id}',
                            'total_evaluations': total_evaluations,
                            'completed_evaluations': completed_evaluations,
                            'current_model': model_id,
                            'current_dataset': dataset_id
                        }
                    )
                    
                    # Run evaluation
                    results = evaluation_service.calculate_model_metrics(
                        model_id=model_id,
                        dataset_id=dataset_id,
                        iou_thresholds=iou_thresholds
                    )
                    
                    benchmark_results['results_matrix'][model_id][dataset_id] = {
                        'status': 'completed',
                        'metrics': results['metrics'],
                        'evaluation_timestamp': results['evaluation_timestamp']
                    }
                    
                    completed_evaluations += 1
                    logger.info(f"Completed evaluation {completed_evaluations}/{total_evaluations}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_id} on dataset {dataset_id}: {str(e)}")
                    benchmark_results['results_matrix'][model_id][dataset_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    completed_evaluations += 1
        
        # Generate summaries
        self.update_state(
            state='RUNNING',
            meta={
                'status': 'summarizing',
                'progress': 95,
                'message': 'Generating benchmark summaries...'
            }
        )
        
        # Model summaries (average performance across datasets)
        for model_id in model_ids:
            model_results = benchmark_results['results_matrix'][model_id]
            successful_results = [r for r in model_results.values() if r['status'] == 'completed']
            
            if successful_results:
                avg_map_50 = sum(r['metrics'].get('mAP@0.5', 0.0) for r in successful_results) / len(successful_results)
                avg_map_50_95 = sum(r['metrics'].get('mAP@0.5:0.95', 0.0) for r in successful_results) / len(successful_results)
                
                benchmark_results['model_summaries'][model_id] = {
                    'datasets_evaluated': len(successful_results),
                    'avg_mAP@0.5': avg_map_50,
                    'avg_mAP@0.5:0.95': avg_map_50_95,
                    'best_mAP@0.5': max(r['metrics'].get('mAP@0.5', 0.0) for r in successful_results),
                    'worst_mAP@0.5': min(r['metrics'].get('mAP@0.5', 0.0) for r in successful_results)
                }
        
        # Dataset summaries (average performance of all models)
        for dataset_id in benchmark_datasets:
            dataset_results = []
            for model_id in model_ids:
                if dataset_id in benchmark_results['results_matrix'][model_id]:
                    result = benchmark_results['results_matrix'][model_id][dataset_id]
                    if result['status'] == 'completed':
                        dataset_results.append(result)
            
            if dataset_results:
                avg_map_50 = sum(r['metrics'].get('mAP@0.5', 0.0) for r in dataset_results) / len(dataset_results)
                avg_map_50_95 = sum(r['metrics'].get('mAP@0.5:0.95', 0.0) for r in dataset_results) / len(dataset_results)
                
                benchmark_results['dataset_summaries'][dataset_id] = {
                    'models_evaluated': len(dataset_results),
                    'avg_mAP@0.5': avg_map_50,
                    'avg_mAP@0.5:0.95': avg_map_50_95,
                    'best_mAP@0.5': max(r['metrics'].get('mAP@0.5', 0.0) for r in dataset_results),
                    'worst_mAP@0.5': min(r['metrics'].get('mAP@0.5', 0.0) for r in dataset_results)
                }
        
        # Overall summary
        all_successful_results = []
        for model_results in benchmark_results['results_matrix'].values():
            for result in model_results.values():
                if result['status'] == 'completed':
                    all_successful_results.append(result)
        
        if all_successful_results:
            benchmark_results['overall_summary'] = {
                'total_evaluations': total_evaluations,
                'successful_evaluations': len(all_successful_results),
                'failed_evaluations': total_evaluations - len(all_successful_results),
                'overall_avg_mAP@0.5': sum(r['metrics'].get('mAP@0.5', 0.0) for r in all_successful_results) / len(all_successful_results),
                'overall_avg_mAP@0.5:0.95': sum(r['metrics'].get('mAP@0.5:0.95', 0.0) for r in all_successful_results) / len(all_successful_results),
                'best_overall_mAP@0.5': max(r['metrics'].get('mAP@0.5', 0.0) for r in all_successful_results),
                'worst_overall_mAP@0.5': min(r['metrics'].get('mAP@0.5', 0.0) for r in all_successful_results)
            }
        
        benchmark_results['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        # Update final status
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'completed',
                'progress': 100,
                'message': 'Benchmark evaluation completed successfully',
                'total_evaluations': total_evaluations,
                'successful_evaluations': len(all_successful_results),
                'failed_evaluations': total_evaluations - len(all_successful_results)
            }
        )
        
        logger.info(f"Benchmark evaluation completed: {len(all_successful_results)}/{total_evaluations} evaluations successful")
        
        return benchmark_results
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Benchmark task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'progress': 0,
                'message': f'Benchmark failed: {error_msg}',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
        )
        
        raise Ignore()


@celery_app.task(bind=True, name='evaluation_worker.get_evaluation_status')
def get_evaluation_status_task(self, evaluation_id: str):
    """
    Get evaluation job status and progress.
    
    Args:
        evaluation_id: ID of the evaluation job
        
    Returns:
        Dict with evaluation job status
    """
    try:
        # For now, return basic status
        # In real implementation, this would query the evaluation service
        return {
            'evaluation_id': evaluation_id,
            'status': 'running',  # This would be retrieved from storage
            'progress': 50,
            'message': 'Evaluation in progress...'
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to get evaluation status {evaluation_id}: {error_msg}")
        raise Exception(error_msg)