"""
Celery workers for ML Evaluation Platform.

This package contains distributed task workers for:
- Training: Model fine-tuning with progress monitoring
- Inference: Single and batch inference processing  
- Evaluation: Performance metrics calculation and model comparison
- Deployment: Model endpoint management and scaling
"""

from .training_worker import (
    start_training_task,
    cancel_training_task, 
    pause_training_task,
    resume_training_task,
    get_training_status_task,
    cleanup_training_job_task
)

from .inference_worker import (
    single_inference_task,
    batch_inference_task,
    cancel_batch_inference_task,
    get_inference_status_task,
    warm_up_model_task
)

from .evaluation_worker import (
    calculate_model_metrics_task,
    compare_models_task,
    benchmark_models_task,
    get_evaluation_status_task
)

from .deployment_worker import (
    create_deployment_task,
    update_deployment_task,
    scale_deployment_task,
    terminate_deployment_task,
    health_check_task,
    get_deployment_status_task,
    collect_deployment_metrics_task
)

__all__ = [
    # Training tasks
    'start_training_task',
    'cancel_training_task',
    'pause_training_task', 
    'resume_training_task',
    'get_training_status_task',
    'cleanup_training_job_task',
    
    # Inference tasks
    'single_inference_task',
    'batch_inference_task',
    'cancel_batch_inference_task',
    'get_inference_status_task',
    'warm_up_model_task',
    
    # Evaluation tasks
    'calculate_model_metrics_task',
    'compare_models_task',
    'benchmark_models_task',
    'get_evaluation_status_task',
    
    # Deployment tasks
    'create_deployment_task',
    'update_deployment_task',
    'scale_deployment_task',
    'terminate_deployment_task',
    'health_check_task',
    'get_deployment_status_task',
    'collect_deployment_metrics_task'
]