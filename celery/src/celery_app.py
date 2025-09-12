"""
Celery application configuration for ML Evaluation Platform.

Handles async tasks for:
- Model training and fine-tuning
- Batch inference processing  
- Performance evaluation
- Model deployment
"""

from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Celery application
celery_app = Celery(
    "ml_eval_platform",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=[
        "src.workers.training_worker",
        "src.workers.inference_worker", 
        "src.workers.evaluation_worker",
        "src.workers.deployment_worker",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Task routing
    task_routes={
        "src.workers.training_worker.*": {"queue": "training"},
        "src.workers.inference_worker.*": {"queue": "inference"},
        "src.workers.evaluation_worker.*": {"queue": "evaluation"},
        "src.workers.deployment_worker.*": {"queue": "deployment"},
    },
    
    # Result backend settings
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Task autodiscovery
celery_app.autodiscover_tasks()


if __name__ == "__main__":
    celery_app.start()