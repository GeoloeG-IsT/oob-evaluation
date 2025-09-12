"""
Training Pipeline Library (T047)

This library provides model fine-tuning capabilities including:
- Model fine-tuning for YOLO11/12, RT-DETR, and SAM2
- Progress tracking and monitoring
- Hyperparameter configuration  
- Training job management
- Dataset preparation and validation
- Training metrics collection

Features:
- Training job creation and management
- Progress monitoring with real-time updates
- Hyperparameter optimization
- Training data validation
- Checkpoint management
- Training metrics and visualization
- Model export and deployment preparation
"""

from .pipeline import (
    TrainingPipeline,
    TrainingJob,
    TrainingConfig,
    TrainingStatus,
    HyperParameters,
    TrainingMetrics,
    CheckpointManager,
)
from .jobs import (
    TrainingJobManager,
    JobScheduler,
    JobProgress,
)
from .datasets import (
    DatasetManager,
    DatasetConfig,
    DatasetSplit,
    DatasetValidator,
)
from .optimizers import (
    HyperParameterOptimizer,
    OptimizationConfig,
    OptimizerType,
)
from .monitoring import (
    TrainingMonitor,
    MetricsCollector,
    TrainingLogger,
)

__all__ = [
    "TrainingPipeline",
    "TrainingJob",
    "TrainingConfig", 
    "TrainingStatus",
    "HyperParameters",
    "TrainingMetrics",
    "CheckpointManager",
    "TrainingJobManager",
    "JobScheduler",
    "JobProgress",
    "DatasetManager",
    "DatasetConfig",
    "DatasetSplit",
    "DatasetValidator",
    "HyperParameterOptimizer",
    "OptimizationConfig",
    "OptimizerType",
    "TrainingMonitor",
    "MetricsCollector",
    "TrainingLogger",
]