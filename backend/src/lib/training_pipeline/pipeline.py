"""
Core training pipeline for ML model fine-tuning.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import uuid
import json
from pathlib import Path

from ..ml_models import ModelType, ModelVariant


class TrainingStatus(str, Enum):
    """Status of training operations."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class HyperParameters:
    """Hyperparameters for model training."""
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 100
    weight_decay: float = 0.0005
    momentum: float = 0.9
    optimizer: str = "adam"  # adam, sgd, rmsprop
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 3
    patience: int = 10  # Early stopping patience
    
    # Data augmentation
    augment_hsv_h: float = 0.015
    augment_hsv_s: float = 0.7
    augment_hsv_v: float = 0.4
    augment_translate: float = 0.1
    augment_scale: float = 0.5
    augment_fliplr: float = 0.5
    augment_flipud: float = 0.0
    augment_mosaic: float = 1.0
    augment_mixup: float = 0.0
    
    # Model-specific parameters
    model_specific: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "patience": self.patience,
            "augment_hsv_h": self.augment_hsv_h,
            "augment_hsv_s": self.augment_hsv_s,
            "augment_hsv_v": self.augment_hsv_v,
            "augment_translate": self.augment_translate,
            "augment_scale": self.augment_scale,
            "augment_fliplr": self.augment_fliplr,
            "augment_flipud": self.augment_flipud,
            "augment_mosaic": self.augment_mosaic,
            "augment_mixup": self.augment_mixup,
            "model_specific": self.model_specific
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperParameters":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training metrics collected during training."""
    epoch: int = 0
    
    # Loss metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    box_loss: float = 0.0
    obj_loss: float = 0.0
    cls_loss: float = 0.0
    
    # Accuracy metrics
    map50: float = 0.0  # mAP@0.5
    map50_95: float = 0.0  # mAP@0.5:0.95
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Performance metrics
    learning_rate: float = 0.0
    epoch_time_seconds: float = 0.0
    eta_seconds: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Additional metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "box_loss": self.box_loss,
            "obj_loss": self.obj_loss,
            "cls_loss": self.cls_loss,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "learning_rate": self.learning_rate,
            "epoch_time_seconds": self.epoch_time_seconds,
            "eta_seconds": self.eta_seconds,
            "gpu_memory_mb": self.gpu_memory_mb,
            "metrics": self.metrics
        }


@dataclass
class TrainingConfig:
    """Configuration for training job."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Model configuration
    base_model_id: str = ""
    model_type: ModelType = ModelType.YOLO11
    variant: ModelVariant = ModelVariant.YOLO_NANO
    
    # Dataset configuration
    dataset_path: str = ""
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0
    num_classes: int = 80
    class_names: List[str] = field(default_factory=list)
    
    # Training configuration
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    resume_from_checkpoint: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Save every N epochs
    
    # Output configuration
    output_dir: str = "./training_outputs"
    experiment_name: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # Hardware configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    multi_gpu: bool = False
    mixed_precision: bool = True
    
    # Monitoring
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        if not self.experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.model_type.value}_{self.variant.value}_{timestamp}"


@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""
    checkpoint_path: str
    epoch: int
    metrics: TrainingMetrics
    timestamp: str
    model_state_size_mb: float
    is_best: bool = False


class CheckpointManager:
    """Manages training checkpoints."""
    
    def __init__(self, output_dir: str, max_checkpoints: int = 5):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[CheckpointInfo] = []
        self.best_checkpoint: Optional[CheckpointInfo] = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict[str, Any],
        metrics: TrainingMetrics,
        is_best: bool = False
    ) -> str:
        """Save a training checkpoint."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Create checkpoint filename
        if is_best:
            checkpoint_filename = "best_model.pt"
        else:
            checkpoint_filename = f"epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.output_dir / checkpoint_filename
        
        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model_state,
            "metrics": metrics.to_dict(),
            "timestamp": timestamp,
            "is_best": is_best
        }
        
        # In a real implementation, this would save to disk
        # For now, we simulate the save operation
        estimated_size_mb = len(json.dumps(checkpoint_data, default=str)) / (1024 * 1024)
        
        # Create checkpoint info
        checkpoint_info = CheckpointInfo(
            checkpoint_path=str(checkpoint_path),
            epoch=epoch,
            metrics=metrics,
            timestamp=timestamp,
            model_state_size_mb=estimated_size_mb,
            is_best=is_best
        )
        
        # Update best checkpoint if needed
        if is_best or (self.best_checkpoint is None and len(self.checkpoints) == 0):
            self.best_checkpoint = checkpoint_info
        elif self.best_checkpoint and metrics.map50_95 > self.best_checkpoint.metrics.map50_95:
            self.best_checkpoint = checkpoint_info
            checkpoint_info.is_best = True
        
        # Add to checkpoint list
        self.checkpoints.append(checkpoint_info)
        
        # Maintain max checkpoints limit (keep best + most recent)
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by epoch (keep most recent) and remove old ones
            sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.epoch)
            checkpoints_to_keep = []
            
            # Always keep the best checkpoint
            if self.best_checkpoint:
                checkpoints_to_keep.append(self.best_checkpoint)
            
            # Keep the most recent checkpoints
            recent_checkpoints = [cp for cp in sorted_checkpoints if not cp.is_best]
            checkpoints_to_keep.extend(recent_checkpoints[-(self.max_checkpoints-1):])
            
            self.checkpoints = checkpoints_to_keep
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from disk."""
        # In a real implementation, this would load from disk
        # For now, return None to simulate file not found
        return None
    
    def get_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the best checkpoint."""
        return self.best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x.epoch)
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x.epoch)


@dataclass
class TrainingJob:
    """Training job with progress tracking."""
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    
    # Progress tracking
    current_epoch: int = 0
    total_epochs: int = 0
    progress_percentage: float = 0.0
    
    # Metrics history
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    latest_metrics: Optional[TrainingMetrics] = None
    best_metrics: Optional[TrainingMetrics] = None
    
    # Checkpoint management
    checkpoint_manager: Optional[CheckpointManager] = None
    
    # Timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Callbacks
    progress_callbacks: List[Callable[["TrainingJob"], None]] = field(default_factory=list)
    
    def __post_init__(self):
        self.total_epochs = self.config.hyperparameters.epochs
        if self.config.save_checkpoints and not self.checkpoint_manager:
            self.checkpoint_manager = CheckpointManager(
                f"{self.config.output_dir}/{self.config.experiment_name}/checkpoints"
            )
    
    @property
    def job_id(self) -> str:
        """Get job ID."""
        return self.config.job_id
    
    @property
    def is_active(self) -> bool:
        """Check if job is currently active."""
        return self.status in [
            TrainingStatus.PREPARING,
            TrainingStatus.TRAINING,
            TrainingStatus.VALIDATING
        ]
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status in [
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED
        ]
    
    def update_progress(self, epoch: int, metrics: Optional[TrainingMetrics] = None) -> None:
        """Update job progress."""
        self.current_epoch = epoch
        self.progress_percentage = (epoch / self.total_epochs) * 100.0
        
        if metrics:
            self.latest_metrics = metrics
            self.metrics_history.append(metrics)
            
            # Update best metrics
            if (self.best_metrics is None or 
                (metrics.map50_95 > self.best_metrics.map50_95)):
                self.best_metrics = metrics
                
                # Save best checkpoint
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        epoch, {}, metrics, is_best=True
                    )
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self)
            except Exception:
                pass  # Don't let callback errors stop training
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.warnings.append(f"[{timestamp}] {message}")
    
    def start_training(self) -> None:
        """Mark training as started."""
        self.status = TrainingStatus.TRAINING
        self.started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    def complete_training(self) -> None:
        """Mark training as completed."""
        self.status = TrainingStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.progress_percentage = 100.0
    
    def fail_training(self, error_message: str) -> None:
        """Mark training as failed."""
        self.status = TrainingStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    def cancel_training(self) -> None:
        """Cancel training."""
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    def pause_training(self) -> None:
        """Pause training."""
        if self.is_active:
            self.status = TrainingStatus.PAUSED
    
    def resume_training(self) -> None:
        """Resume paused training."""
        if self.status == TrainingStatus.PAUSED:
            self.status = TrainingStatus.TRAINING
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        duration_seconds = 0.0
        if self.started_at and self.completed_at:
            start_dt = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
            duration_seconds = (end_dt - start_dt).total_seconds()
        
        return {
            "job_id": self.job_id,
            "status": self.status,
            "config": {
                "model_type": self.config.model_type,
                "variant": self.config.variant,
                "experiment_name": self.config.experiment_name,
                "dataset_path": self.config.dataset_path,
                "num_classes": self.config.num_classes
            },
            "progress": {
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress_percentage": self.progress_percentage
            },
            "metrics": {
                "latest": self.latest_metrics.to_dict() if self.latest_metrics else None,
                "best": self.best_metrics.to_dict() if self.best_metrics else None,
                "history_length": len(self.metrics_history)
            },
            "checkpoints": {
                "total_checkpoints": len(self.checkpoint_manager.checkpoints) if self.checkpoint_manager else 0,
                "best_checkpoint": self.checkpoint_manager.get_best_checkpoint() if self.checkpoint_manager else None,
                "latest_checkpoint": self.checkpoint_manager.get_latest_checkpoint() if self.checkpoint_manager else None
            },
            "timing": {
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_seconds": duration_seconds
            },
            "error_message": self.error_message,
            "warnings_count": len(self.warnings)
        }


class TrainingPipeline:
    """Main training pipeline for ML models."""
    
    def __init__(self):
        self._active_jobs: Dict[str, TrainingJob] = {}
        self._completed_jobs: Dict[str, TrainingJob] = {}
    
    def create_training_job(self, config: TrainingConfig) -> TrainingJob:
        """Create a new training job."""
        job = TrainingJob(config=config)
        self._active_jobs[job.job_id] = job
        return job
    
    async def start_training(self, job: TrainingJob) -> TrainingJob:
        """Start training for a job."""
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"Job {job.job_id} is not in PENDING status")
        
        job.status = TrainingStatus.PREPARING
        job.start_training()
        
        try:
            # Simulate training process
            await self._run_training_loop(job)
            
            if job.status == TrainingStatus.TRAINING:  # Not cancelled
                job.complete_training()
                
        except Exception as e:
            job.fail_training(str(e))
        
        # Move to completed jobs
        if job.job_id in self._active_jobs:
            del self._active_jobs[job.job_id]
            self._completed_jobs[job.job_id] = job
        
        return job
    
    async def _run_training_loop(self, job: TrainingJob) -> None:
        """Run the actual training loop (simulated)."""
        import asyncio
        import random
        
        for epoch in range(1, job.total_epochs + 1):
            # Check for cancellation
            if job.status == TrainingStatus.CANCELLED:
                break
            
            # Simulate epoch training
            await asyncio.sleep(0.1)  # Simulate training time
            
            # Create simulated metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=max(0.1, 2.0 - (epoch * 0.02) + random.uniform(-0.1, 0.1)),
                val_loss=max(0.1, 1.8 - (epoch * 0.018) + random.uniform(-0.1, 0.1)),
                map50=min(0.95, epoch * 0.008 + random.uniform(0, 0.02)),
                map50_95=min(0.85, epoch * 0.006 + random.uniform(0, 0.015)),
                precision=min(0.98, 0.5 + epoch * 0.005 + random.uniform(0, 0.01)),
                recall=min(0.95, 0.4 + epoch * 0.006 + random.uniform(0, 0.01)),
                learning_rate=job.config.hyperparameters.learning_rate * (0.95 ** (epoch // 10)),
                epoch_time_seconds=random.uniform(15, 30),
                gpu_memory_mb=random.uniform(2000, 4000)
            )
            
            # Update job progress
            job.update_progress(epoch, metrics)
            
            # Save checkpoint periodically
            if (job.checkpoint_manager and 
                epoch % job.config.checkpoint_frequency == 0):
                job.checkpoint_manager.save_checkpoint(epoch, {}, metrics)
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return (self._active_jobs.get(job_id) or 
                self._completed_jobs.get(job_id))
    
    def list_active_jobs(self) -> List[TrainingJob]:
        """List all active training jobs."""
        return list(self._active_jobs.values())
    
    def list_completed_jobs(self) -> List[TrainingJob]:
        """List completed training jobs."""
        return list(self._completed_jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = self.get_job(job_id)
        if job and job.is_active:
            job.cancel_training()
            return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a training job."""
        job = self.get_job(job_id)
        if job and job.is_active:
            job.pause_training()
            return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a paused training job."""
        job = self.get_job(job_id)
        if job and job.status == TrainingStatus.PAUSED:
            job.resume_training()
            return True
        return False