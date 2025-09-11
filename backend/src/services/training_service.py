"""
Training service for model fine-tuning.
"""
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from datetime import datetime, timezone

from ..models.training_job import TrainingJobModel, training_job_storage
from ..models.dataset import dataset_storage
from ..models.model import model_storage
from ..lib.training_pipeline import (
    TrainingPipeline, TrainingConfig, HyperParameters,
    TrainingStatus, TrainingJob, JobScheduler, SchedulerConfig, JobPriority
)


class TrainingService:
    """Service for handling training operations."""
    
    def __init__(self, max_concurrent_jobs: int = 2):
        self.storage = training_job_storage
        self.dataset_storage = dataset_storage
        self.model_storage = model_storage
        
        # Initialize training pipeline and scheduler
        self.pipeline = TrainingPipeline()
        scheduler_config = SchedulerConfig(max_concurrent_jobs=max_concurrent_jobs)
        self.scheduler = JobScheduler(scheduler_config)
        
        # Start scheduler
        asyncio.create_task(self._start_scheduler_if_needed())
    
    async def _start_scheduler_if_needed(self) -> None:
        """Start the job scheduler if not already running."""
        if not self.scheduler._scheduler_running:
            await self.scheduler.start()
    
    def create_training_job(self, base_model_id: str, dataset_id: str,
                          hyperparameters: Dict[str, Any],
                          job_name: Optional[str] = None,
                          priority: str = "normal") -> Dict[str, Any]:
        """Create a new training job."""
        # Validate required fields
        if not base_model_id:
            raise ValueError("base_model_id is required")
        if not dataset_id:
            raise ValueError("dataset_id is required")
        if not hyperparameters:
            raise ValueError("hyperparameters are required")
        
        # Validate base model exists
        base_model = self.model_storage.get_by_id(base_model_id)
        if not base_model:
            raise ValueError(f"Base model {base_model_id} not found")
        
        # Validate dataset exists
        dataset = self.dataset_storage.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Validate priority
        valid_priorities = ["low", "normal", "high", "urgent"]
        if priority not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}")
        
        # Create hyperparameters object
        try:
            hyper_params = HyperParameters.from_dict(hyperparameters)
        except Exception as e:
            raise ValueError(f"Invalid hyperparameters: {str(e)}")
        
        # Create training job model for storage
        job_model = TrainingJobModel(
            base_model_id=base_model_id,
            dataset_id=dataset_id,
            hyperparameters=hyperparameters,
            status="queued",
            metadata={
                "job_name": job_name,
                "priority": priority,
                "base_model_name": base_model.name,
                "base_model_type": base_model.type,
                "dataset_name": dataset.name
            }
        )
        
        # Save to storage
        saved_job = self.storage.save(job_model)
        
        # Create training config
        training_config = TrainingConfig(
            base_model_id=base_model_id,
            dataset_path=dataset.data_path,
            output_dir=f"/models/training/{saved_job.id}",
            hyperparameters=hyper_params,
            model_type=base_model.type,
            num_classes=dataset.num_classes or 80
        )
        
        # Create training job for pipeline
        training_job = TrainingJob(
            job_id=saved_job.id,
            config=training_config
        )
        
        # Submit to scheduler
        job_priority = JobPriority(priority)
        success = self.scheduler.submit_job(training_job, job_priority)
        
        if not success:
            # Update job status to failed if couldn't queue
            self.storage.update_status(saved_job.id, "failed", 
                                     execution_logs="Failed to queue job - scheduler queue is full")
            raise RuntimeError("Training queue is full. Please try again later.")
        
        # Start monitoring
        asyncio.create_task(self._monitor_training_job(saved_job.id))
        
        return {
            "job_id": saved_job.id,
            "base_model_id": base_model_id,
            "dataset_id": dataset_id,
            "status": "queued",
            "priority": priority,
            "job_name": job_name,
            "created_at": saved_job.created_at,
            "hyperparameters": hyperparameters
        }
    
    async def _monitor_training_job(self, job_id: str) -> None:
        """Monitor training job progress and update storage."""
        while True:
            try:
                # Get job progress from scheduler
                progress = self.scheduler.get_job_progress(job_id)
                if not progress:
                    # Job might be completed or failed
                    break
                
                # Update storage with progress
                logs = f"Epoch {progress.current_epoch}/{progress.total_epochs} - "
                logs += f"Train Loss: {progress.latest_train_loss:.4f}, "
                logs += f"Val Loss: {progress.latest_val_loss:.4f}, "
                logs += f"mAP@0.5: {progress.latest_map50:.4f}"
                
                self.storage.update_status(
                    job_id,
                    progress.status.value,
                    progress.progress_percentage,
                    execution_logs=logs
                )
                
                # If job is complete, get result model
                if progress.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                    if progress.status == TrainingStatus.COMPLETED:
                        # Create result model
                        base_job = self.storage.get_by_id(job_id)
                        if base_job:
                            result_model = await self._create_result_model(base_job, progress)
                            if result_model:
                                self.storage.update_status(
                                    job_id, 
                                    "completed", 
                                    100.0,
                                    execution_logs=logs,
                                    result_model_id=result_model.id
                                )
                    break
                
                # Wait before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Error monitoring training job {job_id}: {str(e)}")
                break
    
    async def _create_result_model(self, training_job: TrainingJobModel, 
                                 progress) -> Optional[Any]:
        """Create a result model from completed training."""
        try:
            base_model = self.model_storage.get_by_id(training_job.base_model_id)
            if not base_model:
                return None
            
            # Create new model with training results
            from ..models.model import ModelModel
            
            result_model = ModelModel(
                name=f"{base_model.name}-finetuned-{training_job.id[:8]}",
                type=base_model.type,
                variant=base_model.variant,
                version=f"{base_model.version}-ft-{training_job.id[:8]}",
                framework=base_model.framework,
                model_path=f"/models/training/{training_job.id}/best.pt",
                training_status="trained",
                performance_metrics={
                    "mAP@0.5": progress.best_map50,
                    "final_train_loss": progress.latest_train_loss,
                    "final_val_loss": progress.latest_val_loss,
                    "training_epochs": progress.current_epoch,
                    "base_model_id": training_job.base_model_id,
                    "dataset_id": training_job.dataset_id
                },
                metadata={
                    "training_job_id": training_job.id,
                    "hyperparameters": training_job.hyperparameters,
                    "is_finetuned": True
                }
            )
            
            return self.model_storage.save(result_model)
            
        except Exception as e:
            print(f"Error creating result model: {str(e)}")
            return None
    
    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job by ID."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return None
        
        # Get current progress from scheduler if job is running
        progress_info = {}
        if job.status in ["queued", "running"]:
            progress = self.scheduler.get_job_progress(job_id)
            if progress:
                progress_info = {
                    "current_epoch": progress.current_epoch,
                    "total_epochs": progress.total_epochs,
                    "elapsed_time_seconds": progress.elapsed_time_seconds,
                    "estimated_remaining_seconds": progress.estimated_remaining_seconds,
                    "latest_metrics": {
                        "train_loss": progress.latest_train_loss,
                        "val_loss": progress.latest_val_loss,
                        "map50": progress.latest_map50,
                        "best_map50": progress.best_map50
                    },
                    "resource_usage": {
                        "gpu_memory_mb": progress.gpu_memory_usage_mb,
                        "cpu_percent": progress.cpu_usage_percent
                    }
                }
        
        return {
            "job_id": job.id,
            "base_model_id": job.base_model_id,
            "dataset_id": job.dataset_id,
            "status": job.status,
            "progress_percentage": job.progress_percentage,
            "hyperparameters": job.hyperparameters,
            "execution_logs": job.execution_logs,
            "result_model_id": job.result_model_id,
            "created_at": job.created_at,
            "start_time": job.start_time,
            "end_time": job.end_time,
            "metadata": job.metadata,
            **progress_info
        }
    
    def list_training_jobs(self, status: Optional[str] = None,
                          base_model_id: Optional[str] = None,
                          dataset_id: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """List training jobs with optional filtering."""
        jobs, total_count = self.storage.list_training_jobs(
            status=status,
            base_model_id=base_model_id,
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        job_list = []
        for job in jobs:
            job_dict = {
                "job_id": job.id,
                "base_model_id": job.base_model_id,
                "dataset_id": job.dataset_id,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "created_at": job.created_at,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "metadata": job.metadata
            }
            
            # Add current progress for running jobs
            if job.status in ["queued", "running"]:
                progress = self.scheduler.get_job_progress(job.id)
                if progress:
                    job_dict["current_epoch"] = progress.current_epoch
                    job_dict["total_epochs"] = progress.total_epochs
                    job_dict["latest_map50"] = progress.latest_map50
            
            job_list.append(job_dict)
        
        return job_list, total_count
    
    def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return False
        
        if job.status not in ["queued", "running"]:
            return False
        
        # Cancel job in scheduler
        success = self.scheduler.cancel_job(job_id)
        
        if success:
            # Update storage
            self.storage.update_status(
                job_id, 
                "cancelled",
                execution_logs="Job cancelled by user"
            )
        
        return success
    
    def pause_training_job(self, job_id: str) -> bool:
        """Pause a training job."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return False
        
        if job.status != "running":
            return False
        
        # Pause job in scheduler
        success = self.scheduler.pause_job(job_id)
        
        if success:
            self.storage.update_status(
                job_id,
                "paused",
                execution_logs="Job paused by user"
            )
        
        return success
    
    def resume_training_job(self, job_id: str) -> bool:
        """Resume a paused training job."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return False
        
        if job.status != "paused":
            return False
        
        # Resume job in scheduler
        success = self.scheduler.resume_job(job_id)
        
        if success:
            self.storage.update_status(
                job_id,
                "running",
                execution_logs="Job resumed by user"
            )
        
        return success
    
    def get_training_queue_status(self) -> Dict[str, Any]:
        """Get current training queue status."""
        return self.scheduler.get_queue_status()
    
    def get_hyperparameter_recommendations(self, base_model_id: str, 
                                         dataset_id: str,
                                         training_goal: str = "balanced") -> Dict[str, Any]:
        """Get hyperparameter recommendations for training."""
        # Get base model info
        base_model = self.model_storage.get_by_id(base_model_id)
        if not base_model:
            return {"error": "Base model not found"}
        
        # Get dataset info
        dataset = self.dataset_storage.get_by_id(dataset_id)
        if not dataset:
            return {"error": "Dataset not found"}
        
        # Base recommendations
        recommendations = HyperParameters()
        
        # Adjust based on model type
        if base_model.framework == "YOLO11" or base_model.framework == "YOLO12":
            if "nano" in base_model.variant.lower():
                recommendations.learning_rate = 0.01
                recommendations.batch_size = 64
            elif "small" in base_model.variant.lower():
                recommendations.learning_rate = 0.001
                recommendations.batch_size = 32
            elif "large" in base_model.variant.lower() or "xl" in base_model.variant.lower():
                recommendations.learning_rate = 0.0005
                recommendations.batch_size = 8
        elif base_model.framework == "RT-DETR":
            recommendations.learning_rate = 0.0001
            recommendations.batch_size = 16
            recommendations.optimizer = "adamw"
        elif base_model.framework == "SAM2":
            recommendations.learning_rate = 0.00001
            recommendations.batch_size = 4
            recommendations.optimizer = "adamw"
        
        # Adjust based on dataset size
        if dataset.image_count:
            if dataset.image_count < 500:
                recommendations.epochs = 50
                recommendations.learning_rate *= 0.5
            elif dataset.image_count < 2000:
                recommendations.epochs = 100
            else:
                recommendations.epochs = 200
                recommendations.learning_rate *= 1.2
        
        # Adjust based on training goal
        if training_goal == "speed":
            recommendations.epochs = min(recommendations.epochs, 50)
            recommendations.batch_size = min(recommendations.batch_size * 2, 64)
        elif training_goal == "accuracy":
            recommendations.epochs = max(recommendations.epochs, 100)
            recommendations.patience = 20
            recommendations.learning_rate *= 0.8
        
        return {
            "base_model_id": base_model_id,
            "dataset_id": dataset_id,
            "training_goal": training_goal,
            "recommended_hyperparameters": recommendations.to_dict(),
            "reasoning": {
                "model_specific": f"Optimized for {base_model.framework} {base_model.variant}",
                "dataset_specific": f"Adjusted for dataset size of {dataset.image_count or 'unknown'} images",
                "goal_specific": f"Optimized for {training_goal} priority"
            }
        }
    
    def validate_training_config(self, base_model_id: str, dataset_id: str,
                               hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate base model
        base_model = self.model_storage.get_by_id(base_model_id)
        if not base_model:
            validation_result["errors"].append(f"Base model {base_model_id} not found")
            validation_result["is_valid"] = False
        
        # Validate dataset
        dataset = self.dataset_storage.get_by_id(dataset_id)
        if not dataset:
            validation_result["errors"].append(f"Dataset {dataset_id} not found")
            validation_result["is_valid"] = False
        
        # Validate hyperparameters
        try:
            hyper_params = HyperParameters.from_dict(hyperparameters)
            
            # Check for common issues
            if hyper_params.learning_rate > 0.1:
                validation_result["warnings"].append("Learning rate is quite high - may cause unstable training")
            elif hyper_params.learning_rate < 0.00001:
                validation_result["warnings"].append("Learning rate is very low - training may be slow")
            
            if hyper_params.batch_size > 128:
                validation_result["warnings"].append("Large batch size may require significant GPU memory")
            elif hyper_params.batch_size < 4:
                validation_result["warnings"].append("Small batch size may lead to unstable gradients")
            
            if hyper_params.epochs > 500:
                validation_result["warnings"].append("Very high epoch count - consider using early stopping")
            elif hyper_params.epochs < 10:
                validation_result["warnings"].append("Low epoch count may result in underfitting")
            
        except Exception as e:
            validation_result["errors"].append(f"Invalid hyperparameters: {str(e)}")
            validation_result["is_valid"] = False
        
        # Model-dataset compatibility
        if base_model and dataset:
            if base_model.type == "detection" and dataset.annotation_type == "segmentation":
                validation_result["warnings"].append(
                    "Using segmentation dataset with detection model - only bounding boxes will be used"
                )
            elif base_model.type == "segmentation" and dataset.annotation_type == "detection":
                validation_result["errors"].append(
                    "Cannot use detection dataset with segmentation model"
                )
                validation_result["is_valid"] = False
        
        return validation_result
    
    def get_training_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get training statistics for the last N days."""
        jobs, _ = self.storage.list_training_jobs(limit=1000)
        
        # Filter jobs from last N days
        cutoff_date = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
        recent_jobs = []
        
        for job in jobs:
            try:
                job_time = datetime.fromisoformat(job.created_at.replace("Z", "+00:00")).timestamp()
                if job_time >= cutoff_date:
                    recent_jobs.append(job)
            except (ValueError, AttributeError):
                continue
        
        if not recent_jobs:
            return {
                "period_days": days,
                "total_jobs": 0
            }
        
        # Calculate statistics
        status_counts = {}
        for job in recent_jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        completed_jobs = [job for job in recent_jobs if job.status == "completed"]
        
        stats = {
            "period_days": days,
            "total_jobs": len(recent_jobs),
            "status_breakdown": status_counts,
            "completion_rate": (len(completed_jobs) / len(recent_jobs) * 100) if recent_jobs else 0,
            "queue_status": self.scheduler.get_queue_status()
        }
        
        # Average training time for completed jobs
        if completed_jobs:
            training_times = []
            for job in completed_jobs:
                if job.start_time and job.end_time:
                    try:
                        start = datetime.fromisoformat(job.start_time.replace("Z", "+00:00"))
                        end = datetime.fromisoformat(job.end_time.replace("Z", "+00:00"))
                        duration = (end - start).total_seconds()
                        training_times.append(duration)
                    except (ValueError, AttributeError):
                        continue
            
            if training_times:
                avg_time = sum(training_times) / len(training_times)
                stats["average_training_time_hours"] = avg_time / 3600
                stats["min_training_time_hours"] = min(training_times) / 3600
                stats["max_training_time_hours"] = max(training_times) / 3600
        
        return stats