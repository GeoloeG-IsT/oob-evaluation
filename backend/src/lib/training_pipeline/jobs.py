"""
Training job management and scheduling.
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import threading

from .pipeline import TrainingJob, TrainingStatus, TrainingConfig


class JobPriority(str, Enum):
    """Priority levels for training jobs."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class JobProgress:
    """Detailed progress information for a training job."""
    job_id: str
    status: TrainingStatus
    current_epoch: int
    total_epochs: int
    progress_percentage: float
    
    # Time estimates
    elapsed_time_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    average_epoch_time_seconds: float = 0.0
    
    # Performance metrics
    latest_train_loss: float = 0.0
    latest_val_loss: float = 0.0
    latest_map50: float = 0.0
    best_map50: float = 0.0
    
    # Resource usage
    gpu_memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Timestamps
    created_at: str = ""
    started_at: Optional[str] = None
    last_updated_at: str = ""
    
    def __post_init__(self):
        if not self.last_updated_at:
            self.last_updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class SchedulerConfig:
    """Configuration for job scheduler."""
    max_concurrent_jobs: int = 2
    max_queue_size: int = 50
    job_timeout_hours: int = 24
    cleanup_completed_after_hours: int = 72
    enable_auto_scaling: bool = False
    priority_boost_threshold_hours: int = 6
    
    # Resource limits
    max_gpu_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[int] = None
    reserved_memory_mb: int = 1000  # Reserve memory for system


class JobScheduler:
    """Scheduler for managing training job execution."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._job_queue: deque = deque()
        self._running_jobs: Dict[str, TrainingJob] = {}
        self._completed_jobs: Dict[str, TrainingJob] = {}
        self._job_priorities: Dict[str, JobPriority] = {}
        self._job_futures: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    def submit_job(
        self,
        job: TrainingJob,
        priority: JobPriority = JobPriority.NORMAL
    ) -> bool:
        """Submit a job to the scheduler queue."""
        with self._lock:
            if len(self._job_queue) >= self.config.max_queue_size:
                return False
            
            self._job_queue.append(job)
            self._job_priorities[job.job_id] = priority
            
            # Sort queue by priority
            self._sort_queue_by_priority()
            
            return True
    
    def _sort_queue_by_priority(self) -> None:
        """Sort job queue by priority and submission time."""
        priority_order = {
            JobPriority.URGENT: 0,
            JobPriority.HIGH: 1,
            JobPriority.NORMAL: 2,
            JobPriority.LOW: 3
        }
        
        queue_list = list(self._job_queue)
        queue_list.sort(key=lambda job: (
            priority_order.get(self._job_priorities.get(job.job_id, JobPriority.NORMAL), 2),
            job.created_at
        ))
        
        self._job_queue.clear()
        self._job_queue.extend(queue_list)
    
    def start_scheduler(self) -> None:
        """Start the job scheduler."""
        if not self._scheduler_running:
            self._scheduler_running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    def stop_scheduler(self) -> None:
        """Stop the job scheduler."""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                await self._process_queue()
                await self._check_running_jobs()
                await self._cleanup_completed_jobs()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _process_queue(self) -> None:
        """Process jobs from the queue."""
        with self._lock:
            # Check if we can start new jobs
            if (len(self._running_jobs) >= self.config.max_concurrent_jobs or
                not self._job_queue):
                return
            
            # Get next job from queue
            next_job = self._job_queue.popleft()
            
            # Check resource availability
            if not self._check_resource_availability(next_job):
                # Put job back at front of queue
                self._job_queue.appendleft(next_job)
                return
            
            # Start the job
            self._running_jobs[next_job.job_id] = next_job
            
            # Create async task for training
            from .pipeline import TrainingPipeline
            pipeline = TrainingPipeline()
            future = asyncio.create_task(pipeline.start_training(next_job))
            self._job_futures[next_job.job_id] = future
    
    def _check_resource_availability(self, job: TrainingJob) -> bool:
        """Check if resources are available for the job."""
        # Simple resource check - in real implementation, would check GPU/CPU usage
        if self.config.max_gpu_memory_mb:
            # Simulate GPU memory check
            estimated_memory = 2000  # Estimate based on model and batch size
            current_usage = sum(
                getattr(j.latest_metrics, 'gpu_memory_mb', 0) 
                for j in self._running_jobs.values()
            )
            
            if current_usage + estimated_memory > self.config.max_gpu_memory_mb:
                return False
        
        return True
    
    async def _check_running_jobs(self) -> None:
        """Check status of running jobs."""
        completed_jobs = []
        
        with self._lock:
            for job_id, future in self._job_futures.items():
                if future.done():
                    try:
                        completed_job = await future
                        self._completed_jobs[job_id] = completed_job
                        
                        if job_id in self._running_jobs:
                            del self._running_jobs[job_id]
                        
                        completed_jobs.append(job_id)
                    except Exception as e:
                        # Handle job failure
                        if job_id in self._running_jobs:
                            self._running_jobs[job_id].fail_training(str(e))
                            self._completed_jobs[job_id] = self._running_jobs[job_id]
                            del self._running_jobs[job_id]
                        
                        completed_jobs.append(job_id)
        
        # Clean up completed futures
        for job_id in completed_jobs:
            if job_id in self._job_futures:
                del self._job_futures[job_id]
            if job_id in self._job_priorities:
                del self._job_priorities[job_id]
    
    async def _cleanup_completed_jobs(self) -> None:
        """Clean up old completed jobs."""
        if not self.config.cleanup_completed_after_hours:
            return
        
        cutoff_time = time.time() - (self.config.cleanup_completed_after_hours * 3600)
        jobs_to_remove = []
        
        with self._lock:
            for job_id, job in self._completed_jobs.items():
                if job.completed_at:
                    try:
                        completed_time = datetime.fromisoformat(
                            job.completed_at.replace("Z", "+00:00")
                        ).timestamp()
                        
                        if completed_time < cutoff_time:
                            jobs_to_remove.append(job_id)
                    except ValueError:
                        # If parsing fails, remove it
                        jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            if job_id in self._completed_jobs:
                del self._completed_jobs[job_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            # Check if job is in queue
            queue_list = list(self._job_queue)
            for i, job in enumerate(queue_list):
                if job.job_id == job_id:
                    job.cancel_training()
                    del queue_list[i]
                    self._job_queue.clear()
                    self._job_queue.extend(queue_list)
                    return True
            
            # Check if job is running
            if job_id in self._running_jobs:
                job = self._running_jobs[job_id]
                job.cancel_training()
                
                # Cancel the future
                if job_id in self._job_futures:
                    self._job_futures[job_id].cancel()
                
                return True
        
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                "queued_jobs": len(self._job_queue),
                "running_jobs": len(self._running_jobs),
                "completed_jobs": len(self._completed_jobs),
                "max_concurrent": self.config.max_concurrent_jobs,
                "max_queue_size": self.config.max_queue_size,
                "scheduler_running": self._scheduler_running
            }
    
    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get detailed progress for a job."""
        job = None
        
        with self._lock:
            if job_id in self._running_jobs:
                job = self._running_jobs[job_id]
            elif job_id in self._completed_jobs:
                job = self._completed_jobs[job_id]
            else:
                # Check queue
                for queued_job in self._job_queue:
                    if queued_job.job_id == job_id:
                        job = queued_job
                        break
        
        if not job:
            return None
        
        # Calculate timing information
        elapsed_time = 0.0
        estimated_remaining = 0.0
        avg_epoch_time = 0.0
        
        if job.started_at:
            start_time = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
            if job.completed_at:
                end_time = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                elapsed_time = (end_time - start_time).total_seconds()
            else:
                elapsed_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            if job.current_epoch > 0:
                avg_epoch_time = elapsed_time / job.current_epoch
                if job.total_epochs > job.current_epoch:
                    estimated_remaining = avg_epoch_time * (job.total_epochs - job.current_epoch)
        
        # Get latest metrics
        latest_metrics = job.latest_metrics
        best_metrics = job.best_metrics
        
        return JobProgress(
            job_id=job.job_id,
            status=job.status,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            progress_percentage=job.progress_percentage,
            elapsed_time_seconds=elapsed_time,
            estimated_remaining_seconds=estimated_remaining,
            average_epoch_time_seconds=avg_epoch_time,
            latest_train_loss=latest_metrics.train_loss if latest_metrics else 0.0,
            latest_val_loss=latest_metrics.val_loss if latest_metrics else 0.0,
            latest_map50=latest_metrics.map50 if latest_metrics else 0.0,
            best_map50=best_metrics.map50 if best_metrics else 0.0,
            gpu_memory_usage_mb=latest_metrics.gpu_memory_mb if latest_metrics else 0.0,
            created_at=job.created_at,
            started_at=job.started_at
        )


class TrainingJobManager:
    """High-level manager for training jobs."""
    
    def __init__(self, scheduler_config: Optional[SchedulerConfig] = None):
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.scheduler = JobScheduler(self.scheduler_config)
        self._job_callbacks: Dict[str, List[Callable[[TrainingJob], None]]] = defaultdict(list)
        
        # Start scheduler
        self.scheduler.start_scheduler()
    
    def create_and_submit_job(
        self,
        config: TrainingConfig,
        priority: JobPriority = JobPriority.NORMAL,
        progress_callback: Optional[Callable[[TrainingJob], None]] = None
    ) -> str:
        """Create and submit a training job."""
        from .pipeline import TrainingPipeline
        
        pipeline = TrainingPipeline()
        job = pipeline.create_training_job(config)
        
        # Register progress callback if provided
        if progress_callback:
            job.progress_callbacks.append(progress_callback)
            self._job_callbacks[job.job_id].append(progress_callback)
        
        # Submit to scheduler
        success = self.scheduler.submit_job(job, priority)
        
        if not success:
            raise RuntimeError("Failed to submit job - queue is full")
        
        return job.job_id
    
    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get job progress."""
        return self.scheduler.get_job_progress(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        return self.scheduler.cancel_job(job_id)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return self.scheduler.get_queue_status()
    
    def list_jobs(self, status_filter: Optional[TrainingStatus] = None) -> List[Dict[str, Any]]:
        """List jobs with optional status filter."""
        jobs = []
        
        # Get jobs from scheduler
        with self.scheduler._lock:
            # Queued jobs
            for job in self.scheduler._job_queue:
                if not status_filter or job.status == status_filter:
                    jobs.append(self._job_to_dict(job))
            
            # Running jobs
            for job in self.scheduler._running_jobs.values():
                if not status_filter or job.status == status_filter:
                    jobs.append(self._job_to_dict(job))
            
            # Completed jobs
            for job in self.scheduler._completed_jobs.values():
                if not status_filter or job.status == status_filter:
                    jobs.append(self._job_to_dict(job))
        
        # Sort by created_at (most recent first)
        jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jobs
    
    def _job_to_dict(self, job: TrainingJob) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "job_id": job.job_id,
            "status": job.status,
            "experiment_name": job.config.experiment_name,
            "model_type": job.config.model_type,
            "variant": job.config.variant,
            "progress_percentage": job.progress_percentage,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "latest_metrics": job.latest_metrics.to_dict() if job.latest_metrics else None,
            "best_metrics": job.best_metrics.to_dict() if job.best_metrics else None
        }
    
    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job information."""
        job = None
        
        with self.scheduler._lock:
            if job_id in self.scheduler._running_jobs:
                job = self.scheduler._running_jobs[job_id]
            elif job_id in self.scheduler._completed_jobs:
                job = self.scheduler._completed_jobs[job_id]
            else:
                for queued_job in self.scheduler._job_queue:
                    if queued_job.job_id == job_id:
                        job = queued_job
                        break
        
        if not job:
            return None
        
        return job.get_training_summary()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics across all jobs."""
        stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "cancelled_jobs": 0,
            "success_rate": 0.0,
            "average_training_time_hours": 0.0,
            "model_type_breakdown": defaultdict(int),
            "status_breakdown": defaultdict(int)
        }
        
        all_jobs = []
        with self.scheduler._lock:
            all_jobs.extend(self.scheduler._job_queue)
            all_jobs.extend(self.scheduler._running_jobs.values())
            all_jobs.extend(self.scheduler._completed_jobs.values())
        
        if not all_jobs:
            return stats
        
        stats["total_jobs"] = len(all_jobs)
        
        training_times = []
        for job in all_jobs:
            stats["status_breakdown"][job.status] += 1
            stats["model_type_breakdown"][f"{job.config.model_type}_{job.config.variant}"] += 1
            
            if job.status == TrainingStatus.COMPLETED:
                stats["completed_jobs"] += 1
                
                # Calculate training time
                if job.started_at and job.completed_at:
                    start_dt = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                    training_time = (end_dt - start_dt).total_seconds() / 3600  # Convert to hours
                    training_times.append(training_time)
                    
            elif job.status == TrainingStatus.FAILED:
                stats["failed_jobs"] += 1
            elif job.status == TrainingStatus.CANCELLED:
                stats["cancelled_jobs"] += 1
        
        # Calculate rates and averages
        if stats["total_jobs"] > 0:
            stats["success_rate"] = (stats["completed_jobs"] / stats["total_jobs"]) * 100.0
        
        if training_times:
            stats["average_training_time_hours"] = sum(training_times) / len(training_times)
        
        return stats
    
    def register_job_callback(
        self,
        job_id: str,
        callback: Callable[[TrainingJob], None]
    ) -> None:
        """Register a callback for job progress updates."""
        self._job_callbacks[job_id].append(callback)
    
    def unregister_job_callback(
        self,
        job_id: str,
        callback: Callable[[TrainingJob], None]
    ) -> None:
        """Unregister a job callback."""
        if job_id in self._job_callbacks:
            if callback in self._job_callbacks[job_id]:
                self._job_callbacks[job_id].remove(callback)
    
    def shutdown(self) -> None:
        """Shutdown the job manager."""
        self.scheduler.stop_scheduler()
        
        # Cancel all running jobs
        with self.scheduler._lock:
            for job_id in list(self.scheduler._running_jobs.keys()):
                self.scheduler.cancel_job(job_id)