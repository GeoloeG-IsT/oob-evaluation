"""
In-memory storage for training jobs (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class TrainingJobModel:
    """Temporary in-memory training job model for TDD GREEN phase."""
    
    def __init__(self, base_model_id: str, dataset_id: str, hyperparameters: Dict[str, Any],
                 status: str = "queued", progress_percentage: float = 0.0,
                 execution_logs: Optional[str] = None, result_model_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.base_model_id = base_model_id
        self.dataset_id = dataset_id
        self.status = status  # queued, running, completed, failed, cancelled
        self.progress_percentage = progress_percentage
        self.hyperparameters = hyperparameters
        self.execution_logs = execution_logs
        self.start_time = None
        self.end_time = None
        self.result_model_id = result_model_id
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class TrainingJobStorage:
    """Temporary in-memory storage for training jobs."""
    
    def __init__(self):
        self._training_jobs: Dict[str, TrainingJobModel] = {}
    
    def save(self, training_job: TrainingJobModel) -> TrainingJobModel:
        """Save a training job to storage."""
        self._training_jobs[training_job.id] = training_job
        return training_job
    
    def get_by_id(self, training_job_id: str) -> Optional[TrainingJobModel]:
        """Get a training job by ID."""
        return self._training_jobs.get(training_job_id)
    
    def list_training_jobs(self, status: Optional[str] = None, base_model_id: Optional[str] = None,
                          dataset_id: Optional[str] = None, limit: int = 50, 
                          offset: int = 0) -> tuple[List[TrainingJobModel], int]:
        """List training jobs with optional filtering and pagination."""
        jobs = list(self._training_jobs.values())
        
        # Apply filters
        if status:
            jobs = [j for j in jobs if j.status == status]
        if base_model_id:
            jobs = [j for j in jobs if j.base_model_id == base_model_id]
        if dataset_id:
            jobs = [j for j in jobs if j.dataset_id == dataset_id]
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        total_count = len(jobs)
        
        # Apply pagination
        paginated_jobs = jobs[offset:offset + limit]
        
        return paginated_jobs, total_count
    
    def update_status(self, training_job_id: str, status: str, progress_percentage: Optional[float] = None,
                     execution_logs: Optional[str] = None, result_model_id: Optional[str] = None) -> bool:
        """Update training job status and related fields."""
        job = self.get_by_id(training_job_id)
        if not job:
            return False
        
        old_status = job.status
        job.status = status
        
        if progress_percentage is not None:
            job.progress_percentage = progress_percentage
        
        if execution_logs is not None:
            job.execution_logs = execution_logs
            
        if result_model_id is not None:
            job.result_model_id = result_model_id
        
        # Update timestamps based on status transitions
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        if old_status == "queued" and status == "running":
            job.start_time = now
        elif status in ["completed", "failed", "cancelled"]:
            job.end_time = now
        
        return True
    
    def get_jobs_by_model(self, base_model_id: str) -> List[TrainingJobModel]:
        """Get all training jobs for a specific base model."""
        return [job for job in self._training_jobs.values() if job.base_model_id == base_model_id]
    
    def get_jobs_by_dataset(self, dataset_id: str) -> List[TrainingJobModel]:
        """Get all training jobs for a specific dataset."""
        return [job for job in self._training_jobs.values() if job.dataset_id == dataset_id]


# Global storage instance (temporary for TDD)
training_job_storage = TrainingJobStorage()