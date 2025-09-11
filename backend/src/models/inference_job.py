"""
In-memory storage for inference jobs (temporary implementation for TDD GREEN phase).
Later this will be replaced with proper SQLAlchemy models and database storage.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid


class InferenceJobModel:
    """Temporary in-memory inference job model for TDD GREEN phase."""
    
    def __init__(self, model_id: str, target_images: List[str], 
                 status: str = "queued", progress_percentage: float = 0.0,
                 results: Optional[Dict[str, Any]] = None, execution_logs: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.model_id = model_id
        self.target_images = target_images.copy()  # Array of image IDs
        self.status = status  # queued, running, completed, failed, cancelled
        self.progress_percentage = progress_percentage
        self.results = results or {}
        self.execution_logs = execution_logs
        self.start_time = None
        self.end_time = None
        self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.metadata = metadata or {}


class InferenceJobStorage:
    """Temporary in-memory storage for inference jobs."""
    
    def __init__(self):
        self._inference_jobs: Dict[str, InferenceJobModel] = {}
    
    def save(self, inference_job: InferenceJobModel) -> InferenceJobModel:
        """Save an inference job to storage."""
        self._inference_jobs[inference_job.id] = inference_job
        return inference_job
    
    def get_by_id(self, inference_job_id: str) -> Optional[InferenceJobModel]:
        """Get an inference job by ID."""
        return self._inference_jobs.get(inference_job_id)
    
    def list_inference_jobs(self, status: Optional[str] = None, model_id: Optional[str] = None,
                           limit: int = 50, offset: int = 0) -> tuple[List[InferenceJobModel], int]:
        """List inference jobs with optional filtering and pagination."""
        jobs = list(self._inference_jobs.values())
        
        # Apply filters
        if status:
            jobs = [j for j in jobs if j.status == status]
        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        total_count = len(jobs)
        
        # Apply pagination
        paginated_jobs = jobs[offset:offset + limit]
        
        return paginated_jobs, total_count
    
    def update_status(self, inference_job_id: str, status: str, progress_percentage: Optional[float] = None,
                     results: Optional[Dict[str, Any]] = None, execution_logs: Optional[str] = None) -> bool:
        """Update inference job status and related fields."""
        job = self.get_by_id(inference_job_id)
        if not job:
            return False
        
        old_status = job.status
        job.status = status
        
        if progress_percentage is not None:
            job.progress_percentage = progress_percentage
        
        if results is not None:
            job.results = results
            
        if execution_logs is not None:
            job.execution_logs = execution_logs
        
        # Update timestamps based on status transitions
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        if old_status == "queued" and status == "running":
            job.start_time = now
        elif status in ["completed", "failed", "cancelled"]:
            job.end_time = now
        
        return True
    
    def get_jobs_by_model(self, model_id: str) -> List[InferenceJobModel]:
        """Get all inference jobs for a specific model."""
        return [job for job in self._inference_jobs.values() if job.model_id == model_id]
    
    def get_jobs_with_image(self, image_id: str) -> List[InferenceJobModel]:
        """Get all inference jobs that include a specific image."""
        return [job for job in self._inference_jobs.values() if image_id in job.target_images]
    
    def get_job_results_for_image(self, inference_job_id: str, image_id: str) -> Optional[Dict[str, Any]]:
        """Get inference results for a specific image within a job."""
        job = self.get_by_id(inference_job_id)
        if not job or not job.results:
            return None
        
        return job.results.get(image_id)


# Global storage instance (temporary for TDD)
inference_job_storage = InferenceJobStorage()