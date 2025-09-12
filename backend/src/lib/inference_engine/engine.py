"""
Core inference engine for ML model predictions.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..ml_models import ModelFactory, get_model_registry


class InferenceStatus(str, Enum):
    """Status of inference operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    image_path: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference."""
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    total_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: Optional[float] = None
    throughput_fps: float = 0.0


@dataclass
class InferenceResult:
    """Result of model inference."""
    request_id: str
    model_id: str
    status: InferenceStatus
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    completed_at: Optional[str] = None


@dataclass
class BatchInferenceJob:
    """Batch inference job tracking."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    image_paths: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: InferenceStatus = InferenceStatus.PENDING
    results: List[InferenceResult] = field(default_factory=list)
    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.completed_images + self.failed_images) / self.total_images * 100.0
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status in [InferenceStatus.COMPLETED, InferenceStatus.FAILED, InferenceStatus.CANCELLED]


class InferenceEngine:
    """Main inference engine for processing ML model predictions."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_jobs: Dict[str, BatchInferenceJob] = {}
        self._job_futures: Dict[str, Any] = {}
    
    async def single_inference(self, request: InferenceRequest) -> InferenceResult:
        """Perform single image inference."""
        start_time = time.time()
        
        result = InferenceResult(
            request_id=request.request_id,
            model_id=request.model_id,
            status=InferenceStatus.RUNNING
        )
        
        try:
            # Load model if needed
            registry = get_model_registry()
            if request.model_id not in registry._loaded_models:
                model_load_start = time.time()
                registry.load_model(request.model_id)
                load_time = (time.time() - model_load_start) * 1000
                result.performance_metrics.preprocessing_time_ms = load_time
            
            # Perform inference
            inference_start = time.time()
            predictions = ModelFactory.predict(
                request.model_id,
                request.image_path,
                **request.parameters
            )
            inference_time = (time.time() - inference_start) * 1000
            
            # Update result
            result.status = InferenceStatus.COMPLETED
            result.predictions = predictions.get("predictions", [])
            result.performance_metrics.inference_time_ms = inference_time
            result.performance_metrics.total_time_ms = (time.time() - start_time) * 1000
            result.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            # Calculate throughput
            if result.performance_metrics.total_time_ms > 0:
                result.performance_metrics.throughput_fps = 1000.0 / result.performance_metrics.total_time_ms
            
        except Exception as e:
            result.status = InferenceStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        return result
    
    def start_batch_inference(self, job: BatchInferenceJob) -> BatchInferenceJob:
        """Start batch inference job."""
        if job.job_id in self._active_jobs:
            raise ValueError(f"Job {job.job_id} already exists")
        
        job.total_images = len(job.image_paths)
        job.status = InferenceStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        self._active_jobs[job.job_id] = job
        
        # Start async processing
        future = asyncio.create_task(self._process_batch_job(job))
        self._job_futures[job.job_id] = future
        
        return job
    
    async def _process_batch_job(self, job: BatchInferenceJob) -> None:
        """Process batch inference job asynchronously."""
        try:
            # Create inference requests for each image
            requests = []
            for image_path in job.image_paths:
                request = InferenceRequest(
                    model_id=job.model_id,
                    image_path=image_path,
                    parameters=job.parameters.copy()
                )
                requests.append(request)
            
            # Process requests concurrently
            semaphore = asyncio.Semaphore(self.max_workers)
            tasks = [self._process_single_with_semaphore(semaphore, request, job) 
                    for request in requests]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update job status
            if job.failed_images > 0 and job.completed_images == 0:
                job.status = InferenceStatus.FAILED
                job.error_message = "All images failed to process"
            else:
                job.status = InferenceStatus.COMPLETED
            
            job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
        except Exception as e:
            job.status = InferenceStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    async def _process_single_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        request: InferenceRequest,
        job: BatchInferenceJob
    ) -> None:
        """Process single inference with semaphore for concurrency control."""
        async with semaphore:
            result = await self.single_inference(request)
            job.results.append(result)
            
            if result.status == InferenceStatus.COMPLETED:
                job.completed_images += 1
            else:
                job.failed_images += 1
    
    def get_batch_job_status(self, job_id: str) -> Optional[BatchInferenceJob]:
        """Get status of batch inference job."""
        return self._active_jobs.get(job_id)
    
    def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel a running batch inference job."""
        if job_id not in self._active_jobs:
            return False
        
        job = self._active_jobs[job_id]
        if job.is_complete:
            return False
        
        job.status = InferenceStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Cancel the future if it exists
        if job_id in self._job_futures:
            future = self._job_futures[job_id]
            future.cancel()
            del self._job_futures[job_id]
        
        return True
    
    def list_active_jobs(self) -> List[BatchInferenceJob]:
        """List all active batch inference jobs."""
        return [job for job in self._active_jobs.values() if not job.is_complete]
    
    def list_completed_jobs(self, limit: int = 50) -> List[BatchInferenceJob]:
        """List completed batch inference jobs."""
        completed_jobs = [job for job in self._active_jobs.values() if job.is_complete]
        # Sort by completion time (most recent first)
        completed_jobs.sort(key=lambda x: x.completed_at or "", reverse=True)
        return completed_jobs[:limit]
    
    def cleanup_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Cleanup completed jobs older than specified hours."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (older_than_hours * 3600)
        removed_count = 0
        
        jobs_to_remove = []
        for job_id, job in self._active_jobs.items():
            if job.is_complete and job.completed_at:
                try:
                    completed_time = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00")).timestamp()
                    if completed_time < cutoff_time:
                        jobs_to_remove.append(job_id)
                except ValueError:
                    # If parsing fails, consider it old and remove
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self._active_jobs[job_id]
            if job_id in self._job_futures:
                del self._job_futures[job_id]
            removed_count += 1
        
        return removed_count
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of completed inferences."""
        all_results = []
        for job in self._active_jobs.values():
            all_results.extend(job.results)
        
        if not all_results:
            return {"total_inferences": 0}
        
        successful_results = [r for r in all_results if r.status == InferenceStatus.COMPLETED]
        
        if not successful_results:
            return {
                "total_inferences": len(all_results),
                "successful_inferences": 0,
                "success_rate": 0.0
            }
        
        # Calculate averages
        avg_inference_time = sum(r.performance_metrics.inference_time_ms for r in successful_results) / len(successful_results)
        avg_total_time = sum(r.performance_metrics.total_time_ms for r in successful_results) / len(successful_results)
        avg_throughput = sum(r.performance_metrics.throughput_fps for r in successful_results) / len(successful_results)
        
        return {
            "total_inferences": len(all_results),
            "successful_inferences": len(successful_results),
            "failed_inferences": len(all_results) - len(successful_results),
            "success_rate": len(successful_results) / len(all_results) * 100.0,
            "average_inference_time_ms": avg_inference_time,
            "average_total_time_ms": avg_total_time,
            "average_throughput_fps": avg_throughput,
            "min_inference_time_ms": min(r.performance_metrics.inference_time_ms for r in successful_results),
            "max_inference_time_ms": max(r.performance_metrics.inference_time_ms for r in successful_results),
        }
    
    def shutdown(self) -> None:
        """Shutdown the inference engine."""
        # Cancel all active jobs
        for job_id in list(self._active_jobs.keys()):
            self.cancel_batch_job(job_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)