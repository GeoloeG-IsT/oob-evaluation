"""
Specialized processors for different inference scenarios.
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
    BatchInferenceJob,
    InferenceStatus
)


@dataclass
class ProcessingOptions:
    """Options for image processing."""
    max_concurrent: int = 4
    timeout_seconds: int = 30
    retry_count: int = 2
    validate_images: bool = True
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


class SingleImageProcessor:
    """Processor for single image inference operations."""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
    
    async def process(
        self,
        model_id: str,
        image_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[ProcessingOptions] = None
    ) -> InferenceResult:
        """Process a single image with the specified model."""
        if parameters is None:
            parameters = {}
        if options is None:
            options = ProcessingOptions()
        
        # Validate image path
        if options.validate_images:
            image_file = Path(image_path)
            if not image_file.exists():
                return InferenceResult(
                    request_id="",
                    model_id=model_id,
                    status=InferenceStatus.FAILED,
                    error_message=f"Image file not found: {image_path}"
                )
            
            if image_file.suffix.lower() not in options.supported_formats:
                return InferenceResult(
                    request_id="",
                    model_id=model_id,
                    status=InferenceStatus.FAILED,
                    error_message=f"Unsupported image format: {image_file.suffix}"
                )
        
        # Create inference request
        request = InferenceRequest(
            model_id=model_id,
            image_path=image_path,
            parameters=parameters
        )
        
        # Process with retry logic
        last_error = None
        for attempt in range(options.retry_count + 1):
            try:
                # Add timeout
                result = await asyncio.wait_for(
                    self.engine.single_inference(request),
                    timeout=options.timeout_seconds
                )
                
                if result.status == InferenceStatus.COMPLETED:
                    return result
                elif attempt < options.retry_count:
                    # Retry on failure
                    last_error = result.error_message
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    return result
                    
            except asyncio.TimeoutError:
                last_error = f"Inference timeout after {options.timeout_seconds} seconds"
                if attempt < options.retry_count:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < options.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
        
        # All retries failed
        return InferenceResult(
            request_id=request.request_id,
            model_id=model_id,
            status=InferenceStatus.FAILED,
            error_message=last_error or "Unknown error after retries"
        )


class BatchImageProcessor:
    """Processor for batch image inference operations."""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
    
    def create_batch_job(
        self,
        model_id: str,
        image_paths: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        options: Optional[ProcessingOptions] = None
    ) -> BatchInferenceJob:
        """Create a batch inference job."""
        if parameters is None:
            parameters = {}
        if options is None:
            options = ProcessingOptions()
        
        # Validate image paths if requested
        validated_paths = []
        if options.validate_images:
            for path in image_paths:
                image_file = Path(path)
                if image_file.exists() and image_file.suffix.lower() in options.supported_formats:
                    validated_paths.append(path)
        else:
            validated_paths = image_paths
        
        job = BatchInferenceJob(
            model_id=model_id,
            image_paths=validated_paths,
            parameters=parameters
        )
        
        if job_id:
            job.job_id = job_id
        
        return job
    
    def start_batch_processing(
        self,
        job: BatchInferenceJob,
        options: Optional[ProcessingOptions] = None
    ) -> BatchInferenceJob:
        """Start batch processing for a job."""
        if options is None:
            options = ProcessingOptions()
        
        # Update engine max workers if specified
        if hasattr(self.engine, 'max_workers'):
            self.engine.max_workers = min(self.engine.max_workers, options.max_concurrent)
        
        return self.engine.start_batch_inference(job)
    
    def process_batch(
        self,
        model_id: str,
        image_paths: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[ProcessingOptions] = None
    ) -> BatchInferenceJob:
        """Create and start batch processing in one step."""
        job = self.create_batch_job(model_id, image_paths, parameters, options=options)
        return self.start_batch_processing(job, options)
    
    def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress information for a batch job."""
        job = self.engine.get_batch_job_status(job_id)
        if not job:
            return None
        
        # Calculate timing information
        processing_time_seconds = 0.0
        if job.started_at and job.status in [InferenceStatus.RUNNING, InferenceStatus.COMPLETED]:
            start_time = time.time()
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(job.started_at.replace("Z", "+00:00"))
                if job.completed_at:
                    end_dt = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00"))
                    processing_time_seconds = (end_dt - start_dt).total_seconds()
                else:
                    processing_time_seconds = time.time() - start_dt.timestamp()
            except ValueError:
                pass
        
        # Calculate average processing time per image
        avg_time_per_image = 0.0
        if job.completed_images > 0 and processing_time_seconds > 0:
            avg_time_per_image = processing_time_seconds / job.completed_images
        
        # Estimate remaining time
        remaining_time_seconds = 0.0
        if avg_time_per_image > 0 and job.total_images > job.completed_images + job.failed_images:
            remaining_images = job.total_images - job.completed_images - job.failed_images
            remaining_time_seconds = remaining_images * avg_time_per_image
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress_percentage": job.progress_percentage,
            "total_images": job.total_images,
            "completed_images": job.completed_images,
            "failed_images": job.failed_images,
            "remaining_images": job.total_images - job.completed_images - job.failed_images,
            "processing_time_seconds": processing_time_seconds,
            "average_time_per_image": avg_time_per_image,
            "estimated_remaining_time_seconds": remaining_time_seconds,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message
        }
    
    def get_job_results(
        self,
        job_id: str,
        include_failed: bool = True,
        limit: Optional[int] = None
    ) -> Optional[List[InferenceResult]]:
        """Get results from a batch job."""
        job = self.engine.get_batch_job_status(job_id)
        if not job:
            return None
        
        results = job.results
        
        if not include_failed:
            results = [r for r in results if r.status == InferenceStatus.COMPLETED]
        
        if limit:
            results = results[:limit]
        
        return results


class InferenceJobManager:
    """Manager for coordinating multiple inference jobs and processors."""
    
    def __init__(self, max_engines: int = 2):
        self.engines = [InferenceEngine() for _ in range(max_engines)]
        self.single_processors = [SingleImageProcessor(engine) for engine in self.engines]
        self.batch_processors = [BatchImageProcessor(engine) for engine in self.engines]
        self.current_engine_idx = 0
    
    def get_next_engine(self) -> tuple[InferenceEngine, SingleImageProcessor, BatchImageProcessor]:
        """Get the next available engine in round-robin fashion."""
        engine = self.engines[self.current_engine_idx]
        single_proc = self.single_processors[self.current_engine_idx]
        batch_proc = self.batch_processors[self.current_engine_idx]
        
        self.current_engine_idx = (self.current_engine_idx + 1) % len(self.engines)
        return engine, single_proc, batch_proc
    
    def get_least_loaded_engine(self) -> tuple[InferenceEngine, SingleImageProcessor, BatchImageProcessor]:
        """Get the engine with least active jobs."""
        min_jobs = float('inf')
        best_idx = 0
        
        for i, engine in enumerate(self.engines):
            active_jobs = len(engine.list_active_jobs())
            if active_jobs < min_jobs:
                min_jobs = active_jobs
                best_idx = i
        
        return (
            self.engines[best_idx],
            self.single_processors[best_idx],
            self.batch_processors[best_idx]
        )
    
    async def process_single_image(
        self,
        model_id: str,
        image_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[ProcessingOptions] = None
    ) -> InferenceResult:
        """Process single image using least loaded engine."""
        _, single_proc, _ = self.get_least_loaded_engine()
        return await single_proc.process(model_id, image_path, parameters, options)
    
    def process_batch_images(
        self,
        model_id: str,
        image_paths: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[ProcessingOptions] = None
    ) -> BatchInferenceJob:
        """Process batch of images using least loaded engine."""
        _, _, batch_proc = self.get_least_loaded_engine()
        return batch_proc.process_batch(model_id, image_paths, parameters, options)
    
    def get_all_active_jobs(self) -> List[BatchInferenceJob]:
        """Get all active jobs across all engines."""
        all_jobs = []
        for engine in self.engines:
            all_jobs.extend(engine.list_active_jobs())
        return all_jobs
    
    def get_global_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all engines."""
        summaries = [engine.get_performance_summary() for engine in self.engines]
        
        if not summaries or all(s.get("total_inferences", 0) == 0 for s in summaries):
            return {"total_inferences": 0}
        
        # Aggregate summaries
        total_inferences = sum(s.get("total_inferences", 0) for s in summaries)
        successful_inferences = sum(s.get("successful_inferences", 0) for s in summaries)
        
        if successful_inferences == 0:
            return {
                "total_inferences": total_inferences,
                "successful_inferences": 0,
                "success_rate": 0.0
            }
        
        # Calculate weighted averages
        weighted_inference_time = sum(
            s.get("average_inference_time_ms", 0) * s.get("successful_inferences", 0)
            for s in summaries
        ) / successful_inferences
        
        weighted_total_time = sum(
            s.get("average_total_time_ms", 0) * s.get("successful_inferences", 0)
            for s in summaries
        ) / successful_inferences
        
        weighted_throughput = sum(
            s.get("average_throughput_fps", 0) * s.get("successful_inferences", 0)
            for s in summaries
        ) / successful_inferences
        
        return {
            "total_inferences": total_inferences,
            "successful_inferences": successful_inferences,
            "failed_inferences": total_inferences - successful_inferences,
            "success_rate": successful_inferences / total_inferences * 100.0,
            "average_inference_time_ms": weighted_inference_time,
            "average_total_time_ms": weighted_total_time,
            "average_throughput_fps": weighted_throughput,
            "active_engines": len(self.engines),
            "total_active_jobs": len(self.get_all_active_jobs())
        }
    
    def cleanup_all_engines(self, older_than_hours: int = 24) -> int:
        """Cleanup completed jobs across all engines."""
        total_removed = 0
        for engine in self.engines:
            total_removed += engine.cleanup_completed_jobs(older_than_hours)
        return total_removed
    
    def shutdown_all(self) -> None:
        """Shutdown all engines."""
        for engine in self.engines:
            engine.shutdown()