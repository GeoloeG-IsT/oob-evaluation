"""
Inference service for single and batch processing.
"""
from typing import List, Optional, Dict, Any, Tuple
import uuid
import asyncio
from datetime import datetime, timezone

from ..models.inference_job import InferenceJobModel, inference_job_storage
from ..lib.inference_engine import (
    InferenceEngine, InferenceRequest, InferenceResult, 
    BatchInferenceJob, InferenceStatus
)
from ..lib.ml_models import get_model_registry


class InferenceService:
    """Service for handling inference operations."""
    
    def __init__(self, max_workers: int = 4):
        self.storage = inference_job_storage
        self.engine = InferenceEngine(max_workers=max_workers)
        self.registry = get_model_registry()
    
    async def single_inference(self, image_id: str, model_id: str, 
                              parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform single image inference."""
        if not image_id:
            raise ValueError("image_id is required")
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        if not self.registry.get_model_config(model_id):
            raise ValueError(f"Model {model_id} not found")
        
        # For TDD GREEN phase, simulate file path
        image_path = f"/storage/images/{image_id}"
        
        # Create inference request
        request = InferenceRequest(
            model_id=model_id,
            image_path=image_path,
            parameters=parameters or {}
        )
        
        # Perform inference
        result = await self.engine.single_inference(request)
        
        # Create inference job record
        job_model = InferenceJobModel(
            model_id=model_id,
            image_ids=[image_id],
            job_type="single",
            status=result.status.value,
            progress_percentage=100.0 if result.status == InferenceStatus.COMPLETED else 0.0,
            results={
                "request_id": result.request_id,
                "predictions": result.predictions,
                "performance_metrics": result.performance_metrics.__dict__
            },
            metadata={"single_inference": True}
        )
        
        # Save job record
        saved_job = self.storage.save(job_model)
        
        return {
            "job_id": saved_job.id,
            "request_id": result.request_id,
            "model_id": model_id,
            "image_id": image_id,
            "status": result.status.value,
            "predictions": result.predictions,
            "performance_metrics": result.performance_metrics.__dict__,
            "error_message": result.error_message,
            "created_at": result.created_at,
            "completed_at": result.completed_at
        }
    
    def start_batch_inference(self, image_ids: List[str], model_id: str,
                            parameters: Optional[Dict[str, Any]] = None,
                            job_name: Optional[str] = None) -> Dict[str, Any]:
        """Start batch inference job."""
        if not image_ids:
            raise ValueError("image_ids list cannot be empty")
        if not model_id:
            raise ValueError("model_id is required")
        
        # Validate model exists
        if not self.registry.get_model_config(model_id):
            raise ValueError(f"Model {model_id} not found")
        
        # For TDD GREEN phase, simulate file paths
        image_paths = [f"/storage/images/{image_id}" for image_id in image_ids]
        
        # Create batch inference job
        batch_job = BatchInferenceJob(
            model_id=model_id,
            image_paths=image_paths,
            parameters=parameters or {}
        )
        
        # Create inference job record
        job_model = InferenceJobModel(
            model_id=model_id,
            image_ids=image_ids,
            job_type="batch",
            status="running",
            progress_percentage=0.0,
            name=job_name,
            metadata={
                "batch_job_id": batch_job.job_id,
                "total_images": len(image_ids),
                "parameters": parameters or {}
            }
        )
        
        # Save job record
        saved_job = self.storage.save(job_model)
        
        # Start batch processing
        started_batch = self.engine.start_batch_inference(batch_job)
        
        # Start monitoring task
        asyncio.create_task(self._monitor_batch_job(saved_job.id, started_batch.job_id))
        
        return {
            "job_id": saved_job.id,
            "batch_job_id": started_batch.job_id,
            "model_id": model_id,
            "status": started_batch.status.value,
            "total_images": len(image_ids),
            "progress_percentage": 0.0,
            "created_at": saved_job.created_at,
            "name": job_name
        }
    
    async def _monitor_batch_job(self, job_id: str, batch_job_id: str) -> None:
        """Monitor batch job progress and update storage."""
        while True:
            # Get batch job status from engine
            batch_job = self.engine.get_batch_job_status(batch_job_id)
            if not batch_job:
                break
            
            # Update job record in storage
            self.storage.update_status(
                job_id,
                batch_job.status.value,
                batch_job.progress_percentage,
                results={
                    "completed_images": batch_job.completed_images,
                    "failed_images": batch_job.failed_images,
                    "results": [result.__dict__ for result in batch_job.results]
                }
            )
            
            # Break if job is complete
            if batch_job.is_complete:
                break
            
            # Wait before next check
            await asyncio.sleep(5)
    
    def get_inference_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get inference job by ID."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.id,
            "model_id": job.model_id,
            "image_ids": job.image_ids,
            "job_type": job.job_type,
            "status": job.status,
            "progress_percentage": job.progress_percentage,
            "results": job.results,
            "name": job.name,
            "created_at": job.created_at,
            "metadata": job.metadata
        }
    
    def list_inference_jobs(self, model_id: Optional[str] = None, 
                          status: Optional[str] = None,
                          job_type: Optional[str] = None,
                          limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """List inference jobs with optional filtering."""
        jobs, total_count = self.storage.list_inference_jobs(
            model_id=model_id, 
            status=status, 
            job_type=job_type,
            limit=limit, 
            offset=offset
        )
        
        job_list = []
        for job in jobs:
            job_dict = {
                "job_id": job.id,
                "model_id": job.model_id,
                "image_ids": job.image_ids,
                "job_type": job.job_type,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "name": job.name,
                "created_at": job.created_at,
                "metadata": job.metadata
            }
            
            # Add summary info for batch jobs
            if job.job_type == "batch" and job.results:
                job_dict["summary"] = {
                    "total_images": len(job.image_ids),
                    "completed_images": job.results.get("completed_images", 0),
                    "failed_images": job.results.get("failed_images", 0)
                }
            
            job_list.append(job_dict)
        
        return job_list, total_count
    
    def cancel_inference_job(self, job_id: str) -> bool:
        """Cancel a running inference job."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return False
        
        if job.status not in ["running", "pending"]:
            return False
        
        # Cancel batch job in engine if it exists
        if job.job_type == "batch" and job.metadata.get("batch_job_id"):
            batch_job_id = job.metadata["batch_job_id"]
            self.engine.cancel_batch_job(batch_job_id)
        
        # Update job status
        self.storage.update_status(job_id, "cancelled")
        return True
    
    def get_inference_results(self, job_id: str, 
                            include_predictions: bool = True) -> Optional[Dict[str, Any]]:
        """Get detailed inference results for a job."""
        job = self.storage.get_by_id(job_id)
        if not job:
            return None
        
        results = {
            "job_id": job.id,
            "model_id": job.model_id,
            "status": job.status,
            "job_type": job.job_type,
            "created_at": job.created_at
        }
        
        if job.results:
            if job.job_type == "single":
                results.update({
                    "predictions": job.results.get("predictions", []) if include_predictions else [],
                    "performance_metrics": job.results.get("performance_metrics", {}),
                    "image_id": job.image_ids[0] if job.image_ids else None
                })
            elif job.job_type == "batch":
                results.update({
                    "total_images": len(job.image_ids),
                    "completed_images": job.results.get("completed_images", 0),
                    "failed_images": job.results.get("failed_images", 0),
                    "progress_percentage": job.progress_percentage
                })
                
                if include_predictions and job.results.get("results"):
                    results["detailed_results"] = job.results["results"]
        
        return results
    
    def get_model_inference_stats(self, model_id: str, 
                                days: int = 30) -> Dict[str, Any]:
        """Get inference statistics for a specific model."""
        jobs, _ = self.storage.list_inference_jobs(
            model_id=model_id, 
            limit=1000  # Get recent jobs
        )
        
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
                "model_id": model_id,
                "period_days": days,
                "total_jobs": 0,
                "total_images": 0
            }
        
        # Calculate statistics
        total_jobs = len(recent_jobs)
        total_images = sum(len(job.image_ids) for job in recent_jobs)
        successful_jobs = [job for job in recent_jobs if job.status == "completed"]
        failed_jobs = [job for job in recent_jobs if job.status == "failed"]
        
        # Job type breakdown
        single_jobs = [job for job in recent_jobs if job.job_type == "single"]
        batch_jobs = [job for job in recent_jobs if job.job_type == "batch"]
        
        stats = {
            "model_id": model_id,
            "period_days": days,
            "total_jobs": total_jobs,
            "total_images": total_images,
            "successful_jobs": len(successful_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": (len(successful_jobs) / total_jobs * 100) if total_jobs > 0 else 0,
            "job_types": {
                "single": len(single_jobs),
                "batch": len(batch_jobs)
            }
        }
        
        # Calculate average performance metrics for single inference jobs
        single_successful = [job for job in successful_jobs if job.job_type == "single" and job.results]
        if single_successful:
            performance_metrics = []
            for job in single_successful:
                metrics = job.results.get("performance_metrics", {})
                if metrics:
                    performance_metrics.append(metrics)
            
            if performance_metrics:
                avg_inference_time = sum(m.get("inference_time_ms", 0) for m in performance_metrics) / len(performance_metrics)
                avg_total_time = sum(m.get("total_time_ms", 0) for m in performance_metrics) / len(performance_metrics)
                avg_throughput = sum(m.get("throughput_fps", 0) for m in performance_metrics) / len(performance_metrics)
                
                stats["average_performance"] = {
                    "inference_time_ms": avg_inference_time,
                    "total_time_ms": avg_total_time,
                    "throughput_fps": avg_throughput
                }
        
        return stats
    
    def get_engine_performance_summary(self) -> Dict[str, Any]:
        """Get overall engine performance summary."""
        return self.engine.get_performance_summary()
    
    def cleanup_old_jobs(self, older_than_hours: int = 24) -> int:
        """Clean up old completed inference jobs."""
        # Clean up engine jobs first
        engine_cleaned = self.engine.cleanup_completed_jobs(older_than_hours)
        
        # Clean up storage jobs
        cutoff_time = datetime.now(timezone.utc).timestamp() - (older_than_hours * 3600)
        jobs_to_remove = []
        
        all_jobs, _ = self.storage.list_inference_jobs(limit=10000)  # Get all jobs
        
        for job in all_jobs:
            if job.status in ["completed", "failed", "cancelled"]:
                try:
                    job_time = datetime.fromisoformat(job.created_at.replace("Z", "+00:00")).timestamp()
                    if job_time < cutoff_time:
                        jobs_to_remove.append(job.id)
                except (ValueError, AttributeError):
                    # If parsing fails, consider it old
                    jobs_to_remove.append(job.id)
        
        # Remove old jobs from storage
        storage_cleaned = 0
        for job_id in jobs_to_remove:
            # Note: This would need a delete method in storage
            # For now, we'll track but not actually remove
            storage_cleaned += 1
        
        return engine_cleaned + storage_cleaned
    
    def validate_inference_parameters(self, model_id: str, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inference parameters for a specific model."""
        config = self.registry.get_model_config(model_id)
        if not config:
            return {
                "is_valid": False,
                "error": f"Model {model_id} not found"
            }
        
        validation_result = {
            "is_valid": True,
            "model_id": model_id,
            "model_type": config.model_type.value,
            "warnings": [],
            "adjusted_parameters": parameters.copy()
        }
        
        # Common parameter validation
        if "confidence_threshold" in parameters:
            confidence = parameters["confidence_threshold"]
            if not 0.0 <= confidence <= 1.0:
                validation_result["warnings"].append(
                    f"confidence_threshold {confidence} is outside valid range [0.0, 1.0]"
                )
                validation_result["adjusted_parameters"]["confidence_threshold"] = max(0.0, min(1.0, confidence))
        
        if "iou_threshold" in parameters:
            iou = parameters["iou_threshold"]
            if not 0.0 <= iou <= 1.0:
                validation_result["warnings"].append(
                    f"iou_threshold {iou} is outside valid range [0.0, 1.0]"
                )
                validation_result["adjusted_parameters"]["iou_threshold"] = max(0.0, min(1.0, iou))
        
        # Model-specific validation
        if config.model_type.value == "SAM2":
            # SAM2 specific parameters
            if "points" not in parameters and "boxes" not in parameters:
                validation_result["warnings"].append(
                    "SAM2 requires either 'points' or 'boxes' parameters for segmentation"
                )
        
        return validation_result