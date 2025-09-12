"""
Inference execution API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import uuid

from ..services.inference_service import InferenceService
from ..schemas.inference import (
    SingleInferenceRequest, BatchInferenceRequest, SingleInferenceResponse, 
    InferenceJobResponse, InferenceJobListResponse, JobStatus, Priority
)


router = APIRouter(prefix="/api/v1", tags=["inference"])
inference_service = InferenceService()


@router.post("/inference/single", response_model=SingleInferenceResponse)
async def run_single_inference(request: SingleInferenceRequest):
    """Run inference on a single image."""
    # Validate UUID formats
    try:
        uuid.UUID(request.image_id)
        uuid.UUID(request.model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        # Execute single inference
        result = inference_service.run_single_inference(
            request.image_id,
            request.model_id,
            request.confidence_threshold,
            request.nms_threshold,
            request.max_detections,
            request.metadata
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail={"error": "Image or model not found"}
            )
        
        return result
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in run_single_inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/inference/batch", response_model=InferenceJobResponse, status_code=202)
async def run_batch_inference(request: BatchInferenceRequest):
    """Run batch inference on multiple images."""
    # Validate image_ids array
    if not request.image_ids or len(request.image_ids) == 0:
        raise HTTPException(status_code=400, detail={"error": "At least one image_id is required"})
    
    # Validate UUID formats
    try:
        uuid.UUID(request.model_id)
        for image_id in request.image_ids:
            uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        # Create and start batch inference job
        job = inference_service.create_batch_inference_job(
            request.image_ids,
            request.model_id,
            request.confidence_threshold,
            request.nms_threshold,
            request.batch_size,
            request.max_detections,
            request.priority,
            request.metadata
        )
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Model not found"}
            )
        
        return InferenceJobResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            image_ids=job.target_images,
            model_id=job.model_id,
            total_images=len(job.target_images),
            processed_images=int(job.progress_percentage / 100 * len(job.target_images)),
            created_at=job.created_at,
            started_at=job.start_time,
            completed_at=job.end_time,
            results=job.results.get("inference_results", []) if job.results else [],
            progress_percentage=job.progress_percentage,
            priority=request.priority,
            metadata=job.metadata
        )
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in run_batch_inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/inference/jobs", response_model=InferenceJobListResponse)
async def list_inference_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(50, description="Number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip")
):
    """List inference jobs with optional filtering and pagination."""
    # Validate status filter
    if status and status not in [s.value for s in JobStatus]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid status: {status}"})
    
    # Validate model_id format if provided
    if model_id:
        try:
            uuid.UUID(model_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    if offset < 0:
        raise HTTPException(status_code=400, detail={"error": "Offset must be non-negative"})
    
    if limit < 1:
        raise HTTPException(status_code=400, detail={"error": "Limit must be positive"})
    
    # Cap limit at maximum
    if limit > 1000:
        limit = 1000
    
    try:
        jobs, total_count = inference_service.list_inference_jobs(status, model_id, limit, offset)
        
        # Convert to response format
        job_responses = []
        for job in jobs:
            job_response = InferenceJobResponse(
                job_id=job.id,
                status=JobStatus(job.status),
                image_ids=job.target_images,
                model_id=job.model_id,
                total_images=len(job.target_images),
                processed_images=int(job.progress_percentage / 100 * len(job.target_images)),
                created_at=job.created_at,
                started_at=job.start_time,
                completed_at=job.end_time,
                results=job.results.get("inference_results", []) if job.results else [],
                progress_percentage=job.progress_percentage,
                metadata=job.metadata
            )
            job_responses.append(job_response)
        
        return InferenceJobListResponse(
            jobs=job_responses,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        print(f"Unexpected error in list_inference_jobs: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/inference/jobs/{job_id}", response_model=InferenceJobResponse)
async def get_inference_job(job_id: str):
    """Get a single inference job by ID."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    job = inference_service.get_inference_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Inference job not found",
                "message": f"No inference job found with ID: {job_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    return InferenceJobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        image_ids=job.target_images,
        model_id=job.model_id,
        total_images=len(job.target_images),
        processed_images=int(job.progress_percentage / 100 * len(job.target_images)),
        created_at=job.created_at,
        started_at=job.start_time,
        completed_at=job.end_time,
        results=job.results.get("inference_results", []) if job.results else [],
        error_message=job.execution_logs if job.status == "failed" else None,
        progress_percentage=job.progress_percentage,
        metadata=job.metadata
    )


@router.post("/inference/jobs/{job_id}/cancel", response_model=InferenceJobResponse)
async def cancel_inference_job(job_id: str):
    """Cancel a running inference job."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        job = inference_service.cancel_inference_job(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Inference job not found"}
            )
        
        return InferenceJobResponse(
            job_id=job.id,
            status=JobStatus(job.status),
            image_ids=job.target_images,
            model_id=job.model_id,
            total_images=len(job.target_images),
            processed_images=int(job.progress_percentage / 100 * len(job.target_images)),
            created_at=job.created_at,
            started_at=job.start_time,
            completed_at=job.end_time,
            results=job.results.get("inference_results", []) if job.results else [],
            progress_percentage=job.progress_percentage,
            metadata=job.metadata
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in cancel_inference_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})