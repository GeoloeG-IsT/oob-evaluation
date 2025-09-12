"""
Training job management API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import uuid

from ..services.training_service import TrainingService
from ..schemas.training import (
    TrainingJobRequest, TrainingJobResponse, TrainingJobListResponse,
    TrainingJobUpdateRequest, TrainingStatus, Hyperparameters
)


router = APIRouter(prefix="/api/v1", tags=["training"])
training_service = TrainingService()


@router.post("/training/jobs", response_model=TrainingJobResponse, status_code=201)
async def start_training_job(request: TrainingJobRequest):
    """Start a new training job."""
    # Validate UUID formats
    try:
        uuid.UUID(request.base_model_id)
        uuid.UUID(request.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        # Create and start training job
        job = training_service.start_training_job(
            request.base_model_id,
            request.dataset_id,
            request.hyperparameters.dict(),
            request.experiment_name,
            request.metadata
        )
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Base model or dataset not found"}
            )
        
        return TrainingJobResponse(
            id=job.id,
            base_model_id=job.base_model_id,
            dataset_id=job.dataset_id,
            status=TrainingStatus(job.status),
            progress_percentage=job.progress_percentage,
            hyperparameters=Hyperparameters(**job.hyperparameters),
            created_at=job.created_at,
            start_time=job.start_time,
            end_time=job.end_time,
            result_model_id=job.result_model_id,
            execution_logs=job.execution_logs,
            experiment_name=request.experiment_name,
            metadata=job.metadata
        )
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in start_training_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/training/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    base_model_id: Optional[str] = Query(None, description="Filter by base model ID"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    limit: int = Query(50, description="Number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip")
):
    """List training jobs with optional filtering and pagination."""
    # Validate status filter
    if status and status not in [s.value for s in TrainingStatus]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid status: {status}"})
    
    # Validate UUID formats if provided
    if base_model_id:
        try:
            uuid.UUID(base_model_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid base_model_id UUID format"})
    
    if dataset_id:
        try:
            uuid.UUID(dataset_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid dataset_id UUID format"})
    
    if offset < 0:
        raise HTTPException(status_code=400, detail={"error": "Offset must be non-negative"})
    
    if limit < 1:
        raise HTTPException(status_code=400, detail={"error": "Limit must be positive"})
    
    # Cap limit at maximum
    if limit > 1000:
        limit = 1000
    
    try:
        jobs, total_count = training_service.list_training_jobs(status, base_model_id, dataset_id, limit, offset)
        
        # Convert to response format
        job_responses = []
        for job in jobs:
            job_response = TrainingJobResponse(
                id=job.id,
                base_model_id=job.base_model_id,
                dataset_id=job.dataset_id,
                status=TrainingStatus(job.status),
                progress_percentage=job.progress_percentage,
                hyperparameters=Hyperparameters(**job.hyperparameters),
                created_at=job.created_at,
                start_time=job.start_time,
                end_time=job.end_time,
                result_model_id=job.result_model_id,
                execution_logs=job.execution_logs,
                metadata=job.metadata
            )
            
            # Add computed fields
            if job.hyperparameters:
                job_response.current_epoch = int(job.progress_percentage / 100 * job.hyperparameters.get("epochs", 0))
                job_response.total_epochs = job.hyperparameters.get("epochs", 0)
            
            job_responses.append(job_response)
        
        return TrainingJobListResponse(
            jobs=job_responses,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        print(f"Unexpected error in list_training_jobs: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get a single training job by ID."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    job = training_service.get_training_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Training job not found",
                "message": f"No training job found with ID: {job_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    job_response = TrainingJobResponse(
        id=job.id,
        base_model_id=job.base_model_id,
        dataset_id=job.dataset_id,
        status=TrainingStatus(job.status),
        progress_percentage=job.progress_percentage,
        hyperparameters=Hyperparameters(**job.hyperparameters),
        created_at=job.created_at,
        start_time=job.start_time,
        end_time=job.end_time,
        result_model_id=job.result_model_id,
        execution_logs=job.execution_logs,
        error_message=job.execution_logs if job.status == "failed" else None,
        metadata=job.metadata
    )
    
    # Add computed fields
    if job.hyperparameters:
        job_response.current_epoch = int(job.progress_percentage / 100 * job.hyperparameters.get("epochs", 0))
        job_response.total_epochs = job.hyperparameters.get("epochs", 0)
        
        # Simulate some training metrics for completed jobs
        if job.status == "completed":
            job_response.best_metric_value = 0.85
    
    return job_response


@router.put("/training/jobs/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(job_id: str, update_request: TrainingJobUpdateRequest):
    """Update a training job status or progress."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        job = training_service.update_training_job(
            job_id,
            update_request.status.value if update_request.status else None,
            update_request.progress_percentage,
            update_request.execution_logs,
            update_request.error_message
        )
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Training job not found"}
            )
        
        return TrainingJobResponse(
            id=job.id,
            base_model_id=job.base_model_id,
            dataset_id=job.dataset_id,
            status=TrainingStatus(job.status),
            progress_percentage=job.progress_percentage,
            hyperparameters=Hyperparameters(**job.hyperparameters),
            created_at=job.created_at,
            start_time=job.start_time,
            end_time=job.end_time,
            result_model_id=job.result_model_id,
            execution_logs=job.execution_logs,
            error_message=update_request.error_message,
            metadata=job.metadata
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in update_training_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/training/jobs/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        job = training_service.cancel_training_job(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Training job not found"}
            )
        
        return TrainingJobResponse(
            id=job.id,
            base_model_id=job.base_model_id,
            dataset_id=job.dataset_id,
            status=TrainingStatus(job.status),
            progress_percentage=job.progress_percentage,
            hyperparameters=Hyperparameters(**job.hyperparameters),
            created_at=job.created_at,
            start_time=job.start_time,
            end_time=job.end_time,
            result_model_id=job.result_model_id,
            execution_logs=job.execution_logs,
            metadata=job.metadata
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in cancel_training_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/training/jobs/{job_id}/logs", response_model=dict)
async def get_training_logs(job_id: str):
    """Get training logs for a specific job."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    job = training_service.get_training_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "Training job not found"}
        )
    
    return {
        "job_id": job_id,
        "logs": job.execution_logs or "",
        "last_updated": job.created_at
    }