"""
Data export API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import uuid

from ..services.export_service import ExportService
from ..schemas.export import (
    AnnotationExportRequest, DatasetExportRequest, EvaluationExportRequest,
    ModelExportRequest, ExportJobResponse, ExportJobListResponse,
    ExportStatsResponse, BulkExportRequest, BulkExportResponse,
    ExportFormat, ExportType, ExportStatus
)


router = APIRouter(prefix="/api/v1", tags=["export"])
export_service = ExportService()


@router.post("/export/annotations", response_model=ExportJobResponse, status_code=202)
async def export_annotations(request: AnnotationExportRequest):
    """Export annotations in specified format."""
    # Validate UUID formats if provided
    if request.image_ids:
        try:
            for image_id in request.image_ids:
                uuid.UUID(image_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid image_id UUID format"})
    
    if request.annotation_ids:
        try:
            for annotation_id in request.annotation_ids:
                uuid.UUID(annotation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid annotation_id UUID format"})
    
    # Validate that at least one filter is provided
    if not request.image_ids and not request.annotation_ids and not request.dataset_split:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one of image_ids, annotation_ids, or dataset_split must be provided"}
        )
    
    try:
        export_job = export_service.export_annotations(request)
        
        if not export_job:
            raise HTTPException(
                status_code=404,
                detail={"error": "No annotations found matching the criteria"}
            )
        
        return export_job
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in export_annotations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/export/datasets", response_model=ExportJobResponse, status_code=202)
async def export_dataset(request: DatasetExportRequest):
    """Export complete dataset in specified format."""
    # Validate UUID format
    try:
        uuid.UUID(request.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid dataset_id UUID format"})
    
    # Validate splits
    valid_splits = ["train", "validation", "test"]
    if request.splits_to_include:
        for split in request.splits_to_include:
            if split not in valid_splits:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"Invalid dataset split: {split}. Valid options: {valid_splits}"}
                )
    
    try:
        export_job = export_service.export_dataset(request)
        
        if not export_job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Dataset not found"}
            )
        
        return export_job
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in export_dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/export/evaluations", response_model=ExportJobResponse, status_code=202)
async def export_evaluations(request: EvaluationExportRequest):
    """Export evaluation results in specified format."""
    # Validate evaluation IDs
    if not request.evaluation_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one evaluation_id is required"}
        )
    
    try:
        for evaluation_id in request.evaluation_ids:
            uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid evaluation_id UUID format"})
    
    try:
        export_job = export_service.export_evaluations(request)
        
        if not export_job:
            raise HTTPException(
                status_code=404,
                detail={"error": "One or more evaluations not found"}
            )
        
        return export_job
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in export_evaluations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/export/models", response_model=ExportJobResponse, status_code=202)
async def export_model(request: ModelExportRequest):
    """Export model weights and configuration."""
    # Validate UUID format
    try:
        uuid.UUID(request.model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid model_id UUID format"})
    
    # Validate export options
    if not request.export_weights and not request.export_config and not request.export_metadata:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one export option must be enabled (weights, config, or metadata)"}
        )
    
    try:
        export_job = export_service.export_model(request)
        
        if not export_job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Model not found"}
            )
        
        return export_job
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in export_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/export/bulk", response_model=BulkExportResponse, status_code=202)
async def bulk_export(request: BulkExportRequest):
    """Create multiple export jobs in a single request."""
    if not request.export_requests:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one export request is required"}
        )
    
    if len(request.export_requests) > 20:
        raise HTTPException(
            status_code=400,
            detail={"error": "Maximum of 20 export requests allowed in a single bulk request"}
        )
    
    try:
        bulk_result = export_service.bulk_export(request)
        
        return bulk_result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in bulk_export: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/export/jobs", response_model=ExportJobListResponse)
async def list_export_jobs(
    export_type: Optional[str] = Query(None, description="Filter by export type"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    format: Optional[str] = Query(None, description="Filter by export format"),
    limit: int = Query(50, description="Number of jobs to return"),
    offset: int = Query(0, description="Number of jobs to skip")
):
    """List export jobs with optional filtering and pagination."""
    # Validate export_type filter
    if export_type and export_type not in [t.value for t in ExportType]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid export type: {export_type}"})
    
    # Validate status filter
    if status and status not in [s.value for s in ExportStatus]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid status: {status}"})
    
    # Validate format filter
    if format and format not in [f.value for f in ExportFormat]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid format: {format}"})
    
    if offset < 0:
        raise HTTPException(status_code=400, detail={"error": "Offset must be non-negative"})
    
    if limit < 1:
        raise HTTPException(status_code=400, detail={"error": "Limit must be positive"})
    
    # Cap limit at maximum
    if limit > 1000:
        limit = 1000
    
    try:
        jobs, total_count = export_service.list_export_jobs(export_type, status, format, limit, offset)
        
        return ExportJobListResponse(
            jobs=jobs,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        print(f"Unexpected error in list_export_jobs: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/export/jobs/{job_id}", response_model=ExportJobResponse)
async def get_export_job(job_id: str):
    """Get a single export job by ID."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    export_job = export_service.get_export_job(job_id)
    if not export_job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Export job not found",
                "message": f"No export job found with ID: {job_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    return export_job


@router.post("/export/jobs/{job_id}/cancel", response_model=ExportJobResponse)
async def cancel_export_job(job_id: str):
    """Cancel a running export job."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        export_job = export_service.cancel_export_job(job_id)
        if not export_job:
            raise HTTPException(
                status_code=404,
                detail={"error": "Export job not found"}
            )
        
        return export_job
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        elif "cannot" in error_msg or "already" in error_msg:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in cancel_export_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/export/jobs/{job_id}/download")
async def download_export_file(job_id: str):
    """Download the exported file."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        download_info = export_service.get_download_info(job_id)
        if not download_info:
            raise HTTPException(
                status_code=404,
                detail={"error": "Export job not found or file not available"}
            )
        
        return download_info
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "not available" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        elif "expired" in error_msg:
            raise HTTPException(status_code=410, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in download_export_file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.delete("/export/jobs/{job_id}")
async def delete_export_job(job_id: str):
    """Delete an export job and its files."""
    # Basic UUID validation
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        success = export_service.delete_export_job(job_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "Export job not found"}
            )
        
        return {"message": "Export job deleted successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in delete_export_job: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/export/stats", response_model=ExportStatsResponse)
async def get_export_stats():
    """Get overall export statistics."""
    try:
        return export_service.get_export_stats()
        
    except Exception as e:
        print(f"Unexpected error in get_export_stats: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})