"""
Performance evaluation API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import uuid

from ..services.evaluation_service import EvaluationService
from ..schemas.evaluation import (
    EvaluationRequest, EvaluationResult, EvaluationListResponse,
    MetricComparisonRequest, MetricComparisonResponse, MetricType
)


router = APIRouter(prefix="/api/v1", tags=["evaluation"])
evaluation_service = EvaluationService()


@router.post("/evaluation/metrics", response_model=EvaluationResult, status_code=201)
async def calculate_metrics(request: EvaluationRequest):
    """Calculate performance metrics for a model."""
    # Validate UUID formats
    try:
        uuid.UUID(request.model_id)
        uuid.UUID(request.dataset_id)
        for annotation_id in request.ground_truth_annotations:
            uuid.UUID(annotation_id)
        for annotation_id in request.predicted_annotations:
            uuid.UUID(annotation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    # Validate that we have matching counts of ground truth and predicted annotations
    if len(request.ground_truth_annotations) != len(request.predicted_annotations):
        raise HTTPException(
            status_code=400, 
            detail={"error": "Ground truth and predicted annotations must have the same count"}
        )
    
    if not request.ground_truth_annotations:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one annotation pair is required"}
        )
    
    try:
        # Calculate evaluation metrics
        evaluation_result = evaluation_service.calculate_metrics(
            request.model_id,
            request.dataset_id,
            request.ground_truth_annotations,
            request.predicted_annotations,
            request.confidence_threshold,
            request.iou_threshold,
            request.metrics_to_calculate,
            request.metadata
        )
        
        if not evaluation_result:
            raise HTTPException(
                status_code=404,
                detail={"error": "Model, dataset, or annotations not found"}
            )
        
        return evaluation_result
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in calculate_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/evaluation/metrics", response_model=EvaluationListResponse)
async def list_evaluations(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    limit: int = Query(50, description="Number of evaluations to return"),
    offset: int = Query(0, description="Number of evaluations to skip")
):
    """List evaluation results with optional filtering and pagination."""
    # Validate UUID formats if provided
    if model_id:
        try:
            uuid.UUID(model_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid model_id UUID format"})
    
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
        evaluations, total_count = evaluation_service.list_evaluations(model_id, dataset_id, limit, offset)
        
        return EvaluationListResponse(
            evaluations=evaluations,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        print(f"Unexpected error in list_evaluations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/evaluation/metrics/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(evaluation_id: str):
    """Get a single evaluation result by ID."""
    # Basic UUID validation
    try:
        uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    evaluation = evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Evaluation not found",
                "message": f"No evaluation found with ID: {evaluation_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    return evaluation


@router.post("/evaluation/compare", response_model=MetricComparisonResponse)
async def compare_metrics(request: MetricComparisonRequest):
    """Compare metrics across multiple evaluations."""
    # Validate evaluation IDs
    if len(request.evaluation_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least 2 evaluation IDs are required for comparison"}
        )
    
    if len(request.evaluation_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail={"error": "Maximum of 10 evaluations can be compared at once"}
        )
    
    # Validate UUID formats
    try:
        for evaluation_id in request.evaluation_ids:
            uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        comparison_result = evaluation_service.compare_evaluations(
            request.evaluation_ids,
            request.metric_types
        )
        
        if not comparison_result:
            raise HTTPException(
                status_code=404,
                detail={"error": "One or more evaluations not found"}
            )
        
        return comparison_result
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in compare_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.delete("/evaluation/metrics/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation result."""
    # Basic UUID validation
    try:
        uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        success = evaluation_service.delete_evaluation(evaluation_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "Evaluation not found"}
            )
        
        return {"message": "Evaluation deleted successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in delete_evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/evaluation/metrics/{evaluation_id}/export")
async def export_evaluation_results(evaluation_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    """Export evaluation results in specified format."""
    # Basic UUID validation
    try:
        uuid.UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    evaluation = evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=404,
            detail={"error": "Evaluation not found"}
        )
    
    try:
        export_data = evaluation_service.export_evaluation_results(evaluation_id, format)
        
        if format == "csv":
            return {"content_type": "text/csv", "data": export_data}
        else:
            return {"content_type": "application/json", "data": export_data}
            
    except Exception as e:
        print(f"Unexpected error in export_evaluation_results: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/evaluation/stats")
async def get_evaluation_stats():
    """Get overall evaluation statistics."""
    try:
        stats = evaluation_service.get_evaluation_stats()
        return stats
        
    except Exception as e:
        print(f"Unexpected error in get_evaluation_stats: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})