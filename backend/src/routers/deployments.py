"""
Model deployment API routes.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import uuid

from ..services.deployment_service import DeploymentService
from ..schemas.deployment import (
    DeploymentRequest, DeploymentResponse, DeploymentListResponse,
    DeploymentUpdateRequest, DeploymentStatus, DeploymentType,
    DeploymentStatsResponse
)


router = APIRouter(prefix="/api/v1", tags=["deployments"])
deployment_service = DeploymentService()


@router.post("/deployments", response_model=DeploymentResponse, status_code=201)
async def create_deployment(request: DeploymentRequest):
    """Create a new model deployment."""
    # Validate UUID format
    try:
        uuid.UUID(request.model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid model_id UUID format"})
    
    # Validate deployment name
    if not request.config.deployment_name or len(request.config.deployment_name.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail={"error": "Deployment name must be at least 3 characters long"}
        )
    
    try:
        # Create deployment
        deployment = deployment_service.create_deployment(
            request.model_id,
            request.config,
            request.metadata
        )
        
        if not deployment:
            raise HTTPException(
                status_code=404,
                detail={"error": "Model not found"}
            )
        
        return deployment
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        elif "already exists" in error_msg or "conflict" in error_msg:
            raise HTTPException(status_code=409, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in create_deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments", response_model=DeploymentListResponse)
async def list_deployments(
    status: Optional[str] = Query(None, description="Filter by deployment status"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    deployment_type: Optional[str] = Query(None, description="Filter by deployment type"),
    limit: int = Query(50, description="Number of deployments to return"),
    offset: int = Query(0, description="Number of deployments to skip")
):
    """List deployments with optional filtering and pagination."""
    # Validate status filter
    if status and status not in [s.value for s in DeploymentStatus]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid status: {status}"})
    
    # Validate deployment_type filter
    if deployment_type and deployment_type not in [t.value for t in DeploymentType]:
        raise HTTPException(status_code=400, detail={"error": f"Invalid deployment type: {deployment_type}"})
    
    # Validate model_id format if provided
    if model_id:
        try:
            uuid.UUID(model_id)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "Invalid model_id UUID format"})
    
    if offset < 0:
        raise HTTPException(status_code=400, detail={"error": "Offset must be non-negative"})
    
    if limit < 1:
        raise HTTPException(status_code=400, detail={"error": "Limit must be positive"})
    
    # Cap limit at maximum
    if limit > 1000:
        limit = 1000
    
    try:
        deployments, total_count = deployment_service.list_deployments(
            status, model_id, deployment_type, limit, offset
        )
        
        return DeploymentListResponse(
            deployments=deployments,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        print(f"Unexpected error in list_deployments: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(deployment_id: str):
    """Get a single deployment by ID."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    deployment = deployment_service.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Deployment not found",
                "message": f"No deployment found with ID: {deployment_id}",
                "timestamp": "2023-12-07T10:00:00Z"
            }
        )
    
    return deployment


@router.put("/deployments/{deployment_id}", response_model=DeploymentResponse)
async def update_deployment(deployment_id: str, update_request: DeploymentUpdateRequest):
    """Update a deployment configuration or status."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        deployment = deployment_service.update_deployment(
            deployment_id,
            update_request.status.value if update_request.status else None,
            update_request.config,
            update_request.metadata
        )
        
        if not deployment:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return deployment
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in update_deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.delete("/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str):
    """Delete a deployment."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        success = deployment_service.delete_deployment(deployment_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return {"message": "Deployment deleted successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in delete_deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/deployments/{deployment_id}/start", response_model=DeploymentResponse)
async def start_deployment(deployment_id: str):
    """Start a stopped deployment."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        deployment = deployment_service.start_deployment(deployment_id)
        if not deployment:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return deployment
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        elif "already" in error_msg or "cannot" in error_msg:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in start_deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentResponse)
async def stop_deployment(deployment_id: str):
    """Stop a running deployment."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        deployment = deployment_service.stop_deployment(deployment_id)
        if not deployment:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return deployment
        
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail={"error": str(e)})
        elif "already" in error_msg or "cannot" in error_msg:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        else:
            raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in stop_deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments/{deployment_id}/health")
async def get_deployment_health(deployment_id: str):
    """Get deployment health status."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        health_status = deployment_service.get_deployment_health(deployment_id)
        if not health_status:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return health_status
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in get_deployment_health: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments/{deployment_id}/metrics")
async def get_deployment_metrics(deployment_id: str):
    """Get deployment performance metrics."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        metrics = deployment_service.get_deployment_metrics(deployment_id)
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return metrics
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in get_deployment_metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments/{deployment_id}/logs")
async def get_deployment_logs(deployment_id: str, lines: int = Query(100, ge=1, le=1000)):
    """Get deployment logs."""
    # Basic UUID validation
    try:
        uuid.UUID(deployment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": "Invalid UUID format"})
    
    try:
        logs = deployment_service.get_deployment_logs(deployment_id, lines)
        if logs is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "Deployment not found"}
            )
        
        return {"deployment_id": deployment_id, "logs": logs}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        print(f"Unexpected error in get_deployment_logs: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})


@router.get("/deployments/stats", response_model=DeploymentStatsResponse)
async def get_deployment_stats():
    """Get overall deployment statistics."""
    try:
        return deployment_service.get_deployment_stats()
        
    except Exception as e:
        print(f"Unexpected error in get_deployment_stats: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": "Internal server error"})