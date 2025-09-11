"""
Pydantic schemas for deployment-related API models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UPDATING = "updating"


class DeploymentType(str, Enum):
    REST_API = "rest_api"
    GRPC = "grpc"
    BATCH = "batch"


class ScalingPolicy(str, Enum):
    FIXED = "fixed"
    AUTO = "auto"
    ON_DEMAND = "on_demand"


class ResourceRequirements(BaseModel):
    cpu_cores: float = Field(..., ge=0.1, le=64.0)
    memory_mb: int = Field(..., ge=128, le=65536)
    gpu_count: Optional[int] = Field(0, ge=0, le=8)
    gpu_memory_mb: Optional[int] = Field(None, ge=1024, le=32768)
    disk_space_mb: Optional[int] = Field(1024, ge=512, le=102400)


class AutoScalingConfig(BaseModel):
    min_instances: int = Field(..., ge=1, le=100)
    max_instances: int = Field(..., ge=1, le=100)
    target_cpu_utilization: float = Field(0.7, ge=0.1, le=0.95)
    target_memory_utilization: float = Field(0.8, ge=0.1, le=0.95)
    scale_up_cooldown_seconds: int = Field(300, ge=60, le=3600)
    scale_down_cooldown_seconds: int = Field(300, ge=60, le=3600)

    @field_validator('max_instances')
    @classmethod
    def max_instances_must_be_greater_than_min(cls, v, info):
        if info.data.get('min_instances') and v < info.data['min_instances']:
            raise ValueError('max_instances must be greater than or equal to min_instances')
        return v


class DeploymentConfig(BaseModel):
    deployment_name: str = Field(..., min_length=3, max_length=50)
    deployment_type: DeploymentType = DeploymentType.REST_API
    scaling_policy: ScalingPolicy = ScalingPolicy.FIXED
    instance_count: Optional[int] = Field(1, ge=1, le=10)
    resource_requirements: ResourceRequirements
    auto_scaling_config: Optional[AutoScalingConfig] = None
    environment_variables: Optional[Dict[str, str]] = {}
    health_check_path: Optional[str] = "/health"
    health_check_interval_seconds: int = Field(30, ge=10, le=300)
    timeout_seconds: int = Field(300, ge=30, le=3600)


class DeploymentRequest(BaseModel):
    model_id: str
    config: DeploymentConfig
    metadata: Optional[Dict[str, Any]] = None


class DeploymentHealthCheck(BaseModel):
    status: str  # healthy, unhealthy, unknown
    last_check_time: str
    response_time_ms: float
    error_message: Optional[str] = None


class DeploymentMetrics(BaseModel):
    requests_per_minute: float
    average_response_time_ms: float
    error_rate_percent: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    active_connections: int
    throughput_images_per_second: float


class DeploymentInstance(BaseModel):
    instance_id: str
    status: str  # running, starting, stopping, stopped, failed
    created_at: str
    resource_usage: Dict[str, float]
    health_status: str


class DeploymentResponse(BaseModel):
    id: str
    model_id: str
    deployment_name: str
    status: DeploymentStatus
    deployment_type: DeploymentType
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    config: DeploymentConfig
    created_at: str
    updated_at: str
    deployed_at: Optional[str] = None
    health_check: Optional[DeploymentHealthCheck] = None
    metrics: Optional[DeploymentMetrics] = None
    instances: Optional[List[DeploymentInstance]] = []
    error_message: Optional[str] = None
    version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None


class DeploymentListResponse(BaseModel):
    deployments: List[DeploymentResponse]
    total_count: int
    limit: int
    offset: int


class DeploymentUpdateRequest(BaseModel):
    status: Optional[DeploymentStatus] = None
    config: Optional[DeploymentConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class DeploymentStatsResponse(BaseModel):
    total_deployments: int
    running_deployments: int
    failed_deployments: int
    total_requests_today: int
    average_response_time_ms: float
    total_compute_hours: float