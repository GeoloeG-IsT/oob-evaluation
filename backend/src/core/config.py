"""
Comprehensive configuration management for ML Evaluation Platform.

This module provides centralized configuration management with support for:
- Environment-specific settings
- Secure secrets management
- Configuration validation
- GCP Secret Manager integration
- Development and production environments
"""

import os
import logging
import secrets
from typing import Optional, List, Dict, Any
from pathlib import Path
from functools import lru_cache

from pydantic import validator, Field
from pydantic.networks import PostgresDsn, RedisDsn, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'
    )


class DatabaseSettings(BaseConfig):
    """Database configuration settings."""
    
    # Primary database URL (PostgreSQL)
    database_url: PostgresDsn = Field(..., env="DATABASE_URL")
    
    
    # Database pool settings
    db_pool_size: int = Field(10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    db_pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    
    # Database connection retries
    db_retry_attempts: int = Field(3, env="DB_RETRY_ATTEMPTS")
    db_retry_delay: int = Field(1, env="DB_RETRY_DELAY")
    



class RedisSettings(BaseConfig):
    """Redis configuration settings."""
    
    redis_url: RedisDsn = Field(..., env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_timeout: int = Field(30, env="REDIS_TIMEOUT")
    redis_max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS")



class SecuritySettings(BaseConfig):
    """Security and authentication settings."""
    
    # Main application secret key
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # JWT settings
    jwt_secret: Optional[str] = Field(None, env="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(3600, env="JWT_EXPIRATION_SECONDS")
    
    # API keys and tokens
    api_key_header: str = Field("X-API-Key", env="API_KEY_HEADER")
    # admin_api_keys: List[str] = Field([], env="ADMIN_API_KEYS")
    
    # CORS settings
    cors_origins: str = Field("*", env="CORS_ORIGINS")
    cors_methods: str = Field("*", env="CORS_METHODS")
    cors_headers: str = Field("*", env="CORS_HEADERS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(3600, env="RATE_LIMIT_WINDOW")
    
    # Encryption settings
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    # @validator("cors_origins", pre=True)
    # def parse_cors_origins(cls, v):
    #     if isinstance(v, str):
    #         if v.strip() == '':
    #             return ["*"]  # Default to allow all if empty
    #         return [origin.strip() for origin in v.split(",")]
    #     return v
    
    # @validator("admin_api_keys", pre=True)
    # def parse_admin_api_keys(cls, v):
    #     if isinstance(v, str):
    #         if v.strip() == '':
    #             return []
    #         return [key.strip() for key in v.split(",") if key.strip()]
    #     return v



class StorageSettings(BaseConfig):
    """Storage configuration settings."""
    
    # Local storage paths
    model_storage_path: Path = Field(Path("/app/models"), env="MODEL_STORAGE_PATH")
    image_storage_path: Path = Field(Path("/app/images"), env="IMAGE_STORAGE_PATH")
    dataset_storage_path: Path = Field(Path("/app/datasets"), env="DATASET_STORAGE_PATH")
    temp_storage_path: Path = Field(Path("/app/temp"), env="TEMP_STORAGE_PATH")
    
    # Upload settings
    upload_max_size: int = Field(1073741824, env="UPLOAD_MAX_SIZE")  # 1GB default
    upload_allowed_extensions: str = Field(
        ".jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp",
        env="UPLOAD_ALLOWED_EXTENSIONS"
    )
    
    # GCS settings
    gcs_bucket_name: Optional[str] = Field(None, env="GCS_BUCKET_NAME")
    gcs_project_id: Optional[str] = Field(None, env="GCS_PROJECT_ID")
    gcs_credentials_path: Optional[str] = Field(None, env="GCS_CREDENTIALS_PATH")
    use_gcs: bool = Field(False, env="USE_GCS")
    
    # Storage cleanup settings
    temp_file_ttl: int = Field(3600, env="TEMP_FILE_TTL")  # 1 hour
    cleanup_enabled: bool = Field(True, env="STORAGE_CLEANUP_ENABLED")
    
    @validator("model_storage_path", "image_storage_path", "dataset_storage_path", "temp_storage_path")
    def validate_paths(cls, v):
        if isinstance(v, str):
            v = Path(v)
        return v
    
    # @validator("upload_allowed_extensions", pre=True)
    # def parse_extensions(cls, v):
    #     if isinstance(v, str):
    #         if v.strip() == '':
    #             return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]  # Default extensions
    #         return [ext.strip() for ext in v.split(",")]
    #     return v



class MLSettings(BaseConfig):
    """Machine Learning and model configuration settings."""
    
    # CUDA and GPU settings
    cuda_visible_devices: str = Field("0", env="CUDA_VISIBLE_DEVICES")
    torch_home: Path = Field(Path("/app/.torch"), env="TORCH_HOME")
    use_gpu: bool = Field(True, env="USE_GPU")
    gpu_memory_fraction: float = Field(0.8, env="GPU_MEMORY_FRACTION")
    
    # Model settings
    default_model_type: str = Field("yolo11n", env="DEFAULT_MODEL_TYPE")
    model_cache_size: int = Field(5, env="MODEL_CACHE_SIZE")
    model_timeout: int = Field(300, env="MODEL_TIMEOUT_SECONDS")
    
    # Training settings
    default_epochs: int = Field(100, env="DEFAULT_EPOCHS")
    default_batch_size: int = Field(16, env="DEFAULT_BATCH_SIZE")
    default_learning_rate: float = Field(0.01, env="DEFAULT_LEARNING_RATE")
    max_training_jobs: int = Field(2, env="MAX_TRAINING_JOBS")
    
    # Inference settings
    inference_batch_size: int = Field(32, env="INFERENCE_BATCH_SIZE")
    max_inference_jobs: int = Field(5, env="MAX_INFERENCE_JOBS")
    inference_timeout: int = Field(600, env="INFERENCE_TIMEOUT_SECONDS")
    
    # Evaluation settings
    evaluation_metrics: str = Field(
        "mAP,mAP50,precision,recall,f1",
        env="EVALUATION_METRICS"
    )
    confidence_threshold: float = Field(0.25, env="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(0.5, env="IOU_THRESHOLD")
    
    # @validator("evaluation_metrics", pre=True)
    # def parse_metrics(cls, v):
    #     if isinstance(v, str):
    #         if v.strip() == '':
    #             return ["mAP", "IoU", "precision", "recall", "F1"]  # Default metrics
    #         return [metric.strip() for metric in v.split(",")]
    #     return v



class CelerySettings(BaseConfig):
    """Celery task queue configuration settings."""
    
    # Broker settings
    broker_url: RedisDsn = Field(..., env="CELERY_BROKER_URL")
    result_backend: RedisDsn = Field(..., env="CELERY_RESULT_BACKEND")
    
    # Task settings
    task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    accept_content: List[str] = Field(["json"], env="CELERY_ACCEPT_CONTENT")
    timezone: str = Field("UTC", env="CELERY_TIMEZONE")
    
    # Worker settings
    worker_concurrency: int = Field(2, env="CELERY_WORKER_CONCURRENCY")
    worker_max_tasks_per_child: int = Field(1000, env="CELERY_WORKER_MAX_TASKS")
    worker_prefetch_multiplier: int = Field(1, env="CELERY_WORKER_PREFETCH")
    
    # Task routing
    task_routes: Dict[str, Dict[str, Any]] = Field(
        {
            "training.*": {"queue": "training"},
            "inference.*": {"queue": "inference"},
            "evaluation.*": {"queue": "evaluation"},
            "deployment.*": {"queue": "deployment"},
        },
        env="CELERY_TASK_ROUTES"
    )
    
    # Monitoring
    flower_port: int = Field(5555, env="FLOWER_PORT")
    flower_auth: Optional[str] = Field(None, env="FLOWER_AUTH")



class LoggingSettings(BaseConfig):
    """Logging configuration settings."""
    
    log_level: str = Field("INFO", env="LOG_LEVEL")
    structured_logging: bool = Field(True, env="STRUCTURED_LOGGING")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # File logging
    log_file_enabled: bool = Field(False, env="LOG_FILE_ENABLED")
    log_file_path: Optional[Path] = Field(None, env="LOG_FILE_PATH")
    log_file_max_size: int = Field(10485760, env="LOG_FILE_MAX_SIZE")  # 10MB
    log_file_backup_count: int = Field(5, env="LOG_FILE_BACKUP_COUNT")
    
    # External logging services
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    datadog_api_key: Optional[str] = Field(None, env="DATADOG_API_KEY")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()



class GCPSettings(BaseConfig):
    """Google Cloud Platform configuration settings."""
    
    # Project settings
    project_id: Optional[str] = Field(None, env="GCP_PROJECT_ID")
    region: str = Field("us-central1", env="GCP_REGION")
    zone: str = Field("us-central1-a", env="GCP_ZONE")
    
    # Authentication
    credentials_path: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    service_account_email: Optional[str] = Field(None, env="GCP_SERVICE_ACCOUNT_EMAIL")
    
    # Cloud Run settings
    cloud_run_service_name: Optional[str] = Field(None, env="CLOUD_RUN_SERVICE_NAME")
    cloud_run_memory: str = Field("2Gi", env="CLOUD_RUN_MEMORY")
    cloud_run_cpu: str = Field("2", env="CLOUD_RUN_CPU")
    cloud_run_max_instances: int = Field(10, env="CLOUD_RUN_MAX_INSTANCES")
    cloud_run_min_instances: int = Field(0, env="CLOUD_RUN_MIN_INSTANCES")
    
    # Secret Manager
    use_secret_manager: bool = Field(False, env="USE_SECRET_MANAGER")
    secret_manager_project: Optional[str] = Field(None, env="SECRET_MANAGER_PROJECT")
    
    # Cloud Storage
    storage_bucket: Optional[str] = Field(None, env="GCP_STORAGE_BUCKET")
    
    # AI Platform
    ai_platform_region: str = Field("us-central1", env="AI_PLATFORM_REGION")



class ApplicationSettings(BaseConfig):
    """Main application configuration settings."""
    
    # Application info
    app_name: str = Field("ML Evaluation Platform", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    app_description: str = Field(
        "ML Evaluation Platform for Object Detection and Segmentation",
        env="APP_DESCRIPTION"
    )
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    
    # URLs and networking
    backend_url: HttpUrl = Field(..., env="BACKEND_URL")
    frontend_url: HttpUrl = Field(..., env="FRONTEND_URL")
    api_version: str = Field("v1", env="API_VERSION")
    
    # Server settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")
    
    # Feature flags
    enable_training: bool = Field(True, env="ENABLE_TRAINING")
    enable_inference: bool = Field(True, env="ENABLE_INFERENCE")
    enable_evaluation: bool = Field(True, env="ENABLE_EVALUATION")
    enable_deployment: bool = Field(True, env="ENABLE_DEPLOYMENT")
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    
    # Performance settings
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")
    max_request_size: int = Field(104857600, env="MAX_REQUEST_SIZE")  # 100MB
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production", "testing"]
        if v not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of: {valid_envs}")
        return v


class Settings(BaseConfig):
    """Master settings class that combines all configuration sections."""
    
    # Configuration sections
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    # celery: CelerySettings = Field(default_factory=CelerySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_configuration()
        self._setup_directories()
    
    def _validate_configuration(self):
        """Perform cross-section validation."""
        # Ensure Celery broker matches Redis URL
        # if hasattr(self.celery, 'broker_url') and hasattr(self.redis, 'redis_url'):
        #     if str(self.celery.broker_url) != str(self.redis.redis_url):
        #         logger.warning("Celery broker URL differs from Redis URL")
        
        # Validate GCP settings
        if self.gcp.use_secret_manager and not self.gcp.project_id:
            raise ValueError("GCP_PROJECT_ID is required when using Secret Manager")
        
        # Validate production settings
        if self.app.environment == "production":
            if self.app.debug:
                logger.warning("DEBUG should be False in production")
            if self.security.secret_key == "development-secret-key":
                raise ValueError("SECRET_KEY must be changed for production")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.storage.model_storage_path,
            self.storage.image_storage_path,
            self.storage.dataset_storage_path,
            self.storage.temp_storage_path,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
            except PermissionError:
                logger.warning(f"Permission denied creating directory: {directory}")
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.app.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.app.environment == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.app.testing or self.app.environment == "testing"
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """Get database URL with optional async driver."""
        url = str(self.database.database_url)
        if async_driver:
            url = url.replace("postgresql://", "postgresql+asyncpg://")
        return url
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment or Secret Manager."""
        # First try environment variables
        value = os.getenv(key, default)
        
        # If using GCP Secret Manager and no env var found
        if not value and self.gcp.use_secret_manager:
            try:
                from .secrets_manager import get_secret_from_gcp
                value = get_secret_from_gcp(key, self.gcp.project_id)
            except ImportError:
                logger.warning("GCP Secret Manager not available")
            except Exception as e:
                logger.error(f"Error fetching secret {key}: {e}")
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "app": self.app.dict(),
            "database": self.database.dict(),
            "redis": self.redis.dict(),
            "security": self.security.dict(),
            "storage": self.storage.dict(),
            "ml": self.ml.dict(),
            # "celery": self.celery.dict(),
            "logging": self.logging.dict(),
            "gcp": self.gcp.dict(),
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def generate_secret_key() -> str:
    """Generate a secure secret key."""
    return secrets.token_urlsafe(32)


def validate_environment() -> None:
    """Validate current environment configuration."""
    try:
        settings = get_settings()
        logger.info(f"Configuration loaded successfully for environment: {settings.app.environment}")
        
        # Log configuration summary
        logger.info(f"Database: {settings.database.database_url}")
        logger.info(f"Redis: {settings.redis.redis_url}")
        logger.info(f"Storage: {settings.storage.model_storage_path}")
        logger.info(f"ML Backend: GPU={settings.ml.use_gpu}")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


if __name__ == "__main__":
    # CLI for configuration testing
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            validate_environment()
        elif command == "generate-key":
            print(generate_secret_key())
        elif command == "show":
            settings = get_settings()
            import json
            print(json.dumps(settings.to_dict(), indent=2, default=str))
    else:
        print("Usage: python config.py [validate|generate-key|show]")