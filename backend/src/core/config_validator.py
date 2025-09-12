"""
Configuration validation utilities for ML Evaluation Platform.

This module provides comprehensive validation of configuration settings,
environment checks, and security validations to ensure the application
runs safely across different environments.
"""

import logging
import socket
import secrets
from typing import Dict, List, Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class SecurityValidationError(ConfigValidationError):
    """Raised when security validation fails."""
    pass


class ConfigValidator:
    """Validates configuration settings across all environments."""
    
    def __init__(self, settings = None):
        """Initialize validator with settings.
        
        Args:
            settings: Settings instance to validate. If None, gets current settings.
        """
        if settings is None:
            from core.config import get_settings
            settings = get_settings()
        self.settings = settings
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks.
        
        Returns:
            Dictionary with validation results
        """
        self.errors.clear()
        self.warnings.clear()
        self.info.clear()
        
        logger.info("Starting comprehensive configuration validation")
        
        # Run all validation checks
        self._validate_environment()
        self._validate_security()
        self._validate_database()
        self._validate_redis()
        self._validate_storage()
        self._validate_ml_settings()
        self._validate_networking()
        self._validate_gcp_settings()
        self._validate_feature_flags()
        self._validate_performance_settings()
        
        # Compile results
        results = {
            "valid": len(self.errors) == 0,
            "environment": self.settings.app.environment,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "summary": {
                "total_checks": len(self.errors) + len(self.warnings) + len(self.info),
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "info": len(self.info),
            }
        }
        
        if results["valid"]:
            logger.info("Configuration validation passed")
        else:
            logger.error(f"Configuration validation failed with {len(self.errors)} errors")
        
        return results
    
    def _validate_environment(self):
        """Validate environment-specific settings."""
        env = self.settings.app.environment
        
        # Check environment value
        valid_envs = ["development", "staging", "production", "testing"]
        if env not in valid_envs:
            self.errors.append(f"Invalid environment '{env}'. Must be one of: {valid_envs}")
        
        # Environment-specific checks
        if env == "production":
            self._validate_production_environment()
        elif env == "development":
            self._validate_development_environment()
        elif env == "staging":
            self._validate_staging_environment()
        elif env == "testing":
            self._validate_testing_environment()
        
        self.info.append(f"Environment: {env}")
    
    def _validate_production_environment(self):
        """Validate production-specific settings."""
        # Debug should be false in production
        if self.settings.app.debug:
            self.errors.append("DEBUG must be False in production")
        
        # Secret key must be strong
        if self.settings.security.secret_key in [
            "your-secret-key-here",
            "development-secret-key",
            "test-secret-key-for-testing-only-do-not-use-in-production"
        ]:
            self.errors.append("SECRET_KEY must be changed from default for production")
        
        # CORS should be restricted
        if "*" in self.settings.security.cors_origins:
            self.warnings.append("CORS origins should be restricted in production")
        
        # HTTPS should be used
        backend_url = str(self.settings.app.backend_url)
        frontend_url = str(self.settings.app.frontend_url)
        
        if not backend_url.startswith("https://"):
            self.warnings.append("Backend URL should use HTTPS in production")
        
        if not frontend_url.startswith("https://"):
            self.warnings.append("Frontend URL should use HTTPS in production")
        
        # Database should use SSL
        db_url = str(self.settings.database.database_url)
        if "sslmode" not in db_url and not db_url.startswith("sqlite"):
            self.warnings.append("Database should use SSL in production")
        
        # GCP settings should be configured
        if not self.settings.gcp.project_id:
            self.warnings.append("GCP_PROJECT_ID should be set in production")
        
        self.info.append("Production environment validation completed")
    
    def _validate_development_environment(self):
        """Validate development-specific settings."""
        # Development can have relaxed settings
        if self.settings.app.debug:
            self.info.append("Debug mode enabled for development")
        
        # Local URLs are acceptable
        self.info.append("Development environment validation completed")
    
    def _validate_staging_environment(self):
        """Validate staging-specific settings."""
        # Staging should mimic production but can be more relaxed
        if self.settings.app.debug:
            self.warnings.append("Consider disabling debug mode in staging")
        
        self.info.append("Staging environment validation completed")
    
    def _validate_testing_environment(self):
        """Validate testing-specific settings."""
        # Testing should use test databases
        db_url = str(self.settings.database.database_url)
        if "test" not in db_url.lower() and not db_url.startswith("sqlite"):
            self.warnings.append("Testing should use a test database")
        
        # Testing can use mock services
        self.info.append("Testing environment validation completed")
    
    def _validate_security(self):
        """Validate security settings."""
        # Secret key strength
        secret_key = self.settings.security.secret_key
        if len(secret_key) < 32:
            self.errors.append("SECRET_KEY must be at least 32 characters long")
        
        # Check for default/weak keys
        weak_keys = [
            "your-secret-key-here",
            "development-secret-key",
            "test-secret-key",
            "changeme",
            "secret",
            "password"
        ]
        
        if secret_key.lower() in [k.lower() for k in weak_keys]:
            self.errors.append("SECRET_KEY appears to be a default/weak value")
        
        # JWT secret validation
        if self.settings.security.jwt_secret:
            if len(self.settings.security.jwt_secret) < 32:
                self.warnings.append("JWT_SECRET should be at least 32 characters long")
        
        # Rate limiting validation
        if self.settings.security.rate_limit_enabled:
            if self.settings.security.rate_limit_requests <= 0:
                self.errors.append("RATE_LIMIT_REQUESTS must be positive when rate limiting is enabled")
            
            if self.settings.security.rate_limit_window <= 0:
                self.errors.append("RATE_LIMIT_WINDOW must be positive when rate limiting is enabled")
        
        self.info.append("Security validation completed")
    
    def _validate_database(self):
        """Validate database configuration."""
        db_url = str(self.settings.database.database_url)
        
        # Parse database URL
        try:
            parsed = urlparse(db_url)
            
            # Check scheme
            if parsed.scheme not in ["postgresql", "postgres", "sqlite"]:
                self.errors.append(f"Unsupported database scheme: {parsed.scheme}")
            
            # Check host for non-sqlite databases
            if parsed.scheme in ["postgresql", "postgres"] and not parsed.hostname:
                self.errors.append("Database hostname is required for PostgreSQL")
            
            # Check credentials for non-sqlite databases
            if parsed.scheme in ["postgresql", "postgres"]:
                if not parsed.username:
                    self.warnings.append("Database username should be specified")
                if not parsed.password and self.settings.app.environment != "testing":
                    self.warnings.append("Database password should be specified")
        
        except Exception as e:
            self.errors.append(f"Invalid database URL: {e}")
        
        # Pool settings validation
        if self.settings.database.db_pool_size <= 0:
            self.errors.append("DB_POOL_SIZE must be positive")
        
        if self.settings.database.db_max_overflow < 0:
            self.errors.append("DB_MAX_OVERFLOW must be non-negative")
        
        self.info.append("Database validation completed")
    
    def _validate_redis(self):
        """Validate Redis configuration."""
        redis_url = str(self.settings.redis.redis_url)
        
        # Parse Redis URL
        try:
            parsed = urlparse(redis_url)
            
            if parsed.scheme not in ["redis", "rediss"]:
                self.errors.append(f"Invalid Redis scheme: {parsed.scheme}")
            
            if not parsed.hostname:
                self.errors.append("Redis hostname is required")
            
            # Check SSL in production
            if (self.settings.app.environment == "production" and 
                parsed.scheme != "rediss"):
                self.warnings.append("Consider using Redis SSL (rediss://) in production")
        
        except Exception as e:
            self.errors.append(f"Invalid Redis URL: {e}")
        
        # Connection settings
        if self.settings.redis.redis_max_connections <= 0:
            self.errors.append("REDIS_MAX_CONNECTIONS must be positive")
        
        self.info.append("Redis validation completed")
    
    def _validate_storage(self):
        """Validate storage configuration."""
        # Check storage paths
        paths_to_check = [
            ("MODEL_STORAGE_PATH", self.settings.storage.model_storage_path),
            ("IMAGE_STORAGE_PATH", self.settings.storage.image_storage_path),
            ("DATASET_STORAGE_PATH", self.settings.storage.dataset_storage_path),
            ("TEMP_STORAGE_PATH", self.settings.storage.temp_storage_path),
        ]
        
        for name, path in paths_to_check:
            if not path.is_absolute():
                self.warnings.append(f"{name} should be an absolute path")
        
        # Upload settings
        if self.settings.storage.upload_max_size <= 0:
            self.errors.append("UPLOAD_MAX_SIZE must be positive")
        
        # GCS validation
        if self.settings.storage.use_gcs:
            if not self.settings.storage.gcs_bucket_name:
                self.errors.append("GCS_BUCKET_NAME is required when USE_GCS is true")
            
            if not self.settings.storage.gcs_project_id:
                self.errors.append("GCS_PROJECT_ID is required when USE_GCS is true")
        
        self.info.append("Storage validation completed")
    
    def _validate_ml_settings(self):
        """Validate machine learning configuration."""
        # GPU settings
        if self.settings.ml.use_gpu:
            cuda_devices = self.settings.ml.cuda_visible_devices
            if cuda_devices and not cuda_devices.replace(",", "").replace(" ", "").isdigit():
                self.warnings.append("CUDA_VISIBLE_DEVICES format may be invalid")
        
        # Model settings
        if self.settings.ml.model_cache_size <= 0:
            self.errors.append("MODEL_CACHE_SIZE must be positive")
        
        if self.settings.ml.model_timeout <= 0:
            self.errors.append("MODEL_TIMEOUT_SECONDS must be positive")
        
        # Training settings
        if self.settings.ml.default_epochs <= 0:
            self.errors.append("DEFAULT_EPOCHS must be positive")
        
        if self.settings.ml.default_batch_size <= 0:
            self.errors.append("DEFAULT_BATCH_SIZE must be positive")
        
        if not (0 < self.settings.ml.default_learning_rate < 1):
            self.warnings.append("DEFAULT_LEARNING_RATE should be between 0 and 1")
        
        # Threshold validation
        if not (0 <= self.settings.ml.confidence_threshold <= 1):
            self.errors.append("CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if not (0 <= self.settings.ml.iou_threshold <= 1):
            self.errors.append("IOU_THRESHOLD must be between 0 and 1")
        
        self.info.append("ML settings validation completed")
    
    def _validate_networking(self):
        """Validate networking configuration."""
        # URL validation
        urls_to_check = [
            ("BACKEND_URL", self.settings.app.backend_url),
            ("FRONTEND_URL", self.settings.app.frontend_url),
        ]
        
        for name, url in urls_to_check:
            try:
                parsed = urlparse(str(url))
                if not parsed.scheme or not parsed.netloc:
                    self.errors.append(f"{name} is not a valid URL")
                
                if parsed.scheme not in ["http", "https"]:
                    self.warnings.append(f"{name} should use HTTP or HTTPS")
            
            except Exception as e:
                self.errors.append(f"Invalid {name}: {e}")
        
        # Port validation
        if not (1 <= self.settings.app.port <= 65535):
            self.errors.append("PORT must be between 1 and 65535")
        
        # Worker validation
        if self.settings.app.workers <= 0:
            self.errors.append("WORKERS must be positive")
        
        self.info.append("Networking validation completed")
    
    def _validate_gcp_settings(self):
        """Validate Google Cloud Platform configuration."""
        if self.settings.gcp.use_secret_manager:
            if not self.settings.gcp.project_id:
                self.errors.append("GCP_PROJECT_ID is required when using Secret Manager")
            
            if not self.settings.gcp.secret_manager_project:
                self.warnings.append("SECRET_MANAGER_PROJECT should be specified")
        
        # Cloud Run validation
        if self.settings.gcp.cloud_run_max_instances <= 0:
            self.errors.append("CLOUD_RUN_MAX_INSTANCES must be positive")
        
        if self.settings.gcp.cloud_run_min_instances < 0:
            self.errors.append("CLOUD_RUN_MIN_INSTANCES must be non-negative")
        
        if self.settings.gcp.cloud_run_min_instances > self.settings.gcp.cloud_run_max_instances:
            self.errors.append("CLOUD_RUN_MIN_INSTANCES cannot exceed CLOUD_RUN_MAX_INSTANCES")
        
        self.info.append("GCP settings validation completed")
    
    def _validate_feature_flags(self):
        """Validate feature flag configuration."""
        # No specific validation needed for boolean flags
        enabled_features = []
        
        if self.settings.app.enable_training:
            enabled_features.append("training")
        if self.settings.app.enable_inference:
            enabled_features.append("inference")
        if self.settings.app.enable_evaluation:
            enabled_features.append("evaluation")
        if self.settings.app.enable_deployment:
            enabled_features.append("deployment")
        if self.settings.app.enable_monitoring:
            enabled_features.append("monitoring")
        
        self.info.append(f"Enabled features: {', '.join(enabled_features)}")
    
    def _validate_performance_settings(self):
        """Validate performance-related settings."""
        # Request settings
        if self.settings.app.request_timeout <= 0:
            self.errors.append("REQUEST_TIMEOUT must be positive")
        
        if self.settings.app.max_request_size <= 0:
            self.errors.append("MAX_REQUEST_SIZE must be positive")
        
        # Celery settings
        # if self.settings.celery.worker_concurrency <= 0:
        #     self.errors.append("CELERY_WORKER_CONCURRENCY must be positive")
        # 
        # if self.settings.celery.worker_max_tasks_per_child <= 0:
        #     self.warnings.append("CELERY_WORKER_MAX_TASKS should be positive")
        
        self.info.append("Performance settings validation completed")


def validate_configuration(settings = None) -> Dict[str, Any]:
    """Validate configuration and return results.
    
    Args:
        settings: Settings instance to validate
        
    Returns:
        Validation results dictionary
    """
    validator = ConfigValidator(settings)
    return validator.validate_all()


def check_network_connectivity() -> Dict[str, bool]:
    """Check network connectivity to required services.
    
    Returns:
        Dictionary with connectivity status
    """
    from core.config import get_settings
    settings = get_settings()
    results = {}
    
    # Check database connectivity
    try:
        db_url = str(settings.database.database_url)
        parsed = urlparse(db_url)
        
        if parsed.hostname and parsed.port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((parsed.hostname, parsed.port))
            results["database"] = result == 0
            sock.close()
        else:
            results["database"] = True  # Local SQLite or invalid URL
    except Exception:
        results["database"] = False
    
    # Check Redis connectivity
    try:
        redis_url = str(settings.redis.redis_url)
        parsed = urlparse(redis_url)
        
        if parsed.hostname and parsed.port:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((parsed.hostname, parsed.port))
            results["redis"] = result == 0
            sock.close()
        else:
            results["redis"] = False
    except Exception:
        results["redis"] = False
    
    return results


def generate_secure_config() -> Dict[str, str]:
    """Generate secure configuration values.
    
    Returns:
        Dictionary with secure values
    """
    return {
        "SECRET_KEY": secrets.token_urlsafe(32),
        "JWT_SECRET": secrets.token_urlsafe(32),
        "ENCRYPTION_KEY": secrets.token_urlsafe(32),
        "ADMIN_API_KEYS": secrets.token_urlsafe(24),
    }


if __name__ == "__main__":
    # CLI for configuration validation
    import sys
    import json
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            try:
                results = validate_configuration()
                print(json.dumps(results, indent=2))
                sys.exit(0 if results["valid"] else 1)
            except Exception as e:
                print(json.dumps({"error": str(e), "valid": False}, indent=2))
                sys.exit(1)
        
        elif command == "connectivity":
            try:
                results = check_network_connectivity()
                print(json.dumps(results, indent=2))
                all_connected = all(results.values())
                sys.exit(0 if all_connected else 1)
            except Exception as e:
                print(json.dumps({"error": str(e)}, indent=2))
                sys.exit(1)
        
        elif command == "generate":
            # Generate command doesn't need settings to be loaded
            config = generate_secure_config()
            print(json.dumps(config, indent=2))
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("Usage: python config_validator.py [validate|connectivity|generate]")
        sys.exit(1)