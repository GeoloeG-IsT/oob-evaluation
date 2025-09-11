# ML Evaluation Platform - Configuration Guide

This document provides comprehensive guidance for configuring the ML Evaluation Platform across different environments.

## Table of Contents

1. [Overview](#overview)
2. [Environment Configuration](#environment-configuration)
3. [Security Configuration](#security-configuration)
4. [Database Configuration](#database-configuration)
5. [Storage Configuration](#storage-configuration)
6. [ML Configuration](#ml-configuration)
7. [Deployment Configuration](#deployment-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Troubleshooting](#troubleshooting)

## Overview

The ML Evaluation Platform uses a hierarchical configuration system:

- **Environment Variables**: Primary configuration method
- **Configuration Files**: `.env` files for different environments
- **Secret Manager**: Secure storage for sensitive data (production)
- **Configuration Classes**: Type-safe configuration validation

### Configuration Hierarchy

1. Environment variables (highest priority)
2. `.env.local` file (local development)
3. `.env.{environment}` file (environment-specific)
4. `.env` file (defaults)
5. Configuration class defaults (lowest priority)

## Environment Configuration

### Available Environments

| Environment | Description | Use Case |
|-------------|-------------|----------|
| `development` | Local development | Hot reloading, debugging |
| `testing` | Automated testing | CI/CD, unit tests |
| `staging` | Pre-production | Integration testing |
| `production` | Live system | Production deployment |

### Setting Up Environments

1. **Copy the template file:**
   ```bash
   cp .env.template .env.local
   ```

2. **Configure for your environment:**
   ```bash
   # Set the environment
   ENVIRONMENT=development
   
   # Configure basic settings
   SECRET_KEY=your-secure-secret-key-here
   DATABASE_URL=postgresql://user:password@localhost:5432/ml_eval_platform
   REDIS_URL=redis://localhost:6379/0
   ```

3. **Validate configuration:**
   ```bash
   python backend/src/core/config_validator.py validate
   ```

## Security Configuration

### Secret Key Generation

Generate secure keys for production:

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Or use the configuration utility
python backend/src/core/config_validator.py generate
```

### Required Security Settings

| Setting | Description | Example | Required |
|---------|-------------|---------|----------|
| `SECRET_KEY` | Main application secret | `abc123...` | ✅ |
| `JWT_SECRET` | JWT signing key | `def456...` | ⚠️ Optional |
| `ENCRYPTION_KEY` | Data encryption key | `ghi789...` | ⚠️ Optional |
| `CORS_ORIGINS` | Allowed origins | `https://app.example.com` | ✅ |

### Production Security Checklist

- [ ] Use strong, unique secret keys (64+ characters)
- [ ] Enable HTTPS for all URLs
- [ ] Restrict CORS origins to specific domains
- [ ] Use GCP Secret Manager for sensitive data
- [ ] Enable rate limiting
- [ ] Configure SSL for database connections
- [ ] Use service accounts with minimal permissions

## Database Configuration

### PostgreSQL Setup

#### Local Development
```bash
# Using Docker
docker run -d \
  --name ml-eval-postgres \
  -e POSTGRES_DB=ml_eval_platform_dev \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:15
```

#### Production (Cloud SQL)
```bash
# Create Cloud SQL instance using Terraform
cd gcp/terraform
terraform init
terraform plan -var="project_id=your-project-id"
terraform apply
```

### Database Settings

| Setting | Description | Default | Production |
|---------|-------------|---------|------------|
| `DATABASE_URL` | Connection string | Required | From Secret Manager |
| `DB_POOL_SIZE` | Connection pool size | 10 | 20 |
| `DB_MAX_OVERFLOW` | Max overflow connections | 20 | 30 |
| `DB_POOL_TIMEOUT` | Connection timeout | 30s | 30s |
| `DB_RETRY_ATTEMPTS` | Retry attempts | 3 | 5 |

### Running Migrations

```bash
# Development
cd backend
alembic upgrade head

# Production (Cloud Run Job)
gcloud run jobs execute ml-eval-migrate-production --region=us-central1
```

## Storage Configuration

### Local Storage

For development and testing:

```bash
# Create storage directories
mkdir -p data/{models,images,datasets,temp}

# Set permissions
chmod 755 data/{models,images,datasets,temp}
```

### Google Cloud Storage

For production deployment:

```bash
# Create storage bucket
gsutil mb -p your-project-id -c STANDARD -l us-central1 gs://ml-eval-platform-prod-storage

# Set lifecycle policy
gsutil lifecycle set gcp/storage-lifecycle.json gs://ml-eval-platform-prod-storage
```

### Storage Settings

| Setting | Description | Local | GCS |
|---------|-------------|-------|-----|
| `USE_GCS` | Enable GCS | `false` | `true` |
| `GCS_BUCKET_NAME` | Bucket name | - | `ml-eval-prod-storage` |
| `MODEL_STORAGE_PATH` | Model storage | `/app/models` | `/app/models` |
| `IMAGE_STORAGE_PATH` | Image storage | `/app/images` | `/app/images` |
| `UPLOAD_MAX_SIZE` | Max file size | 1GB | 2GB |

## ML Configuration

### GPU Configuration

#### Local Development (NVIDIA Docker)
```bash
# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Production (Cloud Run with GPU)
GPU support is configured in the Cloud Run job specifications.

### ML Settings

| Setting | Description | Development | Production |
|---------|-------------|-------------|------------|
| `USE_GPU` | Enable GPU | `true` (if available) | `true` |
| `CUDA_VISIBLE_DEVICES` | GPU devices | `0` | `0` |
| `DEFAULT_MODEL_TYPE` | Default model | `yolo11n` | `yolo11s` |
| `MODEL_CACHE_SIZE` | Cached models | 5 | 10 |
| `MAX_TRAINING_JOBS` | Concurrent training | 1 | 5 |
| `MAX_INFERENCE_JOBS` | Concurrent inference | 2 | 10 |

### Supported Models

#### YOLO11/12 Variants
- `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`

#### RT-DETR Variants
- `rtdetr-r18`, `rtdetr-r34`, `rtdetr-r50`, `rtdetr-r101`
- `rtdetr-nano`, `rtdetr-small`, `rtdetr-medium`

#### SAM2 Variants
- `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-large`

## Deployment Configuration

### Local Development

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Check services
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend
```

### Staging Deployment

```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run health checks
curl http://localhost:8000/health
```

### Production Deployment

#### Prerequisites
1. GCP project with required APIs enabled
2. Terraform installed and configured
3. Docker images built and pushed to GCR

#### Deployment Steps

1. **Infrastructure Setup:**
   ```bash
   cd gcp/terraform
   terraform init
   terraform plan -var="project_id=your-project-id"
   terraform apply
   ```

2. **Deploy Application:**
   ```bash
   # Trigger Cloud Build
   gcloud builds submit --config=gcp/cloudbuild.yml .
   
   # Or deploy manually
   gcloud run deploy ml-eval-backend-production \
     --image=gcr.io/your-project-id/ml-eval-backend:latest \
     --region=us-central1 \
     --platform=managed
   ```

3. **Verify Deployment:**
   ```bash
   # Check service status
   gcloud run services list --region=us-central1
   
   # Test endpoints
   curl https://ml-eval-backend-production-hash.a.run.app/health
   ```

### Environment-Specific URLs

| Environment | Backend URL | Frontend URL |
|-------------|-------------|--------------|
| Development | `http://localhost:8000` | `http://localhost:3000` |
| Staging | `https://ml-eval-backend-staging-hash.a.run.app` | `https://ml-eval-frontend-staging-hash.a.run.app` |
| Production | `https://ml-eval-backend-production-hash.a.run.app` | `https://ml-eval-frontend-production-hash.a.run.app` |

## Monitoring and Logging

### Logging Configuration

| Setting | Description | Development | Production |
|---------|-------------|-------------|------------|
| `LOG_LEVEL` | Logging level | `DEBUG` | `INFO` |
| `STRUCTURED_LOGGING` | JSON logging | `false` | `true` |
| `SENTRY_DSN` | Error tracking | Optional | Recommended |

### Health Checks

The platform provides several health check endpoints:

- `/health` - Basic health check
- `/health/ready` - Readiness check
- `/health/live` - Liveness check
- `/health/deps` - Dependency health

### Monitoring Setup

```bash
# Enable monitoring APIs
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com

# View logs
gcloud logging read 'resource.type="cloud_run_revision"' --limit=50

# View metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"
```

## Configuration Validation

### Manual Validation

```bash
# Validate current configuration
python backend/src/core/config_validator.py validate

# Check network connectivity
python backend/src/core/config_validator.py connectivity

# Generate secure values
python backend/src/core/config_validator.py generate
```

### Automated Validation

The platform automatically validates configuration on startup:

1. **Environment checks** - Validates environment-specific settings
2. **Security checks** - Ensures secure configuration
3. **Connectivity checks** - Tests database and Redis connections
4. **Resource checks** - Validates storage paths and permissions

### Common Validation Errors

| Error | Solution |
|-------|----------|
| `SECRET_KEY too short` | Generate a 32+ character key |
| `Invalid DATABASE_URL` | Check connection string format |
| `Redis connection failed` | Verify Redis server is running |
| `Storage path not writable` | Check directory permissions |
| `GPU not available` | Set `USE_GPU=false` or install CUDA |

## Troubleshooting

### Common Issues

#### Configuration Loading Issues

```bash
# Check configuration loading
python -c "from backend.src.core.config import get_settings; print(get_settings().app.environment)"

# Validate configuration
python backend/src/core/config_validator.py validate
```

#### Database Connection Issues

```bash
# Test database connection
python -c "
from backend.src.core.config import get_settings
import psycopg2
settings = get_settings()
conn = psycopg2.connect(str(settings.database.database_url))
print('Database connection successful')
"
```

#### Redis Connection Issues

```bash
# Test Redis connection
python -c "
import redis
from backend.src.core.config import get_settings
settings = get_settings()
r = redis.from_url(str(settings.redis.redis_url))
r.ping()
print('Redis connection successful')
"
```

#### GCP Secret Manager Issues

```bash
# Test Secret Manager access
python backend/src/core/secrets_manager.py get SECRET_KEY your-project-id

# List available secrets
python backend/src/core/secrets_manager.py list your-project-id
```

### Debug Mode

Enable debug mode for detailed error information:

```bash
# Development
DEBUG=true LOG_LEVEL=DEBUG python app.py

# Docker
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG your-image
```

### Getting Help

1. Check the [troubleshooting guide](TROUBLESHOOTING.md)
2. Review application logs
3. Validate configuration settings
4. Check service dependencies
5. Contact support with error details

## Configuration Reference

For a complete list of all configuration options, see:
- [Environment Variables Reference](ENV_VARIABLES.md)
- [Security Best Practices](SECURITY.md)
- [Deployment Guide](DEPLOYMENT.md)