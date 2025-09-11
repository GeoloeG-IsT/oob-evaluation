# ML Evaluation Platform - Configuration & Secrets Management

This document provides a complete guide to the environment variable configuration and secrets management system implemented for the ML Evaluation Platform.

## 🎯 Overview

The ML Evaluation Platform features a comprehensive, production-ready configuration management system that:

- **Centralizes all configuration** across development, staging, and production environments
- **Implements secure secrets management** using GCP Secret Manager
- **Provides environment-specific configurations** with validation and error handling
- **Integrates seamlessly with Docker and Cloud Run** deployments
- **Follows security best practices** with encryption and access controls

## 🏗️ Architecture

### Configuration Hierarchy

```
1. Environment Variables (Highest Priority)
2. .env.local (Local Development)
3. .env.{environment} (Environment-Specific)
4. .env.template (Default Template)
5. Configuration Class Defaults (Lowest Priority)
```

### Core Components

```
backend/src/core/
├── config.py              # Main configuration classes
├── config_validator.py    # Validation and error handling
├── secrets_manager.py     # GCP Secret Manager integration
└── security.py           # Security utilities and validation
```

## 🚀 Quick Start

### 1. Initialize Development Environment

```bash
# Copy template and customize
cp .env.template .env.local

# Generate secure keys
python backend/src/core/config_validator.py generate

# Validate configuration
python backend/src/core/config_validator.py validate
```

### 2. Start Development Services

```bash
# Start with development configuration
docker-compose -f docker-compose.dev.yml up -d

# Check service health
docker-compose -f docker-compose.dev.yml ps
```

### 3. Validate Setup

```bash
# Test configuration loading
python -c "from backend.src.core.config import get_settings; print(f'Environment: {get_settings().app.environment}')"

# Check connectivity
python backend/src/core/config_validator.py connectivity
```

## 📝 Configuration Files

### Environment Templates

| File | Purpose | Usage |
|------|---------|--------|
| `.env.template` | Master template with all options | Copy for new environments |
| `.env.development` | Development settings | Local development |
| `.env.testing` | Test configuration | CI/CD and unit tests |
| `.env.staging` | Staging environment | Pre-production testing |
| `.env.production` | Production settings | Live deployment |

### Docker Compose Files

| File | Environment | Features |
|------|-------------|----------|
| `docker-compose.dev.yml` | Development | Hot reload, debugging, admin tools |
| `docker-compose.yml` | Development | Basic setup |
| `docker-compose.staging.yml` | Staging | Production-like with monitoring |
| `docker-compose.production.yml` | Production | Optimized for Cloud Run |

## 🔧 Configuration Sections

### Application Settings

```bash
# Core application configuration
ENVIRONMENT=development
DEBUG=false
APP_NAME="ML Evaluation Platform"
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

### Security Configuration

```bash
# Security and authentication
SECRET_KEY=your-secure-secret-key-here
JWT_SECRET=your-jwt-secret
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
RATE_LIMIT_ENABLED=true
ENCRYPTION_KEY=your-encryption-key
```

### Database Configuration

```bash
# PostgreSQL settings
DATABASE_URL=postgresql://user:password@localhost:5432/ml_eval_platform
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_RETRY_ATTEMPTS=3
```

### ML Configuration

```bash
# Machine learning settings
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
DEFAULT_MODEL_TYPE=yolo11n
MAX_TRAINING_JOBS=2
INFERENCE_BATCH_SIZE=32
```

### Storage Configuration

```bash
# Storage and uploads
MODEL_STORAGE_PATH=/app/models
IMAGE_STORAGE_PATH=/app/images
UPLOAD_MAX_SIZE=1073741824
USE_GCS=false
GCS_BUCKET_NAME=ml-eval-platform-storage
```

## 🔐 Secrets Management

### Development (Local Files)

```bash
# Create local secrets file
touch .env.secrets

# Add sensitive values
echo "SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env.secrets
echo "JWT_SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env.secrets
```

### Production (GCP Secret Manager)

```bash
# Setup GCP Secret Manager
./scripts/config-migration.sh setup-gcp your-project-id

# Deploy secrets
./scripts/config-migration.sh deploy-secrets production

# Access secrets programmatically
python backend/src/core/secrets_manager.py get SECRET_KEY your-project-id
```

### Secret Manager Integration

The platform automatically retrieves secrets from GCP Secret Manager when:
- `USE_SECRET_MANAGER=true`
- `GCP_PROJECT_ID` is set
- Application is running with proper GCP credentials

## 🌍 Environment Setup

### Development Environment

```bash
# Initialize development configuration
./scripts/config-migration.sh init development

# Start development stack
docker-compose -f docker-compose.dev.yml up -d

# Access services
# - Backend API: http://localhost:8000
# - Frontend: http://localhost:3000
# - Flower (Celery): http://localhost:5555
# - pgAdmin: http://localhost:5050
# - Redis Commander: http://localhost:8081
```

### Staging Environment

```bash
# Initialize staging configuration
./scripts/config-migration.sh init staging

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run health checks
curl http://localhost:8000/health
```

### Production Environment

```bash
# Setup GCP infrastructure
cd gcp/terraform
terraform init
terraform plan -var="project_id=your-project-id"
terraform apply

# Deploy application
gcloud builds submit --config=gcp/cloudbuild.yml .

# Verify deployment
gcloud run services list --region=us-central1
```

## 🔍 Validation & Monitoring

### Configuration Validation

```bash
# Validate specific environment
./scripts/config-migration.sh validate production

# Generate security report
python backend/src/core/security.py report

# Check network connectivity
python backend/src/core/config_validator.py connectivity
```

### Health Monitoring

The platform provides comprehensive health checks:

- **`/health`** - Basic application health
- **`/health/ready`** - Readiness for traffic
- **`/health/live`** - Liveness check
- **`/health/deps`** - Dependency status (DB, Redis, etc.)

### Logging & Metrics

```bash
# View application logs
docker-compose logs -f backend

# Production logs (GCP)
gcloud logging read 'resource.type="cloud_run_revision"' --limit=50

# Monitoring (Staging)
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001
```

## 🛠️ Configuration Tools

### Migration Script

```bash
# Available commands
./scripts/config-migration.sh --help

# Initialize new environment
./scripts/config-migration.sh init staging

# Migrate between environments
./scripts/config-migration.sh migrate development staging

# Backup configuration
./scripts/config-migration.sh backup production

# Generate secure values
./scripts/config-migration.sh generate-secrets
```

### Python Configuration API

```python
from backend.src.core.config import get_settings

# Get configuration
settings = get_settings()
print(f"Environment: {settings.app.environment}")
print(f"Database URL: {settings.database.database_url}")

# Validate configuration
from backend.src.core.config_validator import validate_configuration
result = validate_configuration()
print(f"Valid: {result['valid']}")
```

## 🔒 Security Best Practices

### Production Security Checklist

- [ ] **Strong Secret Keys** - Use 64+ character random keys
- [ ] **HTTPS Everywhere** - All URLs use HTTPS in production
- [ ] **Secret Manager** - Sensitive data stored in GCP Secret Manager
- [ ] **Restricted CORS** - CORS origins limited to specific domains
- [ ] **Rate Limiting** - API rate limiting enabled
- [ ] **Database SSL** - Database connections use SSL
- [ ] **Minimal Permissions** - Service accounts with least privilege
- [ ] **Regular Rotation** - Secrets rotated periodically

### Security Validation

```bash
# Run security validation
python backend/src/core/security.py validate

# Generate security report
python backend/src/core/security.py report

# Check for vulnerabilities
./scripts/config-migration.sh validate production
```

## 🚀 Deployment Guide

### Cloud Run Deployment

1. **Prerequisites**
   ```bash
   # Enable APIs
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   ```

2. **Infrastructure Setup**
   ```bash
   cd gcp/terraform
   terraform apply -var="project_id=your-project-id"
   ```

3. **Application Deployment**
   ```bash
   # Build and deploy
   gcloud builds submit --config=gcp/cloudbuild.yml .
   
   # Update configuration
   gcloud run services update ml-eval-backend-production \
     --set-env-vars="NEW_VAR=value" \
     --region=us-central1
   ```

### Scaling Configuration

```yaml
# Cloud Run scaling
run.googleapis.com/cpu: "4"
run.googleapis.com/memory: "8Gi"
autoscaling.knative.dev/maxScale: "50"
autoscaling.knative.dev/minScale: "1"
```

## 📊 Monitoring & Observability

### Application Metrics

- **Request latency and throughput**
- **Error rates and status codes**
- **Database connection pool status**
- **Celery task queue metrics**
- **ML model inference performance**

### Infrastructure Metrics

- **CPU and memory utilization**
- **Database performance**
- **Storage usage**
- **Network traffic**

### Alerting

Configure alerts for:
- High error rates (>5%)
- Slow response times (>2s)
- Database connection failures
- Storage quota exceeded
- GPU utilization issues

## 🔧 Troubleshooting

### Common Issues

#### Configuration Loading Failed
```bash
# Check file permissions
ls -la .env*

# Validate syntax
python backend/src/core/config_validator.py validate
```

#### Database Connection Failed
```bash
# Test connection
python -c "
import psycopg2
conn = psycopg2.connect('your-database-url')
print('Connected successfully')
"
```

#### Secrets Not Loading
```bash
# Check GCP authentication
gcloud auth list

# Test Secret Manager access
python backend/src/core/secrets_manager.py list your-project-id
```

#### GPU Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify Docker GPU support
docker run --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with detailed output
python app.py
```

## 📚 Additional Resources

- [Configuration Reference](docs/CONFIGURATION.md) - Complete configuration guide
- [Security Best Practices](docs/SECURITY.md) - Security guidelines
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment
- [API Documentation](docs/API.md) - API reference
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Common issues

## 🤝 Contributing

When modifying configuration:

1. Update the `.env.template` file
2. Add validation in `config_validator.py`
3. Update documentation
4. Test across all environments
5. Update Docker Compose files if needed

## 📄 License

This configuration system is part of the ML Evaluation Platform and follows the same license terms.

---

**Need Help?** Check the troubleshooting guide or contact the development team with specific error details and environment information.