# ML Evaluation Platform - Production Deployment Guide

This guide provides comprehensive instructions for deploying the ML Evaluation Platform to Google Cloud Platform using Cloud Run.

## Overview

The deployment consists of:

- **Backend Service**: FastAPI application with ML capabilities
- **Frontend Service**: Next.js web application  
- **Celery Workers**: ML training, inference, and evaluation workers
- **Infrastructure**: Cloud SQL (PostgreSQL), Cloud Memorystore (Redis), Cloud Storage
- **Monitoring**: Cloud Operations suite with dashboards and alerts
- **CI/CD**: Cloud Build pipelines for automated deployments

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Cloud CDN     │    │  Cloud Armor    │
└─────────┬───────┘    └─────────────────┘    └─────────────────┘
          │
┌─────────▼───────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │  Celery Workers │
│   (Cloud Run)   │    │   (Cloud Run)   │    │   (Cloud Run)   │
└─────────────────┘    └─────────┬───────┘    └─────────┬───────┘
                                │                       │
                       ┌────────▼────────┐    ┌────────▼────────┐
                       │   Cloud SQL     │    │ Cloud Memorystore│
                       │  (PostgreSQL)   │    │    (Redis)      │
                       └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Required Tools

1. **Google Cloud SDK** (gcloud)
   ```bash
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. **Terraform** (>= 1.0)
   ```bash
   # macOS
   brew install terraform
   
   # Linux
   wget https://releases.hashicorp.com/terraform/1.5.7/terraform_1.5.7_linux_amd64.zip
   unzip terraform_1.5.7_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```

3. **Docker** (for local builds)
   ```bash
   # Follow instructions at https://docs.docker.com/get-docker/
   ```

### GCP Setup

1. **Create or select a GCP project**
   ```bash
   gcloud projects create ml-eval-platform-prod --name="ML Evaluation Platform"
   gcloud config set project ml-eval-platform-prod
   ```

2. **Enable billing** (required for Cloud Run)
   - Go to [GCP Console](https://console.cloud.google.com/)
   - Navigate to Billing and link a billing account

3. **Set up authentication**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

## Quick Start Deployment

### Option 1: Automated Deployment (Recommended)

Use our deployment script for a fully automated setup:

```bash
# Clone the repository
git clone <repository-url>
cd ml-evaluation-platform

# Make script executable (if not already)
chmod +x deployment/scripts/deploy.sh

# Run deployment
./deployment/scripts/deploy.sh --project-id YOUR_PROJECT_ID --region us-central1
```

The script will:
- ✅ Check prerequisites
- ✅ Enable required APIs  
- ✅ Deploy infrastructure with Terraform
- ✅ Build and deploy all services
- ✅ Run health checks
- ✅ Provide service URLs

### Option 2: Manual Step-by-Step Deployment

#### Step 1: Infrastructure Deployment

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Create terraform.tfvars
cat > terraform.tfvars << EOF
project_id = "YOUR_PROJECT_ID"
region     = "us-central1"
environment = "production"
EOF

# Plan and apply
terraform plan
terraform apply
```

#### Step 2: Build and Deploy Services

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Deploy all services
gcloud builds submit \
  --config=deployment/cloudbuild/deploy-all.yaml \
  --substitutions=_REGION=us-central1 \
  .
```

## Configuration

### Environment Variables

Key environment variables are managed through Google Secret Manager:

| Secret | Description | Example |
|--------|-------------|---------|
| `database-url` | PostgreSQL connection | `postgresql://user:pass@host:5432/db` |
| `redis-url` | Redis connection | `redis://user:pass@host:6379/0` |
| `jwt-secret` | JWT signing key | Auto-generated 64-char string |
| `api-keys` | External API keys | JSON with OpenAI, HuggingFace tokens |

### Service Configuration

Each service is configured via Cloud Run service YAML files:

- `backend-service.yaml`: API service with 2 CPU, 4GB RAM
- `frontend-service.yaml`: Web app with 1 CPU, 1GB RAM  
- `celery-*-service.yaml`: Worker services with GPU support

### Infrastructure Configuration

Terraform modules configure:

- **Networking**: VPC, subnets, VPC connectors
- **Databases**: Cloud SQL with high availability
- **Cache**: Redis with persistence
- **Storage**: Multi-bucket setup with lifecycle policies
- **Security**: IAM roles and service accounts
- **Monitoring**: Dashboards and alerting

## CI/CD Pipeline

### Cloud Build Triggers

Set up automated deployments:

```bash
# Create build triggers for each service
gcloud builds triggers create github \
  --repo-name=ml-evaluation-platform \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=deployment/cloudbuild/backend.yaml \
  --name=backend-deploy

gcloud builds triggers create github \
  --repo-name=ml-evaluation-platform \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=deployment/cloudbuild/frontend.yaml \
  --name=frontend-deploy

gcloud builds triggers create github \
  --repo-name=ml-evaluation-platform \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=deployment/cloudbuild/celery.yaml \
  --name=celery-deploy
```

### Build Pipeline Features

- **Multi-stage Docker builds** for optimized images
- **Security scanning** with Container Analysis
- **Automated testing** with health checks
- **Blue-green deployments** with traffic splitting
- **Rollback capabilities** with revision management

## Monitoring and Observability

### Cloud Monitoring Dashboards

Access dashboards at: https://console.cloud.google.com/monitoring

1. **Main Dashboard**: Service health, request rates, latency
2. **Database Dashboard**: Connection pools, query performance
3. **ML Operations Dashboard**: Training jobs, inference throughput

### Alerting

Pre-configured alerts for:

- High error rates (>10%)
- Response time spikes (>5s p95)
- Memory usage (>85%)
- Database connections (>90% of limit)
- Service downtime
- Budget thresholds

### Log Analysis

```bash
# View application logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ml-eval-backend" --limit=100

# Query specific errors
gcloud logging read 'resource.type=cloud_run_revision AND severity>=ERROR' --limit=50
```

## Scaling and Performance

### Auto-scaling Configuration

Services auto-scale based on:

- **Backend**: 1-10 instances, CPU and memory utilization
- **Frontend**: 1-20 instances, request volume
- **Celery Workers**: 0-5 instances, queue depth

### Performance Optimization

1. **Cold Start Optimization**: 
   - Minimum instances configured
   - Startup CPU boost enabled
   - Optimized Docker images

2. **Database Performance**:
   - Connection pooling
   - Read replicas for production
   - Query optimization flags

3. **Caching Strategy**:
   - Redis for session data
   - CDN for static assets
   - Application-level caching

## Security

### Security Features

1. **Network Security**:
   - Private VPC networking
   - VPC Access Connectors
   - Cloud Armor protection

2. **Identity and Access**:
   - Service account isolation
   - Least privilege IAM roles
   - Workload Identity (for GKE migration)

3. **Data Protection**:
   - Encryption at rest and in transit
   - Secret Manager integration
   - Binary Authorization (optional)

4. **Container Security**:
   - Non-root containers
   - Security scanning
   - Distroless base images

### Security Best Practices

```bash
# Enable Binary Authorization (optional)
gcloud container binauthz policy import policy.yaml

# Set up VPC Service Controls (for enhanced security)
gcloud access-context-manager perimeters create ml-eval-perimeter \
  --resources=projects/YOUR_PROJECT_ID \
  --restricted-services=storage.googleapis.com,sql-component.googleapis.com

# Configure firewall rules
gcloud compute firewall-rules create ml-eval-allow-internal \
  --network=ml-eval-vpc \
  --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/8
```

## Disaster Recovery

### Backup Strategy

1. **Database Backups**:
   - Automated daily backups (30-day retention)
   - Point-in-time recovery enabled
   - Cross-region backup replication

2. **Storage Backups**:
   - Multi-regional storage buckets
   - Versioning enabled
   - Lifecycle policies for cost optimization

3. **Configuration Backups**:
   - Terraform state in Cloud Storage
   - Service configurations in version control

### Recovery Procedures

#### Database Recovery

```bash
# List available backups
gcloud sql backups list --instance=ml-eval-db-production

# Restore from backup
gcloud sql backups restore BACKUP_ID --restore-instance=ml-eval-db-restored
```

#### Service Rollback

```bash
# Automated rollback script
./deployment/scripts/rollback.sh --project-id YOUR_PROJECT_ID --service ml-eval-backend

# Manual rollback
gcloud run services update-traffic ml-eval-backend --to-revisions=PREVIOUS_REVISION=100
```

## Cost Optimization

### Cost Monitoring

1. **Budget Alerts**: Configured at $1000/month with notifications
2. **Resource Quotas**: Prevent runaway costs
3. **Rightsizing**: Regular review of resource allocation

### Cost-Saving Features

- **Preemptible instances** for non-critical workloads
- **Auto-scaling to zero** for Celery workers
- **Storage lifecycle policies** for data archiving
- **Regional resources** to minimize egress costs

### Cost Estimation

Expected monthly costs (production):

| Service | Cost Range |
|---------|-----------|
| Cloud Run (all services) | $200-500 |
| Cloud SQL (ha) | $150-300 |
| Cloud Memorystore | $50-100 |
| Cloud Storage | $20-100 |
| Networking | $10-50 |
| **Total** | **$430-1050** |

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check service logs
gcloud run services logs read ml-eval-backend --region=us-central1

# Check revision status
gcloud run revisions describe REVISION_NAME --region=us-central1
```

#### 2. Database Connection Issues

```bash
# Test Cloud SQL connectivity
gcloud sql connect ml-eval-db-production

# Check VPC connector
gcloud compute networks vpc-access connectors describe ml-eval-connector --region=us-central1
```

#### 3. High Memory Usage

```bash
# Check memory metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container/memory"

# Scale up memory
gcloud run services update ml-eval-backend --memory=8Gi --region=us-central1
```

#### 4. Slow Response Times

```bash
# Enable request tracing
gcloud run services update ml-eval-backend --set-env-vars=ENABLE_TRACING=true

# Check database query performance
# Access Cloud SQL Insights in the Console
```

### Support and Debugging

1. **Health Endpoints**: 
   - Backend: `https://backend-url/health`
   - Frontend: `https://frontend-url/api/health`

2. **Debug Mode**: Set `DEBUG=true` in environment variables

3. **Monitoring**: Use Cloud Operations suite for real-time metrics

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review monitoring dashboards
   - Check error rates and performance
   - Validate backup integrity

2. **Monthly**:
   - Update dependencies and base images
   - Review and optimize costs
   - Security audit and updates

3. **Quarterly**:
   - Disaster recovery testing
   - Capacity planning review
   - Performance benchmarking

### Updates and Patches

```bash
# Update infrastructure
cd deployment/terraform
terraform plan
terraform apply

# Update services (automatically via CI/CD or manual)
gcloud builds submit --config=deployment/cloudbuild/deploy-all.yaml
```

## Advanced Configuration

### Custom Domain Setup

```bash
# Map custom domain
gcloud run domain-mappings create --service=ml-eval-frontend --domain=your-domain.com --region=us-central1

# Set up SSL certificate
gcloud compute ssl-certificates create ml-eval-ssl --domains=your-domain.com
```

### Multi-Environment Setup

```bash
# Create staging environment
cd deployment/terraform
terraform workspace new staging
terraform apply -var="environment=staging" -var="project_id=ml-eval-staging"
```

### GPU Configuration

For ML workloads requiring GPU acceleration:

```yaml
# Add to celery service configuration
metadata:
  annotations:
    run.googleapis.com/gpu: "1"
    run.googleapis.com/gpu-type: "nvidia-t4"
```

## Support

For deployment issues or questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review Cloud Run logs and monitoring dashboards
3. Consult Google Cloud documentation
4. Open an issue in the project repository

---

**Next Steps**: After successful deployment, proceed to configure your ML models and datasets using the application interface at your frontend URL.