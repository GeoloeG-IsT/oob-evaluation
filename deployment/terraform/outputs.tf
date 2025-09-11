# Outputs for ML Evaluation Platform Terraform configuration

# Network Outputs
output "vpc_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc.name
}

output "vpc_self_link" {
  description = "Self link of the VPC network"
  value       = google_compute_network.vpc.self_link
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.subnet.name
}

output "vpc_connector_name" {
  description = "Name of the VPC Access Connector"
  value       = google_vpc_access_connector.connector.name
}

# Database Outputs
output "database_instance_name" {
  description = "Name of the Cloud SQL instance"
  value       = module.database.instance_name
}

output "database_connection_name" {
  description = "Connection name for Cloud SQL instance"
  value       = module.database.connection_name
  sensitive   = true
}

output "database_private_ip" {
  description = "Private IP address of the database"
  value       = module.database.private_ip
  sensitive   = true
}

# Redis Outputs
output "redis_instance_name" {
  description = "Name of the Redis instance"
  value       = module.redis.instance_name
}

output "redis_host" {
  description = "Redis host address"
  value       = module.redis.host
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = module.redis.port
}

# Storage Outputs
output "storage_bucket_name" {
  description = "Name of the main storage bucket"
  value       = module.storage.storage_bucket_name
}

output "models_bucket_name" {
  description = "Name of the models storage bucket"
  value       = module.storage.models_bucket_name
}

output "artifacts_bucket_name" {
  description = "Name of the artifacts storage bucket"
  value       = module.storage.artifacts_bucket_name
}

# IAM Outputs
output "backend_service_account_email" {
  description = "Email of the backend service account"
  value       = module.iam.backend_service_account_email
}

output "frontend_service_account_email" {
  description = "Email of the frontend service account"
  value       = module.iam.frontend_service_account_email
}

output "celery_service_account_email" {
  description = "Email of the Celery service account"
  value       = module.iam.celery_service_account_email
}

output "cloudbuild_service_account_email" {
  description = "Email of the Cloud Build service account"
  value       = module.iam.cloudbuild_service_account_email
}

# Secrets Outputs
output "secrets_secret_id" {
  description = "ID of the main secrets in Secret Manager"
  value       = module.secrets.secrets_secret_id
}

# Project Information
output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "zone" {
  description = "GCP Zone"
  value       = var.zone
}

# Application URLs (to be populated after Cloud Run deployment)
output "backend_url" {
  description = "URL of the backend service"
  value       = "https://ml-eval-backend-${var.project_id}.${var.region}.run.app"
}

output "frontend_url" {
  description = "URL of the frontend service"
  value       = "https://ml-eval-frontend-${var.project_id}.${var.region}.run.app"
}

# Environment Configuration
output "environment" {
  description = "Environment name"
  value       = var.environment
}

# Common Labels
output "common_labels" {
  description = "Common labels applied to all resources"
  value       = local.common_labels
}

# Deployment Commands
output "deployment_commands" {
  description = "Commands to deploy the application"
  value = {
    build_backend = "gcloud builds submit --config deployment/cloudbuild/backend.yaml --substitutions=_PROJECT_ID=${var.project_id},_REGION=${var.region}"
    build_frontend = "gcloud builds submit --config deployment/cloudbuild/frontend.yaml --substitutions=_PROJECT_ID=${var.project_id},_REGION=${var.region}"
    build_celery = "gcloud builds submit --config deployment/cloudbuild/celery.yaml --substitutions=_PROJECT_ID=${var.project_id},_REGION=${var.region}"
    deploy_backend = "gcloud run services replace deployment/cloud-run/backend-service.yaml --region ${var.region}"
    deploy_frontend = "gcloud run services replace deployment/cloud-run/frontend-service.yaml --region ${var.region}"
  }
}