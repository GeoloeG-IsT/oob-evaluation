# Terraform configuration for ML Evaluation Platform on GCP
# This file defines all the GCP resources needed for production deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  # Remote state backend
  backend "gcs" {
    bucket = "ml-eval-terraform-state"
    prefix = "terraform/state"
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (production, staging, development)"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Custom domain name for the application"
  type        = string
  default     = ""
}

# Local values
locals {
  name_prefix = "ml-eval-${var.environment}"
  
  labels = {
    project     = "ml-evaluation-platform"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Data sources
data "google_project" "project" {
  project_id = var.project_id
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

# Service Account for ML Evaluation Platform
resource "google_service_account" "ml_eval_service" {
  project      = var.project_id
  account_id   = "${local.name_prefix}-service"
  display_name = "ML Evaluation Platform Service Account"
  description  = "Service account for ML Evaluation Platform services"
}

# IAM bindings for service account
resource "google_project_iam_member" "ml_eval_service_permissions" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/redis.admin",
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.ml_eval_service.email}"
}

# VPC Network
resource "google_compute_network" "ml_eval_vpc" {
  project                 = var.project_id
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460
}

# Subnet
resource "google_compute_subnetwork" "ml_eval_subnet" {
  project       = var.project_id
  name          = "${local.name_prefix}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.ml_eval_vpc.id
  
  private_ip_google_access = true
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Cloud NAT for outbound internet access
resource "google_compute_router" "ml_eval_router" {
  project = var.project_id
  name    = "${local.name_prefix}-router"
  region  = var.region
  network = google_compute_network.ml_eval_vpc.id
}

resource "google_compute_router_nat" "ml_eval_nat" {
  project                = var.project_id
  name                   = "${local.name_prefix}-nat"
  router                 = google_compute_router.ml_eval_router.name
  region                 = var.region
  nat_ip_allocate_option = "AUTO_ONLY"
  
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# VPC Connector for Cloud Run
resource "google_vpc_access_connector" "ml_eval_connector" {
  project       = var.project_id
  name          = "${local.name_prefix}-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.ml_eval_vpc.name
  
  min_throughput = 200
  max_throughput = 1000
  
  depends_on = [google_project_service.apis]
}

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "ml_eval_db" {
  project          = var.project_id
  name             = "${local.name_prefix}-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  deletion_protection = var.environment == "production"
  
  settings {
    tier                        = "db-custom-4-16384"  # 4 vCPU, 16GB RAM
    availability_type           = var.environment == "production" ? "REGIONAL" : "ZONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = 100
    disk_autoresize             = true
    disk_autoresize_limit       = 500
    deletion_protection_enabled = var.environment == "production"
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
      transaction_log_retention_days = 7
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.ml_eval_vpc.id
      enable_private_path_for_google_cloud_services = true
      require_ssl                                   = true
    }
    
    maintenance_window {
      day          = 7  # Sunday
      hour         = 3  # 3 AM
      update_track = "stable"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Database
resource "google_sql_database" "ml_eval_database" {
  project  = var.project_id
  name     = "ml_eval_platform_${var.environment}"
  instance = google_sql_database_instance.ml_eval_db.name
}

# Database user
resource "google_sql_user" "ml_eval_user" {
  project  = var.project_id
  name     = "ml_eval_user"
  instance = google_sql_database_instance.ml_eval_db.name
  password = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Redis instance
resource "google_redis_instance" "ml_eval_redis" {
  project                = var.project_id
  name                   = "${local.name_prefix}-redis"
  tier                   = "STANDARD_HA"
  memory_size_gb         = 4
  region                 = var.region
  location_id            = var.zone
  redis_version          = "REDIS_7_0"
  display_name           = "ML Eval Redis"
  reserved_ip_range      = "10.3.0.0/29"
  
  authorized_network = google_compute_network.ml_eval_vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  auth_enabled           = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"
  
  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWELVE_HOURS"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  depends_on = [google_project_service.apis]
}

# Cloud Storage bucket for models and data
resource "google_storage_bucket" "ml_eval_storage" {
  project       = var.project_id
  name          = "${local.name_prefix}-storage"
  location      = var.region
  force_destroy = var.environment != "production"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  lifecycle_rule {
    condition {
      age                   = 30
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  
  labels = local.labels
}

# Secrets in Secret Manager
resource "google_secret_manager_secret" "secrets" {
  for_each = toset([
    "ml-eval-secret-key",
    "ml-eval-jwt-secret",
    "ml-eval-database-url",
    "ml-eval-redis-url",
    "ml-eval-encryption-key",
  ])
  
  project   = var.project_id
  secret_id = each.value
  
  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }
  
  labels = local.labels
}

# Secret values
resource "google_secret_manager_secret_version" "secret_key" {
  secret      = google_secret_manager_secret.secrets["ml-eval-secret-key"].id
  secret_data = random_password.secret_key.result
}

resource "google_secret_manager_secret_version" "jwt_secret" {
  secret      = google_secret_manager_secret.secrets["ml-eval-jwt-secret"].id
  secret_data = random_password.jwt_secret.result
}

resource "google_secret_manager_secret_version" "database_url" {
  secret = google_secret_manager_secret.secrets["ml-eval-database-url"].id
  secret_data = "postgresql://${google_sql_user.ml_eval_user.name}:${google_sql_user.ml_eval_user.password}@${google_sql_database_instance.ml_eval_db.private_ip_address}:5432/${google_sql_database.ml_eval_database.name}?sslmode=require"
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.secrets["ml-eval-redis-url"].id
  secret_data = "redis://:${google_redis_instance.ml_eval_redis.auth_string}@${google_redis_instance.ml_eval_redis.host}:${google_redis_instance.ml_eval_redis.port}/0"
}

resource "google_secret_manager_secret_version" "encryption_key" {
  secret      = google_secret_manager_secret.secrets["ml-eval-encryption-key"].id
  secret_data = random_password.encryption_key.result
}

resource "random_password" "secret_key" {
  length  = 64
  special = true
}

resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "random_password" "encryption_key" {
  length  = 32
  special = false
}

# Cloud Build trigger for CI/CD
resource "google_cloudbuild_trigger" "ml_eval_deploy" {
  project     = var.project_id
  name        = "${local.name_prefix}-deploy"
  description = "Deploy ML Evaluation Platform"
  
  github {
    owner = "your-github-org"  # Replace with actual GitHub org
    name  = "ml-evaluation-platform"  # Replace with actual repo name
    push {
      branch = var.environment == "production" ? "main" : var.environment
    }
  }
  
  filename = "gcp/cloudbuild.yml"
  
  substitutions = {
    _ENVIRONMENT = var.environment
    _REGION      = var.region
  }
  
  depends_on = [google_project_service.apis]
}

# Cloud Monitoring alerts
resource "google_monitoring_alert_policy" "high_error_rate" {
  project      = var.project_id
  display_name = "ML Eval - High Error Rate"
  combiner     = "OR"
  
  conditions {
    display_name = "High error rate"
    
    condition_threshold {
      filter         = "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"ml-eval-backend-${var.environment}\""
      duration       = "300s"
      comparison     = "COMPARISON_GT"
      threshold_value = 0.1
      
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }
  
  notification_channels = []  # Add notification channels as needed
  
  alert_strategy {
    auto_close = "1800s"  # 30 minutes
  }
}

# Outputs
output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP Region"
  value       = var.region
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.ml_eval_db.connection_name
}

output "database_private_ip" {
  description = "Database private IP"
  value       = google_sql_database_instance.ml_eval_db.private_ip_address
  sensitive   = true
}

output "redis_host" {
  description = "Redis host"
  value       = google_redis_instance.ml_eval_redis.host
  sensitive   = true
}

output "storage_bucket_name" {
  description = "Storage bucket name"
  value       = google_storage_bucket.ml_eval_storage.name
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.ml_eval_service.email
}

output "vpc_connector_name" {
  description = "VPC connector name"
  value       = google_vpc_access_connector.ml_eval_connector.name
}