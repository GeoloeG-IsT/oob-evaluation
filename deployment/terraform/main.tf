# Main Terraform configuration for ML Evaluation Platform on GCP
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
  
  # Configure backend for state management
  backend "gcs" {
    bucket = "PROJECT_ID-terraform-state"
    prefix = "ml-eval-platform/state"
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Local variables
locals {
  app_name = "ml-eval-platform"
  
  common_labels = {
    app         = local.app_name
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "sql.googleapis.com",
    "redis.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "container.googleapis.com",
    "compute.googleapis.com",
    "vpcaccess.googleapis.com",
    "cloudfunctions.googleapis.com",
    "pubsub.googleapis.com",
    "aiplatform.googleapis.com"
  ])
  
  service                    = each.key
  project                    = var.project_id
  disable_dependent_services = true
  disable_on_destroy         = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${local.app_name}-vpc"
  auto_create_subnetworks = false
  
  depends_on = [google_project_service.apis]
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${local.app_name}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  
  # Enable Private Google Access for Cloud SQL and other services
  private_ip_google_access = true
}

# VPC Access Connector for Cloud Run to VPC connectivity
resource "google_vpc_access_connector" "connector" {
  name          = "${local.app_name}-connector"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = var.connector_cidr
  
  min_instances = 2
  max_instances = 10
  
  depends_on = [google_project_service.apis]
}

# Cloud SQL PostgreSQL instance
module "database" {
  source = "./modules/database"
  
  project_id     = var.project_id
  region         = var.region
  app_name       = local.app_name
  environment    = var.environment
  vpc_network    = google_compute_network.vpc.self_link
  
  # Database configuration
  database_version = "POSTGRES_15"
  tier            = var.db_tier
  disk_size       = var.db_disk_size
  backup_enabled  = true
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# Cloud Memorystore Redis
module "redis" {
  source = "./modules/redis"
  
  project_id  = var.project_id
  region      = var.region
  app_name    = local.app_name
  environment = var.environment
  vpc_network = google_compute_network.vpc.id
  
  # Redis configuration
  memory_size_gb = var.redis_memory_gb
  tier          = "STANDARD_HA"
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# Cloud Storage for model and data storage
module "storage" {
  source = "./modules/storage"
  
  project_id  = var.project_id
  region      = var.region
  app_name    = local.app_name
  environment = var.environment
  
  labels = local.common_labels
  
  depends_on = [google_project_service.apis]
}

# IAM and Service Accounts
module "iam" {
  source = "./modules/iam"
  
  project_id  = var.project_id
  app_name    = local.app_name
  environment = var.environment
  
  # Storage bucket names from storage module
  storage_bucket      = module.storage.storage_bucket_name
  models_bucket       = module.storage.models_bucket_name
  artifacts_bucket    = module.storage.artifacts_bucket_name
  
  depends_on = [google_project_service.apis]
}

# Secret Manager for application secrets
module "secrets" {
  source = "./modules/secrets"
  
  project_id  = var.project_id
  app_name    = local.app_name
  environment = var.environment
  
  # Database and Redis connection strings
  database_url = module.database.database_url
  redis_url    = module.redis.redis_url
  
  depends_on = [google_project_service.apis]
}