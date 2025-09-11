# Variables for ML Evaluation Platform Terraform configuration

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

# Network Configuration
variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "connector_cidr" {
  description = "CIDR range for VPC Access Connector"
  type        = string
  default     = "10.0.1.0/28"
}

# Database Configuration
variable "db_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-standard-2"
  
  validation {
    condition = can(regex("^db-(standard|custom)-", var.db_tier))
    error_message = "Database tier must be a valid Cloud SQL tier."
  }
}

variable "db_disk_size" {
  description = "Database disk size in GB"
  type        = number
  default     = 100
  
  validation {
    condition     = var.db_disk_size >= 20 && var.db_disk_size <= 30720
    error_message = "Database disk size must be between 20GB and 30720GB."
  }
}

variable "db_backup_retention_days" {
  description = "Number of days to retain database backups"
  type        = number
  default     = 30
}

# Redis Configuration
variable "redis_memory_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 2
  
  validation {
    condition     = var.redis_memory_gb >= 1 && var.redis_memory_gb <= 300
    error_message = "Redis memory size must be between 1GB and 300GB."
  }
}

# Cloud Run Configuration
variable "backend_min_instances" {
  description = "Minimum number of backend instances"
  type        = number
  default     = 1
}

variable "backend_max_instances" {
  description = "Maximum number of backend instances"
  type        = number
  default     = 10
}

variable "frontend_min_instances" {
  description = "Minimum number of frontend instances"
  type        = number
  default     = 1
}

variable "frontend_max_instances" {
  description = "Maximum number of frontend instances"
  type        = number
  default     = 20
}

variable "celery_training_max_instances" {
  description = "Maximum number of Celery training worker instances"
  type        = number
  default     = 3
}

variable "celery_inference_max_instances" {
  description = "Maximum number of Celery inference worker instances"
  type        = number
  default     = 5
}

# Storage Configuration
variable "storage_location" {
  description = "Location for Cloud Storage buckets"
  type        = string
  default     = "US"
}

variable "storage_class" {
  description = "Storage class for buckets"
  type        = string
  default     = "STANDARD"
  
  validation {
    condition = contains([
      "STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"
    ], var.storage_class)
    error_message = "Storage class must be one of: STANDARD, NEARLINE, COLDLINE, ARCHIVE."
  }
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable Cloud Monitoring and Logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_binary_authorization" {
  description = "Enable Binary Authorization for container images"
  type        = bool
  default     = true
}

variable "allowed_ingress_cidrs" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
}

# Domain Configuration
variable "custom_domain" {
  description = "Custom domain for the application"
  type        = string
  default     = ""
}

variable "ssl_certificate_name" {
  description = "Name of the SSL certificate for custom domain"
  type        = string
  default     = ""
}

# Cost Optimization
variable "enable_preemptible" {
  description = "Enable preemptible instances where applicable"
  type        = bool
  default     = false
}

variable "budget_amount" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 1000
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for disaster recovery"
  type        = bool
  default     = true
}

variable "backup_region" {
  description = "Region for cross-region backups"
  type        = string
  default     = "us-east1"
}