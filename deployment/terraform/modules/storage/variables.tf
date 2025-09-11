# Variables for storage module

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for storage buckets"
  type        = string
}

variable "app_name" {
  description = "Application name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "backup_region" {
  description = "Region for backup storage"
  type        = string
  default     = "us-east1"
}

variable "enable_backup" {
  description = "Enable backup storage bucket"
  type        = bool
  default     = true
}

variable "enable_website" {
  description = "Enable website configuration for main bucket"
  type        = bool
  default     = false
}

variable "kms_key_name" {
  description = "KMS key name for encryption"
  type        = string
  default     = ""
}

variable "allowed_origins" {
  description = "Allowed origins for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "backend_service_account" {
  description = "Backend service account email"
  type        = string
  default     = ""
}

variable "celery_service_account" {
  description = "Celery service account email"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}