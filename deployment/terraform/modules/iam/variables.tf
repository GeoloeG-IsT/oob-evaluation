# Variables for IAM module

variable "project_id" {
  description = "The GCP project ID"
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

variable "storage_bucket" {
  description = "Name of the main storage bucket"
  type        = string
  default     = ""
}

variable "models_bucket" {
  description = "Name of the models storage bucket"
  type        = string
  default     = ""
}

variable "artifacts_bucket" {
  description = "Name of the artifacts storage bucket"
  type        = string
  default     = ""
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity for future GKE integration"
  type        = bool
  default     = false
}