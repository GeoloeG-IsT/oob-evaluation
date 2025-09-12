# Variables for database module

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
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

variable "vpc_network" {
  description = "VPC network self link"
  type        = string
}

variable "database_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "POSTGRES_15"
}

variable "tier" {
  description = "Database instance tier"
  type        = string
  default     = "db-standard-2"
}

variable "disk_size" {
  description = "Database disk size in GB"
  type        = number
  default     = 100
}

variable "backup_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}