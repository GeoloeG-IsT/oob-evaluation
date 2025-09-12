# Variables for Redis module

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
  description = "VPC network ID"
  type        = string
}

variable "memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 2
}

variable "tier" {
  description = "Redis tier (BASIC or STANDARD_HA)"
  type        = string
  default     = "STANDARD_HA"
  
  validation {
    condition     = contains(["BASIC", "STANDARD_HA"], var.tier)
    error_message = "Tier must be either BASIC or STANDARD_HA."
  }
}

variable "enable_read_replica" {
  description = "Enable read replica for production"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default     = {}
}