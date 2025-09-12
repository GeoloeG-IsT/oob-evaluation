# Variables for secrets module

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

variable "database_url" {
  description = "Database connection URL"
  type        = string
  sensitive   = true
}

variable "redis_url" {
  description = "Redis connection URL"
  type        = string
  sensitive   = true
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

# External API keys
variable "openai_api_key" {
  description = "OpenAI API key for AI features"
  type        = string
  default     = ""
  sensitive   = true
}

variable "huggingface_token" {
  description = "Hugging Face token for model downloads"
  type        = string
  default     = ""
  sensitive   = true
}

variable "wandb_api_key" {
  description = "Weights & Biases API key for experiment tracking"
  type        = string
  default     = ""
  sensitive   = true
}

# Custom domain and TLS
variable "enable_custom_domain" {
  description = "Enable custom domain with TLS certificates"
  type        = bool
  default     = false
}

variable "tls_certificate" {
  description = "TLS certificate for custom domain"
  type        = string
  default     = ""
  sensitive   = true
}

variable "tls_private_key" {
  description = "TLS private key for custom domain"
  type        = string
  default     = ""
  sensitive   = true
}

# Monitoring and alerting
variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pagerduty_token" {
  description = "PagerDuty integration token"
  type        = string
  default     = ""
  sensitive   = true
}

variable "email_smtp_password" {
  description = "SMTP password for email notifications"
  type        = string
  default     = ""
  sensitive   = true
}