# Variables for monitoring configuration

variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "notification_channels" {
  description = "List of notification channel IDs for alerts"
  type        = list(string)
  default     = []
}

variable "daily_budget_threshold" {
  description = "Daily budget threshold for cost alerts (USD)"
  type        = number
  default     = 50
}

variable "alert_policy_enabled" {
  description = "Enable alert policies"
  type        = bool
  default     = true
}

variable "dashboard_enabled" {
  description = "Enable monitoring dashboards"
  type        = bool
  default     = true
}