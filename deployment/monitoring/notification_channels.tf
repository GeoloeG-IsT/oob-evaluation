# Notification channels for ML Evaluation Platform alerts

# Email notification channel
resource "google_monitoring_notification_channel" "email" {
  count        = length(var.alert_email_addresses) > 0 ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - Email Notifications"
  type         = "email"

  labels = {
    email_address = var.alert_email_addresses[0]  # Primary email
  }

  description = "Email notifications for ML Evaluation Platform alerts"
  enabled     = true
}

# Additional email channels for team members
resource "google_monitoring_notification_channel" "team_email" {
  count        = length(var.alert_email_addresses) - 1
  project      = var.project_id
  display_name = "ML Eval - Team Email ${count.index + 2}"
  type         = "email"

  labels = {
    email_address = var.alert_email_addresses[count.index + 1]
  }

  description = "Team member email notifications for ML Evaluation Platform"
  enabled     = true
}

# Slack notification channel
resource "google_monitoring_notification_channel" "slack" {
  count        = var.slack_webhook_url != "" ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - Slack Notifications"
  type         = "slack"

  labels = {
    url = var.slack_webhook_url
  }

  description = "Slack notifications for ML Evaluation Platform alerts"
  enabled     = true
}

# PagerDuty notification channel (for critical alerts)
resource "google_monitoring_notification_channel" "pagerduty" {
  count        = var.pagerduty_service_key != "" ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - PagerDuty"
  type         = "pagerduty"

  labels = {
    service_key = var.pagerduty_service_key
  }

  description = "PagerDuty notifications for critical ML Evaluation Platform alerts"
  enabled     = true
}

# SMS notification channel (for critical alerts)
resource "google_monitoring_notification_channel" "sms" {
  count        = var.sms_phone_number != "" ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - SMS Alerts"
  type         = "sms"

  labels = {
    number = var.sms_phone_number
  }

  description = "SMS notifications for critical ML Evaluation Platform alerts"
  enabled     = true
}

# Webhook notification channel (for custom integrations)
resource "google_monitoring_notification_channel" "webhook" {
  count        = var.webhook_url != "" ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - Webhook"
  type         = "webhook_tokenauth"

  labels = {
    url = var.webhook_url
  }

  sensitive_labels {
    auth_token = var.webhook_auth_token
  }

  description = "Webhook notifications for ML Evaluation Platform alerts"
  enabled     = true
}

# Google Chat notification channel
resource "google_monitoring_notification_channel" "google_chat" {
  count        = var.google_chat_webhook_url != "" ? 1 : 0
  project      = var.project_id
  display_name = "ML Eval - Google Chat"
  type         = "googlechat"

  labels = {
    url = var.google_chat_webhook_url
  }

  description = "Google Chat notifications for ML Evaluation Platform alerts"
  enabled     = true
}

# Variables for notification channels
variable "alert_email_addresses" {
  description = "List of email addresses for alert notifications"
  type        = list(string)
  default     = []
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "pagerduty_service_key" {
  description = "PagerDuty service key for critical alerts"
  type        = string
  default     = ""
  sensitive   = true
}

variable "sms_phone_number" {
  description = "Phone number for SMS alerts"
  type        = string
  default     = ""
}

variable "webhook_url" {
  description = "Custom webhook URL for notifications"
  type        = string
  default     = ""
}

variable "webhook_auth_token" {
  description = "Authentication token for webhook notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "google_chat_webhook_url" {
  description = "Google Chat webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

# Output notification channel IDs for use in alert policies
output "notification_channel_ids" {
  description = "List of notification channel IDs"
  value = compact(concat(
    google_monitoring_notification_channel.email[*].id,
    google_monitoring_notification_channel.team_email[*].id,
    google_monitoring_notification_channel.slack[*].id,
    google_monitoring_notification_channel.pagerduty[*].id,
    google_monitoring_notification_channel.sms[*].id,
    google_monitoring_notification_channel.webhook[*].id,
    google_monitoring_notification_channel.google_chat[*].id
  ))
}

output "email_channel_id" {
  description = "Primary email notification channel ID"
  value       = length(google_monitoring_notification_channel.email) > 0 ? google_monitoring_notification_channel.email[0].id : null
}

output "slack_channel_id" {
  description = "Slack notification channel ID"
  value       = length(google_monitoring_notification_channel.slack) > 0 ? google_monitoring_notification_channel.slack[0].id : null
}

output "pagerduty_channel_id" {
  description = "PagerDuty notification channel ID"
  value       = length(google_monitoring_notification_channel.pagerduty) > 0 ? google_monitoring_notification_channel.pagerduty[0].id : null
}