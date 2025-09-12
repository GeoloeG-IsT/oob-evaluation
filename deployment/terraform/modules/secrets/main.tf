# Secret Manager module for ML Evaluation Platform

# Main application secrets
resource "google_secret_manager_secret" "app_secrets" {
  secret_id = "${var.app_name}-secrets-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
  }
}

# Database connection secret
resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.app_secrets.id
  secret_data = jsonencode({
    database-url = var.database_url
  })
}

# Redis connection secret
resource "google_secret_manager_secret" "redis_secrets" {
  secret_id = "${var.app_name}-redis-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
    type        = "redis"
  }
}

resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.redis_secrets.id
  secret_data = jsonencode({
    redis-url = var.redis_url
  })
}

# JWT Secret for authentication (if needed)
resource "random_password" "jwt_secret" {
  length  = 64
  special = true
}

resource "google_secret_manager_secret" "jwt_secret" {
  secret_id = "${var.app_name}-jwt-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
    type        = "jwt"
  }
}

resource "google_secret_manager_secret_version" "jwt_secret_version" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = random_password.jwt_secret.result
}

# API Keys secret (for external services)
resource "google_secret_manager_secret" "api_keys" {
  secret_id = "${var.app_name}-api-keys-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
    type        = "api-keys"
  }
}

resource "google_secret_manager_secret_version" "api_keys_version" {
  secret      = google_secret_manager_secret.api_keys.id
  secret_data = jsonencode({
    # Add your API keys here as needed
    openai_api_key     = var.openai_api_key != "" ? var.openai_api_key : ""
    huggingface_token  = var.huggingface_token != "" ? var.huggingface_token : ""
    wandb_api_key      = var.wandb_api_key != "" ? var.wandb_api_key : ""
  })
}

# SSL/TLS certificates secret (if using custom domain)
resource "google_secret_manager_secret" "tls_certs" {
  count     = var.enable_custom_domain ? 1 : 0
  secret_id = "${var.app_name}-tls-certs-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
    type        = "tls"
  }
}

resource "google_secret_manager_secret_version" "tls_certs_version" {
  count       = var.enable_custom_domain ? 1 : 0
  secret      = google_secret_manager_secret.tls_certs[0].id
  secret_data = jsonencode({
    certificate = var.tls_certificate
    private_key = var.tls_private_key
  })
}

# Monitoring and alerting secrets
resource "google_secret_manager_secret" "monitoring_secrets" {
  secret_id = "${var.app_name}-monitoring-${var.environment}"
  project   = var.project_id
  
  replication {
    auto {}
  }
  
  labels = {
    app         = var.app_name
    environment = var.environment
    managed-by  = "terraform"
    type        = "monitoring"
  }
}

resource "google_secret_manager_secret_version" "monitoring_secrets_version" {
  secret      = google_secret_manager_secret.monitoring_secrets.id
  secret_data = jsonencode({
    slack_webhook_url   = var.slack_webhook_url != "" ? var.slack_webhook_url : ""
    pagerduty_token    = var.pagerduty_token != "" ? var.pagerduty_token : ""
    email_smtp_password = var.email_smtp_password != "" ? var.email_smtp_password : ""
  })
}

# IAM bindings for secret access
resource "google_secret_manager_secret_iam_member" "app_secrets_access" {
  for_each = toset([
    "serviceAccount:${var.backend_service_account}",
    "serviceAccount:${var.celery_service_account}",
  ])
  
  secret_id = google_secret_manager_secret.app_secrets.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "redis_secrets_access" {
  for_each = toset([
    "serviceAccount:${var.backend_service_account}",
    "serviceAccount:${var.celery_service_account}",
  ])
  
  secret_id = google_secret_manager_secret.redis_secrets.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "jwt_secret_access" {
  for_each = toset([
    "serviceAccount:${var.backend_service_account}",
  ])
  
  secret_id = google_secret_manager_secret.jwt_secret.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "api_keys_access" {
  for_each = toset([
    "serviceAccount:${var.backend_service_account}",
    "serviceAccount:${var.celery_service_account}",
  ])
  
  secret_id = google_secret_manager_secret.api_keys.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "monitoring_secrets_access" {
  for_each = toset([
    "serviceAccount:${var.backend_service_account}",
  ])
  
  secret_id = google_secret_manager_secret.monitoring_secrets.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = each.value
  project   = var.project_id
}