# Outputs for secrets module

output "secrets_secret_id" {
  description = "ID of the main secrets in Secret Manager"
  value       = google_secret_manager_secret.app_secrets.secret_id
}

output "secrets_secret_name" {
  description = "Name of the main secrets in Secret Manager"
  value       = google_secret_manager_secret.app_secrets.name
}

output "redis_secret_id" {
  description = "ID of the Redis secret in Secret Manager"
  value       = google_secret_manager_secret.redis_secrets.secret_id
}

output "redis_secret_name" {
  description = "Name of the Redis secret in Secret Manager"
  value       = google_secret_manager_secret.redis_secrets.name
}

output "jwt_secret_id" {
  description = "ID of the JWT secret in Secret Manager"
  value       = google_secret_manager_secret.jwt_secret.secret_id
}

output "jwt_secret_name" {
  description = "Name of the JWT secret in Secret Manager"
  value       = google_secret_manager_secret.jwt_secret.name
}

output "api_keys_secret_id" {
  description = "ID of the API keys secret in Secret Manager"
  value       = google_secret_manager_secret.api_keys.secret_id
}

output "api_keys_secret_name" {
  description = "Name of the API keys secret in Secret Manager"
  value       = google_secret_manager_secret.api_keys.name
}

output "monitoring_secret_id" {
  description = "ID of the monitoring secret in Secret Manager"
  value       = google_secret_manager_secret.monitoring_secrets.secret_id
}

output "monitoring_secret_name" {
  description = "Name of the monitoring secret in Secret Manager"
  value       = google_secret_manager_secret.monitoring_secrets.name
}

output "tls_certs_secret_id" {
  description = "ID of the TLS certificates secret in Secret Manager"
  value       = var.enable_custom_domain ? google_secret_manager_secret.tls_certs[0].secret_id : null
}

output "tls_certs_secret_name" {
  description = "Name of the TLS certificates secret in Secret Manager"
  value       = var.enable_custom_domain ? google_secret_manager_secret.tls_certs[0].name : null
}