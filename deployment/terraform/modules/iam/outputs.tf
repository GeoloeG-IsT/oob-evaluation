# Outputs for IAM module

output "backend_service_account_email" {
  description = "Email of the backend service account"
  value       = google_service_account.backend.email
}

output "backend_service_account_name" {
  description = "Name of the backend service account"
  value       = google_service_account.backend.name
}

output "frontend_service_account_email" {
  description = "Email of the frontend service account"
  value       = google_service_account.frontend.email
}

output "frontend_service_account_name" {
  description = "Name of the frontend service account"
  value       = google_service_account.frontend.name
}

output "celery_service_account_email" {
  description = "Email of the Celery service account"
  value       = google_service_account.celery.email
}

output "celery_service_account_name" {
  description = "Name of the Celery service account"
  value       = google_service_account.celery.name
}

output "cloudbuild_service_account_email" {
  description = "Email of the Cloud Build service account"
  value       = google_service_account.cloudbuild.email
}

output "cloudbuild_service_account_name" {
  description = "Name of the Cloud Build service account"
  value       = google_service_account.cloudbuild.name
}

output "ml_ops_role_name" {
  description = "Name of the custom ML operations role"
  value       = google_project_iam_custom_role.ml_ops_role.name
}