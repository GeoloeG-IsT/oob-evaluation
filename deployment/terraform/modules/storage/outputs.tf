# Outputs for storage module

output "storage_bucket_name" {
  description = "Name of the main storage bucket"
  value       = google_storage_bucket.storage_bucket.name
}

output "storage_bucket_url" {
  description = "URL of the main storage bucket"
  value       = google_storage_bucket.storage_bucket.url
}

output "models_bucket_name" {
  description = "Name of the models storage bucket"
  value       = google_storage_bucket.models_bucket.name
}

output "models_bucket_url" {
  description = "URL of the models storage bucket"
  value       = google_storage_bucket.models_bucket.url
}

output "artifacts_bucket_name" {
  description = "Name of the artifacts storage bucket"
  value       = google_storage_bucket.artifacts_bucket.name
}

output "artifacts_bucket_url" {
  description = "URL of the artifacts storage bucket"
  value       = google_storage_bucket.artifacts_bucket.url
}

output "backup_bucket_name" {
  description = "Name of the backup storage bucket"
  value       = var.enable_backup ? google_storage_bucket.backup_bucket[0].name : null
}

output "backup_bucket_url" {
  description = "URL of the backup storage bucket"
  value       = var.enable_backup ? google_storage_bucket.backup_bucket[0].url : null
}

output "logs_bucket_name" {
  description = "Name of the logs storage bucket"
  value       = google_storage_bucket.logs_bucket.name
}

output "logs_bucket_url" {
  description = "URL of the logs storage bucket"
  value       = google_storage_bucket.logs_bucket.url
}