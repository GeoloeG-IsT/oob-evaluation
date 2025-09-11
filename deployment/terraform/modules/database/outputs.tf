# Outputs for database module

output "instance_name" {
  description = "Name of the Cloud SQL instance"
  value       = google_sql_database_instance.main.name
}

output "connection_name" {
  description = "Connection name for Cloud SQL instance"
  value       = google_sql_database_instance.main.connection_name
}

output "private_ip" {
  description = "Private IP address of the database"
  value       = google_sql_database_instance.main.private_ip_address
}

output "database_url" {
  description = "Database connection URL"
  value       = "postgresql://${google_sql_user.user.name}:${random_password.db_password.result}@${google_sql_database_instance.main.private_ip_address}:5432/${google_sql_database.database.name}"
  sensitive   = true
}

output "database_name" {
  description = "Name of the database"
  value       = google_sql_database.database.name
}

output "database_user" {
  description = "Database user name"
  value       = google_sql_user.user.name
}

output "database_password" {
  description = "Database user password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "read_replica_connection_name" {
  description = "Connection name for read replica (if exists)"
  value       = var.environment == "production" ? google_sql_database_instance.read_replica[0].connection_name : null
}