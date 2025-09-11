# Outputs for Redis module

output "instance_name" {
  description = "Name of the Redis instance"
  value       = google_redis_instance.cache.name
}

output "host" {
  description = "Redis host address"
  value       = google_redis_instance.cache.host
}

output "port" {
  description = "Redis port"
  value       = google_redis_instance.cache.port
}

output "auth_string" {
  description = "Redis AUTH string"
  value       = google_redis_instance.cache.auth_string
  sensitive   = true
}

output "redis_url" {
  description = "Redis connection URL"
  value       = "redis://default:${google_redis_instance.cache.auth_string}@${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
  sensitive   = true
}

output "current_location_id" {
  description = "Current location ID of the Redis instance"
  value       = google_redis_instance.cache.current_location_id
}

output "read_replica_host" {
  description = "Redis read replica host address (if exists)"
  value       = var.environment == "production" && var.enable_read_replica ? google_redis_instance.read_replica[0].host : null
}

output "read_replica_port" {
  description = "Redis read replica port (if exists)"
  value       = var.environment == "production" && var.enable_read_replica ? google_redis_instance.read_replica[0].port : null
}