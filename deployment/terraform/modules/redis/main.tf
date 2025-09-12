# Cloud Memorystore Redis module for ML Evaluation Platform

# Redis instance
resource "google_redis_instance" "cache" {
  name           = "${var.app_name}-redis-${var.environment}"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.memory_size_gb
  tier           = var.tier

  # Network configuration
  authorized_network = var.vpc_network
  connect_mode      = "PRIVATE_SERVICE_ACCESS"

  # Redis configuration
  redis_version     = "REDIS_7_0"
  display_name     = "${var.app_name} Redis Cache"
  
  # Enable AUTH for security
  auth_enabled = true
  
  # Persistence configuration for production
  persistence_config {
    persistence_mode    = var.environment == "production" ? "RDB" : "DISABLED"
    rdb_snapshot_period = var.environment == "production" ? "TWELVE_HOURS" : null
    rdb_snapshot_start_time = var.environment == "production" ? "03:00" : null
  }

  # Maintenance policy
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 4
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }

  # Redis configuration parameters
  redis_configs = {
    # Memory management
    maxmemory-policy = "allkeys-lru"
    
    # Timeout settings
    timeout = "300"
    
    # Enable keyspace notifications for Celery
    notify-keyspace-events = "Ex"
    
    # Connection settings
    tcp-keepalive = "60"
    
    # Logging
    loglevel = "notice"
  }

  # Labels
  labels = var.labels
}

# Redis instance for production read replica (if needed)
resource "google_redis_instance" "read_replica" {
  count          = var.environment == "production" && var.enable_read_replica ? 1 : 0
  name           = "${var.app_name}-redis-replica-${var.environment}"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.memory_size_gb
  tier           = "STANDARD_HA"

  # Network configuration
  authorized_network = var.vpc_network
  connect_mode      = "PRIVATE_SERVICE_ACCESS"

  # Redis configuration
  redis_version = "REDIS_7_0"
  display_name  = "${var.app_name} Redis Read Replica"
  
  # Enable AUTH
  auth_enabled = true

  # Read replica configuration
  read_replicas_mode = "READ_REPLICAS_ENABLED"
  replica_count     = 1

  # Maintenance policy
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 5
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }

  # Redis configuration parameters
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout         = "300"
    notify-keyspace-events = "Ex"
    tcp-keepalive   = "60"
    loglevel       = "notice"
  }

  # Labels
  labels = merge(var.labels, {
    replica = "true"
  })
}