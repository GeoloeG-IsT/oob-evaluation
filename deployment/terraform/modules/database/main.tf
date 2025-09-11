# Cloud SQL PostgreSQL module for ML Evaluation Platform

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Private IP range for Cloud SQL
resource "google_compute_global_address" "private_ip_range" {
  name          = "${var.app_name}-db-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = var.vpc_network
}

# Private connection for Cloud SQL
resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = var.vpc_network
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]
}

# Cloud SQL instance
resource "google_sql_database_instance" "main" {
  name             = "${var.app_name}-db-${var.environment}"
  database_version = var.database_version
  region           = var.region
  project          = var.project_id

  deletion_protection = var.environment == "production" ? true : false

  depends_on = [google_service_networking_connection.private_vpc_connection]

  settings {
    tier              = var.tier
    disk_size         = var.disk_size
    disk_type         = "SSD"
    disk_autoresize   = true
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    # Backup configuration
    backup_configuration {
      enabled                        = var.backup_enabled
      start_time                     = "03:00"
      point_in_time_recovery_enabled = var.environment == "production"
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
      transaction_log_retention_days = 7
    }

    # IP configuration for private networking
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = var.vpc_network
      enable_private_path_for_google_cloud_services = true
    }

    # Database flags for performance optimization
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,pg_hint_plan"
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }

    database_flags {
      name  = "shared_buffers"
      value = "256MB"
    }

    database_flags {
      name  = "effective_cache_size"
      value = "1GB"
    }

    database_flags {
      name  = "maintenance_work_mem"
      value = "64MB"
    }

    database_flags {
      name  = "checkpoint_completion_target"
      value = "0.9"
    }

    database_flags {
      name  = "wal_buffers"
      value = "16MB"
    }

    database_flags {
      name  = "default_statistics_target"
      value = "100"
    }

    # Maintenance window
    maintenance_window {
      day          = 7  # Sunday
      hour         = 4  # 4 AM
      update_track = "stable"
    }

    # Enable query insights
    insights_config {
      query_insights_enabled  = true
      record_application_tags = true
      record_client_address   = true
    }

    # User labels
    user_labels = var.labels
  }
}

# Database
resource "google_sql_database" "database" {
  name     = "ml_eval_platform"
  instance = google_sql_database_instance.main.name
  project  = var.project_id
}

# Database user
resource "google_sql_user" "user" {
  name     = "ml_eval_user"
  instance = google_sql_database_instance.main.name
  project  = var.project_id
  password = random_password.db_password.result
}

# Additional database for testing (non-production)
resource "google_sql_database" "test_database" {
  count    = var.environment != "production" ? 1 : 0
  name     = "ml_eval_platform_test"
  instance = google_sql_database_instance.main.name
  project  = var.project_id
}

# Read replica for production
resource "google_sql_database_instance" "read_replica" {
  count               = var.environment == "production" ? 1 : 0
  name                = "${var.app_name}-db-replica-${var.environment}"
  master_instance_name = google_sql_database_instance.main.name
  region              = var.region
  project             = var.project_id

  replica_configuration {
    failover_target = false
  }

  settings {
    tier              = var.tier
    disk_size         = var.disk_size
    disk_type         = "SSD"
    availability_type = "ZONAL"

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.vpc_network
    }

    user_labels = merge(var.labels, {
      replica = "true"
    })
  }
}