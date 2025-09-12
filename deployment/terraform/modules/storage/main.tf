# Cloud Storage module for ML Evaluation Platform

# Main storage bucket for user uploads and data
resource "google_storage_bucket" "storage_bucket" {
  name                        = "${var.app_name}-storage-${var.project_id}"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = var.environment != "production"
  uniform_bucket_level_access = true

  # Versioning for data protection
  versioning {
    enabled = var.environment == "production"
  }

  # Lifecycle management
  lifecycle_rule {
    condition {
      age                   = 90
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age                   = 365
      matches_storage_class = ["NEARLINE"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Delete old versions after 30 days
  lifecycle_rule {
    condition {
      age                        = 30
      with_state                 = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  # CORS configuration for web uploads
  cors {
    origin          = var.allowed_origins
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  # Encryption
  encryption {
    default_kms_key_name = var.kms_key_name != "" ? var.kms_key_name : null
  }

  # Labels
  labels = var.labels

  # Website configuration for static assets
  dynamic "website" {
    for_each = var.enable_website ? [1] : []
    content {
      main_page_suffix = "index.html"
      not_found_page   = "404.html"
    }
  }
}

# Models storage bucket for ML models
resource "google_storage_bucket" "models_bucket" {
  name                        = "${var.app_name}-models-${var.project_id}"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = var.environment != "production"
  uniform_bucket_level_access = true

  # Versioning for model management
  versioning {
    enabled = true
  }

  # Lifecycle management for models
  lifecycle_rule {
    condition {
      age = 180
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Keep model versions for 2 years
  lifecycle_rule {
    condition {
      age        = 730
      with_state = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  # Encryption
  encryption {
    default_kms_key_name = var.kms_key_name != "" ? var.kms_key_name : null
  }

  # Labels
  labels = merge(var.labels, {
    bucket-type = "models"
  })
}

# Build artifacts bucket for CI/CD
resource "google_storage_bucket" "artifacts_bucket" {
  name                        = "${var.app_name}-artifacts-${var.project_id}"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = true
  uniform_bucket_level_access = true

  # Short lifecycle for build artifacts
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  # Labels
  labels = merge(var.labels, {
    bucket-type = "artifacts"
  })
}

# Backup bucket for cross-region disaster recovery
resource "google_storage_bucket" "backup_bucket" {
  count                       = var.enable_backup ? 1 : 0
  name                        = "${var.app_name}-backup-${var.project_id}"
  location                    = var.backup_region
  project                     = var.project_id
  force_destroy               = var.environment != "production"
  uniform_bucket_level_access = true
  storage_class              = "COLDLINE"

  # Versioning
  versioning {
    enabled = true
  }

  # Long-term retention for backups
  lifecycle_rule {
    condition {
      age = 2555  # 7 years
    }
    action {
      type = "Delete"
    }
  }

  # Encryption
  encryption {
    default_kms_key_name = var.kms_key_name != "" ? var.kms_key_name : null
  }

  # Labels
  labels = merge(var.labels, {
    bucket-type = "backup"
  })
}

# Logs bucket for application logs
resource "google_storage_bucket" "logs_bucket" {
  name                        = "${var.app_name}-logs-${var.project_id}"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = var.environment != "production"
  uniform_bucket_level_access = true
  storage_class              = "NEARLINE"

  # Lifecycle management for logs
  lifecycle_rule {
    condition {
      age = 30
      matches_storage_class = ["NEARLINE"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 90
      matches_storage_class = ["COLDLINE"]
    }
    action {
      type = "Delete"
    }
  }

  # Labels
  labels = merge(var.labels, {
    bucket-type = "logs"
  })
}

# IAM bindings for buckets
resource "google_storage_bucket_iam_binding" "storage_bucket_binding" {
  bucket = google_storage_bucket.storage_bucket.name
  role   = "roles/storage.objectAdmin"
  
  members = [
    "serviceAccount:${var.backend_service_account}",
    "serviceAccount:${var.celery_service_account}",
  ]
}

resource "google_storage_bucket_iam_binding" "models_bucket_binding" {
  bucket = google_storage_bucket.models_bucket.name
  role   = "roles/storage.objectAdmin"
  
  members = [
    "serviceAccount:${var.backend_service_account}",
    "serviceAccount:${var.celery_service_account}",
  ]
}

# Public access prevention
resource "google_storage_bucket_iam_binding" "prevent_public_access" {
  for_each = toset([
    google_storage_bucket.storage_bucket.name,
    google_storage_bucket.models_bucket.name,
    google_storage_bucket.artifacts_bucket.name,
    google_storage_bucket.logs_bucket.name,
  ])
  
  bucket = each.value
  role   = "roles/storage.publicAccessPrevention"
  
  members = [
    "allUsers",
  ]
}