# IAM module for ML Evaluation Platform service accounts and permissions

# Backend service account
resource "google_service_account" "backend" {
  account_id   = "${var.app_name}-backend"
  display_name = "${var.app_name} Backend Service Account"
  project      = var.project_id
}

# Frontend service account
resource "google_service_account" "frontend" {
  account_id   = "${var.app_name}-frontend"
  display_name = "${var.app_name} Frontend Service Account"
  project      = var.project_id
}

# Celery service account
resource "google_service_account" "celery" {
  account_id   = "${var.app_name}-celery"
  display_name = "${var.app_name} Celery Workers Service Account"
  project      = var.project_id
}

# Cloud Build service account
resource "google_service_account" "cloudbuild" {
  account_id   = "${var.app_name}-cloudbuild"
  display_name = "${var.app_name} Cloud Build Service Account"
  project      = var.project_id
}

# IAM bindings for backend service account
resource "google_project_iam_member" "backend_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

resource "google_project_iam_member" "backend_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# IAM bindings for frontend service account
resource "google_project_iam_member" "frontend_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_project_iam_member" "frontend_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

resource "google_project_iam_member" "frontend_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.frontend.email}"
}

# IAM bindings for Celery service account
resource "google_project_iam_member" "celery_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_ai_platform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

resource "google_project_iam_member" "celery_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.celery.email}"
}

# IAM bindings for Cloud Build service account
resource "google_project_iam_member" "cloudbuild_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_service_account_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

resource "google_project_iam_member" "cloudbuild_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloudbuild.email}"
}

# Custom IAM role for ML operations
resource "google_project_iam_custom_role" "ml_ops_role" {
  role_id     = "${replace(var.app_name, "-", "_")}_ml_ops"
  title       = "${var.app_name} ML Operations"
  description = "Custom role for ML operations including model training and inference"
  
  permissions = [
    "aiplatform.batchPredictionJobs.create",
    "aiplatform.batchPredictionJobs.get",
    "aiplatform.batchPredictionJobs.list",
    "aiplatform.customJobs.create",
    "aiplatform.customJobs.get",
    "aiplatform.customJobs.list",
    "aiplatform.models.create",
    "aiplatform.models.get",
    "aiplatform.models.list",
    "aiplatform.models.update",
    "compute.instances.get",
    "compute.instances.list",
    "compute.machineTypes.get",
    "compute.machineTypes.list",
    "compute.zones.get",
    "compute.zones.list",
    "storage.buckets.get",
    "storage.objects.create",
    "storage.objects.delete",
    "storage.objects.get",
    "storage.objects.list",
    "storage.objects.update"
  ]
}

# Assign custom role to Celery service account
resource "google_project_iam_member" "celery_ml_ops" {
  project = var.project_id
  role    = google_project_iam_custom_role.ml_ops_role.name
  member  = "serviceAccount:${google_service_account.celery.email}"
}

# Workload Identity for future GKE integration (if needed)
resource "google_service_account_iam_member" "workload_identity_backend" {
  count              = var.enable_workload_identity ? 1 : 0
  service_account_id = google_service_account.backend.name
  role              = "roles/iam.workloadIdentityUser"
  member            = "serviceAccount:${var.project_id}.svc.id.goog[default/ml-eval-backend]"
}

resource "google_service_account_iam_member" "workload_identity_celery" {
  count              = var.enable_workload_identity ? 1 : 0
  service_account_id = google_service_account.celery.name
  role              = "roles/iam.workloadIdentityUser"
  member            = "serviceAccount:${var.project_id}.svc.id.goog[default/ml-eval-celery]"
}