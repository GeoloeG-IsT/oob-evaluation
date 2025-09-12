# Staging environment configuration for ML Evaluation Platform

# Project Configuration
project_id  = "ml-eval-platform-staging"  # Replace with your actual staging project ID
region      = "us-central1"
zone        = "us-central1-b"
environment = "staging"

# Network Configuration
subnet_cidr    = "10.1.0.0/24"
connector_cidr = "10.1.1.0/28"

# Database Configuration (smaller for staging)
db_tier                     = "db-standard-2"      # Smaller tier for staging
db_disk_size               = 50                     # 50GB for staging
db_backup_retention_days   = 7                     # 7 days backup retention

# Redis Configuration (smaller for staging)
redis_memory_gb = 1                               # 1GB Redis for staging

# Cloud Run Configuration (reduced capacity)
backend_min_instances              = 0            # Scale to zero when idle
backend_max_instances              = 5            # Max 5 backend instances
frontend_min_instances             = 0            # Scale to zero when idle
frontend_max_instances             = 10           # Max 10 frontend instances
celery_training_max_instances      = 2            # Max 2 training workers
celery_inference_max_instances     = 3            # Max 3 inference workers

# Storage Configuration
storage_location = "US-CENTRAL1"                 # Regional for cost savings
storage_class    = "STANDARD"                     # Standard class

# Monitoring Configuration
enable_monitoring  = true                         # Enable monitoring
log_retention_days = 30                          # 30 days log retention

# Security Configuration (relaxed for staging)
enable_binary_authorization = false             # Disable for easier development
allowed_ingress_cidrs      = ["0.0.0.0/0"]     # Allow all for testing

# Cost Management
enable_preemptible          = true              # Use preemptible for cost savings
budget_amount              = 500                # $500/month budget limit

# Disaster Recovery (minimal for staging)
enable_cross_region_backup = false             # No cross-region backup needed
backup_region             = "us-east1"         # Not used for staging