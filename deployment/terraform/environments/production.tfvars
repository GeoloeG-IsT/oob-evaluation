# Production environment configuration for ML Evaluation Platform

# Project Configuration
project_id  = "ml-eval-platform-prod"  # Replace with your actual project ID
region      = "us-central1"
zone        = "us-central1-a"
environment = "production"

# Network Configuration
subnet_cidr    = "10.0.0.0/24"
connector_cidr = "10.0.1.0/28"

# Database Configuration
db_tier                     = "db-standard-4"      # Higher tier for production
db_disk_size               = 200                   # 200GB for production data
db_backup_retention_days   = 30                    # 30 days backup retention

# Redis Configuration
redis_memory_gb = 4                               # 4GB Redis for production

# Cloud Run Configuration
backend_min_instances              = 2            # Always have 2 backend instances
backend_max_instances              = 20           # Scale up to 20 instances
frontend_min_instances             = 2            # Always have 2 frontend instances  
frontend_max_instances             = 50           # Scale up to 50 instances
celery_training_max_instances      = 5            # Up to 5 training workers
celery_inference_max_instances     = 10           # Up to 10 inference workers

# Storage Configuration
storage_location = "US"                           # Multi-region for production
storage_class    = "STANDARD"                     # Standard class for frequent access

# Monitoring Configuration
enable_monitoring  = true                         # Enable full monitoring
log_retention_days = 90                          # 90 days log retention

# Security Configuration
enable_binary_authorization = true               # Enable container security
allowed_ingress_cidrs      = ["0.0.0.0/0"]     # Allow all (restrict as needed)

# Cost Management
enable_preemptible          = false             # No preemptible for production
budget_amount              = 2000               # $2000/month budget limit

# Disaster Recovery
enable_cross_region_backup = true              # Enable cross-region backups
backup_region             = "us-east1"         # Backup to different region