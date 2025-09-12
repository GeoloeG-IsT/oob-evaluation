#!/bin/bash

# ML Evaluation Platform - Project Setup Script
# This script sets up a new GCP project for the ML Evaluation Platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID=""
BILLING_ACCOUNT_ID=""
ORGANIZATION_ID=""
FOLDER_ID=""
ENVIRONMENT="production"
REGION="us-central1"

# Help function
show_help() {
    echo "ML Evaluation Platform - Project Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --project-id PROJECT_ID          GCP Project ID (required)"
    echo "  -b, --billing-account ACCOUNT_ID     Billing Account ID (required)"
    echo "  -o, --organization-id ORG_ID         Organization ID (optional)"
    echo "  -f, --folder-id FOLDER_ID            Folder ID (optional)"
    echo "  -e, --environment ENV                Environment (production|staging) (default: production)"
    echo "  -r, --region REGION                  GCP Region (default: us-central1)"
    echo "  -h, --help                           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --project-id ml-eval-prod --billing-account 0X0X0X-0X0X0X-0X0X0X"
    echo "  $0 -p ml-eval-staging -b ACCOUNT_ID -e staging"
    echo ""
    echo "To find your billing account ID:"
    echo "  gcloud billing accounts list"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -b|--billing-account)
            BILLING_ACCOUNT_ID="$2"
            shift 2
            ;;
        -o|--organization-id)
            ORGANIZATION_ID="$2"
            shift 2
            ;;
        -f|--folder-id)
            FOLDER_ID="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$PROJECT_ID" ]]; then
    echo -e "${RED}Error: Project ID is required${NC}"
    show_help
    exit 1
fi

if [[ -z "$BILLING_ACCOUNT_ID" ]]; then
    echo -e "${RED}Error: Billing Account ID is required${NC}"
    echo "Run 'gcloud billing accounts list' to find your billing account ID"
    exit 1
fi

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        log_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check billing account access
    if ! gcloud billing accounts describe "$BILLING_ACCOUNT_ID" &> /dev/null; then
        log_error "Cannot access billing account $BILLING_ACCOUNT_ID. Please check permissions."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create project
create_project() {
    log_info "Creating project $PROJECT_ID..."
    
    # Build create command
    local create_cmd="gcloud projects create $PROJECT_ID --name='ML Evaluation Platform ($ENVIRONMENT)'"
    
    if [[ -n "$ORGANIZATION_ID" ]]; then
        create_cmd="$create_cmd --organization=$ORGANIZATION_ID"
    elif [[ -n "$FOLDER_ID" ]]; then
        create_cmd="$create_cmd --folder=$FOLDER_ID"
    fi
    
    # Create project
    if eval "$create_cmd"; then
        log_success "Project $PROJECT_ID created successfully"
    else
        log_error "Failed to create project $PROJECT_ID"
        exit 1
    fi
}

# Link billing account
link_billing() {
    log_info "Linking billing account to project..."
    
    if gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"; then
        log_success "Billing account linked successfully"
    else
        log_error "Failed to link billing account"
        exit 1
    fi
}

# Set project as default
set_default_project() {
    log_info "Setting project as default..."
    
    gcloud config set project "$PROJECT_ID"
    gcloud config set compute/region "$REGION"
    
    log_success "Project set as default"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required APIs..."
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "sql.googleapis.com"
        "redis.googleapis.com"
        "storage.googleapis.com"
        "secretmanager.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "container.googleapis.com"
        "compute.googleapis.com"
        "vpcaccess.googleapis.com"
        "cloudfunctions.googleapis.com"
        "pubsub.googleapis.com"
        "aiplatform.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "servicenetworking.googleapis.com"
        "sqladmin.googleapis.com"
    )
    
    log_info "Enabling ${#apis[@]} APIs (this may take a few minutes)..."
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        if gcloud services enable "$api" --project="$PROJECT_ID"; then
            log_success "Enabled $api"
        else
            log_warning "Failed to enable $api (might already be enabled)"
        fi
    done
    
    log_success "API enablement completed"
}

# Create service accounts
create_service_accounts() {
    log_info "Creating service accounts..."
    
    # Cloud Build service account
    if gcloud iam service-accounts create ml-eval-cloudbuild \
        --display-name="ML Eval Cloud Build Service Account" \
        --project="$PROJECT_ID"; then
        log_success "Created Cloud Build service account"
    else
        log_warning "Cloud Build service account might already exist"
    fi
    
    # Grant necessary roles to Cloud Build service account
    local cloudbuild_roles=(
        "roles/cloudbuild.builds.builder"
        "roles/run.developer"
        "roles/storage.admin"
        "roles/iam.serviceAccountUser"
        "roles/logging.logWriter"
    )
    
    for role in "${cloudbuild_roles[@]}"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:ml-eval-cloudbuild@${PROJECT_ID}.iam.gserviceaccount.com" \
            --role="$role" &> /dev/null
    done
    
    log_success "Service accounts configured"
}

# Create storage buckets for Terraform state
create_terraform_bucket() {
    log_info "Creating Terraform state storage bucket..."
    
    local bucket_name="${PROJECT_ID}-terraform-state"
    
    if gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$bucket_name" 2>/dev/null; then
        log_success "Created Terraform state bucket: gs://$bucket_name"
    else
        log_warning "Terraform state bucket might already exist"
    fi
    
    # Enable versioning
    gsutil versioning set on "gs://$bucket_name" 2>/dev/null || true
    
    log_success "Terraform state bucket configured"
}

# Create environment-specific configuration
create_environment_config() {
    log_info "Creating environment-specific configuration..."
    
    local config_dir="deployment/terraform/environments"
    local config_file="$config_dir/${ENVIRONMENT}.tfvars"
    
    # Create directory if it doesn't exist
    mkdir -p "$config_dir"
    
    # Create tfvars file
    cat > "$config_file" << EOF
# ${ENVIRONMENT^} environment configuration for ML Evaluation Platform
# Generated by setup-project.sh on $(date)

# Project Configuration
project_id  = "$PROJECT_ID"
region      = "$REGION"
zone        = "${REGION}-a"
environment = "$ENVIRONMENT"

# Network Configuration
subnet_cidr    = "10.0.0.0/24"
connector_cidr = "10.0.1.0/28"

EOF

    # Add environment-specific settings
    if [[ "$ENVIRONMENT" == "production" ]]; then
        cat >> "$config_file" << EOF
# Database Configuration (Production)
db_tier                     = "db-standard-4"
db_disk_size               = 200
db_backup_retention_days   = 30

# Redis Configuration (Production)
redis_memory_gb = 4

# Cloud Run Configuration (Production)
backend_min_instances              = 2
backend_max_instances              = 20
frontend_min_instances             = 2
frontend_max_instances             = 50
celery_training_max_instances      = 5
celery_inference_max_instances     = 10

# Storage Configuration
storage_location = "US"
storage_class    = "STANDARD"

# Security and Monitoring
enable_binary_authorization = true
enable_monitoring          = true
log_retention_days        = 90
budget_amount            = 2000

# Disaster Recovery
enable_cross_region_backup = true
backup_region             = "us-east1"
EOF
    else
        cat >> "$config_file" << EOF
# Database Configuration (Staging/Development)
db_tier                     = "db-standard-2"
db_disk_size               = 50
db_backup_retention_days   = 7

# Redis Configuration (Staging/Development)
redis_memory_gb = 1

# Cloud Run Configuration (Staging/Development)
backend_min_instances              = 0
backend_max_instances              = 5
frontend_min_instances             = 0
frontend_max_instances             = 10
celery_training_max_instances      = 2
celery_inference_max_instances     = 3

# Storage Configuration
storage_location = "${REGION^^}"
storage_class    = "STANDARD"

# Security and Monitoring (relaxed for non-production)
enable_binary_authorization = false
enable_monitoring          = true
log_retention_days        = 30
budget_amount            = 500
enable_preemptible        = true

# Disaster Recovery (minimal for non-production)
enable_cross_region_backup = false
backup_region             = "us-east1"
EOF
    fi
    
    log_success "Created configuration file: $config_file"
}

# Setup summary
show_summary() {
    echo ""
    echo "========================================"
    echo "Project Setup Complete!"
    echo "========================================"
    echo ""
    echo "Project Details:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Environment: $ENVIRONMENT"
    echo "  Region: $REGION"
    echo "  Billing Account: $BILLING_ACCOUNT_ID"
    echo ""
    echo "Next Steps:"
    echo "  1. Review the generated configuration:"
    echo "     deployment/terraform/environments/${ENVIRONMENT}.tfvars"
    echo ""
    echo "  2. Deploy the infrastructure:"
    echo "     cd deployment/terraform"
    echo "     terraform init"
    echo "     terraform apply -var-file=\"environments/${ENVIRONMENT}.tfvars\""
    echo ""
    echo "  3. Or use the automated deployment script:"
    echo "     ./deployment/scripts/deploy.sh --project-id $PROJECT_ID --region $REGION"
    echo ""
    echo "Project Console: https://console.cloud.google.com/home/dashboard?project=$PROJECT_ID"
    echo ""
}

# Main setup flow
main() {
    echo "========================================"
    echo "ML Evaluation Platform - Project Setup"
    echo "========================================"
    echo ""
    
    log_info "Setting up project: $PROJECT_ID ($ENVIRONMENT)"
    echo ""
    
    check_prerequisites
    create_project
    link_billing
    set_default_project
    enable_apis
    create_service_accounts
    create_terraform_bucket
    create_environment_config
    show_summary
    
    log_success "Project setup completed successfully!"
}

# Run main function
main "$@"