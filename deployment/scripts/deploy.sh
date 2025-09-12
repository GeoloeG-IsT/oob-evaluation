#!/bin/bash

# ML Evaluation Platform - Production Deployment Script
# This script deploys the entire ML Evaluation Platform to Google Cloud Run

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID=""
REGION="us-central1"
DEPLOY_INFRASTRUCTURE="true"
DEPLOY_SERVICES="true"
RUN_TESTS="true"
SKIP_CONFIRMATION="false"

# Help function
show_help() {
    echo "ML Evaluation Platform Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --project-id PROJECT_ID    GCP Project ID (required)"
    echo "  -r, --region REGION            GCP Region (default: us-central1)"
    echo "  -i, --skip-infrastructure      Skip infrastructure deployment"
    echo "  -s, --skip-services           Skip service deployment"
    echo "  -t, --skip-tests              Skip post-deployment tests"
    echo "  -y, --yes                     Skip confirmation prompts"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --project-id my-project --region us-central1"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -i|--skip-infrastructure)
            DEPLOY_INFRASTRUCTURE="false"
            shift
            ;;
        -s|--skip-services)
            DEPLOY_SERVICES="false"
            shift
            ;;
        -t|--skip-tests)
            RUN_TESTS="false"
            shift
            ;;
        -y|--yes)
            SKIP_CONFIRMATION="true"
            shift
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
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        log_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Cannot access project $PROJECT_ID. Please check permissions."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Set gcloud project
setup_gcloud() {
    log_info "Setting up gcloud configuration..."
    gcloud config set project "$PROJECT_ID"
    gcloud config set run/region "$REGION"
    log_success "gcloud configured for project: $PROJECT_ID, region: $REGION"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
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
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        if gcloud services enable "$api" --project="$PROJECT_ID"; then
            log_success "Enabled $api"
        else
            log_warning "Failed to enable $api (might already be enabled)"
        fi
    done
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    if [[ "$DEPLOY_INFRASTRUCTURE" == "false" ]]; then
        log_info "Skipping infrastructure deployment"
        return 0
    fi
    
    log_info "Deploying infrastructure with Terraform..."
    
    cd deployment/terraform
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init
    
    # Create terraform.tfvars if it doesn't exist
    if [[ ! -f terraform.tfvars ]]; then
        log_info "Creating terraform.tfvars file..."
        cat > terraform.tfvars << EOF
project_id = "$PROJECT_ID"
region     = "$REGION"
environment = "production"
EOF
    fi
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -var="project_id=$PROJECT_ID" -var="region=$REGION"
    
    if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
        echo -e "${YELLOW}Do you want to proceed with the infrastructure deployment? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Infrastructure deployment cancelled"
            cd ../..
            return 0
        fi
    fi
    
    # Apply deployment
    log_info "Applying Terraform configuration..."
    terraform apply -auto-approve -var="project_id=$PROJECT_ID" -var="region=$REGION"
    
    cd ../..
    log_success "Infrastructure deployment completed"
}

# Build and deploy services
deploy_services() {
    if [[ "$DEPLOY_SERVICES" == "false" ]]; then
        log_info "Skipping service deployment"
        return 0
    fi
    
    log_info "Building and deploying services..."
    
    # Submit build for all services
    log_info "Submitting Cloud Build for all services..."
    gcloud builds submit \
        --config=deployment/cloudbuild/deploy-all.yaml \
        --substitutions=_REGION="$REGION" \
        --project="$PROJECT_ID" \
        .
    
    log_success "Services deployed successfully"
}

# Run post-deployment tests
run_tests() {
    if [[ "$RUN_TESTS" == "false" ]]; then
        log_info "Skipping post-deployment tests"
        return 0
    fi
    
    log_info "Running post-deployment tests..."
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 60
    
    # Get service URLs
    local backend_url
    local frontend_url
    
    backend_url=$(gcloud run services describe ml-eval-backend \
        --region="$REGION" \
        --format='value(status.url)' \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    frontend_url=$(gcloud run services describe ml-eval-frontend \
        --region="$REGION" \
        --format='value(status.url)' \
        --project="$PROJECT_ID" 2>/dev/null || echo "")
    
    if [[ -z "$backend_url" || -z "$frontend_url" ]]; then
        log_error "Could not retrieve service URLs"
        return 1
    fi
    
    log_info "Backend URL: $backend_url"
    log_info "Frontend URL: $frontend_url"
    
    # Test backend health
    log_info "Testing backend health..."
    if curl -f "$backend_url/health" --connect-timeout 30 --max-time 60; then
        log_success "Backend health check passed"
    else
        log_error "Backend health check failed"
        return 1
    fi
    
    # Test frontend health
    log_info "Testing frontend health..."
    if curl -f "$frontend_url/api/health" --connect-timeout 30 --max-time 60; then
        log_success "Frontend health check passed"
    else
        log_error "Frontend health check failed"
        return 1
    fi
    
    # Test API documentation
    log_info "Testing API documentation..."
    if curl -f "$backend_url/docs" --connect-timeout 30 --max-time 60 > /dev/null; then
        log_success "API documentation accessible"
    else
        log_warning "API documentation not accessible"
    fi
    
    log_success "Post-deployment tests completed"
}

# Deployment summary
show_summary() {
    log_info "Deployment Summary:"
    echo "=================="
    echo "Project ID: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Infrastructure: $([[ "$DEPLOY_INFRASTRUCTURE" == "true" ]] && echo "Deployed" || echo "Skipped")"
    echo "Services: $([[ "$DEPLOY_SERVICES" == "true" ]] && echo "Deployed" || echo "Skipped")"
    echo "Tests: $([[ "$RUN_TESTS" == "true" ]] && echo "Executed" || echo "Skipped")"
    echo ""
    
    # Get service URLs
    local backend_url
    local frontend_url
    
    backend_url=$(gcloud run services describe ml-eval-backend \
        --region="$REGION" \
        --format='value(status.url)' \
        --project="$PROJECT_ID" 2>/dev/null || echo "Not deployed")
    
    frontend_url=$(gcloud run services describe ml-eval-frontend \
        --region="$REGION" \
        --format='value(status.url)' \
        --project="$PROJECT_ID" 2>/dev/null || echo "Not deployed")
    
    echo "Service URLs:"
    echo "  Backend:  $backend_url"
    echo "  Frontend: $frontend_url"
    echo ""
    
    log_success "ML Evaluation Platform deployment completed!"
    log_info "Access your application at: $frontend_url"
}

# Main deployment flow
main() {
    echo "========================================"
    echo "ML Evaluation Platform Deployment"
    echo "========================================"
    echo ""
    
    check_prerequisites
    setup_gcloud
    
    if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
        echo -e "${YELLOW}This will deploy the ML Evaluation Platform to:${NC}"
        echo "  Project: $PROJECT_ID"
        echo "  Region: $REGION"
        echo ""
        echo -e "${YELLOW}Do you want to continue? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    enable_apis
    deploy_infrastructure
    deploy_services
    run_tests
    show_summary
}

# Run main function
main "$@"