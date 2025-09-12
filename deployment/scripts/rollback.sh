#!/bin/bash

# ML Evaluation Platform - Rollback Script
# This script rolls back the deployment to a previous version

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
TARGET_REVISION=""
SERVICE_NAME=""
SKIP_CONFIRMATION="false"

# Help function
show_help() {
    echo "ML Evaluation Platform Rollback Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --project-id PROJECT_ID    GCP Project ID (required)"
    echo "  -r, --region REGION            GCP Region (default: us-central1)"
    echo "  -s, --service SERVICE_NAME     Service to rollback (optional, all if not specified)"
    echo "  -t, --target-revision REVISION Target revision to rollback to"
    echo "  -y, --yes                      Skip confirmation prompts"
    echo "  -h, --help                     Show this help message"
    echo ""
    echo "Services:"
    echo "  - ml-eval-backend"
    echo "  - ml-eval-frontend"
    echo "  - ml-eval-celery-training"
    echo "  - ml-eval-celery-inference"
    echo "  - ml-eval-celery-evaluation"
    echo ""
    echo "Examples:"
    echo "  $0 --project-id my-project --service ml-eval-backend --target-revision backend-v1-2-3"
    echo "  $0 --project-id my-project  # Rollback all services to previous revision"
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
        -s|--service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -t|--target-revision)
            TARGET_REVISION="$2"
            shift 2
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

# List available revisions for a service
list_revisions() {
    local service_name="$1"
    
    log_info "Available revisions for $service_name:"
    gcloud run revisions list \
        --service="$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="table(metadata.name:label=REVISION,status.conditions[0].lastTransitionTime:label=CREATED,spec.containers[0].image:label=IMAGE,status.allocatedTraffic:label=TRAFFIC)" \
        2>/dev/null || {
        log_warning "Could not list revisions for $service_name (service might not exist)"
        return 1
    }
}

# Get the previous revision for a service
get_previous_revision() {
    local service_name="$1"
    
    # Get the current revision receiving traffic
    local current_revision
    current_revision=$(gcloud run services describe "$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.traffic[0].revisionName)" \
        2>/dev/null)
    
    if [[ -z "$current_revision" ]]; then
        log_error "Could not determine current revision for $service_name"
        return 1
    fi
    
    # Get the previous revision (second most recent)
    local previous_revision
    previous_revision=$(gcloud run revisions list \
        --service="$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(metadata.name)" \
        --sort-by="~metadata.creationTimestamp" \
        --limit=2 \
        2>/dev/null | tail -n1)
    
    if [[ -z "$previous_revision" ]]; then
        log_error "Could not determine previous revision for $service_name"
        return 1
    fi
    
    echo "$previous_revision"
}

# Rollback a single service
rollback_service() {
    local service_name="$1"
    local target_revision="$2"
    
    log_info "Rolling back $service_name to revision $target_revision..."
    
    # Verify the target revision exists
    if ! gcloud run revisions describe "$target_revision" \
        --region="$REGION" \
        --project="$PROJECT_ID" &>/dev/null; then
        log_error "Target revision $target_revision does not exist for $service_name"
        return 1
    fi
    
    # Perform the rollback by updating traffic allocation
    log_info "Updating traffic allocation to revision $target_revision..."
    if gcloud run services update-traffic "$service_name" \
        --to-revisions="$target_revision=100" \
        --region="$REGION" \
        --project="$PROJECT_ID"; then
        log_success "Successfully rolled back $service_name to $target_revision"
        return 0
    else
        log_error "Failed to rollback $service_name"
        return 1
    fi
}

# Verify rollback success
verify_rollback() {
    local service_name="$1"
    local expected_revision="$2"
    
    log_info "Verifying rollback for $service_name..."
    
    # Wait for rollback to complete
    sleep 30
    
    # Check current revision
    local current_revision
    current_revision=$(gcloud run services describe "$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(status.traffic[0].revisionName)" \
        2>/dev/null)
    
    if [[ "$current_revision" == "$expected_revision" ]]; then
        log_success "Rollback verified: $service_name is running revision $current_revision"
        
        # Test service health
        local service_url
        service_url=$(gcloud run services describe "$service_name" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --format="value(status.url)" \
            2>/dev/null)
        
        if [[ -n "$service_url" ]]; then
            if [[ "$service_name" == *"backend"* ]]; then
                if curl -f "$service_url/health" --connect-timeout 10 --max-time 30 >/dev/null 2>&1; then
                    log_success "Health check passed for $service_name"
                else
                    log_warning "Health check failed for $service_name"
                fi
            elif [[ "$service_name" == *"frontend"* ]]; then
                if curl -f "$service_url/api/health" --connect-timeout 10 --max-time 30 >/dev/null 2>&1; then
                    log_success "Health check passed for $service_name"
                else
                    log_warning "Health check failed for $service_name"
                fi
            fi
        fi
        
        return 0
    else
        log_error "Rollback verification failed: expected $expected_revision, got $current_revision"
        return 1
    fi
}

# Rollback all services
rollback_all_services() {
    local services=(
        "ml-eval-backend"
        "ml-eval-frontend"
        "ml-eval-celery-training"
        "ml-eval-celery-inference"
        "ml-eval-celery-evaluation"
    )
    
    log_info "Rolling back all services..."
    
    local failed_services=()
    local success_count=0
    
    for service in "${services[@]}"; do
        # Check if service exists
        if ! gcloud run services describe "$service" \
            --region="$REGION" \
            --project="$PROJECT_ID" &>/dev/null; then
            log_warning "Service $service does not exist, skipping..."
            continue
        fi
        
        # Show available revisions
        if ! list_revisions "$service"; then
            log_warning "Skipping $service (could not list revisions)"
            continue
        fi
        
        # Get previous revision
        local previous_revision
        if previous_revision=$(get_previous_revision "$service"); then
            log_info "Previous revision for $service: $previous_revision"
        else
            log_warning "Could not determine previous revision for $service, skipping..."
            failed_services+=("$service")
            continue
        fi
        
        # Ask for confirmation for each service if not skipping
        if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
            echo -e "${YELLOW}Rollback $service to $previous_revision? (y/N/s=skip)${NC}"
            read -r response
            if [[ "$response" =~ ^[Ss]$ ]]; then
                log_info "Skipping $service"
                continue
            elif [[ ! "$response" =~ ^[Yy]$ ]]; then
                log_info "Skipping $service"
                continue
            fi
        fi
        
        # Perform rollback
        if rollback_service "$service" "$previous_revision"; then
            if verify_rollback "$service" "$previous_revision"; then
                ((success_count++))
            else
                failed_services+=("$service")
            fi
        else
            failed_services+=("$service")
        fi
    done
    
    # Summary
    log_info "Rollback Summary:"
    echo "  Successful: $success_count"
    echo "  Failed: ${#failed_services[@]}"
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        echo "  Failed services: ${failed_services[*]}"
        return 1
    fi
    
    return 0
}

# Rollback single service
rollback_single_service() {
    local service_name="$1"
    local target_revision="$2"
    
    # Check if service exists
    if ! gcloud run services describe "$service_name" \
        --region="$REGION" \
        --project="$PROJECT_ID" &>/dev/null; then
        log_error "Service $service_name does not exist"
        exit 1
    fi
    
    # If no target revision specified, get the previous one
    if [[ -z "$target_revision" ]]; then
        list_revisions "$service_name"
        
        if target_revision=$(get_previous_revision "$service_name"); then
            log_info "Previous revision for $service_name: $target_revision"
        else
            log_error "Could not determine previous revision"
            exit 1
        fi
    fi
    
    # Confirm rollback
    if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
        echo -e "${YELLOW}Rollback $service_name to $target_revision? (y/N)${NC}"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Perform rollback
    if rollback_service "$service_name" "$target_revision"; then
        verify_rollback "$service_name" "$target_revision"
    else
        exit 1
    fi
}

# Main rollback flow
main() {
    echo "========================================"
    echo "ML Evaluation Platform Rollback"
    echo "========================================"
    echo ""
    
    check_prerequisites
    setup_gcloud
    
    if [[ -n "$SERVICE_NAME" ]]; then
        rollback_single_service "$SERVICE_NAME" "$TARGET_REVISION"
    else
        if [[ "$SKIP_CONFIRMATION" == "false" ]]; then
            echo -e "${YELLOW}This will rollback ML Evaluation Platform services in:${NC}"
            echo "  Project: $PROJECT_ID"
            echo "  Region: $REGION"
            echo ""
            echo -e "${YELLOW}Do you want to continue? (y/N)${NC}"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                log_info "Rollback cancelled"
                exit 0
            fi
        fi
        
        rollback_all_services
    fi
    
    log_success "Rollback operation completed"
}

# Run main function
main "$@"