#!/bin/bash

# T085 - ML Evaluation Platform Comprehensive Validation Runner
# This script orchestrates the complete validation process

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
VALIDATION_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.validation.yml"
LOG_FILE="$SCRIPT_DIR/t085_validation_runner.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}$message${NC}"
    log "$message"
}

# Check prerequisites
check_prerequisites() {
    print_status $BLUE "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_status $RED "Error: Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_status $RED "Error: Docker daemon is not running"
        exit 1
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_status $RED "Error: docker-compose is not installed"
        exit 1
    fi
    
    # Check if Python 3.8+ is available
    if ! command -v python3 &> /dev/null; then
        print_status $RED "Error: Python 3 is not installed"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    local required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_status $RED "Error: Python 3.8+ required, found $python_version"
        exit 1
    fi
    
    print_status $GREEN "✓ Prerequisites check passed"
}

# Install Python dependencies
install_dependencies() {
    print_status $BLUE "Installing Python dependencies..."
    
    if [ -f "$SCRIPT_DIR/t085_requirements.txt" ]; then
        python3 -m pip install --user -r "$SCRIPT_DIR/t085_requirements.txt" || {
            print_status $RED "Failed to install Python dependencies"
            exit 1
        }
    fi
    
    print_status $GREEN "✓ Python dependencies installed"
}

# Clean up any existing containers
cleanup_containers() {
    print_status $BLUE "Cleaning up existing containers..."
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
    
    # Clean up validation data
    if [ -d "$SCRIPT_DIR/t085_validation_data" ]; then
        rm -rf "$SCRIPT_DIR/t085_validation_data"
    fi
    
    # Clean up test images
    if [ -d "$SCRIPT_DIR/test_images" ]; then
        rm -rf "$SCRIPT_DIR/test_images"
    fi
    
    print_status $GREEN "✓ Cleanup completed"
}

# Start Docker Compose services
start_services() {
    print_status $BLUE "Starting Docker Compose services..."
    
    # Create validation data directory
    mkdir -p "$SCRIPT_DIR/t085_validation_data"
    
    # Build and start services
    docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" build --parallel || {
        print_status $RED "Failed to build services"
        exit 1
    }
    
    docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" up -d || {
        print_status $RED "Failed to start services"
        exit 1
    }
    
    print_status $GREEN "✓ Services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status $BLUE "Waiting for services to become healthy..."
    
    local max_wait=300  # 5 minutes
    local wait_interval=10
    local elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        local healthy_count=0
        local total_services=0
        
        # Check health of key services
        local services=("db" "redis" "backend" "frontend")
        
        for service in "${services[@]}"; do
            total_services=$((total_services + 1))
            local health=$(docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" ps -q "$service" | xargs docker inspect --format='{{.State.Health.Status}}' 2>/dev/null)
            
            if [ "$health" = "healthy" ] || { [ -z "$health" ] && docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" ps "$service" | grep -q "Up"; }; then
                healthy_count=$((healthy_count + 1))
            fi
        done
        
        print_status $YELLOW "Health check: $healthy_count/$total_services services ready"
        
        if [ $healthy_count -eq $total_services ]; then
            print_status $GREEN "✓ All services are healthy"
            return 0
        fi
        
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
    done
    
    print_status $RED "Timeout waiting for services to become healthy"
    
    # Show service status for debugging
    print_status $YELLOW "Current service status:"
    docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" ps
    
    # Show logs for debugging
    print_status $YELLOW "Recent logs:"
    docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" logs --tail=10
    
    return 1
}

# Run the validation
run_validation() {
    print_status $BLUE "Running T085 comprehensive validation..."
    
    cd "$SCRIPT_DIR"
    
    # Set environment variables for validation
    export BACKEND_URL="http://localhost:8000"
    export FRONTEND_URL="http://localhost:3000"
    export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ml_eval_platform"
    export REDIS_URL="redis://localhost:6379/0"
    
    # Run the validation with manual mode (services already started)
    python3 t085_comprehensive_validator.py --manual || {
        local exit_code=$?
        print_status $RED "Validation failed with exit code $exit_code"
        
        # Collect logs for analysis
        print_status $BLUE "Collecting service logs for analysis..."
        mkdir -p "$SCRIPT_DIR/validation_logs"
        
        local services=("db" "redis" "backend" "frontend" "celery-training" "celery-inference" "celery-evaluation" "celery-deployment" "flower")
        for service in "${services[@]}"; do
            docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" logs --no-color "$service" > "$SCRIPT_DIR/validation_logs/${service}.log" 2>&1 || true
        done
        
        print_status $YELLOW "Service logs saved to validation_logs/ directory"
        return $exit_code
    }
    
    print_status $GREEN "✓ Validation completed successfully"
}

# Show validation results
show_results() {
    print_status $BLUE "Validation Results Summary:"
    
    # Find the latest validation report
    local latest_report=$(ls -t t085_validation_report_*.txt 2>/dev/null | head -1)
    
    if [ -n "$latest_report" ] && [ -f "$latest_report" ]; then
        echo
        echo "================================================================================"
        echo "LATEST VALIDATION REPORT:"
        echo "================================================================================"
        head -30 "$latest_report"
        echo "..."
        echo "Full report available in: $latest_report"
        
        # Also show JSON summary if available
        local json_report="${latest_report%.txt}.json"
        if [ -f "$json_report" ]; then
            echo "JSON report available in: $json_report"
        fi
    else
        print_status $YELLOW "No validation report found"
    fi
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    print_status $GREEN "Starting T085 ML Evaluation Platform Comprehensive Validation"
    print_status $GREEN "=============================================================="
    
    # Initialize log file
    echo "T085 Validation Run Started: $(date)" > "$LOG_FILE"
    
    # Parse command line arguments
    local cleanup_after=true
    local skip_deps=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cleanup)
                cleanup_after=false
                shift
                ;;
            --skip-deps)
                skip_deps=false
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --no-cleanup    Don't stop services after validation"
                echo "  --skip-deps     Skip dependency installation"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_status $RED "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute validation steps
    check_prerequisites
    
    if [ "$skip_deps" = false ]; then
        install_dependencies
    fi
    
    cleanup_containers
    start_services
    
    if wait_for_services; then
        if run_validation; then
            local validation_success=true
        else
            local validation_success=false
        fi
    else
        local validation_success=false
        print_status $RED "Services failed to start properly"
    fi
    
    show_results
    
    # Cleanup
    if [ "$cleanup_after" = true ]; then
        print_status $BLUE "Stopping services..."
        docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" down -v
        print_status $GREEN "✓ Services stopped"
    else
        print_status $YELLOW "Services left running (--no-cleanup specified)"
        print_status $YELLOW "To stop services manually: docker-compose -f $COMPOSE_FILE -f $VALIDATION_COMPOSE_FILE down -v"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_status $BLUE "Total validation time: ${duration} seconds"
    
    if [ "$validation_success" = true ]; then
        print_status $GREEN "🎉 T085 VALIDATION COMPLETED SUCCESSFULLY!"
        exit 0
    else
        print_status $RED "❌ T085 VALIDATION FAILED"
        exit 1
    fi
}

# Trap signals for cleanup
trap 'print_status $YELLOW "Interrupted! Cleaning up..."; docker-compose -f "$COMPOSE_FILE" -f "$VALIDATION_COMPOSE_FILE" down -v 2>/dev/null || true; exit 130' INT TERM

# Run main function
main "$@"