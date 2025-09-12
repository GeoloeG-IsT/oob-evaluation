#!/bin/bash

# ML Evaluation Platform - Configuration Migration Script
# This script helps migrate configuration between environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Usage information
usage() {
    cat << EOF
ML Evaluation Platform - Configuration Migration Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    init <environment>        Initialize configuration for environment
    migrate <from> <to>       Migrate configuration between environments
    validate <environment>    Validate configuration for environment
    backup <environment>      Backup configuration
    restore <backup_file>     Restore configuration from backup
    generate-secrets         Generate secure configuration values
    setup-gcp <project_id>    Setup GCP Secret Manager
    deploy-secrets <env>      Deploy secrets to GCP Secret Manager

Environments:
    development, staging, production, testing

Examples:
    $0 init development
    $0 migrate development staging
    $0 validate production
    $0 setup-gcp my-project-id
    $0 deploy-secrets production

Options:
    -h, --help               Show this help message
    -v, --verbose            Enable verbose output
    --dry-run               Show what would be done without making changes
    --force                 Force operations without confirmation

EOF
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check required tools
    for tool in python3 docker gcloud; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
    
    # Check Python dependencies
    if ! python3 -c "import pydantic, sqlalchemy" &> /dev/null; then
        log_warning "Python dependencies missing. Installing..."
        pip install -r "$PROJECT_ROOT/backend/requirements.txt"
    fi
}

# Generate secure configuration values
generate_secrets() {
    log_info "Generating secure configuration values..."
    
    python3 -c "
import secrets
import json

config = {
    'SECRET_KEY': secrets.token_urlsafe(32),
    'JWT_SECRET': secrets.token_urlsafe(32),
    'ENCRYPTION_KEY': secrets.token_urlsafe(32),
    'ADMIN_API_KEY': secrets.token_urlsafe(24),
    'FLOWER_AUTH': f'admin:{secrets.token_urlsafe(16)}',
    'DB_PASSWORD': secrets.token_urlsafe(24),
    'REDIS_PASSWORD': secrets.token_urlsafe(24)
}

print(json.dumps(config, indent=2))
"
}

# Initialize environment configuration
init_environment() {
    local env="$1"
    local template_file="$CONFIG_DIR/.env.template"
    local env_file="$CONFIG_DIR/.env.$env"
    
    log_info "Initializing configuration for environment: $env"
    
    # Check if template exists
    if [ ! -f "$template_file" ]; then
        log_error "Template file not found: $template_file"
        exit 1
    fi
    
    # Check if environment file already exists
    if [ -f "$env_file" ] && [ "$FORCE" != "true" ]; then
        log_warning "Environment file already exists: $env_file"
        read -p "Do you want to overwrite it? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborted"
            exit 0
        fi
    fi
    
    # Copy template and customize for environment
    cp "$template_file" "$env_file"
    
    # Environment-specific customizations
    case "$env" in
        "development")
            sed -i 's/ENVIRONMENT=development/ENVIRONMENT=development/' "$env_file"
            sed -i 's/DEBUG=false/DEBUG=true/' "$env_file"
            sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' "$env_file"
            sed -i 's/ml_eval_platform/ml_eval_platform_dev/g' "$env_file"
            ;;
        "testing")
            sed -i 's/ENVIRONMENT=development/ENVIRONMENT=testing/' "$env_file"
            sed -i 's/DEBUG=false/DEBUG=true/' "$env_file"
            sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' "$env_file"
            sed -i 's/postgresql:\/\/user:password@localhost:5432\/ml_eval_platform/sqlite:\/\/\/\.\/test_ml_eval_platform.db/' "$env_file"
            sed -i 's/redis:\/\/localhost:6379\/0/redis:\/\/localhost:6379\/15/' "$env_file"
            ;;
        "staging")
            sed -i 's/ENVIRONMENT=development/ENVIRONMENT=staging/' "$env_file"
            sed -i 's/ml_eval_platform/ml_eval_platform_staging/g' "$env_file"
            sed -i 's/USE_SECRET_MANAGER=false/USE_SECRET_MANAGER=true/' "$env_file"
            ;;
        "production")
            sed -i 's/ENVIRONMENT=development/ENVIRONMENT=production/' "$env_file"
            sed -i 's/DEBUG=false/DEBUG=false/' "$env_file"
            sed -i 's/localhost:8000/your-backend-url/' "$env_file"
            sed -i 's/localhost:3000/your-frontend-url/' "$env_file"
            sed -i 's/USE_GCS=false/USE_GCS=true/' "$env_file"
            sed -i 's/USE_SECRET_MANAGER=false/USE_SECRET_MANAGER=true/' "$env_file"
            ;;
    esac
    
    # Generate and set secure values
    if [ "$env" != "testing" ]; then
        log_info "Generating secure values..."
        local secrets
        secrets=$(generate_secrets)
        
        # Extract values and update file
        local secret_key
        secret_key=$(echo "$secrets" | python3 -c "import sys, json; print(json.load(sys.stdin)['SECRET_KEY'])")
        sed -i "s/your-secret-key-here-replace-in-production/$secret_key/" "$env_file"
        
        if [ "$env" = "production" ]; then
            # For production, use placeholder values that will be replaced by Secret Manager
            sed -i 's/SECRET_KEY=.*/SECRET_KEY=${SECRET_KEY}/' "$env_file"
            sed -i 's/DATABASE_URL=.*/DATABASE_URL=${DATABASE_URL}/' "$env_file"
            sed -i 's/REDIS_URL=.*/REDIS_URL=${REDIS_URL}/' "$env_file"
        fi
    fi
    
    log_success "Environment configuration initialized: $env_file"
    log_info "Please review and customize the configuration for your needs"
}

# Migrate configuration between environments
migrate_configuration() {
    local from_env="$1"
    local to_env="$2"
    local from_file="$CONFIG_DIR/.env.$from_env"
    local to_file="$CONFIG_DIR/.env.$to_env"
    
    log_info "Migrating configuration from $from_env to $to_env"
    
    # Check source file exists
    if [ ! -f "$from_file" ]; then
        log_error "Source configuration file not found: $from_file"
        exit 1
    fi
    
    # Backup existing target file if it exists
    if [ -f "$to_file" ]; then
        local backup_file="$to_file.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$to_file" "$backup_file"
        log_info "Backed up existing configuration to: $backup_file"
    fi
    
    # Copy and customize
    cp "$from_file" "$to_file"
    
    # Update environment-specific values
    sed -i "s/ENVIRONMENT=$from_env/ENVIRONMENT=$to_env/" "$to_file"
    
    # Environment-specific migrations
    case "$to_env" in
        "production")
            sed -i 's/DEBUG=true/DEBUG=false/' "$to_file"
            sed -i 's/LOG_LEVEL=DEBUG/LOG_LEVEL=INFO/' "$to_file"
            sed -i 's/USE_GCS=false/USE_GCS=true/' "$to_file"
            sed -i 's/USE_SECRET_MANAGER=false/USE_SECRET_MANAGER=true/' "$to_file"
            ;;
        "development")
            sed -i 's/DEBUG=false/DEBUG=true/' "$to_file"
            sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' "$to_file"
            ;;
    esac
    
    log_success "Configuration migrated successfully"
    
    # Validate the new configuration
    if validate_configuration "$to_env"; then
        log_success "Migration validation passed"
    else
        log_warning "Migration validation failed - please review the configuration"
    fi
}

# Validate configuration
validate_configuration() {
    local env="$1"
    local env_file="$CONFIG_DIR/.env.$env"
    
    log_info "Validating configuration for environment: $env"
    
    if [ ! -f "$env_file" ]; then
        log_error "Configuration file not found: $env_file"
        return 1
    fi
    
    # Export environment variables from file
    set -a
    source "$env_file"
    set +a
    
    # Run Python validation
    cd "$PROJECT_ROOT/backend"
    if python3 -c "
import sys
sys.path.append('src')
from core.config_validator import validate_configuration
result = validate_configuration()
if result['valid']:
    print('✅ Configuration validation passed')
    exit(0)
else:
    print('❌ Configuration validation failed:')
    for error in result['errors']:
        print(f'  - {error}')
    exit(1)
"; then
        log_success "Configuration validation passed"
        return 0
    else
        log_error "Configuration validation failed"
        return 1
    fi
}

# Setup GCP Secret Manager
setup_gcp_secrets() {
    local project_id="$1"
    
    log_info "Setting up GCP Secret Manager for project: $project_id"
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Please authenticate with gcloud first: gcloud auth login"
        exit 1
    fi
    
    # Set project
    gcloud config set project "$project_id"
    
    # Enable Secret Manager API
    log_info "Enabling Secret Manager API..."
    gcloud services enable secretmanager.googleapis.com
    
    # Create secrets
    local secrets=(
        "ml-eval-secret-key"
        "ml-eval-jwt-secret"
        "ml-eval-database-url"
        "ml-eval-redis-url"
        "ml-eval-encryption-key"
    )
    
    for secret in "${secrets[@]}"; do
        log_info "Creating secret: $secret"
        if gcloud secrets describe "$secret" &> /dev/null; then
            log_warning "Secret $secret already exists, skipping"
        else
            gcloud secrets create "$secret" --replication-policy="automatic"
            log_success "Created secret: $secret"
        fi
    done
    
    log_success "GCP Secret Manager setup completed"
}

# Deploy secrets to GCP
deploy_secrets() {
    local env="$1"
    local env_file="$CONFIG_DIR/.env.$env"
    
    log_info "Deploying secrets to GCP Secret Manager for environment: $env"
    
    if [ ! -f "$env_file" ]; then
        log_error "Configuration file not found: $env_file"
        exit 1
    fi
    
    # Generate secure values
    local secrets_json
    secrets_json=$(generate_secrets)
    
    # Deploy each secret
    echo "$secrets_json" | python3 -c "
import sys
import json
import subprocess

secrets = json.load(sys.stdin)
secret_mapping = {
    'SECRET_KEY': 'ml-eval-secret-key',
    'JWT_SECRET': 'ml-eval-jwt-secret', 
    'ENCRYPTION_KEY': 'ml-eval-encryption-key'
}

for key, gcp_secret in secret_mapping.items():
    value = secrets[key]
    cmd = ['gcloud', 'secrets', 'versions', 'add', gcp_secret, '--data-file=-']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
    proc.communicate(input=value)
    if proc.returncode == 0:
        print(f'✅ Deployed {gcp_secret}')
    else:
        print(f'❌ Failed to deploy {gcp_secret}')
"
    
    log_success "Secrets deployed to GCP Secret Manager"
}

# Backup configuration
backup_configuration() {
    local env="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="$PROJECT_ROOT/backups/config"
    local backup_file="$backup_dir/${env}_config_backup_${timestamp}.tar.gz"
    
    log_info "Creating configuration backup for environment: $env"
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Create backup
    tar -czf "$backup_file" -C "$CONFIG_DIR" \
        ".env.$env" \
        docker-compose.yml \
        docker-compose.${env}.yml 2>/dev/null || true
    
    log_success "Configuration backed up to: $backup_file"
    echo "$backup_file"
}

# Restore configuration
restore_configuration() {
    local backup_file="$1"
    
    log_info "Restoring configuration from: $backup_file"
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Extract backup
    tar -xzf "$backup_file" -C "$CONFIG_DIR"
    
    log_success "Configuration restored from backup"
}

# Parse command line arguments
VERBOSE=false
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Check if command is provided
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Main command processing
COMMAND="$1"
shift

case "$COMMAND" in
    "init")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 init <environment>"
            exit 1
        fi
        check_prerequisites
        init_environment "$1"
        ;;
    "migrate")
        if [ $# -ne 2 ]; then
            log_error "Usage: $0 migrate <from_env> <to_env>"
            exit 1
        fi
        check_prerequisites
        migrate_configuration "$1" "$2"
        ;;
    "validate")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 validate <environment>"
            exit 1
        fi
        check_prerequisites
        validate_configuration "$1"
        ;;
    "backup")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 backup <environment>"
            exit 1
        fi
        backup_configuration "$1"
        ;;
    "restore")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 restore <backup_file>"
            exit 1
        fi
        restore_configuration "$1"
        ;;
    "generate-secrets")
        generate_secrets
        ;;
    "setup-gcp")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 setup-gcp <project_id>"
            exit 1
        fi
        setup_gcp_secrets "$1"
        ;;
    "deploy-secrets")
        if [ $# -ne 1 ]; then
            log_error "Usage: $0 deploy-secrets <environment>"
            exit 1
        fi
        deploy_secrets "$1"
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac