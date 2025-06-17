#!/bin/bash

# SearXNG Secret Key Generator
# Generates a cryptographically secure secret key for SearXNG
# Usage: ./generate_secret_key.sh [options]

set -euo pipefail

# Default configuration
SETTINGS_FILE="settings.yml"
KEY_LENGTH=64
CREATE_BACKUP=true
GENERATE_ONLY=false
VERBOSE=false

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}🔍 $1${NC}"
    fi
}

# Function to print header
print_header() {
    echo -e "${PURPLE}🔐 SearXNG Secret Key Generator${NC}"
    echo -e "${PURPLE}==================================${NC}"
    echo
}

# Function to generate cryptographically secure secret key
generate_secret_key() {
    local length=${1:-64}
    local secret_key=""
    
    log_debug "Attempting to generate ${length}-character secret key"
    
    # Method 1: OpenSSL (preferred)
    if command -v openssl >/dev/null 2>&1; then
        log_debug "Using OpenSSL for key generation"
        secret_key=$(openssl rand -base64 $((length * 3 / 4)) | tr -d "=+/" | cut -c1-${length})
    
    # Method 2: Python3 with secrets module
    elif command -v python3 >/dev/null 2>&1; then
        log_debug "Using Python3 for key generation"
        secret_key=$(python3 -c "
import secrets
import string
alphabet = string.ascii_letters + string.digits + '!@#\$%^&*()-_=+[]{}|;:,.<>?'
print(''.join(secrets.choice(alphabet) for _ in range($length)))
")
    
    # Method 3: /dev/urandom fallback
    elif [[ -f /dev/urandom ]]; then
        log_debug "Using /dev/urandom for key generation"
        secret_key=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9!@#$%^&*()-_=+[]{}|;:,.<>?' | fold -w ${length} | head -n 1)
    
    # Method 4: Basic fallback (less secure)
    else
        log_warning "No secure random generator found, using basic fallback"
        secret_key=$(date +%s | sha256sum | base64 | head -c ${length})
    fi
    
    # Validate generated key
    if [[ ${#secret_key} -lt $length ]]; then
        log_error "Generated key is too short (${#secret_key} < $length)"
        return 1
    fi
    
    echo "$secret_key"
}

# Function to create backup of settings file
create_backup() {
    local file=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="${file}.backup.${timestamp}"
    
    if cp "$file" "$backup_file" 2>/dev/null; then
        log_success "Backup created: $backup_file"
        return 0
    else
        log_error "Failed to create backup"
        return 1
    fi
}

# Function to update settings.yml file
update_settings_file() {
    local settings_file=$1
    local new_secret=$2
    local create_backup=$3
    
    # Check if file exists
    if [[ ! -f "$settings_file" ]]; then
        log_error "Settings file '$settings_file' not found!"
        log_info "Create the file first or use -g flag to just generate a key"
        return 1
    fi
    
    # Create backup if requested
    if [[ "$create_backup" == "true" ]]; then
        if ! create_backup "$settings_file"; then
            log_warning "Continuing without backup..."
        fi
    fi
    
    # Check if we can modify the file
    if [[ ! -w "$settings_file" ]]; then
        log_error "No write permission for '$settings_file'"
        return 1
    fi
    
    # Escape special characters in the secret key for sed
    local escaped_secret=$(printf '%s\n' "$new_secret" | sed 's/[[\.*^$()+?{|]/\\&/g')
    
    # Check if secret_key line exists
    if grep -q "secret_key:" "$settings_file"; then
        log_debug "Found existing secret_key line, updating..."
        
        # Different sed syntax for macOS vs Linux
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed syntax
            sed -i '' "s/secret_key:.*/secret_key: \"$escaped_secret\"/" "$settings_file"
        else
            # Linux sed syntax
            sed -i "s/secret_key:.*/secret_key: \"$escaped_secret\"/" "$settings_file"
        fi
        
        # Verify the update
        if grep -q "secret_key: \"$escaped_secret\"" "$settings_file"; then
            log_success "Secret key updated successfully in $settings_file"
            return 0
        else
            log_error "Failed to verify secret key update"
            return 1
        fi
    else
        log_warning "secret_key not found in $settings_file"
        log_info "Please add this line under the 'server:' section:"
        echo -e "${CYAN}  secret_key: \"$new_secret\"${NC}"
        return 1
    fi
}

# Function to display generated key with usage examples
display_key_info() {
    local secret_key=$1
    
    echo
    log_success "Generated Secret Key:"
    echo "   $secret_key"
    echo
    log_info "Usage examples:"
    echo -e "${CYAN}   YAML format:${NC}"
    echo "   secret_key: \"$secret_key\""
    echo
    echo -e "${CYAN}   Environment variable:${NC}"
    echo "   export SEARXNG_SECRET_KEY=\"$secret_key\""
    echo
    echo -e "${CYAN}   Docker Compose:${NC}"
    echo "   environment:"
    echo "     - SEARXNG_SECRET_KEY=$secret_key"
    echo
    echo -e "${CYAN}   .env file:${NC}"
    echo "   SEARXNG_SECRET_KEY=$secret_key"
}

# Function to show usage information
show_usage() {
    cat << EOF
🔐 SearXNG Secret Key Generator

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message and exit
    -g, --generate-only     Generate key without updating any files
    -f, --file FILE         Specify settings file (default: settings.yml)
    -l, --length LENGTH     Specify key length (default: 64)
    -v, --verbose           Enable verbose output
    --no-backup             Don't create backup file when updating
    
EXAMPLES:
    $0                              # Update settings.yml with new secret
    $0 -g                           # Just generate and display key
    $0 -f config.yml                # Update specific settings file  
    $0 -l 128                       # Generate 128-character key
    $0 -g -l 32                     # Generate 32-char key without updating
    $0 --generate-only --verbose    # Generate with verbose output
    $0 -f settings.yml --no-backup  # Update without creating backup

REQUIREMENTS:
    One of: openssl, python3, or /dev/urandom (for secure key generation)
    For file updates: sed, read/write permissions

SECURITY:
    - Uses cryptographically secure random number generation
    - Creates timestamped backups before modifying files
    - Supports various secure random sources as fallbacks

EOF
}

# Function to validate requirements
check_requirements() {
    local missing_tools=()
    
    # Check for random generation tools
    if ! command -v openssl >/dev/null 2>&1 && \
       ! command -v python3 >/dev/null 2>&1 && \
       [[ ! -f /dev/urandom ]]; then
        missing_tools+=("openssl OR python3 OR /dev/urandom")
    fi
    
    # Check for file manipulation tools (only if not generate-only)
    if [[ "$GENERATE_ONLY" == "false" ]]; then
        if ! command -v sed >/dev/null 2>&1; then
            missing_tools+=("sed")
        fi
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        return 1
    fi
    
    return 0
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -g|--generate-only)
                GENERATE_ONLY=true
                shift
                ;;
            -f|--file)
                if [[ -n "${2:-}" ]]; then
                    SETTINGS_FILE="$2"
                    shift 2
                else
                    log_error "Option -f requires a filename argument"
                    exit 1
                fi
                ;;
            -l|--length)
                if [[ -n "${2:-}" ]] && [[ "$2" =~ ^[0-9]+$ ]] && [[ "$2" -gt 0 ]]; then
                    KEY_LENGTH="$2"
                    shift 2
                else
                    log_error "Option -l requires a positive integer argument"
                    exit 1
                fi
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --no-backup)
                CREATE_BACKUP=false
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                echo "Use -h for help"
                exit 1
                ;;
            *)
                log_error "Unexpected argument: $1"
                echo "Use -h for help"
                exit 1
                ;;
        esac
    done
}

# Main execution function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show header
    print_header
    
    # Check requirements
    if ! check_requirements; then
        log_error "Requirements check failed"
        exit 1
    fi
    
    # Generate the secret key
    log_info "Generating ${KEY_LENGTH}-character secret key..."
    
    if ! SECRET_KEY=$(generate_secret_key $KEY_LENGTH); then
        log_error "Failed to generate secret key"
        exit 1
    fi
    
    log_debug "Generated key: ${SECRET_KEY:0:8}..."
    
    if [[ "$GENERATE_ONLY" == "true" ]]; then
        # Just display the key
        display_key_info "$SECRET_KEY"
    else
        # Update the settings file
        log_info "Updating settings file: $SETTINGS_FILE"
        
        if update_settings_file "$SETTINGS_FILE" "$SECRET_KEY" "$CREATE_BACKUP"; then
            echo
            log_success "Configuration updated successfully!"
            log_info "New secret key: ${SECRET_KEY:0:8}...${SECRET_KEY: -8}"
            log_info "Updated file: $SETTINGS_FILE"
            
            echo
            log_info "Next steps:"
            echo "   1. Restart SearXNG container:"
            echo "      docker-compose restart searxng"
            echo "   2. Verify configuration:"
            echo "      curl -s http://localhost:8080/config | jq '.general'"
            echo "   3. Test search functionality:"
            echo "      curl 'http://localhost:8080/search?q=test&format=json'"
        else
            log_error "Failed to update settings file"
            log_info "Generated key (update manually):"
            echo "   $SECRET_KEY"
            exit 1
        fi
    fi
    
    echo
    log_success "Secret key generation completed!"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi