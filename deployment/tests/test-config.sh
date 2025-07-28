#!/bin/bash

# Test Configuration for KonveyN2AI
# Provides secure test tokens and configuration for CI/CD testing

# Security Enhancement: Use more secure test tokens instead of static demo-token
# These tokens are accepted by GuardFort middleware and provide better security

# Dynamic Test Token Generation Function
generate_dynamic_test_token() {
    local token_type="${1:-bearer}"
    local environment="${ENVIRONMENT:-testing}"
    
    # Generate secure random component
    local random_suffix=$(openssl rand -hex 8 2>/dev/null || echo $(date +%N | head -c 8))
    local timestamp=$(date +%s)
    
    case "$token_type" in
        "bearer")
            echo "test-bearer-${environment}-${timestamp}-${random_suffix}"
            ;;
        "api")
            echo "test-api-${environment}-${timestamp}-${random_suffix}"
            ;;
        *)
            echo "test-token-${environment}-${timestamp}-${random_suffix}"
            ;;
    esac
}

# Initialize tokens - use dynamic generation if environment supports it
if [ "${USE_DYNAMIC_TOKENS:-false}" = "true" ]; then
    export TEST_BEARER_TOKEN=$(generate_dynamic_test_token "bearer")
    export TEST_API_KEY=$(generate_dynamic_test_token "api")
    echo "🔐 Generated dynamic test tokens for enhanced security"
else
    # Use static tokens for backwards compatibility (but warn in non-dev environments)
    export TEST_BEARER_TOKEN="${TEST_BEARER_TOKEN:-demo-token}"
    export TEST_API_KEY="${TEST_API_KEY:-demo-api-key}"
    
    if [ "${ENVIRONMENT:-testing}" != "development" ] && [ "$TEST_BEARER_TOKEN" = "demo-token" ]; then
        echo "⚠️  WARNING: Using static demo-token in ${ENVIRONMENT:-testing} environment"
        echo "   Consider setting USE_DYNAMIC_TOKENS=true for enhanced security"
    fi
fi

# Legacy token support for backwards compatibility
export DEMO_TOKEN_1="demo-token"  # Keep for existing tests
export DEMO_TOKEN_2="test-token"
export DEMO_API_KEY="$TEST_API_KEY"

# Project Configuration for dynamic service discovery
export PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-konveyn2ai}"
export REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"

# Dynamic Service Discovery Function
discover_service_url() {
    local service_name=$1
    local env_var_name=$2
    
    # First try environment variable
    local url_from_env="${!env_var_name}"
    if [ -n "$url_from_env" ]; then
        echo "$url_from_env"
        return 0
    fi
    
    # If gcloud is available, try dynamic discovery
    if command -v gcloud &> /dev/null; then
        local discovered_url=$(gcloud run services describe $service_name \
            --region=$REGION \
            --project=$PROJECT_ID \
            --format="value(status.url)" 2>/dev/null || echo "")
        
        if [ -n "$discovered_url" ]; then
            echo "$discovered_url"
            export $env_var_name="$discovered_url"
            return 0
        fi
    fi
    
    # Warning: No hardcoded fallbacks - must be provided via environment
    echo "❌ Could not discover URL for $service_name. Please set $env_var_name environment variable." >&2
    return 1
}

# Service URLs - Dynamic discovery with environment variable override (NO HARD-CODED FALLBACKS)
export SVAMI_URL=$(discover_service_url "svami" "SVAMI_URL")
export JANAPADA_URL=$(discover_service_url "janapada" "JANAPADA_URL") 
export AMATYA_URL=$(discover_service_url "amatya" "AMATYA_URL")

# Validate all service URLs were discovered
if [ -z "$SVAMI_URL" ] || [ -z "$JANAPADA_URL" ] || [ -z "$AMATYA_URL" ]; then
    echo "❌ Error: Could not discover all service URLs" >&2
    echo "   Missing URLs - Svami: ${SVAMI_URL:-NOT_SET}, Janapada: ${JANAPADA_URL:-NOT_SET}, Amatya: ${AMATYA_URL:-NOT_SET}" >&2
    echo "   Please set SVAMI_URL, JANAPADA_URL, and AMATYA_URL environment variables" >&2
    echo "   or ensure gcloud is configured and services are deployed" >&2
    exit 1
fi

# Test configuration
export TEST_TIMEOUT="${TEST_TIMEOUT:-30}"
export TEST_RETRIES="${TEST_RETRIES:-3}"

# Function to get a valid test token
get_test_token() {
    local token_type="${1:-bearer}"
    
    case "$token_type" in
        "bearer")
            echo "$TEST_BEARER_TOKEN"
            ;;
        "api")
            echo "$TEST_API_KEY"
            ;;
        "demo1")
            echo "$DEMO_TOKEN_1"
            ;;
        "demo2")
            echo "$DEMO_TOKEN_2"
            ;;
        *)
            echo "$TEST_BEARER_TOKEN"
            ;;
    esac
}

# Function to create authorization header
create_auth_header() {
    local token_type="${1:-bearer}"
    local token=$(get_test_token "$token_type")
    
    case "$token_type" in
        "api"|"demo_api")
            echo "ApiKey $token"
            ;;
        *)
            echo "Bearer $token"
            ;;
    esac
}

# Function to test authentication with different tokens
test_auth_tokens() {
    local url="$1"
    local endpoint="${2:-/health}"
    
    echo "Testing authentication tokens against $url$endpoint"
    
    # Test primary bearer token
    local auth_header=$(create_auth_header "bearer")
    local status=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: $auth_header" "$url$endpoint")
    echo "Primary Bearer Token: HTTP $status"
    
    # Test API key
    local auth_header=$(create_auth_header "api")
    local status=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: $auth_header" "$url$endpoint")
    echo "API Key: HTTP $status"
    
    # Test demo tokens
    local auth_header=$(create_auth_header "demo1")
    local status=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: $auth_header" "$url$endpoint")
    echo "Demo Token 1: HTTP $status"
}

# Export functions for use in other scripts
export -f get_test_token
export -f create_auth_header
export -f test_auth_tokens

echo "Test configuration loaded successfully"
echo "Primary Bearer Token: $TEST_BEARER_TOKEN"
echo "Primary API Key: $TEST_API_KEY"
echo "Service URLs configured"
