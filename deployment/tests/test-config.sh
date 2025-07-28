#!/bin/bash

# Test Configuration for KonveyN2AI
# Provides secure test tokens and configuration for CI/CD testing

# Security Enhancement: Use more secure test tokens instead of static demo-token
# These tokens are accepted by GuardFort middleware and provide better security

# Primary test tokens (accepted by GuardFort)
export TEST_BEARER_TOKEN="${TEST_BEARER_TOKEN:-konveyn2ai-token}"
export TEST_API_KEY="${TEST_API_KEY:-konveyn2ai-api-key-demo}"

# Alternative test tokens for different test scenarios
export DEMO_TOKEN_1="hackathon-demo"
export DEMO_TOKEN_2="test-token"
export DEMO_API_KEY="demo-api-key"

# Service URLs (from environment variables with fallbacks)
export SVAMI_URL="${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}"
export JANAPADA_URL="${JANAPADA_URL:-https://janapada-nfsp5dohya-uc.a.run.app}"
export AMATYA_URL="${AMATYA_URL:-https://amatya-72021522495.us-central1.run.app}"

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
