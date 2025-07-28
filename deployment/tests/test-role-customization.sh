#!/bin/bash
# Phase 3: Role-Based Customization Evaluation with Secure Configuration

# Load secure test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test-config.sh"

echo "👥 Phase 3: Role-Based Customization Evaluation"
echo "=============================================="

# Test same question across different roles
test_question="How do I implement authentication in this system?"

echo "Testing role differentiation with authentication question..."
echo "Using secure authentication token..."

auth_header=$(create_auth_header "bearer")

# Backend Developer Response
backend_response=$(curl -s -X POST "$SVAMI_URL/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: $auth_header" \
    -d "{\"question\": \"$test_question\", \"role\": \"backend_developer\"}")

# Security Engineer Response  
security_response=$(curl -s -X POST "$SVAMI_URL/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: $auth_header" \
    -d "{\"question\": \"$test_question\", \"role\": \"security_engineer\"}")

if echo "$backend_response" | jq -e '.answer' > /dev/null 2>&1 && echo "$security_response" | jq -e '.answer' > /dev/null 2>&1; then
    backend_answer=$(echo "$backend_response" | jq -r '.answer')
    security_answer=$(echo "$security_response" | jq -r '.answer')
    
    role_score=0
    
    # Check for backend-specific terms
    if echo "$backend_answer" | grep -qiE "(api|endpoint|middleware|framework)"; then
        role_score=$((role_score + 2))
    fi
    
    # Check for security-specific terms
    if echo "$security_answer" | grep -qiE "(security|secure|threat|vulnerability)"; then
        role_score=$((role_score + 2))
    fi
    
    # Check for different perspectives (responses should be different)
    if [ "$backend_answer" != "$security_answer" ]; then
        role_score=$((role_score + 2))
    fi
    
    echo "📊 Role-Based Customization Score: $role_score/6"
    
    if [ $role_score -ge 5 ]; then
        echo "✅ Phase 3: PASSED (Good role differentiation)"
        exit 0
    elif [ $role_score -ge 3 ]; then
        echo "⚠️ Phase 3: PASSED (Basic role differentiation)"
        exit 0
    else
        echo "❌ Phase 3: FAILED (Needs improvement)"
        exit 1
    fi
else
    echo "❌ Phase 3: FAILED (No response received)"
    exit 1
fi
