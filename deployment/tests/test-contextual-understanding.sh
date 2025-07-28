#!/bin/bash
# Phase 2: Contextual Understanding Evaluation

echo "🧠 Phase 2: Contextual Understanding Evaluation"
echo "=============================================="

# Test contextual understanding with complex technical question
echo "Testing contextual understanding with complex JWT security question..."

response=$(curl -s -X POST "https://svami-72021522495.us-central1.run.app/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer demo-token" \
    -d '{"question": "How do I implement secure JWT authentication with refresh tokens and role-based access control?", "role": "security_engineer"}')

if echo "$response" | jq -e '.answer' > /dev/null 2>&1; then
    answer=$(echo "$response" | jq -r '.answer')
    
    # Check for contextual indicators
    context_score=0
    if echo "$answer" | grep -qiE "(jwt|token|authentication|security)"; then
        context_score=$((context_score + 2))
    fi
    if echo "$answer" | grep -qiE "(refresh|role|access.control)"; then
        context_score=$((context_score + 2))
    fi
    if echo "$answer" | grep -qiE "(implement|configure|secure)"; then
        context_score=$((context_score + 2))
    fi
    if [ ${#answer} -gt 200 ]; then
        context_score=$((context_score + 2))
    fi
    
    echo "📊 Contextual Understanding Score: $context_score/8"
    
    if [ $context_score -ge 7 ]; then
        echo "✅ Phase 2: PASSED (Excellent contextual understanding)"
        exit 0
    elif [ $context_score -ge 5 ]; then
        echo "⚠️ Phase 2: PASSED (Good contextual understanding)"
        exit 0
    else
        echo "❌ Phase 2: FAILED (Needs improvement)"
        exit 1
    fi
else
    echo "❌ Phase 2: FAILED (No response received)"
    exit 1
fi
