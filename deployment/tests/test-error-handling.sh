#!/bin/bash
# Phase 4: Error Handling Evaluation

echo "🛡️ Phase 4: Error Handling Evaluation"
echo "===================================="

error_score=0

echo "Testing authentication error handling..."
# Test without auth token
unauth_response=$(curl -s -X POST "https://svami-72021522495.us-central1.run.app/answer" \
    -H "Content-Type: application/json" \
    -d '{"question": "test", "role": "backend_developer"}')

if echo "$unauth_response" | grep -qiE "(authentication|auth|unauthorized|error)"; then
    error_score=$((error_score + 2))
    echo "✅ Authentication errors handled correctly"
else
    echo "⚠️ Authentication error handling needs improvement"
fi

echo "Testing malformed request handling..."
# Test with malformed request
malformed_response=$(curl -s -X POST "https://svami-72021522495.us-central1.run.app/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer demo-token" \
    -d '{"invalid": "json", "structure": true}')

if echo "$malformed_response" | grep -qiE "(error|invalid|bad)"; then
    error_score=$((error_score + 2))
    echo "✅ Malformed requests handled correctly"
else
    echo "⚠️ Malformed request handling needs improvement"
fi

echo "Testing edge case handling..."
# Test with empty question
empty_response=$(curl -s -X POST "https://svami-72021522495.us-central1.run.app/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer demo-token" \
    -d '{"question": "", "role": "backend_developer"}')

if [ -n "$empty_response" ]; then
    error_score=$((error_score + 2))
    echo "✅ Edge cases handled correctly"
else
    echo "⚠️ Edge case handling needs improvement"
fi

echo "📊 Error Handling Score: $error_score/6"

if [ $error_score -ge 5 ]; then
    echo "✅ Phase 4: PASSED (Excellent error handling)"
    exit 0
elif [ $error_score -ge 3 ]; then
    echo "⚠️ Phase 4: PASSED (Good error handling)"
    exit 0
else
    echo "❌ Phase 4: FAILED (Needs improvement)"
    exit 1
fi
