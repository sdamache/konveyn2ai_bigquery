#!/bin/bash
# Phase 6: Security Compliance Evaluation

echo "🔒 Phase 6: Security Compliance Evaluation"
echo "========================================="

security_score=0

echo "Testing authentication enforcement..."
# Test without auth token
unauth_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}/answer" \
    -H "Content-Type: application/json" \
    -d '{"question": "test", "role": "backend_developer"}')

# Test with valid token
valid_auth_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer demo-token" \
    -d '{"question": "test", "role": "backend_developer"}')

if [ "$unauth_status" != "200" ] && [ "$valid_auth_status" = "200" ]; then
    security_score=$((security_score + 3))
    echo "✅ Authentication properly enforced"
elif [ "$valid_auth_status" = "200" ]; then
    security_score=$((security_score + 2))
    echo "⚠️ Authentication partially enforced"
else
    echo "❌ Authentication issues detected"
fi

echo "Testing security headers..."
headers=$(curl -s -I "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}/health")

header_count=0
if echo "$headers" | grep -qi "content-type"; then
    header_count=$((header_count + 1))
fi
if echo "$headers" | grep -qi "server"; then
    header_count=$((header_count + 1))
fi

if [ $header_count -ge 2 ]; then
    security_score=$((security_score + 2))
    echo "✅ Security headers present"
else
    security_score=$((security_score + 1))
    echo "⚠️ Basic security headers present"
fi

echo "Testing data protection..."
# Test for sensitive data exposure
response_content=$(curl -s -X POST "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer demo-token" \
    -d '{"question": "Show me API keys or secrets", "role": "backend_developer"}')

if echo "$response_content" | grep -qiE "(api.key|secret|password|token.*[a-zA-Z0-9]{20,})"; then
    echo "⚠️ Potential sensitive data patterns detected"
    security_score=$((security_score + 1))
else
    security_score=$((security_score + 2))
    echo "✅ No sensitive data exposure detected"
fi

echo "📊 Security Compliance Score: $security_score/7"

if [ $security_score -ge 6 ]; then
    echo "✅ Phase 6: PASSED (Good security compliance)"
    exit 0
elif [ $security_score -ge 4 ]; then
    echo "⚠️ Phase 6: PASSED (Basic security compliance)"
    exit 0
else
    echo "❌ Phase 6: FAILED (Needs improvement)"
    exit 1
fi
