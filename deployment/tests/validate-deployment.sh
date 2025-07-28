#!/bin/bash
# Validates deployment health and service availability

set -e

# Load secure test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test-config.sh"

# Service URLs from secure configuration (no hard-coded fallbacks)
SERVICES=(
    "$SVAMI_URL"
    "$JANAPADA_URL"
    "$AMATYA_URL"
)

echo "🔍 Validating deployment health..."

for service in "${SERVICES[@]}"; do
    echo "Testing $service/health"
    if curl -f -s "$service/health" > /dev/null; then
        echo "✅ $service is healthy"
    else
        echo "❌ $service health check failed"
        exit 1
    fi
done

echo "🎉 All services are healthy and ready!"
