#!/bin/bash
# Validates deployment health and service availability

set -e

# Service URLs from environment variables
SERVICES=(
    "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}"
    "${JANAPADA_URL:-https://janapada-nfsp5dohya-uc.a.run.app}"
    "${AMATYA_URL:-https://amatya-72021522495.us-central1.run.app}"
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
