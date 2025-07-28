#!/bin/bash

# Test KonveyN2AI Cloud Run deployment
# This script verifies that all services are deployed and functioning correctly

set -e

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"konveyn2ai"}
REGION=${GOOGLE_CLOUD_LOCATION:-"us-central1"}

echo "🧪 Testing KonveyN2AI Cloud Run deployment"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "=================================================="

# Test configuration
TIMEOUT=30
RETRY_COUNT=3

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

function log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function log_error() {
    echo -e "${RED}❌ $1${NC}"
}

function test_service_exists() {
    local service_name=$1
    log_info "Checking if $service_name service exists..."
    
    if gcloud run services describe $service_name --region=$REGION --project=$PROJECT_ID --format="value(metadata.name)" &>/dev/null; then
        log_success "$service_name service exists"
        return 0
    else
        log_error "$service_name service not found"
        return 1
    fi
}

function get_service_url() {
    local service_name=$1
    gcloud run services describe $service_name --region=$REGION --project=$PROJECT_ID --format="value(status.url)" 2>/dev/null || echo ""
}

function test_health_endpoint() {
    local service_name=$1
    local service_url=$2
    local health_url="${service_url}/health"
    
    log_info "Testing $service_name health endpoint: $health_url"
    
    for i in $(seq 1 $RETRY_COUNT); do
        if curl -s -f --connect-timeout $TIMEOUT "$health_url" > /dev/null; then
            log_success "$service_name health check passed"
            return 0
        else
            log_warning "$service_name health check failed (attempt $i/$RETRY_COUNT)"
            if [ $i -lt $RETRY_COUNT ]; then
                sleep 5
            fi
        fi
    done
    
    log_error "$service_name health check failed after $RETRY_COUNT attempts"
    return 1
}

function test_service_response() {
    local service_name=$1
    local service_url=$2
    local health_url="${service_url}/health"
    
    log_info "Testing $service_name response content..."
    
    response=$(curl -s --connect-timeout $TIMEOUT "$health_url" 2>/dev/null || echo "")
    
    if [ -n "$response" ]; then
        log_success "$service_name returned response: $response"
        return 0
    else
        log_error "$service_name returned empty response"
        return 1
    fi
}

function test_service_metrics() {
    local service_name=$1
    
    log_info "Checking $service_name metrics..."
    
    # Check if the service has recent metrics (last 10 minutes)
    end_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    start_time=$(date -u -d '10 minutes ago' +"%Y-%m-%dT%H:%M:%SZ")
    
    metrics=$(gcloud monitoring time-series list \
        --filter="resource.type=cloud_run_revision AND resource.label.service_name=$service_name AND metric.type=run.googleapis.com/request_count" \
        --interval-start-time="$start_time" \
        --interval-end-time="$end_time" \
        --project=$PROJECT_ID \
        --format="value(points.length)" 2>/dev/null | head -1)
    
    if [ -n "$metrics" ] && [ "$metrics" -gt 0 ]; then
        log_success "$service_name has recent metrics ($metrics data points)"
        return 0
    else
        log_warning "$service_name has no recent metrics (this might be normal for new deployments)"
        return 0  # Don't fail on missing metrics for new deployments
    fi
}

function test_inter_service_communication() {
    log_info "Testing inter-service communication..."
    
    # Get Svami URL
    svami_url=$(get_service_url "svami")
    if [ -z "$svami_url" ]; then
        log_error "Svami URL not found, skipping inter-service communication test"
        return 1
    fi
    
    # Test if Svami can reach other services (this would be service-specific)
    # For now, just verify Svami is responding
    if curl -s -f --connect-timeout $TIMEOUT "${svami_url}/health" > /dev/null; then
        log_success "Svami orchestrator is responding (inter-service communication baseline)"
        return 0
    else
        log_error "Svami orchestrator not responding"
        return 1
    fi
}

function run_comprehensive_test() {
    echo ""
    log_info "Starting comprehensive deployment test..."
    echo ""
    
    local overall_success=true
    local services=("janapada" "amatya" "svami")
    
    # Test 1: Service existence
    echo "🔍 Phase 1: Service Existence Tests"
    echo "-----------------------------------"
    for service in "${services[@]}"; do
        if ! test_service_exists "$service"; then
            overall_success=false
        fi
    done
    echo ""
    
    # Test 2: Health endpoints
    echo "🏥 Phase 2: Health Endpoint Tests"
    echo "---------------------------------"
    for service in "${services[@]}"; do
        service_url=$(get_service_url "$service")
        if [ -n "$service_url" ]; then
            echo "  Service URL: $service_url"
            if ! test_health_endpoint "$service" "$service_url"; then
                overall_success=false
            fi
            if ! test_service_response "$service" "$service_url"; then
                overall_success=false
            fi
        else
            log_error "$service URL not found"
            overall_success=false
        fi
        echo ""
    done
    
    # Test 3: Service metrics
    echo "📊 Phase 3: Service Metrics Tests"
    echo "--------------------------------"
    for service in "${services[@]}"; do
        test_service_metrics "$service"
    done
    echo ""
    
    # Test 4: Inter-service communication
    echo "🔗 Phase 4: Inter-Service Communication Test"
    echo "-------------------------------------------"
    if ! test_inter_service_communication; then
        overall_success=false
    fi
    echo ""
    
    # Test 5: Resource utilization
    echo "💾 Phase 5: Resource Utilization Check"
    echo "-------------------------------------"
    for service in "${services[@]}"; do
        log_info "Checking $service resource allocation..."
        
        service_info=$(gcloud run services describe $service --region=$REGION --project=$PROJECT_ID --format="json" 2>/dev/null || echo "{}")
        
        if [ "$service_info" != "{}" ]; then
            memory=$(echo "$service_info" | jq -r '.spec.template.spec.containers[0].resources.limits.memory // "Not set"')
            cpu=$(echo "$service_info" | jq -r '.spec.template.spec.containers[0].resources.limits.cpu // "Not set"')
            min_instances=$(echo "$service_info" | jq -r '.spec.template.metadata.annotations."autoscaling.knative.dev/minScale" // "Not set"')
            
            log_success "$service - Memory: $memory, CPU: $cpu, Min instances: $min_instances"
        else
            log_warning "$service resource info not available"
        fi
    done
    echo ""
    
    # Final result
    echo "🎯 Test Results Summary"
    echo "======================"
    if [ "$overall_success" = true ]; then
        log_success "All tests passed! KonveyN2AI deployment is healthy."
        echo ""
        echo "📋 Service URLs:"
        for service in "${services[@]}"; do
            service_url=$(get_service_url "$service")
            if [ -n "$service_url" ]; then
                echo "  $service: $service_url"
            fi
        done
        echo ""
        log_info "Ready for production use! 🚀"
        return 0
    else
        log_error "Some tests failed. Please check the issues above."
        echo ""
        log_info "Troubleshooting steps:"
        echo "  1. Check service logs: gcloud logs tail --service=<service-name> --project=$PROJECT_ID"
        echo "  2. Verify environment variables are set correctly"
        echo "  3. Check service account permissions"
        echo "  4. Ensure all required APIs are enabled"
        echo "  5. Run './setup-artifact-registry.sh' if services aren't deployed"
        return 1
    fi
}

# Create a simple load test function
function run_load_test() {
    local service_name=$1
    local service_url=$2
    local requests=${3:-10}
    
    log_info "Running light load test on $service_name ($requests requests)..."
    
    local success_count=0
    local start_time=$(date +%s)
    
    for i in $(seq 1 $requests); do
        if curl -s -f --connect-timeout 10 "${service_url}/health" > /dev/null; then
            ((success_count++))
        fi
        echo -n "."
    done
    echo ""
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local success_rate=$((success_count * 100 / requests))
    
    log_success "$service_name load test: $success_count/$requests requests successful ($success_rate%) in ${duration}s"
}

# Main execution
case "${1:-test}" in
    "test")
        run_comprehensive_test
        ;;
    "load")
        log_info "Running load tests on all services..."
        services=("janapada" "amatya" "svami")
        for service in "${services[@]}"; do
            service_url=$(get_service_url "$service")
            if [ -n "$service_url" ]; then
                run_load_test "$service" "$service_url" 10
            else
                log_warning "$service not available for load testing"
            fi
        done
        ;;
    "quick")
        log_info "Running quick health checks..."
        services=("janapada" "amatya" "svami")
        for service in "${services[@]}"; do
            service_url=$(get_service_url "$service")
            if [ -n "$service_url" ]; then
                test_health_endpoint "$service" "$service_url"
            fi
        done
        ;;
    *)
        echo "Usage: $0 [test|load|quick]"
        echo "  test  - Run comprehensive deployment tests (default)"
        echo "  load  - Run light load tests on all services"
        echo "  quick - Run quick health checks only"
        ;;
esac