#!/bin/bash
# Phase 5: Performance Evaluation

echo "⚡ Phase 5: Performance Evaluation"
echo "================================"

total_time=0
test_count=5
fast_responses=0

echo "Measuring response times across $test_count requests..."

for i in $(seq 1 $test_count); do
    start_time=$(date +%s.%N)
    curl -s -X POST "${SVAMI_URL:-https://svami-72021522495.us-central1.run.app}/answer" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer demo-token" \
        -d '{"question": "What is JWT authentication?", "role": "backend_developer"}' > /dev/null
    end_time=$(date +%s.%N)
    response_time=$(echo "$end_time - $start_time" | bc -l)
    total_time=$(echo "$total_time + $response_time" | bc -l)
    
    echo "Request $i: ${response_time}s"
    
    # Count fast responses (<3 seconds)
    if (( $(echo "$response_time < 3.0" | bc -l) )); then
        fast_responses=$((fast_responses + 1))
    fi
done

avg_time=$(echo "scale=2; $total_time / $test_count" | bc -l)

# Score based on average response time
performance_score=0
if (( $(echo "$avg_time < 2.0" | bc -l) )); then
    performance_score=10  # Excellent (<2s)
elif (( $(echo "$avg_time < 3.0" | bc -l) )); then
    performance_score=8   # Good (<3s)
elif (( $(echo "$avg_time < 5.0" | bc -l) )); then
    performance_score=6   # Acceptable (<5s)
else
    performance_score=4   # Slow (>5s)
fi

echo "📊 Average response time: ${avg_time}s"
echo "📊 Fast responses (<3s): $fast_responses/$test_count"
echo "📊 Performance Score: $performance_score/10"

if [ $performance_score -ge 8 ]; then
    echo "✅ Phase 5: PASSED (Excellent performance)"
    exit 0
elif [ $performance_score -ge 6 ]; then
    echo "⚠️ Phase 5: PASSED (Good performance)"
    exit 0
else
    echo "❌ Phase 5: FAILED (Needs improvement)"
    exit 1
fi
