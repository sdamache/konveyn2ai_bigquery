#!/bin/bash
# Phase 1: Response Relevance Evaluation

source deployment/tests/test-real-integration.sh

echo "🎯 Phase 1: Response Relevance Evaluation"
echo "========================================"

relevance_score=$(evaluate_response_relevance)
echo "📊 Response Relevance Score: $relevance_score/10"

if [ $relevance_score -ge 8 ]; then
    echo "✅ Phase 1: PASSED (Excellent relevance)"
    exit 0
elif [ $relevance_score -ge 6 ]; then
    echo "⚠️ Phase 1: PASSED (Good relevance)"
    exit 0
else
    echo "❌ Phase 1: FAILED (Needs improvement)"
    exit 1
fi
