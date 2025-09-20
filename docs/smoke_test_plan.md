# Smoke Test Plan: Pipeline Coverage Validation

## Overview

This document defines the comprehensive smoke test plan for the gap analysis pipeline, ensuring that all system components work correctly together and that rule coverage meets quality standards across all artifact types. The plan follows industry best practices for integration testing and provides measurable success criteria.

## Test Objectives

### Primary Objectives
1. **End-to-End Functionality**: Verify complete pipeline from source ingestion to gap metrics output
2. **Rule Coverage Validation**: Ensure all artifact types and rule categories are properly tested
3. **Integration Verification**: Validate BigQuery integration, confidence calculation, and template rendering
4. **Performance Baseline**: Establish performance benchmarks for pipeline execution
5. **Error Handling**: Verify graceful handling of edge cases and error conditions

### Secondary Objectives
1. **Data Quality Assurance**: Validate output data meets schema and business requirements
2. **Idempotency Verification**: Ensure repeated executions produce consistent results
3. **Security Compliance**: Verify proper handling of sensitive data and access controls
4. **Operational Readiness**: Confirm monitoring, logging, and alerting functionality

---

## Test Environment Setup

### Prerequisites

#### Infrastructure Requirements
```bash
# BigQuery dataset and tables
export GOOGLE_CLOUD_PROJECT_ID="konveyn2ai-test"
export BIGQUERY_DATASET_ID="gap_analysis_test"

# Required tables
bq mk --dataset ${GOOGLE_CLOUD_PROJECT_ID}:${BIGQUERY_DATASET_ID}
bq mk --table ${BIGQUERY_DATASET_ID}.source_metadata
bq mk --table ${BIGQUERY_DATASET_ID}.source_embeddings  
bq mk --table ${BIGQUERY_DATASET_ID}.gap_metrics
```

#### Test Data Requirements
```bash
# Minimum test data volumes per artifact type
KUBERNETES_SAMPLES=50      # Various workload types
FASTAPI_SAMPLES=30         # Different endpoint patterns
COBOL_SAMPLES=20           # Copybooks and procedures
IRS_SAMPLES=15             # Different form layouts
MUMPS_SAMPLES=25           # FileMan and routine samples

# Total minimum test chunks: 140
# Expected test execution time: <10 minutes
```

#### Rule Configuration
```yaml
# Test-specific rule configuration
test_rules:
  enabled_artifact_types: 
    - kubernetes
    - fastapi
    - cobol
    - irs
    - mumps
  confidence_threshold: 0.1  # Low threshold for comprehensive testing
  severity_levels: [1, 2, 3, 4, 5]  # All severity levels
  rule_categories:
    - documentation
    - security
    - compliance
    - validation
```

### Test Data Preparation

#### Synthetic Test Data Generation
```python
# Generate test data with known characteristics
def generate_test_samples():
    """Generate synthetic test samples with predictable outcomes."""
    
    samples = {
        'perfect_quality': {
            'kubernetes': generate_perfect_k8s_samples(10),
            'fastapi': generate_perfect_api_samples(6),
            'cobol': generate_perfect_cobol_samples(4),
            'irs': generate_perfect_irs_samples(3),
            'mumps': generate_perfect_mumps_samples(5)
        },
        'high_quality': {
            # Similar structure for each quality level
        },
        'moderate_quality': {
            # 30% of samples
        },
        'low_quality': {
            # 20% of samples
        },
        'poor_quality': {
            # 10% of samples
        }
    }
    
    return samples
```

#### Real-World Sample Integration
```bash
# Include real samples from project repositories
./test_data/
├── kubernetes/
│   ├── production_deployments/     # Anonymized prod samples
│   ├── development_configs/        # Dev environment samples
│   └── edge_cases/                 # Known problematic configs
├── fastapi/
│   ├── well_documented_apis/       # High-quality examples
│   ├── legacy_endpoints/           # Older code samples
│   └── minimal_examples/           # Basic implementations
└── [similar structure for other artifact types]
```

---

## Test Suite Structure

### Tier 1: Component Smoke Tests
**Execution Time**: 2-3 minutes  
**Purpose**: Verify individual components function correctly

#### Rule Engine Validation
```python
def test_rule_engine_smoke():
    """Smoke test for rule evaluation engine."""
    
    test_cases = [
        {
            'name': 'kubernetes_deployment_documentation',
            'artifact_type': 'kubernetes',
            'input_metadata': sample_k8s_metadata,
            'expected_pass': True,
            'expected_confidence_range': (0.8, 1.0)
        },
        {
            'name': 'fastapi_endpoint_missing_docs',
            'artifact_type': 'fastapi', 
            'input_metadata': minimal_fastapi_metadata,
            'expected_pass': False,
            'expected_confidence_range': (0.2, 0.5)
        }
        # Additional test cases for each rule
    ]
    
    for test_case in test_cases:
        result = evaluate_rule(test_case['name'], test_case['input_metadata'])
        
        # Validate basic functionality
        assert result.passed == test_case['expected_pass']
        assert test_case['expected_confidence_range'][0] <= result.confidence <= test_case['expected_confidence_range'][1]
        assert result.severity in [1, 2, 3, 4, 5]
        assert len(result.suggested_fix) > 0
```

#### Confidence Calculator Smoke Test
```python
def test_confidence_calculator_smoke():
    """Verify confidence calculation produces expected ranges."""
    
    calculator = ConfidenceCalculator()
    
    # Test with golden sample data
    perfect_field_analysis = create_perfect_field_analysis()
    perfect_content_quality = create_perfect_content_quality()
    no_penalties = create_empty_penalty_assessment()
    
    result = calculator.calculate_confidence(
        perfect_field_analysis, 
        perfect_content_quality, 
        no_penalties,
        "kubernetes"
    )
    
    # Perfect inputs should yield high confidence
    assert 0.9 <= result.final_confidence <= 1.0
    assert result.component_scores[ConfidenceComponent.FIELD_COMPLETENESS] > 0.9
    assert result.total_penalties == 0.0
```

#### Template Rendering Smoke Test
```python
def test_template_rendering_smoke():
    """Verify template rendering produces valid output."""
    
    test_contexts = [
        {
            'template': 'kubernetes_deployment_documentation',
            'context': {
                'missing_description': True,
                'missing_app_label': False,
                'missing_resources': True
            },
            'expected_output_pattern': r'Add missing documentation:.*description.*resource limits'
        }
    ]
    
    for test_context in test_contexts:
        rendered = render_template(
            test_context['template'], 
            test_context['context']
        )
        
        assert len(rendered) > 0
        assert len(rendered) <= 500  # Character limit
        assert re.match(test_context['expected_output_pattern'], rendered, re.IGNORECASE)
```

### Tier 2: Integration Smoke Tests
**Execution Time**: 4-5 minutes  
**Purpose**: Verify component integration and data flow

#### BigQuery Integration Test
```sql
-- Test BigQuery integration with minimal dataset
WITH test_metadata AS (
  SELECT 
    'test_chunk_001' as chunk_id,
    'kubernetes' as artifact_type,
    JSON_OBJECT(
      'kind', 'Deployment',
      'metadata', JSON_OBJECT(
        'name', 'test-app',
        'labels', JSON_OBJECT('app', 'test-app'),
        'annotations', JSON_OBJECT('description', 'Test deployment for smoke test')
      )
    ) as metadata,
    CURRENT_TIMESTAMP() as created_at
)

-- Execute sample rule evaluation
SELECT
  chunk_id,
  artifact_type,
  'kubernetes_deployment_documentation' as rule_name,
  CASE 
    WHEN JSON_EXTRACT_SCALAR(metadata, '$.metadata.annotations.description') IS NOT NULL
    THEN true 
    ELSE false 
  END as passed,
  0.85 as confidence,
  2 as severity,
  'Test evaluation successful' as suggested_fix,
  'smoke_test_001' as analysis_id,
  CURRENT_TIMESTAMP() as created_at,
  'v1.0.0' as ruleset_version
FROM test_metadata;
```

#### End-to-End Pipeline Test
```python
def test_end_to_end_pipeline_smoke():
    """Test complete pipeline from input to output."""
    
    # Step 1: Prepare test input
    test_chunks = generate_minimal_test_chunks()
    
    # Step 2: Execute pipeline
    analysis_id = f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    pipeline_result = execute_gap_analysis_pipeline(
        chunks=test_chunks,
        analysis_id=analysis_id,
        ruleset_version="v1.0.0"
    )
    
    # Step 3: Validate output
    assert pipeline_result.status == "SUCCESS"
    assert pipeline_result.processed_chunks == len(test_chunks)
    assert pipeline_result.total_evaluations > 0
    assert pipeline_result.execution_time < 300  # 5 minutes max
    
    # Step 4: Verify BigQuery results
    results = query_gap_metrics(analysis_id=analysis_id)
    
    assert len(results) > 0
    assert all(r.analysis_id == analysis_id for r in results)
    assert all(0 <= r.confidence <= 1 for r in results)
    assert all(1 <= r.severity <= 5 for r in results)
```

### Tier 3: Coverage Validation Tests
**Execution Time**: 3-4 minutes  
**Purpose**: Ensure comprehensive rule coverage and quality distribution

#### Artifact Type Coverage Test
```python
def test_artifact_type_coverage():
    """Verify all artifact types are processed."""
    
    required_artifact_types = ['kubernetes', 'fastapi', 'cobol', 'irs', 'mumps']
    test_data = load_comprehensive_test_dataset()
    
    results = execute_gap_analysis_pipeline(test_data)
    
    # Verify each artifact type produced results
    coverage_by_type = group_results_by_artifact_type(results)
    
    for artifact_type in required_artifact_types:
        assert artifact_type in coverage_by_type
        assert len(coverage_by_type[artifact_type]) > 0
        
        # Verify rule diversity within artifact type
        rules_tested = set(r.rule_name for r in coverage_by_type[artifact_type])
        assert len(rules_tested) >= 3  # Minimum rule coverage per type
```

#### Rule Category Coverage Test
```python
def test_rule_category_coverage():
    """Verify all rule categories are tested."""
    
    required_categories = ['documentation', 'security', 'compliance', 'validation']
    results = execute_gap_analysis_pipeline(load_comprehensive_test_dataset())
    
    # Group results by rule category (inferred from rule names)
    coverage_by_category = categorize_rules(results)
    
    for category in required_categories:
        assert category in coverage_by_category
        assert len(coverage_by_category[category]) > 0
        
        # Verify severity distribution within category
        severities = [r.severity for r in coverage_by_category[category]]
        if category in ['security', 'compliance']:
            assert max(severities) >= 4  # Should have high-severity rules
```

#### Confidence Score Distribution Test
```python
def test_confidence_distribution():
    """Verify confidence scores show expected distribution."""
    
    results = execute_gap_analysis_pipeline(load_stratified_test_dataset())
    confidence_scores = [r.confidence for r in results]
    
    # Verify reasonable distribution
    assert min(confidence_scores) >= 0.0
    assert max(confidence_scores) <= 1.0
    assert len(confidence_scores) > 50  # Sufficient sample size
    
    # Verify distribution characteristics
    mean_confidence = statistics.mean(confidence_scores)
    std_confidence = statistics.stdev(confidence_scores)
    
    assert 0.3 <= mean_confidence <= 0.8  # Reasonable average
    assert 0.1 <= std_confidence <= 0.4   # Reasonable variance
    
    # Verify all confidence ranges represented
    confidence_ranges = {
        'low': len([c for c in confidence_scores if c < 0.3]),
        'medium': len([c for c in confidence_scores if 0.3 <= c < 0.7]),
        'high': len([c for c in confidence_scores if c >= 0.7])
    }
    
    assert confidence_ranges['low'] > 0
    assert confidence_ranges['medium'] > 0
    assert confidence_ranges['high'] > 0
```

---

## Performance Benchmarks

### Execution Time Targets

#### Per-Rule Performance
```python
PERFORMANCE_TARGETS = {
    'rule_evaluation': {
        'max_time_ms': 50,      # Individual rule evaluation
        'target_time_ms': 25,   # Target performance
        'timeout_ms': 200       # Hard timeout
    },
    'confidence_calculation': {
        'max_time_ms': 10,      # Confidence score calculation
        'target_time_ms': 5,    # Target performance
    },
    'template_rendering': {
        'max_time_ms': 5,       # Template rendering
        'target_time_ms': 2,    # Target performance
    }
}
```

#### Pipeline Performance
```python
PIPELINE_PERFORMANCE = {
    'small_dataset': {        # 100 chunks
        'max_time_minutes': 2,
        'target_time_minutes': 1
    },
    'medium_dataset': {       # 1000 chunks
        'max_time_minutes': 10,
        'target_time_minutes': 6
    },
    'large_dataset': {        # 10000 chunks
        'max_time_minutes': 60,
        'target_time_minutes': 40
    }
}
```

### Resource Usage Targets
```python
RESOURCE_LIMITS = {
    'memory_usage_mb': 512,     # Maximum memory per process
    'cpu_usage_percent': 80,    # Maximum CPU utilization
    'bigquery_slots': 100,      # BigQuery slot allocation
    'disk_usage_gb': 2,         # Temporary disk usage
}
```

---

## Error Handling Validation

### Edge Case Testing

#### Malformed Input Handling
```python
def test_malformed_input_handling():
    """Verify graceful handling of malformed input data."""
    
    edge_cases = [
        {
            'name': 'empty_metadata',
            'input': {'chunk_id': 'test_001', 'metadata': {}},
            'expected_behavior': 'fail_gracefully'
        },
        {
            'name': 'invalid_json_metadata', 
            'input': {'chunk_id': 'test_002', 'metadata': 'invalid_json'},
            'expected_behavior': 'skip_with_warning'
        },
        {
            'name': 'missing_chunk_id',
            'input': {'metadata': {'valid': 'json'}},
            'expected_behavior': 'validation_error'
        },
        {
            'name': 'extremely_large_metadata',
            'input': {'chunk_id': 'test_003', 'metadata': generate_large_metadata(10000)},
            'expected_behavior': 'process_or_truncate'
        }
    ]
    
    for case in edge_cases:
        try:
            result = process_chunk(case['input'])
            
            if case['expected_behavior'] == 'fail_gracefully':
                assert result.status == 'FAILED'
                assert result.error_message is not None
            elif case['expected_behavior'] == 'skip_with_warning':
                assert result.status == 'SKIPPED'
                assert 'warning' in result.messages
                
        except ValidationError as e:
            if case['expected_behavior'] == 'validation_error':
                assert True  # Expected
            else:
                raise
```

#### Database Connection Failures
```python
def test_database_failure_handling():
    """Test behavior during BigQuery connection issues."""
    
    # Simulate connection failure
    with mock_bigquery_failure():
        try:
            result = execute_gap_analysis_pipeline(minimal_test_data)
            assert False, "Should have raised connection error"
        except DatabaseConnectionError as e:
            assert "Unable to connect" in str(e)
            assert e.recovery_suggestions is not None
```

#### Resource Exhaustion
```python
def test_resource_exhaustion_handling():
    """Test behavior under resource constraints."""
    
    # Test with oversized dataset
    large_dataset = generate_test_chunks(count=50000)
    
    with resource_limits(memory_mb=256, timeout_seconds=300):
        result = execute_gap_analysis_pipeline(large_dataset)
        
        # Should complete or fail gracefully
        assert result.status in ['SUCCESS', 'PARTIAL_SUCCESS', 'FAILED']
        
        if result.status == 'PARTIAL_SUCCESS':
            assert result.processed_chunks > 0
            assert result.error_chunks > 0
            assert result.total_chunks == len(large_dataset)
```

---

## Data Quality Validation

### Output Schema Compliance
```python
def test_output_schema_compliance():
    """Verify all output data meets schema requirements."""
    
    results = execute_gap_analysis_pipeline(comprehensive_test_data)
    
    for result in results:
        # Required fields
        assert result.chunk_id is not None and len(result.chunk_id) > 0
        assert result.artifact_type in ['kubernetes', 'fastapi', 'cobol', 'irs', 'mumps']
        assert result.rule_name is not None and len(result.rule_name) > 0
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert 1 <= result.severity <= 5
        assert result.suggested_fix is not None
        assert len(result.suggested_fix) <= 500
        assert result.analysis_id is not None
        assert result.created_at is not None
        assert result.ruleset_version is not None
        
        # Data type validation
        assert isinstance(result.confidence, float)
        assert isinstance(result.severity, int)
        assert isinstance(result.suggested_fix, str)
```

### Business Rule Validation
```python
def test_business_rule_compliance():
    """Verify output complies with business rules."""
    
    results = execute_gap_analysis_pipeline(comprehensive_test_data)
    
    # Rule: High severity should correlate with low confidence
    high_severity_results = [r for r in results if r.severity >= 4]
    for result in high_severity_results:
        if result.confidence > 0.8:
            warnings.warn(f"High severity ({result.severity}) with high confidence ({result.confidence}) for rule {result.rule_name}")
    
    # Rule: Security rules should have appropriate severity
    security_results = [r for r in results if 'security' in r.rule_name.lower()]
    for result in security_results:
        assert result.severity >= 3, f"Security rule {result.rule_name} has low severity {result.severity}"
    
    # Rule: All results should have actionable suggestions
    for result in results:
        assert not result.suggested_fix.isspace(), f"Empty suggestion for rule {result.rule_name}"
        assert len(result.suggested_fix.strip()) >= 10, f"Too short suggestion for rule {result.rule_name}"
```

---

## Idempotency Verification

### Reproducibility Testing
```python
def test_pipeline_idempotency():
    """Verify repeated executions produce identical results."""
    
    test_data = load_fixed_test_dataset()
    
    # Execute pipeline twice with identical inputs
    result1 = execute_gap_analysis_pipeline(
        chunks=test_data,
        analysis_id="idempotency_test_001",
        ruleset_version="v1.0.0"
    )
    
    result2 = execute_gap_analysis_pipeline(
        chunks=test_data,
        analysis_id="idempotency_test_002", 
        ruleset_version="v1.0.0"
    )
    
    # Compare results (excluding analysis_id and timestamps)
    assert result1.processed_chunks == result2.processed_chunks
    assert result1.total_evaluations == result2.total_evaluations
    
    # Compare individual evaluation results
    results1_normalized = normalize_results_for_comparison(result1.evaluation_results)
    results2_normalized = normalize_results_for_comparison(result2.evaluation_results)
    
    assert len(results1_normalized) == len(results2_normalized)
    
    for r1, r2 in zip(results1_normalized, results2_normalized):
        assert r1.chunk_id == r2.chunk_id
        assert r1.rule_name == r2.rule_name
        assert r1.passed == r2.passed
        assert abs(r1.confidence - r2.confidence) < 0.001  # Float precision tolerance
        assert r1.severity == r2.severity
        assert r1.suggested_fix == r2.suggested_fix
```

### Version Consistency Testing
```python
def test_version_consistency():
    """Verify different rule versions produce separate result sets."""
    
    test_data = load_fixed_test_dataset()
    
    # Execute with version 1.0.0
    result_v1 = execute_gap_analysis_pipeline(
        chunks=test_data,
        ruleset_version="v1.0.0"
    )
    
    # Execute with version 1.1.0 (with modified rules)
    result_v2 = execute_gap_analysis_pipeline(
        chunks=test_data,
        ruleset_version="v1.1.0"
    )
    
    # Verify both result sets exist in database
    v1_count = count_results_for_version("v1.0.0")
    v2_count = count_results_for_version("v1.1.0")
    
    assert v1_count > 0
    assert v2_count > 0
    assert v1_count + v2_count == query_total_result_count()
```

---

## Success Criteria

### Functional Requirements
- [ ] **All test tiers pass**: Tier 1, 2, and 3 tests complete successfully
- [ ] **Coverage targets met**: All artifact types and rule categories tested
- [ ] **Performance benchmarks**: Execution times within target ranges
- [ ] **Error handling**: Graceful handling of all edge cases
- [ ] **Data quality**: Output meets schema and business rule requirements
- [ ] **Idempotency**: Reproducible results across multiple executions

### Quality Metrics
```python
SUCCESS_CRITERIA = {
    'test_pass_rate': 100,           # All smoke tests must pass
    'artifact_coverage': 100,        # All 5 artifact types covered
    'rule_coverage': 90,             # 90% of rules tested
    'performance_compliance': 95,    # 95% of operations within target time
    'error_handling_coverage': 100,  # All error scenarios tested
    'data_quality_score': 95,       # 95% of outputs meet quality standards
}
```

### Operational Readiness
- [ ] **Monitoring integration**: All pipeline metrics collected
- [ ] **Logging completeness**: Adequate logging for troubleshooting  
- [ ] **Alerting configuration**: Alerts configured for failure scenarios
- [ ] **Documentation currency**: All test documentation up to date
- [ ] **Team training**: Team members can execute and interpret tests

---

## Continuous Integration

### Automated Execution
```yaml
# CI/CD pipeline integration
smoke_test_pipeline:
  trigger:
    - merge_to_main
    - rule_configuration_changes
    - schema_modifications
  
  stages:
    - setup_test_environment
    - execute_tier_1_tests
    - execute_tier_2_tests  
    - execute_tier_3_tests
    - validate_success_criteria
    - cleanup_test_environment
  
  success_criteria:
    - all_tests_pass: true
    - performance_regression: false
    - coverage_maintained: true
```

### Regression Detection
```python
def detect_performance_regression():
    """Compare current performance against baseline."""
    
    current_metrics = execute_performance_benchmark()
    baseline_metrics = load_baseline_performance_metrics()
    
    regressions = []
    
    for metric_name, current_value in current_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)
        if baseline_value:
            regression_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            if regression_percent > REGRESSION_THRESHOLD:
                regressions.append({
                    'metric': metric_name,
                    'current': current_value,
                    'baseline': baseline_value,
                    'regression_percent': regression_percent
                })
    
    if regressions:
        raise PerformanceRegressionError(regressions)
```

### Quality Gates
```python
def validate_quality_gates():
    """Enforce quality gates before deployment."""
    
    smoke_test_results = execute_comprehensive_smoke_tests()
    
    quality_checks = [
        ('Test Pass Rate', smoke_test_results.pass_rate, 100),
        ('Coverage Score', smoke_test_results.coverage_score, 90),
        ('Performance Score', smoke_test_results.performance_score, 95),
        ('Data Quality Score', smoke_test_results.data_quality_score, 95)
    ]
    
    failures = []
    for check_name, actual_value, required_value in quality_checks:
        if actual_value < required_value:
            failures.append(f"{check_name}: {actual_value}% < {required_value}%")
    
    if failures:
        raise QualityGateFailure(failures)
    
    return True  # All quality gates passed
```

This comprehensive smoke test plan ensures thorough validation of the gap analysis pipeline while maintaining fast execution times suitable for continuous integration workflows.