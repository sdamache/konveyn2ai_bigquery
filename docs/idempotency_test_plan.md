# Idempotency Test Plan: Verification Methods for Deterministic Pipeline Execution

## Overview

This document defines comprehensive test procedures for validating the idempotency characteristics of the gap analysis pipeline. The plan ensures that repeated executions with identical inputs produce consistent, reproducible results while properly handling version changes, data updates, and system state transitions.

## Idempotency Requirements

### Core Idempotency Principles
1. **Deterministic Results**: Same inputs always produce same outputs
2. **Safe Re-execution**: Multiple runs don't corrupt existing data
3. **Atomic Operations**: All-or-nothing execution semantics
4. **State Consistency**: System state remains valid after any execution
5. **Audit Trail Integrity**: Complete history of operations maintained

### Conceptual Key Validation
```sql
-- Primary idempotency constraint
CONCEPTUAL_KEY = (chunk_id, rule_name, ruleset_version)

-- Business rules enforced by idempotency
- One result per unique (chunk_id, rule_name, ruleset_version) combination
- Version changes create new result sets, don't overwrite history
- Failed executions don't leave partial results
- Concurrent executions are properly serialized
```

---

## Test Environment Setup

### Infrastructure Configuration

#### Test Database Setup
```sql
-- Create isolated test environment
CREATE SCHEMA IF NOT EXISTS gap_analysis_idempotency_test;

-- Test tables with same structure as production
CREATE TABLE gap_analysis_idempotency_test.source_metadata (
  chunk_id STRING NOT NULL,
  artifact_type STRING NOT NULL,
  source_path STRING,
  content_hash STRING,
  metadata JSON,
  created_at TIMESTAMP,
  modified_at TIMESTAMP
)
PARTITION BY DATE(created_at)
CLUSTER BY artifact_type, chunk_id;

CREATE TABLE gap_analysis_idempotency_test.gap_metrics (
  chunk_id STRING NOT NULL,
  artifact_type STRING NOT NULL,
  rule_name STRING NOT NULL,
  passed BOOLEAN NOT NULL,
  confidence FLOAT64 NOT NULL,
  severity INT64 NOT NULL,
  suggested_fix STRING,
  analysis_id STRING NOT NULL,
  created_at TIMESTAMP NOT NULL,
  ruleset_version STRING NOT NULL,
  partition_date DATE NOT NULL
)
PARTITION BY partition_date
CLUSTER BY analysis_id, rule_name, chunk_id;

-- Execution locks table for concurrency testing
CREATE TABLE gap_analysis_idempotency_test.execution_locks (
  lock_name STRING,
  analysis_id STRING,
  locked_at TIMESTAMP,
  locked_by STRING,
  expires_at TIMESTAMP
);
```

#### Test Data Fixtures
```python
class IdempotencyTestFixtures:
    """Standardized test data for idempotency validation."""
    
    def __init__(self):
        self.fixed_timestamp = datetime(2023, 1, 15, 10, 30, 0)
        self.fixed_uuid_seed = "idempotency_test_seed"
        
    def create_deterministic_chunks(self, count: int = 10) -> List[Dict]:
        """Create chunks with deterministic content for reproducible tests."""
        
        chunks = []
        for i in range(count):
            chunk = {
                'chunk_id': f'test_chunk_{i:03d}',
                'artifact_type': ['kubernetes', 'fastapi', 'cobol'][i % 3],
                'source_path': f'/test/path/file_{i:03d}.yaml',
                'content_hash': hashlib.md5(f'test_content_{i}'.encode()).hexdigest(),
                'metadata': self._create_deterministic_metadata(i),
                'created_at': self.fixed_timestamp,
                'modified_at': self.fixed_timestamp
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_deterministic_metadata(self, index: int) -> Dict:
        """Create metadata with known characteristics."""
        
        if index % 3 == 0:  # kubernetes
            return {
                'kind': 'Deployment',
                'metadata': {
                    'name': f'test-app-{index}',
                    'labels': {'app': f'test-app-{index}'}
                }
            }
        elif index % 3 == 1:  # fastapi
            return {
                'function_name': f'get_user_{index}',
                'docstring': f'Get user {index} information'
            }
        else:  # cobol
            return {
                'copybook_name': f'TEST-COPY-{index:03d}',
                'fields': [{'name': f'FIELD-{index}', 'type': 'PIC X(10)'}]
            }
```

### Rule Configuration Stability
```yaml
# Fixed rule configuration for idempotency testing
test_rules_v1_0_0:
  version: "v1.0.0"
  rules:
    kubernetes_deployment_documentation:
      severity: 3
      confidence_weights:
        field_completeness: 0.6
        content_quality: 0.4
    
    fastapi_endpoint_documentation:
      severity: 2
      confidence_weights:
        field_completeness: 0.7
        content_quality: 0.3

# Modified rules for version testing  
test_rules_v1_1_0:
  version: "v1.1.0"
  rules:
    kubernetes_deployment_documentation:
      severity: 4  # Severity increased
      confidence_weights:
        field_completeness: 0.6
        content_quality: 0.4
    
    fastapi_endpoint_documentation:
      severity: 2
      confidence_weights:
        field_completeness: 0.65  # Weights adjusted
        content_quality: 0.35
```

---

## Test Suite Categories

### Category 1: Basic Idempotency Validation
**Purpose**: Verify core idempotency principles with identical inputs

#### Test 1.1: Identical Input Reproducibility
```python
def test_identical_input_reproducibility():
    """Verify identical inputs produce identical outputs."""
    
    # Setup
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(5)
    ruleset_version = "v1.0.0"
    
    # Execute first analysis
    analysis_id_1 = "idempotency_test_001"
    result_1 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_1,
        ruleset_version=ruleset_version
    )
    
    # Execute second analysis with identical inputs
    analysis_id_2 = "idempotency_test_002"
    result_2 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_2,
        ruleset_version=ruleset_version
    )
    
    # Validation
    assert result_1.status == "SUCCESS"
    assert result_2.status == "SUCCESS"
    assert result_1.processed_chunks == result_2.processed_chunks
    assert result_1.total_evaluations == result_2.total_evaluations
    
    # Compare individual results (excluding analysis_id and timestamps)
    results_1 = get_gap_metrics(analysis_id=analysis_id_1)
    results_2 = get_gap_metrics(analysis_id=analysis_id_2)
    
    assert len(results_1) == len(results_2)
    
    # Sort for deterministic comparison
    results_1_sorted = sorted(results_1, key=lambda r: (r.chunk_id, r.rule_name))
    results_2_sorted = sorted(results_2, key=lambda r: (r.chunk_id, r.rule_name))
    
    for r1, r2 in zip(results_1_sorted, results_2_sorted):
        assert r1.chunk_id == r2.chunk_id
        assert r1.artifact_type == r2.artifact_type
        assert r1.rule_name == r2.rule_name
        assert r1.passed == r2.passed
        assert abs(r1.confidence - r2.confidence) < 1e-6  # Float precision
        assert r1.severity == r2.severity
        assert r1.suggested_fix == r2.suggested_fix
        assert r1.ruleset_version == r2.ruleset_version
```

#### Test 1.2: Multiple Sequential Executions
```python
def test_multiple_sequential_executions():
    """Verify consistency across multiple sequential runs."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(3)
    ruleset_version = "v1.0.0"
    
    results = []
    for i in range(5):  # Execute 5 times
        analysis_id = f"sequential_test_{i:03d}"
        result = execute_gap_analysis(
            chunks=test_chunks,
            analysis_id=analysis_id,
            ruleset_version=ruleset_version
        )
        results.append(result)
    
    # All executions should succeed
    assert all(r.status == "SUCCESS" for r in results)
    
    # All executions should process same number of chunks
    chunk_counts = [r.processed_chunks for r in results]
    assert len(set(chunk_counts)) == 1  # All identical
    
    # All executions should produce same number of evaluations
    evaluation_counts = [r.total_evaluations for r in results]
    assert len(set(evaluation_counts)) == 1  # All identical
    
    # Verify database contains results from all executions
    total_results = count_total_gap_metrics()
    expected_results = len(test_chunks) * get_rule_count() * 5  # 5 executions
    assert total_results == expected_results
```

#### Test 1.3: Deterministic Confidence Calculation
```python
def test_deterministic_confidence_calculation():
    """Verify confidence scores are deterministic."""
    
    # Create chunk with known characteristics
    test_chunk = {
        'chunk_id': 'confidence_test_001',
        'artifact_type': 'kubernetes',
        'metadata': {
            'kind': 'Deployment',
            'metadata': {
                'name': 'test-app',
                'labels': {'app': 'test-app'},
                'annotations': {
                    'description': 'Test deployment with specific characteristics for confidence testing'
                }
            },
            'spec': {
                'replicas': 3,
                'template': {
                    'spec': {
                        'containers': [{
                            'name': 'test-container',
                            'resources': {
                                'requests': {'memory': '64Mi', 'cpu': '100m'},
                                'limits': {'memory': '256Mi', 'cpu': '500m'}
                            }
                        }]
                    }
                }
            }
        }
    }
    
    # Execute multiple times and collect confidence scores
    confidence_scores = []
    for i in range(10):
        analysis_id = f"confidence_test_{i:03d}"
        execute_gap_analysis(
            chunks=[test_chunk],
            analysis_id=analysis_id,
            ruleset_version="v1.0.0"
        )
        
        results = get_gap_metrics(analysis_id=analysis_id)
        for result in results:
            confidence_scores.append((result.rule_name, result.confidence))
    
    # Group by rule and verify consistency
    scores_by_rule = {}
    for rule_name, confidence in confidence_scores:
        if rule_name not in scores_by_rule:
            scores_by_rule[rule_name] = []
        scores_by_rule[rule_name].append(confidence)
    
    # All confidence scores for each rule should be identical
    for rule_name, scores in scores_by_rule.items():
        assert len(set(scores)) == 1, f"Inconsistent confidence for rule {rule_name}: {scores}"
```

### Category 2: Version and State Management
**Purpose**: Verify proper handling of rule version changes and system state

#### Test 2.1: Rule Version Coexistence
```python
def test_rule_version_coexistence():
    """Verify different rule versions create separate result sets."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(3)
    
    # Execute with version 1.0.0
    analysis_id_v1 = "version_test_v1_0_0"
    result_v1 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_v1,
        ruleset_version="v1.0.0"
    )
    
    # Execute with version 1.1.0 (modified rules)
    analysis_id_v2 = "version_test_v1_1_0"
    result_v2 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_v2,
        ruleset_version="v1.1.0"
    )
    
    # Both executions should succeed
    assert result_v1.status == "SUCCESS"
    assert result_v2.status == "SUCCESS"
    
    # Verify both result sets exist
    results_v1 = get_gap_metrics(analysis_id=analysis_id_v1)
    results_v2 = get_gap_metrics(analysis_id=analysis_id_v2)
    
    assert len(results_v1) > 0
    assert len(results_v2) > 0
    
    # Verify version isolation
    assert all(r.ruleset_version == "v1.0.0" for r in results_v1)
    assert all(r.ruleset_version == "v1.1.0" for r in results_v2)
    
    # Results may differ due to rule changes
    # Verify that at least some rule produced different results
    different_results_found = False
    for r1 in results_v1:
        for r2 in results_v2:
            if (r1.chunk_id == r2.chunk_id and 
                r1.rule_name == r2.rule_name and
                (r1.confidence != r2.confidence or r1.severity != r2.severity)):
                different_results_found = True
                break
    
    assert different_results_found, "Rule version changes should affect results"
```

#### Test 2.2: Incremental Data Updates
```python
def test_incremental_data_updates():
    """Verify idempotency with incremental data changes."""
    
    # Initial dataset
    initial_chunks = IdempotencyTestFixtures().create_deterministic_chunks(3)
    
    # Execute initial analysis
    analysis_id_1 = "incremental_test_001"
    result_1 = execute_gap_analysis(
        chunks=initial_chunks,
        analysis_id=analysis_id_1,
        ruleset_version="v1.0.0"
    )
    
    # Add more chunks
    additional_chunks = IdempotencyTestFixtures().create_deterministic_chunks(2)
    # Modify chunk IDs to avoid conflicts
    for i, chunk in enumerate(additional_chunks):
        chunk['chunk_id'] = f'additional_chunk_{i:03d}'
    
    all_chunks = initial_chunks + additional_chunks
    
    # Execute analysis with expanded dataset
    analysis_id_2 = "incremental_test_002"
    result_2 = execute_gap_analysis(
        chunks=all_chunks,
        analysis_id=analysis_id_2,
        ruleset_version="v1.0.0"
    )
    
    # Validate results
    assert result_1.status == "SUCCESS"
    assert result_2.status == "SUCCESS"
    assert result_2.processed_chunks == len(all_chunks)
    assert result_2.processed_chunks > result_1.processed_chunks
    
    # Original chunk results should be identical
    results_1 = get_gap_metrics(analysis_id=analysis_id_1)
    results_2 = get_gap_metrics(analysis_id=analysis_id_2)
    
    original_chunk_ids = {chunk['chunk_id'] for chunk in initial_chunks}
    
    results_1_original = [r for r in results_1 if r.chunk_id in original_chunk_ids]
    results_2_original = [r for r in results_2 if r.chunk_id in original_chunk_ids]
    
    # Sort for comparison
    results_1_original.sort(key=lambda r: (r.chunk_id, r.rule_name))
    results_2_original.sort(key=lambda r: (r.chunk_id, r.rule_name))
    
    assert len(results_1_original) == len(results_2_original)
    
    for r1, r2 in zip(results_1_original, results_2_original):
        assert r1.chunk_id == r2.chunk_id
        assert r1.rule_name == r2.rule_name
        assert r1.passed == r2.passed
        assert abs(r1.confidence - r2.confidence) < 1e-6
        assert r1.severity == r2.severity
```

#### Test 2.3: Overwrite Behavior Validation
```python
def test_overwrite_behavior():
    """Verify proper overwrite behavior for same version re-execution."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(2)
    ruleset_version = "v1.0.0"
    
    # First execution
    analysis_id_1 = "overwrite_test_001"
    result_1 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_1,
        ruleset_version=ruleset_version
    )
    
    initial_result_count = count_gap_metrics_for_version(ruleset_version)
    
    # Second execution with same version (should overwrite)
    analysis_id_2 = "overwrite_test_002"
    result_2 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_2,
        ruleset_version=ruleset_version
    )
    
    final_result_count = count_gap_metrics_for_version(ruleset_version)
    
    # Verify overwrite occurred (same total count)
    assert initial_result_count == final_result_count
    
    # Verify only second analysis results exist
    results_1 = get_gap_metrics(analysis_id=analysis_id_1)
    results_2 = get_gap_metrics(analysis_id=analysis_id_2)
    
    assert len(results_1) == 0  # First analysis results should be gone
    assert len(results_2) > 0   # Second analysis results should exist
```

### Category 3: Concurrency and Error Handling
**Purpose**: Verify idempotency under concurrent access and error conditions

#### Test 3.1: Concurrent Execution Prevention
```python
def test_concurrent_execution_prevention():
    """Verify concurrent executions are properly serialized."""
    
    import threading
    import time
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(5)
    ruleset_version = "v1.0.0"
    
    execution_results = []
    execution_errors = []
    
    def execute_analysis(thread_id):
        try:
            analysis_id = f"concurrent_test_{thread_id:03d}"
            # Add delay to increase chance of concurrency
            time.sleep(0.1)
            result = execute_gap_analysis(
                chunks=test_chunks,
                analysis_id=analysis_id,
                ruleset_version=ruleset_version
            )
            execution_results.append((thread_id, result))
        except Exception as e:
            execution_errors.append((thread_id, e))
    
    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=execute_analysis, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify execution behavior
    successful_executions = len(execution_results)
    failed_executions = len(execution_errors)
    
    # At least one execution should succeed
    assert successful_executions >= 1
    
    # Failed executions should be due to lock conflicts
    for thread_id, error in execution_errors:
        assert "execution lock" in str(error).lower() or "concurrent" in str(error).lower()
    
    # Successful executions should produce consistent results
    if successful_executions > 1:
        first_result = execution_results[0][1]
        for thread_id, result in execution_results[1:]:
            assert result.processed_chunks == first_result.processed_chunks
            assert result.total_evaluations == first_result.total_evaluations
```

#### Test 3.2: Partial Failure Recovery
```python
def test_partial_failure_recovery():
    """Verify failed executions don't leave partial results."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(5)
    
    # Establish baseline
    baseline_count = count_total_gap_metrics()
    
    # Simulate execution failure mid-process
    with mock.patch('gap_analysis.confidence_calculator.ConfidenceCalculator.calculate_confidence') as mock_calc:
        # First few calculations succeed, then fail
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return create_mock_confidence_result()
            else:
                raise DatabaseConnectionError("Simulated database failure")
        
        mock_calc.side_effect = side_effect
        
        # Attempt execution that will fail
        with pytest.raises(DatabaseConnectionError):
            execute_gap_analysis(
                chunks=test_chunks,
                analysis_id="failure_test_001",
                ruleset_version="v1.0.0"
            )
    
    # Verify no partial results remain
    post_failure_count = count_total_gap_metrics()
    assert post_failure_count == baseline_count
    
    # Verify execution lock is released
    assert not is_execution_locked("gap_analysis_execution")
    
    # Verify subsequent execution can proceed normally
    result = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id="recovery_test_001",
        ruleset_version="v1.0.0"
    )
    
    assert result.status == "SUCCESS"
    assert result.processed_chunks == len(test_chunks)
```

#### Test 3.3: System State Consistency
```python
def test_system_state_consistency():
    """Verify system state remains consistent after various operations."""
    
    # Perform series of operations
    operations = [
        ('initial_load', lambda: execute_gap_analysis(
            IdempotencyTestFixtures().create_deterministic_chunks(3),
            "state_test_001", "v1.0.0")),
        ('version_change', lambda: execute_gap_analysis(
            IdempotencyTestFixtures().create_deterministic_chunks(3),
            "state_test_002", "v1.1.0")),
        ('incremental_update', lambda: execute_gap_analysis(
            IdempotencyTestFixtures().create_deterministic_chunks(5),
            "state_test_003", "v1.0.0")),
        ('reversion', lambda: execute_gap_analysis(
            IdempotencyTestFixtures().create_deterministic_chunks(3),
            "state_test_004", "v1.0.0"))
    ]
    
    state_snapshots = []
    
    for operation_name, operation_func in operations:
        # Execute operation
        result = operation_func()
        assert result.status == "SUCCESS"
        
        # Capture state snapshot
        snapshot = {
            'operation': operation_name,
            'total_results': count_total_gap_metrics(),
            'unique_versions': get_unique_ruleset_versions(),
            'unique_analysis_ids': get_unique_analysis_ids(),
            'orphaned_results': count_orphaned_results(),
            'duplicate_conceptual_keys': count_duplicate_conceptual_keys()
        }
        state_snapshots.append(snapshot)
    
    # Validate state consistency rules
    for snapshot in state_snapshots:
        # No orphaned results
        assert snapshot['orphaned_results'] == 0
        
        # No duplicate conceptual keys
        assert snapshot['duplicate_conceptual_keys'] == 0
        
        # Version list only grows (no deletions)
        if len(state_snapshots) > 1:
            previous_versions = state_snapshots[state_snapshots.index(snapshot) - 1]['unique_versions']
            current_versions = snapshot['unique_versions']
            assert set(previous_versions).issubset(set(current_versions))
```

### Category 4: Data Integrity and Audit Trail
**Purpose**: Verify data integrity and complete audit trail maintenance

#### Test 4.1: Audit Trail Completeness
```python
def test_audit_trail_completeness():
    """Verify complete audit trail for all operations."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(2)
    
    # Execute operations and track expected audit events
    expected_audit_events = []
    
    # Operation 1: Initial analysis
    analysis_id_1 = "audit_test_001"
    result_1 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_1,
        ruleset_version="v1.0.0"
    )
    expected_audit_events.append({
        'operation': 'gap_analysis_execution',
        'analysis_id': analysis_id_1,
        'operation_type': 'INSERT'
    })
    
    # Operation 2: Overwrite with same version
    analysis_id_2 = "audit_test_002"
    result_2 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id_2,
        ruleset_version="v1.0.0"
    )
    expected_audit_events.extend([
        {
            'operation': 'gap_analysis_execution',
            'analysis_id': analysis_id_1,
            'operation_type': 'DELETE'
        },
        {
            'operation': 'gap_analysis_execution',
            'analysis_id': analysis_id_2,
            'operation_type': 'INSERT'
        }
    ])
    
    # Verify audit trail
    audit_records = get_audit_trail()
    
    for expected_event in expected_audit_events:
        matching_records = [
            r for r in audit_records 
            if (r.operation == expected_event['operation'] and
                r.analysis_id == expected_event['analysis_id'] and
                r.operation_type == expected_event['operation_type'])
        ]
        assert len(matching_records) > 0, f"Missing audit record for {expected_event}"
```

#### Test 4.2: Data Consistency Validation
```python
def test_data_consistency_validation():
    """Verify referential integrity and data consistency."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(4)
    
    # Execute analysis
    analysis_id = "consistency_test_001"
    result = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id=analysis_id,
        ruleset_version="v1.0.0"
    )
    
    # Validate data consistency rules
    consistency_checks = [
        {
            'name': 'No orphaned gap_metrics',
            'query': '''
                SELECT COUNT(*) as count
                FROM gap_metrics gm
                LEFT JOIN source_metadata sm ON gm.chunk_id = sm.chunk_id
                WHERE sm.chunk_id IS NULL
            ''',
            'expected_count': 0
        },
        {
            'name': 'No duplicate conceptual keys',
            'query': '''
                SELECT chunk_id, rule_name, ruleset_version, COUNT(*) as count
                FROM gap_metrics
                GROUP BY chunk_id, rule_name, ruleset_version
                HAVING COUNT(*) > 1
            ''',
            'expected_count': 0
        },
        {
            'name': 'All confidence scores in valid range',
            'query': '''
                SELECT COUNT(*) as count
                FROM gap_metrics
                WHERE confidence < 0.0 OR confidence > 1.0
            ''',
            'expected_count': 0
        },
        {
            'name': 'All severity values valid',
            'query': '''
                SELECT COUNT(*) as count
                FROM gap_metrics
                WHERE severity < 1 OR severity > 5
            ''',
            'expected_count': 0
        },
        {
            'name': 'All suggested_fix fields populated',
            'query': '''
                SELECT COUNT(*) as count
                FROM gap_metrics
                WHERE suggested_fix IS NULL OR LENGTH(TRIM(suggested_fix)) = 0
            ''',
            'expected_count': 0
        }
    ]
    
    for check in consistency_checks:
        result_count = execute_query(check['query'])[0]['count']
        assert result_count == check['expected_count'], f"Consistency check failed: {check['name']}"
```

#### Test 4.3: Historical Data Preservation
```python
def test_historical_data_preservation():
    """Verify historical data is preserved correctly."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(2)
    
    # Execute with multiple versions over time
    version_history = [
        ("v1.0.0", "2023-01-15"),
        ("v1.1.0", "2023-02-15"),
        ("v1.2.0", "2023-03-15")
    ]
    
    all_analysis_ids = []
    
    for version, date_str in version_history:
        analysis_id = f"history_test_{version.replace('.', '_')}"
        all_analysis_ids.append(analysis_id)
        
        # Mock the current date for historical simulation
        with mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.strptime(date_str, "%Y-%m-%d")
            
            result = execute_gap_analysis(
                chunks=test_chunks,
                analysis_id=analysis_id,
                ruleset_version=version
            )
            
            assert result.status == "SUCCESS"
    
    # Verify all versions coexist
    for version in [v[0] for v in version_history]:
        version_results = get_gap_metrics_for_version(version)
        assert len(version_results) > 0, f"No results found for version {version}"
    
    # Verify historical timeline
    all_results = get_all_gap_metrics()
    results_by_version = {}
    
    for result in all_results:
        if result.ruleset_version not in results_by_version:
            results_by_version[result.ruleset_version] = []
        results_by_version[result.ruleset_version].append(result)
    
    # Each version should have its own complete result set
    expected_results_per_version = len(test_chunks) * get_active_rule_count()
    
    for version, results in results_by_version.items():
        assert len(results) >= expected_results_per_version, \
            f"Incomplete results for version {version}: {len(results)} < {expected_results_per_version}"
```

---

## Performance Impact Validation

### Idempotency Overhead Testing
```python
def test_idempotency_overhead():
    """Measure performance impact of idempotency mechanisms."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(10)
    
    # Measure baseline performance (first execution)
    start_time = time.time()
    result_1 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id="performance_test_001",
        ruleset_version="v1.0.0"
    )
    baseline_time = time.time() - start_time
    
    # Measure idempotent execution performance (overwrite)
    start_time = time.time()
    result_2 = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id="performance_test_002",
        ruleset_version="v1.0.0"
    )
    idempotent_time = time.time() - start_time
    
    # Validate performance
    assert result_1.status == "SUCCESS"
    assert result_2.status == "SUCCESS"
    
    # Idempotent execution should not be significantly slower
    overhead_ratio = idempotent_time / baseline_time
    assert overhead_ratio < 1.5, f"Idempotency overhead too high: {overhead_ratio:.2f}x"
    
    # Both executions should produce identical processing metrics
    assert result_1.processed_chunks == result_2.processed_chunks
    assert result_1.total_evaluations == result_2.total_evaluations
```

### Lock Contention Impact
```python
def test_lock_contention_impact():
    """Measure impact of execution locks on performance."""
    
    test_chunks = IdempotencyTestFixtures().create_deterministic_chunks(5)
    
    # Measure execution time without contention
    start_time = time.time()
    result = execute_gap_analysis(
        chunks=test_chunks,
        analysis_id="lock_test_001",
        ruleset_version="v1.0.0"
    )
    uncontended_time = time.time() - start_time
    
    # Simulate lock contention (rapid sequential executions)
    execution_times = []
    for i in range(3):
        start_time = time.time()
        result = execute_gap_analysis(
            chunks=test_chunks,
            analysis_id=f"lock_test_{i:03d}",
            ruleset_version="v1.0.0"
        )
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
    
    # Verify minimal impact from lock management
    max_execution_time = max(execution_times)
    lock_overhead_ratio = max_execution_time / uncontended_time
    
    assert lock_overhead_ratio < 2.0, f"Lock overhead too high: {lock_overhead_ratio:.2f}x"
```

---

## Automated Test Execution

### Continuous Integration Pipeline
```yaml
# CI/CD integration for idempotency testing
idempotency_test_pipeline:
  trigger:
    - code_changes
    - rule_configuration_changes
    - schema_modifications
  
  test_stages:
    - basic_idempotency:
        tests: 
          - test_identical_input_reproducibility
          - test_multiple_sequential_executions
          - test_deterministic_confidence_calculation
        timeout: 10_minutes
        
    - version_management:
        tests:
          - test_rule_version_coexistence
          - test_incremental_data_updates
          - test_overwrite_behavior
        timeout: 15_minutes
        
    - concurrency_and_errors:
        tests:
          - test_concurrent_execution_prevention
          - test_partial_failure_recovery
          - test_system_state_consistency
        timeout: 20_minutes
        
    - data_integrity:
        tests:
          - test_audit_trail_completeness
          - test_data_consistency_validation
          - test_historical_data_preservation
        timeout: 15_minutes
        
    - performance_validation:
        tests:
          - test_idempotency_overhead
          - test_lock_contention_impact
        timeout: 10_minutes
```

### Test Environment Cleanup
```python
def cleanup_idempotency_test_environment():
    """Clean up test environment after test execution."""
    
    # Drop test schema
    execute_query("DROP SCHEMA IF EXISTS gap_analysis_idempotency_test CASCADE")
    
    # Clear execution locks
    execute_query("DELETE FROM execution_locks WHERE locked_by LIKE 'idempotency_test_%'")
    
    # Clean up test files
    test_data_path = Path("./test_data/idempotency")
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
    
    # Reset any cached configurations
    clear_rule_configuration_cache()
    clear_template_cache()
```

---

## Success Criteria and Reporting

### Idempotency Validation Metrics
```python
IDEMPOTENCY_SUCCESS_CRITERIA = {
    'reproducibility_accuracy': 100,     # % of identical inputs producing identical outputs
    'version_isolation': 100,            # % of version changes properly isolated
    'concurrency_safety': 100,           # % of concurrent executions properly handled
    'failure_recovery': 100,             # % of failures leaving clean state
    'data_consistency': 100,             # % of consistency checks passing
    'audit_trail_completeness': 100,     # % of operations properly audited
    'performance_overhead': 50,          # Maximum % overhead from idempotency
}
```

### Test Result Reporting
```python
def generate_idempotency_test_report():
    """Generate comprehensive idempotency test report."""
    
    report = {
        'test_execution_summary': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'execution_time': 0
        },
        'idempotency_metrics': {},
        'performance_impact': {},
        'consistency_validation': {},
        'recommendations': []
    }
    
    # Execute all test categories and collect metrics
    test_categories = [
        ('Basic Idempotency', run_basic_idempotency_tests),
        ('Version Management', run_version_management_tests),
        ('Concurrency/Errors', run_concurrency_error_tests),
        ('Data Integrity', run_data_integrity_tests),
        ('Performance', run_performance_validation_tests)
    ]
    
    for category_name, test_function in test_categories:
        try:
            category_results = test_function()
            report['test_execution_summary']['total_tests'] += category_results.total_tests
            report['test_execution_summary']['passed_tests'] += category_results.passed_tests
            report['test_execution_summary']['failed_tests'] += category_results.failed_tests
            
            report[category_name.lower().replace(' ', '_')] = category_results.metrics
            
        except Exception as e:
            report['recommendations'].append(f"Failed to execute {category_name} tests: {e}")
    
    # Generate recommendations based on results
    if report['test_execution_summary']['failed_tests'] > 0:
        report['recommendations'].append("Some idempotency tests failed - review implementation")
    
    if report.get('performance_impact', {}).get('overhead_ratio', 0) > 1.5:
        report['recommendations'].append("High idempotency overhead detected - optimize implementation")
    
    return report
```

This comprehensive idempotency test plan ensures that the gap analysis pipeline maintains deterministic, reproducible behavior while properly handling version changes, concurrent access, and error conditions. The test suite validates all aspects of the idempotency design and provides measurable criteria for system reliability.