# Idempotency Design: Safe Re-run Semantics

## Overview

This document specifies the idempotency strategy for gap analysis rule evaluation, ensuring that repeated executions with the same inputs produce consistent results without data duplication or corruption. The design follows functional data engineering principles and BigQuery best practices.

## Core Idempotency Principles

### 1. Deterministic Results
- **Same Inputs → Same Outputs**: Identical rule configurations and source data always produce identical gap_metrics results
- **Reproducible Calculations**: Confidence scores, pass/fail decisions, and suggested fixes are deterministic
- **Version Consistency**: Results depend only on rule version, not execution time or environment

### 2. Safe Re-execution
- **No Side Effects**: Re-running analysis doesn't corrupt existing data
- **Atomic Operations**: All-or-nothing execution prevents partial updates
- **Rollback Capability**: Failed executions leave previous results intact

### 3. Audit Trail Preservation
- **Historical Results**: Previous rule versions remain accessible
- **Change Tracking**: Analysis runs are logged with correlation IDs
- **Version Evolution**: Rule updates create new result sets, don't overwrite history

## Conceptual Idempotency Key

### Primary Uniqueness Constraint

The logical primary key for gap_metrics ensures one evaluation result per unique combination:

```sql
CONCEPTUAL_KEY = (chunk_id, rule_name, ruleset_version)
```

### Key Component Semantics

#### chunk_id
- **Source**: Links to `source_metadata.chunk_id`
- **Scope**: Identifies the specific content chunk being evaluated
- **Stability**: Immutable for the lifetime of the content
- **Format**: String following pattern `{source_path}:{chunk_identifier}`

#### rule_name  
- **Source**: Rule configuration `rule_name` field
- **Scope**: Identifies the specific rule being applied
- **Stability**: Immutable within a rule version
- **Format**: Lowercase alphanumeric with underscores (e.g., `kubernetes_deployment_security`)

#### ruleset_version
- **Source**: Rule configuration version manifest
- **Scope**: Identifies the rule configuration version
- **Stability**: Immutable once published
- **Format**: Semantic version string `vX.Y.Z` (e.g., `v1.2.3`)

### Key Benefits

1. **Uniqueness**: Prevents duplicate evaluations for the same content-rule-version combination
2. **Versioning**: Allows multiple rule versions to coexist with separate results  
3. **Traceability**: Every result can be traced to specific rule and content versions
4. **Evolution**: Rule updates create new result sets without losing historical data

## Overwrite Strategy: Partition-Based Replacement

### Strategy Selection Rationale

**Chosen Approach**: Partition-based overwrite using BigQuery transactions

**Alternative Approaches Considered**:
- **Upsert/Merge**: Complex logic, potential for race conditions
- **Append-only**: Requires manual cleanup, storage bloat
- **Full table replacement**: Loss of historical data, poor performance

### Partition Design

#### Partitioning Schema
```sql
-- BigQuery table partitioning
PARTITION BY DATE(partition_date)
CLUSTER BY (analysis_id, rule_name, chunk_id)
```

#### Partition Key Components
- **partition_date**: Daily partitions based on analysis execution date
- **analysis_id**: Batch execution identifier for atomic operations
- **Clustering**: Optimizes query performance for common access patterns

### Overwrite Process Flow

#### Phase 1: Pre-execution Setup
```sql
-- 1. Generate execution metadata
analysis_id = FORMAT("YYYYMMDD-HHMMSS-%s", GENERATE_UUID())
partition_date = CURRENT_DATE()
ruleset_version = CALCULATE_RULESET_VERSION_HASH()

-- 2. Validate prerequisites  
ASSERT source_metadata table accessible
ASSERT rule configurations valid
ASSERT BigQuery permissions sufficient
```

#### Phase 2: Atomic Transaction Execution
```sql
BEGIN TRANSACTION;

-- 3. Delete existing results for this ruleset version
DELETE FROM gap_metrics 
WHERE ruleset_version = @current_ruleset_version
  AND partition_date = @target_partition_date;

-- 4. Insert new evaluation results
INSERT INTO gap_metrics (
  chunk_id, artifact_type, rule_name, passed, confidence, 
  severity, suggested_fix, analysis_id, created_at, 
  ruleset_version, partition_date
)
SELECT 
  chunk_id, artifact_type, rule_name, passed, confidence,
  severity, suggested_fix, @analysis_id, CURRENT_TIMESTAMP(),
  @current_ruleset_version, @target_partition_date
FROM evaluation_results_temp;

-- 5. Verify transaction success
ASSERT ROW_COUNT(gap_metrics WHERE analysis_id = @analysis_id) > 0;

COMMIT TRANSACTION;
```

#### Phase 3: Post-execution Validation
```sql
-- 6. Verify no duplicates exist
SELECT chunk_id, rule_name, ruleset_version, COUNT(*) as duplicate_count
FROM gap_metrics 
WHERE ruleset_version = @current_ruleset_version
GROUP BY chunk_id, rule_name, ruleset_version
HAVING COUNT(*) > 1;
-- Should return 0 rows

-- 7. Verify referential integrity
SELECT COUNT(*) as orphaned_records
FROM gap_metrics gm
LEFT JOIN source_metadata sm ON gm.chunk_id = sm.chunk_id
WHERE sm.chunk_id IS NULL;
-- Should return 0
```

## Stale Result Detection and Cleanup

### Staleness Indicators

#### 1. Version-Based Staleness
```sql
-- Results from older rule versions
SELECT DISTINCT ruleset_version 
FROM gap_metrics 
WHERE ruleset_version != @current_ruleset_version
ORDER BY ruleset_version;
```

#### 2. Age-Based Staleness  
```sql
-- Results older than retention period
SELECT analysis_id, created_at, COUNT(*) as record_count
FROM gap_metrics 
WHERE created_at < DATE_SUB(CURRENT_DATE(), INTERVAL @retention_days DAY)
GROUP BY analysis_id, created_at
ORDER BY created_at;
```

#### 3. Orphaned Results
```sql
-- Results referencing non-existent source data
SELECT gm.chunk_id, gm.analysis_id, gm.created_at
FROM gap_metrics gm
LEFT JOIN source_metadata sm ON gm.chunk_id = sm.chunk_id
WHERE sm.chunk_id IS NULL;
```

### Cleanup Procedures

#### Automated Cleanup (Built into pipeline)
```sql
-- Remove results older than retention policy
DELETE FROM gap_metrics 
WHERE created_at < DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY);

-- Remove orphaned results
DELETE FROM gap_metrics 
WHERE chunk_id NOT IN (
  SELECT DISTINCT chunk_id FROM source_metadata
);
```

#### Manual Cleanup (Administrative operations)
```sql
-- Remove specific rule version results
DELETE FROM gap_metrics 
WHERE ruleset_version = @deprecated_version;

-- Remove results for specific analysis run
DELETE FROM gap_metrics 
WHERE analysis_id = @failed_analysis_id;
```

### Retention Policy Configuration
```yaml
retention_policy:
  default_retention_days: 90
  critical_rule_retention_days: 365
  compliance_rule_retention_days: 2555  # 7 years
  version_retention_count: 5  # Keep last 5 rule versions
```

## Re-run Execution Semantics

### Execution Modes

#### 1. Incremental Re-run (Default)
- **Scope**: Process only new or modified source_metadata records
- **Idempotency**: Update results for changed content, preserve unchanged
- **Performance**: Optimized for frequent updates

```sql
-- Incremental processing logic
WITH changed_chunks AS (
  SELECT chunk_id 
  FROM source_metadata 
  WHERE created_at > @last_analysis_timestamp
     OR modified_at > @last_analysis_timestamp
)
-- Process only changed chunks
```

#### 2. Full Re-run (Explicit)
- **Scope**: Re-evaluate all source_metadata records
- **Idempotency**: Replace all results for current ruleset version
- **Performance**: Complete rebuild, use for rule changes

```sql
-- Full processing logic
-- Process all chunks in source_metadata
SELECT chunk_id FROM source_metadata 
WHERE artifact_type IN @enabled_artifact_types
```

#### 3. Rule-Specific Re-run
- **Scope**: Re-evaluate specific rules across all content
- **Idempotency**: Replace results for specified rules only
- **Performance**: Targeted updates for rule modifications

```sql
-- Rule-specific processing
DELETE FROM gap_metrics 
WHERE rule_name IN @target_rules 
  AND ruleset_version = @current_version;
-- Then re-evaluate only target rules
```

### Execution Safety Mechanisms

#### Pre-execution Validation
```python
def validate_execution_safety():
    """Validate system state before starting analysis."""
    checks = [
        check_source_metadata_accessibility(),
        check_rule_configuration_validity(), 
        check_bigquery_permissions(),
        check_disk_space_availability(),
        check_concurrent_execution_lock()
    ]
    
    if not all(checks):
        raise ExecutionValidationError("Pre-execution validation failed")
```

#### Concurrent Execution Prevention
```sql
-- Execution lock mechanism
CREATE TABLE IF NOT EXISTS execution_locks (
  lock_name STRING,
  analysis_id STRING, 
  locked_at TIMESTAMP,
  locked_by STRING
);

-- Acquire lock before execution
INSERT INTO execution_locks (lock_name, analysis_id, locked_at, locked_by)
VALUES ('gap_analysis_execution', @analysis_id, CURRENT_TIMESTAMP(), @executor);

-- Release lock after completion
DELETE FROM execution_locks 
WHERE lock_name = 'gap_analysis_execution' 
  AND analysis_id = @analysis_id;
```

#### Failure Recovery
```python
def handle_execution_failure(analysis_id, error):
    """Handle failed execution with proper cleanup."""
    try:
        # Rollback partial results
        delete_partial_results(analysis_id)
        
        # Release execution lock
        release_execution_lock(analysis_id)
        
        # Log failure details
        log_execution_failure(analysis_id, error)
        
        # Preserve previous results (no deletion)
        
    except Exception as cleanup_error:
        log_critical_error(f"Cleanup failed: {cleanup_error}")
        raise
```

## Performance Optimization

### Query Optimization Strategies

#### 1. Partition Pruning
```sql
-- Leverage partition pruning for efficient queries
SELECT * FROM gap_metrics 
WHERE partition_date = CURRENT_DATE()  -- Partition pruning
  AND rule_name = 'kubernetes_deployment_security'  -- Clustering
  AND analysis_id = @target_analysis_id;  -- Clustering
```

#### 2. Clustering Benefits
```sql
-- Clustered columns optimize common query patterns
-- Analysis-based queries (dashboards)
SELECT rule_name, AVG(confidence), COUNT(*) 
FROM gap_metrics 
WHERE analysis_id = @recent_analysis_id
GROUP BY rule_name;

-- Rule-based queries (rule performance analysis)
SELECT analysis_id, AVG(confidence), COUNT(*)
FROM gap_metrics 
WHERE rule_name = @target_rule
  AND partition_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY analysis_id;
```

#### 3. Batch Processing Optimization
```python
def process_chunks_in_batches(chunks, batch_size=1000):
    """Process chunks in optimized batches for BigQuery."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Prepare batch SQL with parameter arrays
        chunk_ids = [chunk['chunk_id'] for chunk in batch]
        
        # Single query processes entire batch
        execute_batch_evaluation(chunk_ids)
```

### Memory and Resource Management

#### BigQuery Slot Management
```yaml
resource_limits:
  max_concurrent_slots: 500
  query_timeout_seconds: 1800
  max_batch_size_mb: 100
  
performance_targets:
  chunks_per_second: 100
  max_execution_time_minutes: 30
  memory_limit_gb: 8
```

#### Error Rate Monitoring
```sql
-- Monitor execution quality
WITH execution_stats AS (
  SELECT 
    analysis_id,
    COUNT(*) as total_evaluations,
    COUNT(CASE WHEN passed = false THEN 1 END) as failed_evaluations,
    AVG(confidence) as avg_confidence,
    COUNTIF(confidence < 0.3) as low_confidence_count
  FROM gap_metrics 
  WHERE partition_date = CURRENT_DATE()
  GROUP BY analysis_id
)
SELECT 
  analysis_id,
  total_evaluations,
  ROUND(failed_evaluations * 100.0 / total_evaluations, 2) as failure_rate_pct,
  ROUND(avg_confidence, 3) as avg_confidence,
  ROUND(low_confidence_count * 100.0 / total_evaluations, 2) as low_confidence_rate_pct
FROM execution_stats;
```

## Testing and Validation

### Idempotency Test Scenarios

#### Test 1: Identical Re-run
```python
def test_identical_rerun_idempotency():
    """Verify identical inputs produce identical outputs."""
    # Execute analysis with fixed input
    result1 = execute_gap_analysis(source_data, rules_v1_0_0)
    
    # Re-execute with identical input
    result2 = execute_gap_analysis(source_data, rules_v1_0_0)
    
    # Results should be identical
    assert result1.metrics_count == result2.metrics_count
    assert result1.confidence_scores == result2.confidence_scores
    assert result1.duplicate_count == 0
    assert result2.duplicate_count == 0
```

#### Test 2: Rule Version Evolution
```python
def test_rule_version_coexistence():
    """Verify different rule versions create separate result sets."""
    # Execute with rule version 1.0.0
    result_v1 = execute_gap_analysis(source_data, rules_v1_0_0)
    
    # Execute with rule version 1.1.0
    result_v2 = execute_gap_analysis(source_data, rules_v1_1_0)
    
    # Both result sets should exist
    v1_count = count_results_for_version("v1.0.0")
    v2_count = count_results_for_version("v1.1.0")
    
    assert v1_count > 0
    assert v2_count > 0
    assert v1_count + v2_count == total_result_count()
```

#### Test 3: Partial Failure Recovery
```python
def test_partial_failure_recovery():
    """Verify failed executions don't corrupt existing data."""
    # Establish baseline
    baseline_count = count_total_results()
    
    # Attempt execution that will fail mid-process
    with pytest.raises(ExecutionError):
        execute_gap_analysis_with_forced_failure(source_data, rules)
    
    # Verify no partial results remain
    post_failure_count = count_total_results()
    assert post_failure_count == baseline_count
    
    # Verify execution lock is released
    assert not is_execution_locked()
```

### Performance Validation

#### Execution Time Targets
```python
performance_requirements = {
    "10k_chunks": {"max_time_minutes": 5, "target_time_minutes": 3},
    "100k_chunks": {"max_time_minutes": 30, "target_time_minutes": 20},
    "1m_chunks": {"max_time_minutes": 180, "target_time_minutes": 120}
}

def test_execution_performance(chunk_count):
    """Validate execution meets performance targets."""
    start_time = time.time()
    
    execute_gap_analysis(generate_test_chunks(chunk_count), rules)
    
    execution_time = time.time() - start_time
    target_time = performance_requirements[f"{chunk_count}_chunks"]["max_time_minutes"] * 60
    
    assert execution_time < target_time, f"Execution took {execution_time}s, target {target_time}s"
```

This idempotency design ensures safe, reliable, and performant gap analysis execution while maintaining data integrity and providing clear audit trails for all operations.