# Technical Design Plan: Issue #5 - Gap Analysis Rules System

**Reference:** Issue #5 "M3: Define coverage and confidence metrics for knowledge gaps"  
**Research Base:** Research.md - Industry best practices for deterministic rule engines

## 1. Architecture Overview

### High-Level Flow
```
[Rule Configs] → [Config Loader] → [Rule Evaluator] → [gap_metrics Table]
      ↓              ↓                    ↓                    ↓
   rules/*.yaml   Validation        SQL Generation      Idempotent Write
   versioned     + Parsing         + Execution         + Audit Trail
```

**Process Steps:**
1. **Config Loading:** Parse versioned YAML rule definitions with validation
2. **Metadata Query:** Load source_metadata records for evaluation scope  
3. **Rule Evaluation:** Apply deterministic SQL-based rules with confidence scoring
4. **Metrics Emission:** Upsert results to gap_metrics with conceptual key deduplication
5. **Audit Logging:** Record execution metadata and rule version tracking

### Interface Definitions

#### Rule Config Interface
```yaml
# Contract: Rule definition structure
rule_name: string (unique identifier)
artifact_type: enum (kubernetes|fastapi|cobol|irs|mumps)
version: semver (X.Y.Z)
evaluation_sql: string (parameterized BigQuery SQL)
confidence_weights: object (field completion factors)
severity: integer (1-5)
suggested_fix_template: string (mustache template)
```

#### Evaluator Interface  
```python
# Conceptual interface (no implementation)
class RuleEvaluator:
    def load_rules(config_path: str, version: str) -> List[Rule]
    def evaluate_chunk(chunk_data: dict, rules: List[Rule]) -> List[GapMetric]
    def compute_confidence(field_data: dict, weights: dict) -> float
    def generate_suggested_fix(template: str, context: dict) -> str
```

#### Metrics Emission Interface
```python
# Conceptual interface for BigQuery integration
class MetricsEmitter:
    def upsert_metrics(metrics: List[GapMetric], analysis_id: str) -> None
    def cleanup_stale_results(ruleset_version: str) -> None
    def validate_schema_compatibility() -> bool
```

## 2. Rule Configuration Design

### File Structure Strategy
Based on Airbnb's Wall Framework and Uber's data quality patterns:

```
configs/
├── rules/
│   ├── kubernetes/
│   │   ├── deployment.yaml      # Deployment-specific rules
│   │   ├── service.yaml         # Service-specific rules  
│   │   └── common.yaml          # Cross-resource rules
│   ├── fastapi/
│   │   ├── endpoints.yaml       # API endpoint rules
│   │   └── schemas.yaml         # Request/response rules
│   ├── cobol/
│   │   ├── copybooks.yaml       # Data structure rules
│   │   └── procedures.yaml      # Logic documentation rules
│   ├── irs/
│   │   └── record_layouts.yaml  # IRS format rules
│   └── mumps/
│       └── fileman.yaml         # VistA/FileMan rules
├── versions/
│   ├── v1.0.0.yaml             # Rule set version manifest
│   └── v1.1.0.yaml             # Updated rule set
└── schemas/
    └── rule_schema.json        # JSON Schema validation
```

### Versioning Strategy (Semantic Versioning)
Following industry standards from research:
- **Major (X):** Breaking changes to rule evaluation logic
- **Minor (Y):** New rules added, backward compatible changes
- **Patch (Z):** Bug fixes, clarifications, no behavior change

### Validation Checks
Based on BigQuery Dataplex and enterprise data quality patterns:
1. **Schema Validation:** JSON Schema compliance for rule structure
2. **SQL Syntax:** Parse and validate BigQuery SQL templates
3. **Cross-Reference:** Verify artifact_type exists in source_metadata
4. **Confidence Weights:** Sum to 1.0, all values in [0,1] range
5. **Template Validation:** Mustache syntax and required variables

### Example Rule Definitions

#### Kubernetes Deployment Rule
```yaml
rule_name: "k8s_deployment_documentation"
artifact_type: "kubernetes"
version: "1.0.0"
description: "Validates Kubernetes Deployment has proper documentation"
evaluation_sql: |
  WITH rule_eval AS (
    SELECT 
      chunk_id,
      CASE WHEN JSON_EXTRACT_SCALAR(metadata, '$.annotations.description') IS NOT NULL 
           AND LENGTH(JSON_EXTRACT_SCALAR(metadata, '$.annotations.description')) > 10
           THEN true ELSE false END as has_description,
      CASE WHEN JSON_EXTRACT_SCALAR(metadata, '$.labels.app') IS NOT NULL
           THEN true ELSE false END as has_app_label
    FROM source_metadata 
    WHERE artifact_type = 'kubernetes' AND kind = 'Deployment'
  )
  SELECT 
    chunk_id,
    (has_description AND has_app_label) as passed,
    (CAST(has_description as INT64) * 0.7 + CAST(has_app_label as INT64) * 0.3) as confidence
  FROM rule_eval
confidence_weights:
  description_present: 0.7
  labels_complete: 0.3
severity: 3
suggested_fix_template: "Add description annotation: metadata.annotations.description = '{{description_suggestion}}'"
```

#### FastAPI Endpoint Rule  
```yaml
rule_name: "fastapi_endpoint_docstring"
artifact_type: "fastapi"
version: "1.0.0"
evaluation_sql: |
  SELECT 
    chunk_id,
    REGEXP_CONTAINS(text_content, r'"""[\s\S]*?"""') as has_docstring,
    LENGTH(REGEXP_EXTRACT(text_content, r'"""([\s\S]*?)"""')) > 50 as has_adequate_docs
  FROM source_metadata 
  WHERE artifact_type = 'fastapi' AND api_path LIKE '%endpoint%'
confidence_weights:
  docstring_present: 0.6
  content_length: 0.4
severity: 2
suggested_fix_template: "Add comprehensive docstring to {{endpoint_name}} explaining parameters and return values"
```

## 3. Evaluator Design

### Deterministic Pass/Fail Logic
Based on Microsoft Azure AI confidence patterns:

**Decision Tree:**
1. **Required Fields Check:** All mandatory fields present → Continue
2. **Quality Threshold:** Content meets minimum standards → Continue  
3. **Validation Rules:** Custom business rules pass → PASS
4. **Any Failure:** Missing required or below threshold → FAIL

### Confidence Formula Components
Following LLM-based confidence research patterns:

```
confidence = base_completeness * quality_multiplier - penalty_deductions

base_completeness = (present_fields / required_fields)
quality_multiplier = min(1.0, content_quality_score / quality_threshold)  
penalty_deductions = critical_missing_fields * 0.15 + format_violations * 0.05

content_quality_score = (
  length_score * 0.4 +           # Adequate content length
  structure_score * 0.3 +        # Proper formatting/structure  
  vocabulary_score * 0.3         # Domain-appropriate terminology
)
```

### Severity Mapping (5×5 Risk Matrix Standard)
Based on healthcare and regulatory industry patterns:

| Severity | Impact | Examples |
|----------|--------|----------|
| 1 (Low) | Cosmetic issues | Missing optional descriptions |
| 2 (Moderate) | Minor functionality gaps | Incomplete parameter docs |
| 3 (Medium) | Operational impact | Missing resource constraints |
| 4 (High) | Service degradation risk | No error handling docs |
| 5 (Critical) | System failure risk | No security specifications |

### Suggested Fix Templating Rules
Following McKinsey AI4DQ framework:

**Template Structure (Mustache Format):**
```mustache
{{#if missing_description}}
Add description: {{field_name}} = "{{suggested_content}}"
{{/if}}
{{#if inadequate_length}}  
Expand {{field_name}} to minimum {{min_length}} characters
{{/if}}
{{#if missing_required_fields}}
Required fields: {{#each required_fields}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}
{{/if}}
```

### Error Handling Strategy
1. **Missing Fields:** Record partial evaluation with reduced confidence
2. **Unknown Artifact Types:** Skip evaluation, log warning, continue processing
3. **SQL Errors:** Fail-fast with detailed error context
4. **Template Errors:** Use fallback generic message, log issue

## 4. Idempotency Strategy

### Conceptual Keying (Functional Data Engineering)
**Primary Key:** `(chunk_id, rule_name, ruleset_version)`

**Key Benefits:**
- Same rule version + chunk always produces identical results
- Rule updates create new version, preserving historical results
- Parallel execution safe through unique key constraints

### Overwrite vs Insert Strategy
Following Apache Airflow and dbt patterns:

**Chosen Approach: Partition-Based Overwrite**
1. **Pre-execution:** Calculate target partition: `analysis_date + ruleset_version`
2. **Cleanup:** Delete existing partition data for idempotency  
3. **Insert:** Batch insert all new results for partition
4. **Atomicity:** Use BigQuery transactions for all-or-nothing semantics

### Detection of Stale Outputs
**Staleness Indicators:**
- Results exist for older ruleset_version than current
- Analysis_date older than configured retention period (default: 30 days)
- Orphaned results where source_metadata.chunk_id no longer exists

**Cleanup Strategy:**
```sql
-- Conceptual cleanup logic
DELETE FROM gap_metrics 
WHERE ruleset_version != @current_version 
  AND analysis_date < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
```

### Re-run Semantics
**Safe Re-execution Pattern:**
1. **Version Check:** Compare current vs last execution ruleset_version
2. **If Same Version:** Overwrite existing results (idempotent)
3. **If New Version:** Insert alongside historical results
4. **Audit Trail:** Log all executions with timestamps and row counts

## 5. Data Model Definition (Conceptual)

### gap_metrics Table Columns

| Column | Type | Constraints | Purpose |
|--------|------|-------------|---------|
| `chunk_id` | STRING | NOT NULL, FK to source_metadata | Links to evaluated content |
| `artifact_type` | STRING | NOT NULL, ENUM values | Artifact classification |
| `rule_name` | STRING | NOT NULL | Rule identifier for traceability |
| `passed` | BOOLEAN | NOT NULL | Binary evaluation result |
| `confidence` | FLOAT64 | [0.0, 1.0] | Evaluation certainty score |
| `severity` | INTEGER | [1, 5] | Impact level for failed rules |
| `suggested_fix` | STRING | MAX 500 chars | Actionable remediation guidance |
| `analysis_id` | STRING | NOT NULL | Batch execution identifier |
| `created_at` | TIMESTAMP | NOT NULL, DEFAULT NOW() | Evaluation timestamp |
| `ruleset_version` | STRING | NOT NULL | Rule configuration version |
| `rule_metadata` | JSON | NULLABLE | Additional rule context |
| `partition_date` | DATE | NOT NULL | For BigQuery partitioning |

### Field Semantics and Constraints

**confidence Range Logic:**
- `1.0`: Perfect compliance, all criteria met
- `0.8-0.99`: High compliance, minor gaps
- `0.5-0.79`: Moderate compliance, notable issues
- `0.2-0.49`: Low compliance, significant gaps  
- `0.0-0.19`: Critical gaps, major remediation needed

**severity Escalation Rules:**
- Confidence < 0.2 → Minimum severity 4
- Missing critical fields → Severity 5
- Format violations → Severity 2-3
- Optional improvements → Severity 1

**suggested_fix Content Guidelines:**
- Action-oriented language ("Add", "Update", "Include")
- Specific field names and expected values
- No implementation details or code snippets
- Maximum 500 characters for readability

## 6. Operational Runbook (Conceptual)

### Single Command Entrypoint
**Primary Command:** `make compute_metrics`

**Command Options:**
```bash
make compute_metrics                    # Full evaluation with latest rules
make compute_metrics DRY_RUN=true     # Validation only, no data writes  
make compute_metrics RULES_VERSION=v1.0.0  # Specific version
make compute_metrics ARTIFACT_TYPE=kubernetes  # Single type only
```

### Configuration Validation Step
**Pre-execution Validation:**
1. **Schema Compliance:** Validate all rule files against JSON Schema
2. **SQL Syntax:** Parse BigQuery SQL templates for syntax errors
3. **Version Consistency:** Verify version manifest matches file versions
4. **Dependency Check:** Confirm source_metadata table accessibility
5. **Permissions:** Validate BigQuery write permissions for gap_metrics

### Dry-Run Mode Implementation
**Dry-Run Behavior:**
- Parse and validate all rule configurations
- Execute SQL queries with `LIMIT 0` to check syntax
- Generate sample output schema without data writes
- Report validation results and estimated execution time
- Exit with success/failure status for CI/CD integration

### Logs and Telemetry Expectations
**Structured Logging (JSON Format):**
```json
{
  "timestamp": "2025-01-09T10:00:00Z",
  "level": "INFO", 
  "component": "rule_evaluator",
  "analysis_id": "20250109-100000-abc123",
  "ruleset_version": "v1.2.0",
  "chunk_id": "k8s-deployment-001",
  "rule_name": "k8s_deployment_documentation", 
  "result": "PASSED",
  "confidence": 0.85,
  "execution_time_ms": 245
}
```

**Key Metrics to Track:**
- Rules processed per second
- Confidence score distribution
- Pass/fail rate by artifact type
- Execution time per rule evaluation
- Memory usage and BigQuery slot consumption

### Runtime Expectations
**Performance Envelope:**
- **10K records:** <5 minutes end-to-end
- **100K records:** <30 minutes with parallel processing  
- **Memory:** <2GB peak usage for rule evaluation
- **BigQuery Slots:** <500 concurrent slots during execution

**Rerun Frequency:**
- **Development:** On-demand via `make compute_metrics`
- **Staging:** Daily automated execution
- **Production:** Weekly or on rule configuration changes

## 7. Testing Strategy

### Unit Tests for Rules Engine

**Rule Parsing Tests:**
- Valid YAML configurations parse correctly
- Invalid configurations fail with descriptive errors  
- Version manifest integrity checks
- SQL template parameter validation

**Evaluator Math Tests:**
- Confidence calculation accuracy with known inputs
- Boundary condition handling (0.0, 1.0 confidence)
- Severity mapping consistency
- Suggested fix template rendering

### Golden Samples Per Artifact Type

**Test Data Structure:**
```
tests/
├── golden_samples/
│   ├── kubernetes/
│   │   ├── complete_deployment.yaml     # Should pass all rules
│   │   ├── minimal_deployment.yaml      # Should fail documentation rules
│   │   └── malformed_deployment.yaml    # Should fail validation
│   ├── fastapi/
│   │   ├── documented_endpoint.py       # Complete docstrings
│   │   └── undocumented_endpoint.py     # Missing documentation
│   └── ...other artifact types
└── expected_results/
    ├── kubernetes_expected.json         # Expected gap_metrics output
    └── ...other expectations
```

**Golden Sample Tests:**
- Load sample artifacts into test source_metadata table
- Execute rule evaluation against samples
- Compare actual gap_metrics output with expected results
- Verify confidence scores within acceptable tolerance (±0.05)

### Smoke Test: ≥1 Rule Per Record

**Coverage Validation:**
```sql
-- Test query to verify rule coverage
WITH rule_coverage AS (
  SELECT 
    sm.chunk_id,
    sm.artifact_type,
    COUNT(gm.rule_name) as rules_applied
  FROM source_metadata sm
  LEFT JOIN gap_metrics gm ON sm.chunk_id = gm.chunk_id
  GROUP BY sm.chunk_id, sm.artifact_type
)
SELECT 
  artifact_type,
  COUNT(*) as total_records,
  COUNT(CASE WHEN rules_applied >= 1 THEN 1 END) as covered_records,
  COUNT(CASE WHEN rules_applied = 0 THEN 1 END) as uncovered_records
FROM rule_coverage
GROUP BY artifact_type
```

**Success Criteria:** uncovered_records = 0 for all artifact types

### Idempotency Test Plan

**Test Scenarios:**
1. **Identical Re-run:** Execute same ruleset version twice, verify identical results
2. **Version Upgrade:** Execute newer ruleset, verify historical results preserved  
3. **Partial Failure Recovery:** Simulate failure mid-execution, verify cleanup
4. **Concurrent Execution:** Run multiple instances, verify no race conditions

**Verification Queries:**
```sql
-- Idempotency verification
SELECT 
  chunk_id,
  rule_name,
  COUNT(*) as duplicate_count
FROM gap_metrics 
WHERE ruleset_version = @test_version
GROUP BY chunk_id, rule_name
HAVING COUNT(*) > 1
-- Should return 0 rows
```

## 8. AC Traceability Matrix

| Acceptance Criteria | Design Artifacts | Implementation Evidence |
|---------------------|------------------|------------------------|
| "documented set of rules exists for all artifact types" | **Section 2:** Rule Configuration Design<br/>- File structure per artifact type<br/>- Example rules for kubernetes, fastapi, cobol, irs, mumps | **Deliverables:**<br/>- `configs/rules/{artifact_type}/*.yaml`<br/>- JSON Schema validation<br/>- Version manifest files |
| "Running `make compute_metrics` populates gap_metrics table" | **Section 6:** Operational Runbook<br/>- Single command entrypoint<br/>- Configuration validation<br/>**Section 5:** Data Model Definition | **Deliverables:**<br/>- Makefile target implementation<br/>- gap_metrics table population<br/>- Structured logging output |
| "at least one rule applied per record" | **Section 7:** Testing Strategy<br/>- Smoke test coverage validation<br/>- SQL query for rule coverage verification | **Deliverables:**<br/>- Coverage validation SQL<br/>- Test assertions for 100% coverage<br/>- CI/CD integration checks |
| "Confidence scores are calculated and stored" | **Section 3:** Evaluator Design<br/>- Confidence formula components<br/>- Deterministic calculation logic<br/>**Section 5:** confidence column [0.0, 1.0] | **Deliverables:**<br/>- Confidence calculation implementation<br/>- Unit tests for math accuracy<br/>- gap_metrics.confidence population |
| "rerun without duplicating results" | **Section 4:** Idempotency Strategy<br/>- Conceptual keying strategy<br/>- Partition-based overwrite approach<br/>**Section 7:** Idempotency test plan | **Deliverables:**<br/>- Upsert logic implementation<br/>- Duplicate detection tests<br/>- Re-run safety verification |

**Verification Methods:**
- **Unit Tests:** Automated validation of individual components
- **Integration Tests:** End-to-end pipeline execution with golden samples  
- **Performance Tests:** Runtime envelope validation with load testing
- **Acceptance Tests:** Direct AC validation with stakeholder review

---

**Next Steps:** Implementation prioritization based on AC dependencies and technical complexity assessment.