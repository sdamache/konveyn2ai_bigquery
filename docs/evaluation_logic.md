# Gap Analysis Evaluation Logic

## Overview

This document defines the deterministic pass/fail logic and quality assessment rubric for gap analysis rule evaluation across all artifact types. The evaluation system follows a structured decision tree to ensure consistent, reproducible results.

## Core Evaluation Principles

### 1. Deterministic Processing
- Same inputs always produce identical outputs
- No randomness or time-dependent factors
- Clear tie-breaking rules for edge cases
- Floating-point calculations rounded to 3 decimal places

### 2. Binary Pass/Fail Decision
- **PASS (true)**: All requirements met AND quality thresholds satisfied
- **FAIL (false)**: Any requirement missing OR below quality standards
- No partial credit or gray areas

### 3. Evidence-Based Assessment
- Rules operate on concrete, observable criteria
- Text pattern matching using deterministic regex
- Field presence validation using structured data paths
- Quantifiable quality metrics (length, completeness, format)

## Decision Tree Framework

```
Rule Evaluation Input (chunk_id, rule_config, source_metadata)
    ↓
1. PREREQUISITE CHECK
    ↓
    artifact_type matches rule scope? → NO → SKIP (not evaluated)
    ↓ YES
2. REQUIRED FIELDS CHECK  
    ↓
    All required fields present? → NO → FAIL (confidence penalty applied)
    ↓ YES
3. QUALITY THRESHOLD CHECK
    ↓
    Content meets minimum standards? → NO → FAIL (quality score applied)
    ↓ YES  
4. VALIDATION RULES CHECK
    ↓
    Custom business rules pass? → NO → FAIL (rule-specific scoring)
    ↓ YES
5. PASS (full confidence calculation)
```

## Field Presence Validation

### Required Field Categories

#### Critical Fields (Must be present for PASS)
- **Documentation Fields**: Description, purpose, function explanation
- **Security Fields**: Security contexts, access controls, validation rules
- **Operational Fields**: Resource limits, error handling, monitoring

#### Important Fields (Impact confidence if missing)
- **Metadata Fields**: Labels, annotations, categorization
- **Quality Fields**: Examples, usage guidance, parameter docs
- **Maintenance Fields**: Author, version, review dates

#### Optional Fields (Minor confidence impact)
- **Enhancement Fields**: Additional documentation, cross-references
- **Formatting Fields**: Consistent style, organization structure

### Validation Logic by Artifact Type

#### Kubernetes Resources
```sql
-- Example: Deployment Documentation Rule
WITH field_check AS (
  SELECT 
    chunk_id,
    -- Critical: Description present and meaningful
    CASE WHEN JSON_EXTRACT_SCALAR(metadata, '$.annotations.description') IS NOT NULL 
         AND LENGTH(JSON_EXTRACT_SCALAR(metadata, '$.annotations.description')) > 10
         THEN 1 ELSE 0 END as has_description,
    
    -- Critical: Resource limits defined for containers
    CASE WHEN JSON_EXTRACT_SCALAR(metadata, '$.spec.template.spec.containers[0].resources.limits') IS NOT NULL
         THEN 1 ELSE 0 END as has_resource_limits,
    
    -- Important: Proper labeling for organization
    CASE WHEN JSON_EXTRACT_SCALAR(metadata, '$.labels.app') IS NOT NULL
         THEN 1 ELSE 0 END as has_app_label
  FROM source_metadata 
  WHERE artifact_type = 'kubernetes' AND kind = 'Deployment'
)
SELECT 
  chunk_id,
  -- Pass only if ALL critical fields present
  (has_description = 1 AND has_resource_limits = 1) as passed
FROM field_check
```

#### FastAPI Endpoints
```sql
-- Example: Endpoint Documentation Rule  
WITH content_analysis AS (
  SELECT 
    chunk_id,
    -- Critical: Docstring present with adequate content
    CASE WHEN REGEXP_CONTAINS(text_content, r'"""[\s\S]*?"""') 
         AND LENGTH(REGEXP_EXTRACT(text_content, r'"""([\s\S]*?)"""')) > 50
         THEN 1 ELSE 0 END as has_adequate_docstring,
    
    -- Critical: Response model defined
    CASE WHEN REGEXP_CONTAINS(text_content, r'response_model\s*=')
         THEN 1 ELSE 0 END as has_response_model,
    
    -- Important: HTTP status codes documented
    CASE WHEN REGEXP_CONTAINS(text_content, r'status_code\s*=\s*[0-9]{3}')
         THEN 1 ELSE 0 END as has_status_code
  FROM source_metadata 
  WHERE artifact_type = 'fastapi' AND api_path LIKE '%endpoint%'
)
SELECT 
  chunk_id,
  -- Pass if critical documentation present
  (has_adequate_docstring = 1 AND has_response_model = 1) as passed
FROM content_analysis
```

## Quality Threshold Assessment

### Content Quality Metrics

#### 1. Length-Based Quality
```
quality_length_score = min(1.0, actual_length / minimum_required_length)

Thresholds by content type:
- Descriptions: minimum 10 characters
- Docstrings: minimum 50 characters  
- Comments: minimum 5 characters
- Documentation blocks: minimum 100 characters
```

#### 2. Structure-Based Quality
```
quality_structure_score = (
  has_proper_formatting * 0.4 +
  has_consistent_style * 0.3 +
  has_required_sections * 0.3
)

Structure Requirements:
- Proper indentation and formatting
- Consistent naming conventions
- Required sections present (e.g., Purpose, Parameters, Returns)
```

#### 3. Vocabulary-Based Quality
```
quality_vocabulary_score = (
  uses_domain_terminology * 0.5 +
  avoids_placeholder_text * 0.3 +
  has_specific_details * 0.2
)

Vocabulary Checks:
- Domain-specific terms present (Kubernetes: "pod", "service"; FastAPI: "endpoint", "schema")
- No placeholder text ("TODO", "FIXME", "TBD")
- Specific rather than generic descriptions
```

## Custom Business Rules

### Rule-Specific Validation Logic

#### Security-Focused Rules
```
security_compliance_check = (
  has_security_context AND
  runs_as_non_root AND
  no_privilege_escalation AND
  readonly_root_filesystem
)

-- If any security requirement fails → FAIL with severity 4-5
```

#### Compliance-Focused Rules (IRS, Healthcare)
```
compliance_documentation_check = (
  has_field_descriptions AND
  has_validation_rules AND
  has_data_classification AND
  has_retention_policy
)

-- Missing compliance docs → FAIL with severity 4-5
```

#### Operational-Focused Rules
```
operational_readiness_check = (
  has_resource_limits AND
  has_health_checks AND
  has_monitoring_config AND
  has_error_handling
)

-- Production readiness gaps → FAIL with severity 3-4
```

## Error Handling Logic

### Missing Fields Strategy
1. **Critical Field Missing**: Immediate FAIL, confidence = 0.0
2. **Important Field Missing**: FAIL, confidence reduced by 50%
3. **Optional Field Missing**: PASS possible, confidence reduced by 10%

### Unknown Artifact Types
1. **Rule doesn't apply**: Skip evaluation (no result generated)
2. **Artifact type mismatch**: Log warning, skip evaluation
3. **Invalid content**: FAIL with confidence = 0.0, severity = 1

### Malformed Content
1. **Unparseable JSON/YAML**: Extract what's possible, reduce confidence
2. **Invalid regex patterns**: Use fallback patterns, log issue
3. **Encoding issues**: Attempt best-effort parsing, flag for review

## Tie-Breaking Rules

### Multiple Rules for Same Content
1. **Same rule, different versions**: Use latest rule version
2. **Overlapping rules**: Apply all applicable rules separately
3. **Conflicting severity**: Use highest severity level
4. **Confidence conflicts**: Average confidence scores

### Edge Cases
1. **Empty content**: FAIL with confidence = 0.0, severity = 1
2. **Content exactly at threshold**: Round up to PASS (inclusive thresholds)
3. **Floating point precision**: Round to 3 decimal places for consistency

## Confidence Calculation Framework

### Base Completeness Score
```
base_completeness = (
  required_fields_present / total_required_fields
)
```

### Quality Multiplier
```
quality_multiplier = (
  content_length_score * 0.4 +
  structure_score * 0.3 +
  vocabulary_score * 0.3
)
```

### Penalty Deductions
```
penalties = (
  critical_missing_fields * 0.20 +  # Max 0.20 per critical field
  security_violations * 0.15 +      # Security issues
  compliance_gaps * 0.10            # Regulatory gaps
)
```

### Final Confidence Formula
```
confidence = max(0.0, min(1.0, 
  base_completeness * quality_multiplier - penalties
))
```

## Validation Examples

### Example 1: Kubernetes Deployment (PASS)
```yaml
# Input
metadata:
  annotations:
    description: "Nginx web server for production traffic handling"
  labels:
    app: "nginx"
spec:
  template:
    spec:
      containers:
      - resources:
          limits:
            memory: "512Mi"
            cpu: "500m"

# Evaluation
has_description: true (length=52 > 10)
has_resource_limits: true  
has_app_label: true

# Result
passed: true
confidence: 0.95 (high completeness + good quality)
severity: 3 (rule-defined baseline)
```

### Example 2: FastAPI Endpoint (FAIL)
```python
# Input
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user": user_id}

# Evaluation  
has_adequate_docstring: false (no docstring)
has_response_model: false (no response_model)
has_status_code: false (no explicit status)

# Result
passed: false
confidence: 0.1 (critical elements missing)
severity: 2 (documentation gap)
suggested_fix: "Add comprehensive docstring and response_model parameter"
```

### Example 3: COBOL Copybook (PARTIAL)
```cobol
# Input
01 CUSTOMER-RECORD.
   05 CUSTOMER-ID    PIC 9(10).
   05 CUSTOMER-NAME  PIC X(30).

# Evaluation
has_record_structure: true (01-level present)
has_pic_comments: false (no PIC descriptions)  
has_field_descriptions: false (no business meaning)

# Result
passed: false
confidence: 0.33 (structure present but documentation missing)
severity: 3 (operational impact)
suggested_fix: "Add PIC clause comments and field descriptions with business meaning"
```

## Performance Considerations

### Optimization Strategies
1. **Batch Processing**: Evaluate multiple chunks in single SQL query
2. **Parallel Execution**: Process different artifact types concurrently  
3. **Caching**: Cache compiled regex patterns and JSON paths
4. **Indexing**: Optimize BigQuery queries with proper clustering

### Scalability Targets
- **10K records**: <5 minutes total processing time
- **100K records**: <30 minutes with parallel processing
- **Memory usage**: <2GB peak for rule evaluation engine
- **BigQuery slots**: <500 concurrent slots during execution

This evaluation logic ensures consistent, auditable gap analysis results while maintaining the flexibility to accommodate different artifact types and evolving documentation standards.