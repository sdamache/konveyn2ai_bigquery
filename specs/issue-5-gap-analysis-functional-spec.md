# Functional Specification: Issue #5 - Gap Analysis Rules & Metrics

**Reference:** Issue #5 "M3: Define coverage and confidence metrics for knowledge gaps"

## 1. Problem & Goals

**Objective:** Create a deterministic, configurable rules system that evaluates documentation completeness across all artifact types (Kubernetes, FastAPI, COBOL, IRS, MUMPS) and generates quantified gap metrics for downstream visualization consumption.

**User-Visible Goal:** Analysts and engineers can identify insufficient documentation/metadata through transparent, reproducible scoring that powers heat maps and progress tracking without requiring code changes.

## 2. User Stories (Stakeholders)

### Analyst validating documentation completeness
- **As a** data analyst reviewing system documentation coverage
- **I want** transparent gap metrics with confidence scores and severity levels
- **So that** I can identify critical documentation gaps and prioritize improvement efforts

### Engineer adjusting rules without code changes  
- **As a** platform engineer maintaining documentation standards
- **I want** configurable rules that I can modify through configuration files
- **So that** I can adapt evaluation criteria without deployment cycles

### Reviewer re-running metrics safely without duplication
- **As a** team lead conducting periodic documentation reviews
- **I want** idempotent metric computation that replaces previous results
- **So that** I can safely re-run analysis without data corruption or duplicates

## 3. Scope (In/Out)

### In Scope
- **Rules catalog:** Complete rule definitions for all artifact types (Kubernetes, FastAPI, COBOL, IRS, MUMPS)
- **Evaluator behavior:** Deterministic pass/fail logic with confidence scoring
- **gap_metrics fields:** chunk_id, artifact_type, rule_name, passed, confidence, severity, suggested_fix semantics
- **Idempotent re-runs:** Safe metric recomputation without duplication
- **Confidence/severity definitions:** Standardized 0-1 confidence scale and 1-5 severity rubric
- **Suggested_fix semantics:** Actionable guidance for remediation

### Out of Scope
- **Visualization dashboards:** Frontend heat map rendering
- **Progress trend reporting:** Historical analysis and trending
- **Infrastructure/CI specifics:** Build pipeline integration details

## 4. Data Contracts

### Input Contract: source_metadata
```sql
-- Required fields from normalized source_metadata
chunk_id: STRING (unique identifier)
artifact_type: STRING (kubernetes|fastapi|cobol|irs|mumps)  
text_content: STRING (raw content for evaluation)
kind: STRING (resource type - Deployment, Service, etc.)
api_path: STRING (endpoint paths, file paths)
metadata: JSON (extracted structured data)
```

### Output Contract: gap_metrics
```sql
-- Output schema with stable semantics
chunk_id: STRING (links to source_metadata.chunk_id)
artifact_type: STRING (matches source_metadata.artifact_type)
rule_name: STRING (unique rule identifier)
passed: BOOLEAN (binary evaluation result)
confidence: FLOAT64 (0.0-1.0 completeness score)
severity: INTEGER (1-5 impact scale)
suggested_fix: STRING (actionable remediation guidance)
analysis_id: STRING (batch execution identifier)
created_at: TIMESTAMP (evaluation timestamp)
ruleset_version: STRING (rule configuration version hash)
```

**Idempotency Key:** `(chunk_id, rule_name, ruleset_version)` - conceptual uniqueness constraint

## 5. Rules Model

### Rule Structure
```yaml
rule_name: "kubernetes_deployment_description"
artifact_type: "kubernetes" 
required_fields: ["metadata.description", "spec.template.metadata.labels"]
evaluation_logic: "Check presence of description annotation and required labels"
severity: 3  # 1=low, 5=critical
confidence_factors:
  - field_completeness: 0.6  # weight for required fields present
  - content_quality: 0.4     # weight for content length/structure
suggested_fix_template: "Add description annotation: metadata.annotations.description"
```

### Configurability Requirements
- **Human-editable:** YAML/JSON configuration files in version control
- **Versioned:** Configuration changes tracked with semantic versioning
- **Auditable:** Rule modifications logged with timestamps and authors

### Artifact Type Coverage
- **Kubernetes:** Description annotations, resource limits, label standards
- **FastAPI:** Endpoint docstrings, parameter documentation, response schemas  
- **COBOL:** PIC clause comments, data item descriptions, section documentation
- **IRS:** Record layout field descriptions, format specifications
- **MUMPS:** FileMan field definitions, routine documentation, input templates

## 6. Evaluation Semantics

### Binary Pass Definition
- **PASSED (true):** All required fields present AND meet quality thresholds
- **FAILED (false):** Any required field missing OR below quality thresholds

### Confidence Scoring [0,1]
```
confidence = (completeness_score * 0.6) + (quality_score * 0.4) - penalties

completeness_score = (fields_present / fields_required)
quality_score = min(1.0, content_length / min_length_threshold)  
penalties = missing_critical_fields * 0.2  # max 0.2 penalty per critical field
```

### Severity Rubric [1-5]
- **1 (Low):** Minor documentation improvements
- **2 (Moderate):** Missing non-critical descriptions  
- **3 (Medium):** Missing required fields or incomplete specs
- **4 (High):** Critical metadata absent, impacts operations
- **5 (Critical):** Fundamental documentation missing, blocks understanding

### Suggested Fix Guidance
- **Format:** Action-oriented instructions with specific field names
- **Examples:** "Add description annotation: metadata.annotations.description"
- **Constraints:** Maximum 200 characters, no implementation details

### Determinism Requirements
- **Reproducible:** Same inputs always produce identical outputs
- **Tie-breaking:** When multiple rules overlap, apply severity-ordered evaluation
- **Floating point:** Round confidence to 3 decimal places for consistency

## 7. Idempotency & Re-Run Behavior

### Re-Run Logic
1. **Pre-execution:** Generate analysis_id and ruleset_version hash
2. **Deletion:** Remove existing rows matching `(chunk_id, rule_name, ruleset_version)`
3. **Insertion:** Insert new evaluation results
4. **Atomicity:** Entire operation succeeds or fails as unit

### Duplicate Prevention
- **Conceptual Key:** `(chunk_id, rule_name, ruleset_version)` prevents duplicates
- **Version Tracking:** Different rule versions create separate result sets
- **Cleanup:** Old rule versions can be archived based on retention policy

### Safe Re-execution
- `make compute_metrics` can be run multiple times safely
- Previous results for same rule version are replaced, not appended
- Partial failures leave previous results intact

## 8. Acceptance Criteria Mapping

| AC Requirement | Specification Clause |
|----------------|---------------------|
| "documented set of rules exists for all artifact types" | Section 5: Rules Model + Artifact Type Coverage |
| "Running `make compute_metrics` populates gap_metrics table" | Section 4: Data Contracts + Section 7: Re-Run Behavior |
| "at least one rule applied per record" | Section 9: Success Metrics - Coverage requirement |
| "Confidence scores are calculated and stored" | Section 6: Evaluation Semantics - Confidence Scoring |
| "rerun without duplicating results" | Section 7: Idempotency & Re-Run Behavior |

## 9. Success Metrics

### Coverage Requirements
- **Artifact Type Coverage:** 100% of artifact types have ≥1 rule defined
- **Record Coverage:** ≥1 rule applied per source_metadata record
- **Rule Application:** Every gap_metrics row has valid confidence score (0.0-1.0)

### Quality Thresholds  
- **Confidence Distribution:** <10% of records with confidence <0.3
- **Severity Balance:** Each severity level (1-5) represented in results
- **Suggested Fix Quality:** 100% of failed rules include actionable suggested_fix

### Operational Requirements
- **Re-run Safety:** Zero duplicate rows after repeated execution
- **Performance:** Complete metric computation in <5 minutes for 10K records
- **Auditability:** Every gap_metrics row traceable to specific rule version

---

**Implementation Command:** `make compute_metrics`
**Dependencies:** source_metadata table populated, rules configuration deployed
**Output:** Populated gap_metrics table ready for visualization consumption