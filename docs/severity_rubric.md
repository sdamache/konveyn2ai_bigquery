# Severity Rubric: 5-Level Impact Assessment

## Overview

This document defines the standardized severity rubric for gap analysis results, following the industry-standard 5×5 risk matrix approach used in healthcare, regulatory, and enterprise systems. The rubric provides consistent impact assessment across all artifact types and rule categories.

## Severity Scale Definition

### Level 1: Low Impact
**Label:** `Low`  
**Numeric Value:** `1`  
**Risk Color:** `Green`

#### Impact Description
Cosmetic issues and minor documentation improvements that have no operational impact. These are "nice-to-have" enhancements that improve code quality and maintainability but don't affect functionality or safety.

#### Operational Characteristics
- **Service Impact:** None
- **User Impact:** None  
- **Developer Impact:** Minor convenience
- **Remediation Urgency:** When time permits
- **Business Risk:** Negligible

#### Example Scenarios

**Kubernetes:**
- Missing optional annotations (owner, team, cost-center)
- Inconsistent label formatting
- Absent documentation links in ConfigMaps
- Non-standard naming conventions that still function

**FastAPI:**
- Missing examples in response schemas
- Absent optional parameter descriptions
- Inconsistent docstring formatting
- Missing type hints for optional parameters

**COBOL:**
- Missing optional comment headers
- Inconsistent spacing in copybooks
- Absent author information
- Non-standard but functional naming

**IRS/Compliance:**
- Missing non-required field explanations
- Absent formatting guidelines
- Optional documentation sections

**MUMPS/Healthcare:**
- Missing routine version information
- Absent optional help prompts
- Non-critical documentation gaps

#### Remediation Examples
- "Add optional description annotation for better documentation"
- "Include example values in API schema for developer convenience"
- "Add author and date comments to copybook header"

---

### Level 2: Moderate Impact
**Label:** `Moderate`  
**Numeric Value:** `2`  
**Risk Color:** `Yellow`

#### Impact Description
Minor functionality gaps and incomplete parameter documentation that cause minor inconvenience to developers or operators. These issues don't break systems but reduce efficiency and increase learning curve.

#### Operational Characteristics
- **Service Impact:** Minor inconvenience
- **User Impact:** Slight confusion possible
- **Developer Impact:** Increased time to understand/use
- **Remediation Urgency:** Next sprint/iteration
- **Business Risk:** Low

#### Example Scenarios

**Kubernetes:**
- Missing probe configurations (but defaults work)
- Incomplete resource request specifications
- Absent troubleshooting annotations
- Missing but non-critical environment variables

**FastAPI:**
- Incomplete parameter documentation
- Missing HTTP status code documentation
- Absent request/response examples
- Incomplete error message descriptions

**COBOL:**
- Missing procedure purpose comments
- Incomplete parameter documentation
- Absent business rule explanations
- Missing data validation descriptions

**IRS/Compliance:**
- Incomplete field usage guidance
- Missing validation rule explanations
- Absent cross-reference documentation

**MUMPS/Healthcare:**
- Missing input parameter documentation
- Incomplete routine purpose descriptions
- Absent usage examples

#### Confidence Correlation
- Typically associated with confidence scores: 0.4-0.7
- May be escalated to Level 3 if confidence < 0.3

#### Remediation Examples
- "Add comprehensive docstring explaining all parameters and return values"
- "Include HTTP status codes and error response documentation"
- "Add PURPOSE comment explaining business logic and data flow"

---

### Level 3: Medium Impact  
**Label:** `Medium`  
**Numeric Value:** `3`  
**Risk Color:** `Orange`

#### Impact Description
Operational impact issues including missing required fields or incomplete specifications that may cause confusion, inefficiency, or minor operational problems. These gaps affect system maintainability and operational effectiveness.

#### Operational Characteristics
- **Service Impact:** May cause operational confusion
- **User Impact:** Potential for errors or misunderstanding
- **Developer Impact:** Significant time spent researching/debugging
- **Remediation Urgency:** Current sprint/month
- **Business Risk:** Medium

#### Example Scenarios

**Kubernetes:**
- Missing resource limits (pods may consume excess resources)
- Absent health check endpoints
- Missing service documentation
- Incomplete network policy specifications

**FastAPI:**
- Missing input validation descriptions
- Absent error handling documentation
- Incomplete authentication requirements
- Missing rate limiting specifications

**COBOL:**
- Missing critical data item descriptions
- Absent file handling procedures
- Incomplete error handling logic
- Missing data transformation rules

**IRS/Compliance:**
- Missing required field descriptions
- Absent validation criteria
- Incomplete audit trail specifications
- Missing data retention requirements

**MUMPS/Healthcare:**
- Missing critical field definitions in FileMan
- Absent data validation rules
- Incomplete cross-reference documentation
- Missing integration specifications

#### Confidence Correlation
- Typically associated with confidence scores: 0.2-0.6
- Default severity for many rule failures
- May be escalated to Level 4 if multiple critical elements missing

#### Remediation Examples
- "Add resource limits to prevent resource starvation"
- "Include comprehensive error handling and validation documentation"
- "Add critical field descriptions with business impact explanation"

---

### Level 4: High Impact
**Label:** `High`  
**Numeric Value:** `4`  
**Risk Color:** `Red`

#### Impact Description
Service degradation risk where critical metadata is absent, creating potential for operational issues, security vulnerabilities, or compliance violations. These gaps pose significant risk to system reliability and safety.

#### Operational Characteristics
- **Service Impact:** Risk of service degradation or failure
- **User Impact:** Potential security exposure or data loss
- **Developer Impact:** High risk of incorrect implementation
- **Remediation Urgency:** Immediate (within days)
- **Business Risk:** High

#### Example Scenarios

**Kubernetes:**
- Missing security contexts (containers may run as root)
- Absent network policies (open network access)
- Missing admission controller configurations
- Absent RBAC specifications

**FastAPI:**
- Missing authentication/authorization documentation
- Absent input sanitization specifications
- Missing rate limiting configurations
- Incomplete security header documentation

**COBOL:**
- Missing critical data validation rules
- Absent security access controls
- Missing data encryption specifications
- Incomplete audit logging requirements

**IRS/Compliance:**
- Missing PII identification and handling
- Absent data classification specifications
- Missing compliance validation rules
- Incomplete audit requirements

**MUMPS/Healthcare:**
- Missing patient data protection rules
- Absent HIPAA compliance documentation
- Missing critical data access controls
- Incomplete audit trail specifications

#### Automatic Escalation Rules
- **Confidence-Based:** Any rule with confidence < 0.2 → minimum severity 4
- **Security Keywords:** References to authentication, authorization, encryption → severity 4
- **Compliance Contexts:** PII, PHI, financial data → severity 4
- **Critical Operations:** Data deletion, system configuration → severity 4

#### Remediation Examples
- "Add security context with runAsNonRoot: true and readOnlyRootFilesystem: true"
- "Include authentication requirements and authorization matrix"
- "Add critical data validation rules and security access controls"

---

### Level 5: Critical Impact
**Label:** `Critical`  
**Numeric Value:** `5`  
**Risk Color:** `Dark Red`

#### Impact Description
System failure risk where fundamental documentation is missing, creating high probability of system failure, security breach, or severe compliance violation. These are show-stopper issues requiring immediate attention.

#### Operational Characteristics
- **Service Impact:** High risk of system failure or breach
- **User Impact:** Potential data corruption or unauthorized access
- **Developer Impact:** Cannot safely implement without information
- **Remediation Urgency:** Immediate (within hours)
- **Business Risk:** Critical

#### Example Scenarios

**Kubernetes:**
- Missing admission controller security policies
- Absent Pod Security Standards enforcement
- Missing network segmentation requirements
- Absent backup and disaster recovery procedures

**FastAPI:**
- Missing authentication system entirely
- Absent SQL injection prevention measures
- Missing critical business logic validation
- Absent data encryption specifications

**COBOL:**
- Missing critical business logic documentation
- Absent data integrity validation
- Missing transaction rollback procedures
- Absent critical error handling

**IRS/Compliance:**
- Missing regulatory compliance validation
- Absent critical audit requirements
- Missing data breach response procedures
- Absent required regulatory reporting

**MUMPS/Healthcare:**
- Missing patient safety validation rules
- Absent critical clinical decision support logic
- Missing drug interaction checking requirements
- Absent emergency access procedures

#### Automatic Escalation Rules
- **Critical Missing Fields:** Any field marked as "critical" in rule configuration
- **Safety Systems:** Healthcare, financial, safety-critical applications
- **Security Fundamentals:** Authentication, encryption, access control
- **Regulatory Requirements:** SOX, HIPAA, PCI-DSS, GDPR compliance
- **Data Protection:** PII/PHI handling, data classification

#### Remediation Examples
- "Add comprehensive security policies and admission controller configurations"
- "Implement complete authentication and authorization system documentation"
- "Add critical patient safety validation rules and emergency procedures"

## Severity Assignment Algorithm

### Primary Assignment Rules

1. **Rule-Defined Baseline**
   - Each rule specifies a default severity level
   - Based on the type of requirement and business impact

2. **Confidence-Based Escalation**
   ```
   IF confidence < 0.2 THEN severity = max(severity, 4)
   IF confidence < 0.1 THEN severity = 5
   ```

3. **Context-Based Escalation**
   ```
   IF content contains security keywords THEN severity = max(severity, 4)
   IF content contains compliance keywords THEN severity = max(severity, 4)
   IF content contains safety keywords THEN severity = max(severity, 5)
   ```

4. **Field-Based Escalation**
   ```
   IF critical_fields_missing > 0 THEN severity = max(severity, 4)
   IF critical_fields_missing > 2 THEN severity = 5
   ```

### Context Keywords for Escalation

#### Security Keywords (→ severity 4+)
- authentication, authorization, encrypt, decrypt, token, credential
- password, secret, key, certificate, signature, hash
- permission, access, role, privilege, audit, log
- security, vulnerability, attack, breach, threat

#### Compliance Keywords (→ severity 4+)
- PII, PHI, GDPR, HIPAA, SOX, PCI, compliance, regulation
- audit, log, trace, monitor, report, privacy
- retention, classification, protection, handling

#### Safety Keywords (→ severity 5)
- patient, safety, critical, emergency, life, death
- drug, medication, dosage, allergy, interaction
- alarm, alert, warning, failure, backup, recovery

### Severity Distribution Guidelines

#### Target Distribution (Healthy System)
- **Level 1 (Low):** 40-50% of issues
- **Level 2 (Moderate):** 25-35% of issues  
- **Level 3 (Medium):** 15-25% of issues
- **Level 4 (High):** 5-10% of issues
- **Level 5 (Critical):** 0-5% of issues

#### Warning Indicators
- **Too many Level 5:** May indicate over-classification or serious systemic issues
- **Too many Level 1:** May indicate under-classification or excellent documentation
- **No Level 4-5:** May indicate insufficient security/compliance rule coverage

## Integration with Confidence Scoring

### Confidence-Severity Correlation Matrix

| Confidence Range | Default Severity | Escalation Rules |
|------------------|------------------|------------------|
| 0.9 - 1.0        | Rule baseline    | No escalation    |
| 0.7 - 0.89       | Rule baseline    | +1 if critical content |
| 0.5 - 0.69       | Rule baseline + 1| +1 if security content |
| 0.3 - 0.49       | Rule baseline + 1| +2 if compliance content |
| 0.2 - 0.29       | Minimum level 4  | +1 for safety content |
| 0.0 - 0.19       | Level 5          | No further escalation |

### Severity Validation Rules

1. **Consistency Check**
   ```sql
   -- High severity should correlate with low confidence
   SELECT COUNT(*) FROM gap_metrics 
   WHERE severity >= 4 AND confidence > 0.8
   -- Should be minimal
   ```

2. **Distribution Check**
   ```sql
   -- Check severity distribution per artifact type
   SELECT artifact_type, severity, COUNT(*) as count,
          ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY artifact_type), 1) as percentage
   FROM gap_metrics 
   GROUP BY artifact_type, severity 
   ORDER BY artifact_type, severity
   ```

3. **Escalation Audit**
   ```sql
   -- Verify automatic escalations are working
   SELECT rule_name, AVG(severity) as avg_severity, AVG(confidence) as avg_confidence
   FROM gap_metrics 
   WHERE confidence < 0.2
   GROUP BY rule_name
   HAVING AVG(severity) < 4  -- Should be empty
   ```

## Usage in Downstream Systems

### Heat Map Visualization
- **Color Coding:** Map severity levels to visual intensity
- **Priority Filtering:** Allow filtering by severity level
- **Trend Analysis:** Track severity distribution over time

### Alerting and Notifications
- **Level 5:** Immediate alerts to security/compliance teams
- **Level 4:** Daily digest to development teams
- **Level 3:** Weekly reports to product owners
- **Level 1-2:** Monthly quality reports

### Remediation Prioritization
1. **Critical (5):** Drop everything, fix immediately
2. **High (4):** Current sprint, security review
3. **Medium (3):** Next sprint, architectural review
4. **Moderate (2):** Backlog, quality improvement
5. **Low (1):** Technical debt, when time permits

This severity rubric ensures consistent risk assessment across all gap analysis results while providing clear guidance for remediation prioritization and resource allocation.