# Feature Specification: M1 - Parse and Ingest Multi-Source Artifacts

**Feature Branch**: `002-m1-parse-and`
**Created**: 2025-09-14
**Status**: Draft
**Input**: User description: "M1: Parse and ingest Kubernetes, FastAPI, COBOL, IRS, and MUMPS artifacts"

## Execution Flow (main)
```
1. Parse user description from Input
   ’ Milestone 1: Multi-source artifact ingestion pipeline
2. Extract key concepts from description
   ’ Actors: Data engineers, AI system
   ’ Actions: Parse, chunk, normalize, ingest
   ’ Data: K8s manifests, FastAPI specs, COBOL copybooks, IRS layouts, MUMPS globals
   ’ Constraints: Idempotent writes, shared schemas, e100 rows per source
3. Clear requirements - no ambiguities identified
4. Fill User Scenarios & Testing section
   ’ Clear ingestion workflow for each source type
5. Generate Functional Requirements
   ’ All requirements are testable and measurable
6. Identify Key Entities (source_metadata table schema)
7. Run Review Checklist
   ’ SUCCESS - all requirements clear and testable
8. Return: SUCCESS (spec ready for planning)
```

---

## ¡ Quick Guidelines
-  Focus on WHAT data needs to be ingested and WHY
- L Avoid HOW to implement (no tech stack, APIs, code structure)
- =e Written for business stakeholders and data engineers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a data engineer, I need to ingest knowledge artifacts from multiple legacy and modern systems (Kubernetes, FastAPI, COBOL, IRS, MUMPS) into a unified BigQuery schema so that downstream AI systems can perform embedding generation and gap analysis on a consistent data corpus.

### Acceptance Scenarios
1. **Given** a Kubernetes cluster with YAML/JSON resources, **When** the K8s ingestion script runs, **Then** each deployment, service, and resource is captured as a separate record with metadata (kind, api_path, namespace)
2. **Given** a FastAPI project with OpenAPI spec, **When** the FastAPI ingestion script runs, **Then** each endpoint and Pydantic model is chunked separately with metadata (http_method, route_path, model_names)
3. **Given** COBOL copybooks with .cpy/.cbl files, **When** the COBOL ingestion script runs, **Then** each 01-level data structure is captured with field names and PIC clauses
4. **Given** IRS IMF copybook definitions, **When** the IRS ingestion script runs, **Then** record layouts are extracted with start positions, lengths, and field types
5. **Given** MUMPS/VistA FileMan dictionaries, **When** the MUMPS ingestion script runs, **Then** each FileMan file and global node is captured with file numbers and field numbers
6. **Given** any ingestion script runs multiple times, **When** processing the same source data, **Then** records are upserted idempotently without duplication

### Edge Cases
- What happens when source files are malformed or missing required metadata?
- How does the system handle partial ingestion failures across multiple source types?
- What occurs when BigQuery schema changes between ingestion runs?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST parse Kubernetes YAML/JSON manifests and extract deployments, services, and resources as separate records
- **FR-002**: System MUST parse FastAPI OpenAPI specifications and source code to extract endpoint definitions and Pydantic models
- **FR-003**: System MUST parse COBOL copybooks (.cpy/.cbl) and split on 01-level data structures with field metadata
- **FR-004**: System MUST parse IRS IMF copybook definitions and extract record layouts with positioning data
- **FR-005**: System MUST parse MUMPS globals and FileMan dictionaries to capture file and field structures
- **FR-006**: System MUST normalize all extracted chunks into the shared source_metadata table schema
- **FR-007**: System MUST write at least 100 sample rows per source family for demonstration purposes
- **FR-008**: System MUST perform idempotent upsert operations to prevent duplicate records
- **FR-009**: System MUST tag each record with appropriate source identifiers and artifact type classifications
- **FR-010**: System MUST provide independent make targets for each source type (make ingest_k8s, make ingest_fastapi, etc.)

### Chunking Strategies
- **Kubernetes**: One chunk per resource (deployment, service, configmap, etc.) with full YAML/JSON content
- **FastAPI**: Separate chunks for each endpoint handler and each Pydantic model definition
- **COBOL**: One chunk per 01-level data structure, capturing hierarchical field relationships
- **IRS**: One chunk per record layout section, preserving field positioning and type information
- **MUMPS**: One chunk per FileMan file definition and one per top-level global node structure

### Metadata Fields Per Source
- **Common Fields**: source_id, artifact_type, content_text, created_at, updated_at
- **Kubernetes**: kind, api_version, namespace, resource_name, labels
- **FastAPI**: http_method, route_path, model_names, operation_id, tags
- **COBOL**: structure_level, field_names, pic_clauses, occurs_count
- **IRS**: start_position, field_length, data_type, record_section
- **MUMPS**: file_number, field_number, global_name, data_type

### Idempotent Write Pattern
- **Primary Key**: Composite key using source_type + source_id + artifact_hash
- **Upsert Logic**: INSERT ON DUPLICATE KEY UPDATE for changed content only
- **Change Detection**: Content hash comparison to identify modifications
- **Timestamp Tracking**: created_at (immutable) and updated_at (on content changes)

### Key Entities *(include if feature involves data)*
- **source_metadata**: Central table storing normalized chunks from all source types with common schema fields and source-specific metadata
- **ingestion_log**: Tracking table for ingestion run status, row counts, and error reporting per source type
- **artifact_hash**: Content-based hash for change detection and idempotent operations

## Test Plan

### Unit Testing
- **Parser Tests**: Validate each source type parser handles valid and malformed input correctly
- **Schema Tests**: Verify all extracted metadata conforms to source_metadata table schema
- **Chunking Tests**: Confirm chunking strategies produce expected record counts and content boundaries

### Integration Testing
- **End-to-End Tests**: Run full ingestion pipeline from sample source files to BigQuery insertion
- **Idempotency Tests**: Verify repeated ingestion runs produce identical results without duplicates
- **Schema Compatibility**: Test ingestion against shared schemas from Issue #1

### Acceptance Testing
- **Row Count Validation**: Confirm e100 rows generated per source family
- **Make Target Tests**: Verify all make commands (ingest_k8s, ingest_fastapi, etc.) execute successfully
- **Cross-Source Tests**: Validate all five source types can be ingested simultaneously without conflicts

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (e100 rows per source)
- [x] Scope is clearly bounded (5 specific source types)
- [x] Dependencies identified (shared schemas from Issue #1)

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none identified)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed