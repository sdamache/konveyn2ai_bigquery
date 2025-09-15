# Data Model: M1 Multi-Source Ingestion

## Core Entities

### source_metadata (Primary Table)
Central table storing normalized chunks from all source types with unified schema.

**Common Fields:**
- `source_type` (STRING, REQUIRED) - Source family: 'kubernetes', 'fastapi', 'cobol', 'irs', 'mumps'
- `artifact_id` (STRING, REQUIRED) - Deterministic semantic identifier
- `parent_id` (STRING, NULLABLE) - Parent artifact for hierarchical relationships
- `parent_type` (STRING, NULLABLE) - Type of parent artifact
- `content_text` (STRING, REQUIRED) - Chunk content for embedding, max 16KB
- `content_tokens` (INTEGER, NULLABLE) - Token count for the content
- `content_hash` (STRING, REQUIRED) - SHA256 hash for idempotency
- `created_at` (TIMESTAMP, REQUIRED) - First ingestion timestamp
- `updated_at` (TIMESTAMP, REQUIRED) - Last modification timestamp
- `collected_at` (TIMESTAMP, REQUIRED) - Source collection timestamp
- `source_uri` (STRING, REQUIRED) - Original source location
- `repo_ref` (STRING, NULLABLE) - Git commit SHA or image tag
- `tool_version` (STRING, REQUIRED) - Parser version for reproducibility

**Kubernetes Fields:**
- `k8s_api_version` (STRING, NULLABLE) - e.g., "apps/v1"
- `k8s_kind` (STRING, NULLABLE) - Resource type: "Deployment", "Service"
- `k8s_namespace` (STRING, NULLABLE) - Kubernetes namespace
- `k8s_resource_name` (STRING, NULLABLE) - Resource name
- `k8s_labels` (JSON, NULLABLE) - Labels as JSON object
- `k8s_annotations` (JSON, NULLABLE) - Annotations as JSON object
- `k8s_resource_version` (STRING, NULLABLE) - Kubernetes resource version

**FastAPI Fields:**
- `fastapi_http_method` (STRING, NULLABLE) - "GET", "POST", "PUT", "DELETE", "PATCH"
- `fastapi_route_path` (STRING, NULLABLE) - API route path, e.g., "/users/{id}"
- `fastapi_operation_id` (STRING, NULLABLE) - OpenAPI operation identifier
- `fastapi_request_model` (STRING, NULLABLE) - Pydantic request model name
- `fastapi_response_model` (STRING, NULLABLE) - Pydantic response model name
- `fastapi_status_codes` (JSON, NULLABLE) - Expected HTTP status codes
- `fastapi_dependencies` (JSON, NULLABLE) - Route dependencies
- `fastapi_src_path` (STRING, NULLABLE) - Source file path
- `fastapi_start_line` (INTEGER, NULLABLE) - Function start line number
- `fastapi_end_line` (INTEGER, NULLABLE) - Function end line number

**COBOL Fields:**
- `cobol_structure_level` (INTEGER, NULLABLE) - COBOL level: 01, 05, 10, etc.
- `cobol_field_names` (JSON, NULLABLE) - Array of field names in structure
- `cobol_pic_clauses` (JSON, NULLABLE) - PIC clause specifications
- `cobol_occurs_count` (INTEGER, NULLABLE) - OCCURS repetition count
- `cobol_redefines` (STRING, NULLABLE) - REDEFINES target field
- `cobol_usage` (STRING, NULLABLE) - USAGE clause specification

**IRS Fields:**
- `irs_record_type` (STRING, NULLABLE) - IMF record type identifier
- `irs_layout_version` (STRING, NULLABLE) - Layout specification version
- `irs_start_position` (INTEGER, NULLABLE) - Field start byte position
- `irs_field_length` (INTEGER, NULLABLE) - Field length in bytes
- `irs_data_type` (STRING, NULLABLE) - Field data type
- `irs_section` (STRING, NULLABLE) - Record layout section

**MUMPS Fields:**
- `mumps_global_name` (STRING, NULLABLE) - Global variable name (without ^)
- `mumps_node_path` (STRING, NULLABLE) - Node subscription path
- `mumps_file_no` (INTEGER, NULLABLE) - FileMan file number
- `mumps_field_no` (FLOAT, NULLABLE) - FileMan field number
- `mumps_xrefs` (JSON, NULLABLE) - Cross-reference definitions
- `mumps_input_transform` (STRING, NULLABLE) - Input validation transform

**Primary Key:** Composite key on (source_type, artifact_id, content_hash)
**Partitioning:** BY collected_at (daily partitions)
**Clustering:** BY source_type, artifact_id

### source_metadata_errors (Error Tracking)
Stores parsing errors and ingestion failures for debugging and quality monitoring.

**Fields:**
- `error_id` (STRING, REQUIRED) - ULID-generated unique error identifier
- `source_type` (STRING, REQUIRED) - Source family where error occurred
- `source_uri` (STRING, REQUIRED) - Original source location
- `error_class` (STRING, REQUIRED) - Error classification: 'parsing', 'validation', 'ingestion'
- `error_msg` (STRING, REQUIRED) - Detailed error message
- `sample_text` (STRING, NULLABLE) - Sample content that caused error, max 1KB
- `collected_at` (TIMESTAMP, REQUIRED) - Error occurrence timestamp
- `tool_version` (STRING, REQUIRED) - Parser version for debugging
- `stack_trace` (STRING, NULLABLE) - Full stack trace if available

**Partitioning:** BY collected_at (daily partitions)
**Clustering:** BY error_class, source_type

### ingestion_log (Run Tracking)
Tracks ingestion run statistics and performance metrics for monitoring and optimization.

**Fields:**
- `run_id` (STRING, REQUIRED) - ULID-generated unique run identifier
- `source_type` (STRING, REQUIRED) - Source family processed
- `started_at` (TIMESTAMP, REQUIRED) - Run start timestamp
- `completed_at` (TIMESTAMP, NULLABLE) - Run completion timestamp
- `status` (STRING, REQUIRED) - 'running', 'completed', 'failed', 'partial'
- `files_processed` (INTEGER, REQUIRED) - Number of source files processed
- `rows_written` (INTEGER, REQUIRED) - Successful row insertions
- `rows_skipped` (INTEGER, REQUIRED) - Duplicate/unchanged rows
- `errors_count` (INTEGER, REQUIRED) - Number of errors encountered
- `bytes_written` (INTEGER, REQUIRED) - Total bytes written to BigQuery
- `processing_duration_ms` (INTEGER, NULLABLE) - Total processing time
- `avg_chunk_size_tokens` (FLOAT, NULLABLE) - Average tokens per chunk
- `tool_version` (STRING, REQUIRED) - Parser version for tracking
- `config_params` (JSON, NULLABLE) - Run configuration parameters

**Primary Key:** run_id
**Partitioning:** BY started_at (daily partitions)
**Clustering:** BY source_type, status

## Entity Relationships

### Parent-Child Hierarchies
- **COBOL:** 01-level structures (parents) contain 05/10-level fields (children)
- **MUMPS:** FileMan files (parents) contain field definitions (children)
- **FastAPI:** Modules (parents) contain endpoint handlers and models (children)
- **Kubernetes:** Namespaces (parents) contain resource manifests (children)
- **IRS:** Record types (parents) contain field layout sections (children)

### Cross-Entity References
- **source_metadata → ingestion_log:** via tool_version and collected_at range
- **source_metadata_errors → ingestion_log:** via run_id correlation
- **Idempotency:** content_hash enables duplicate detection across runs

## Validation Rules

### Data Quality Constraints
1. **content_text:** Must be non-empty and ≤ 16KB
2. **artifact_id:** Must follow format: `{source_type}://{semantic_path}`
3. **content_hash:** Must be 64-character SHA256 hex string
4. **source_type:** Must be one of: 'kubernetes', 'fastapi', 'cobol', 'irs', 'mumps'
5. **timestamps:** created_at ≤ updated_at, collected_at within reasonable range

### Source-Specific Validations
- **Kubernetes:** k8s_kind required when k8s_api_version present
- **FastAPI:** fastapi_route_path required when fastapi_http_method present
- **COBOL:** cobol_structure_level required when COBOL fields populated
- **IRS:** irs_start_position and irs_field_length required together
- **MUMPS:** mumps_global_name or mumps_file_no must be present

### Performance Considerations
- **Partitioning:** Daily partitions by collected_at for efficient time-range queries
- **Clustering:** Multi-level clustering (source_type, artifact_id) for fast lookups
- **Indexing:** No additional indexes needed due to clustering strategy
- **Storage:** Estimated 1GB per 100K chunks with metadata

This data model supports the full M1 ingestion pipeline requirements while maintaining query performance and data quality standards for downstream AI processing.