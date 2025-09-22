-- BigQuery DDL for M1 Multi-Source Ingestion
-- Creates tables for source metadata, errors, and ingestion tracking
-- Optimized for high-volume ingestion with partitioning and clustering

-- Main source metadata table with unified schema
CREATE TABLE `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_INGESTION_DATASET_ID}.source_metadata` (
  -- Common fields for all source types
  source_type STRING NOT NULL OPTIONS(description="Source family: kubernetes, fastapi, cobol, irs, mumps"),
  artifact_id STRING NOT NULL OPTIONS(description="Deterministic semantic identifier"),
  parent_id STRING OPTIONS(description="Parent artifact for hierarchical relationships"),
  parent_type STRING OPTIONS(description="Type of parent artifact"),
  content_text STRING NOT NULL OPTIONS(description="Chunk content for embedding, max 16KB"),
  content_tokens INT64 OPTIONS(description="Token count for the content"),
  content_hash STRING NOT NULL OPTIONS(description="SHA256 hash for idempotency"),
  created_at TIMESTAMP NOT NULL OPTIONS(description="First ingestion timestamp"),
  updated_at TIMESTAMP NOT NULL OPTIONS(description="Last modification timestamp"),
  collected_at TIMESTAMP NOT NULL OPTIONS(description="Source collection timestamp"),
  source_uri STRING NOT NULL OPTIONS(description="Original source location"),
  repo_ref STRING OPTIONS(description="Git commit SHA or image tag"),
  tool_version STRING NOT NULL OPTIONS(description="Parser version for reproducibility"),

  -- Kubernetes-specific fields
  k8s_api_version STRING OPTIONS(description="Kubernetes API version"),
  k8s_kind STRING OPTIONS(description="Resource type: Deployment, Service, etc."),
  k8s_namespace STRING OPTIONS(description="Kubernetes namespace"),
  k8s_resource_name STRING OPTIONS(description="Resource name"),
  k8s_labels JSON OPTIONS(description="Labels as JSON object"),
  k8s_annotations JSON OPTIONS(description="Annotations as JSON object"),
  k8s_resource_version STRING OPTIONS(description="Kubernetes resource version"),

  -- FastAPI-specific fields
  fastapi_http_method STRING OPTIONS(description="HTTP method: GET, POST, PUT, DELETE, PATCH"),
  fastapi_route_path STRING OPTIONS(description="API route path"),
  fastapi_operation_id STRING OPTIONS(description="OpenAPI operation identifier"),
  fastapi_request_model STRING OPTIONS(description="Pydantic request model name"),
  fastapi_response_model STRING OPTIONS(description="Pydantic response model name"),
  fastapi_status_codes JSON OPTIONS(description="Expected HTTP status codes"),
  fastapi_dependencies JSON OPTIONS(description="Route dependencies"),
  fastapi_src_path STRING OPTIONS(description="Source file path"),
  fastapi_start_line INT64 OPTIONS(description="Function start line number"),
  fastapi_end_line INT64 OPTIONS(description="Function end line number"),

  -- COBOL-specific fields
  cobol_structure_level INT64 OPTIONS(description="COBOL level: 01, 05, 10, etc."),
  cobol_field_names JSON OPTIONS(description="Array of field names in structure"),
  cobol_pic_clauses JSON OPTIONS(description="PIC clause specifications"),
  cobol_occurs_count INT64 OPTIONS(description="OCCURS repetition count"),
  cobol_redefines STRING OPTIONS(description="REDEFINES target field"),
  cobol_usage STRING OPTIONS(description="USAGE clause specification"),

  -- IRS-specific fields
  irs_record_type STRING OPTIONS(description="IMF record type identifier"),
  irs_layout_version STRING OPTIONS(description="Layout specification version"),
  irs_start_position INT64 OPTIONS(description="Field start byte position"),
  irs_field_length INT64 OPTIONS(description="Field length in bytes"),
  irs_data_type STRING OPTIONS(description="Field data type"),
  irs_section STRING OPTIONS(description="Record layout section"),

  -- MUMPS-specific fields
  mumps_global_name STRING OPTIONS(description="Global variable name without ^"),
  mumps_node_path STRING OPTIONS(description="Node subscription path"),
  mumps_file_no INT64 OPTIONS(description="FileMan file number"),
  mumps_field_no FLOAT64 OPTIONS(description="FileMan field number"),
  mumps_xrefs JSON OPTIONS(description="Cross-reference definitions"),
  mumps_input_transform STRING OPTIONS(description="Input validation transform")
)
PARTITION BY DATE(collected_at)
CLUSTER BY source_type, artifact_id
OPTIONS(
  description="Normalized chunks from multi-source artifact ingestion",
  partition_expiration_days=null,
  require_partition_filter=false
);

-- Error tracking table for parsing and ingestion failures
CREATE TABLE `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_INGESTION_DATASET_ID}.source_metadata_errors` (
  error_id STRING NOT NULL OPTIONS(description="ULID-generated unique error identifier"),
  source_type STRING NOT NULL OPTIONS(description="Source family where error occurred"),
  source_uri STRING NOT NULL OPTIONS(description="Original source location"),
  error_class STRING NOT NULL OPTIONS(description="Error classification: parsing, validation, ingestion"),
  error_msg STRING NOT NULL OPTIONS(description="Detailed error message"),
  sample_text STRING OPTIONS(description="Sample content that caused error, max 1KB"),
  collected_at TIMESTAMP NOT NULL OPTIONS(description="Error occurrence timestamp"),
  tool_version STRING NOT NULL OPTIONS(description="Parser version for debugging"),
  stack_trace STRING OPTIONS(description="Full stack trace if available")
)
PARTITION BY DATE(collected_at)
CLUSTER BY error_class, source_type
OPTIONS(
  description="Parsing errors and ingestion failures for debugging",
  partition_expiration_days=90,
  require_partition_filter=false
);

-- Ingestion run tracking for monitoring and performance analysis
CREATE TABLE `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_INGESTION_DATASET_ID}.ingestion_log` (
  run_id STRING NOT NULL OPTIONS(description="ULID-generated unique run identifier"),
  source_type STRING NOT NULL OPTIONS(description="Source family processed"),
  started_at TIMESTAMP NOT NULL OPTIONS(description="Run start timestamp"),
  completed_at TIMESTAMP OPTIONS(description="Run completion timestamp"),
  status STRING NOT NULL OPTIONS(description="running, completed, failed, partial"),
  files_processed INT64 NOT NULL OPTIONS(description="Number of source files processed"),
  rows_written INT64 NOT NULL OPTIONS(description="Successful row insertions"),
  rows_skipped INT64 NOT NULL OPTIONS(description="Duplicate/unchanged rows"),
  errors_count INT64 NOT NULL OPTIONS(description="Number of errors encountered"),
  bytes_written INT64 NOT NULL OPTIONS(description="Total bytes written to BigQuery"),
  processing_duration_ms INT64 OPTIONS(description="Total processing time"),
  avg_chunk_size_tokens FLOAT64 OPTIONS(description="Average tokens per chunk"),
  tool_version STRING NOT NULL OPTIONS(description="Parser version for tracking"),
  config_params JSON OPTIONS(description="Run configuration parameters")
)
PARTITION BY DATE(started_at)
CLUSTER BY source_type, status
OPTIONS(
  description="Ingestion run statistics and performance metrics",
  partition_expiration_days=365,
  require_partition_filter=false
);

-- Create dataset-level permissions and labels
-- Note: Replace ${GOOGLE_CLOUD_PROJECT} and ${BIGQUERY_INGESTION_DATASET_ID} with actual values during deployment
ALTER SCHEMA `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_INGESTION_DATASET_ID}`
SET OPTIONS(
  description="M1 Multi-Source Ingestion for KonveyN2AI BigQuery Hackathon",
  labels=[("environment", "hackathon"), ("milestone", "m1"), ("purpose", "ingestion")]
);