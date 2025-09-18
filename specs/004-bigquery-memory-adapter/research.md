# Research: BigQuery Memory Adapter Integration

## Research Tasks Completed

### BigQuery Vector Search Implementation
**Decision**: Use VECTOR_SEARCH function with approximate_neighbors for similarity queries
**Rationale**:
- VECTOR_SEARCH provides built-in approximate nearest neighbor search
- Supports DOT_PRODUCT and COSINE distance metrics
- Optimized for large-scale vector similarity operations
- Native BigQuery function with query optimization support

**Alternatives considered**:
- Manual distance calculations: Too slow for production scale
- Client-side filtering: Defeats purpose of BigQuery backend
- Vertex AI Matching Engine: Adds complexity and separate service dependency

### Configuration Management Patterns
**Decision**: Environment-based configuration with Application Default Credentials (ADC)
**Rationale**:
- Follows Google Cloud best practices for authentication
- Supports multiple deployment environments (local, staging, production)
- Leverages existing .env pattern in codebase
- ADC handles credential rotation automatically

**Alternatives considered**:
- Service account key files: Security risk, manual rotation
- Hard-coded configuration: No environment flexibility
- Complex configuration framework: Violates constitutional simplicity

### Fallback Strategy Implementation
**Decision**: In-memory vector index using existing vector utilities as fallback
**Rationale**:
- Preserves system availability when BigQuery unavailable
- Reuses proven vector indexing code from existing implementation
- Transparent to Svami orchestrator (same interface contract)
- Enables local development without BigQuery dependency

**Alternatives considered**:
- Fail fast on BigQuery errors: Poor user experience
- External vector database fallback: Additional infrastructure complexity
- No fallback: Violates resilience requirements

### Interface Contract Preservation
**Decision**: Maintain existing vector_index method signatures and return types
**Rationale**:
- Zero breaking changes to Svami orchestrator integration
- Preserves established testing contracts and mocks
- Enables gradual rollout with feature flags
- Satisfies constitutional requirement for stable interfaces

**Alternatives considered**:
- New BigQuery-specific interface: Breaks existing integrations
- Optional BigQuery methods: Interface proliferation
- Versioned interfaces: Unnecessary complexity for this change

### Error Handling and Observability
**Decision**: Structured logging with correlation IDs and BigQuery job references
**Rationale**:
- Enables distributed tracing across BigQuery operations
- Facilitates debugging of query performance issues
- Supports operational monitoring and alerting
- Aligns with constitutional observability requirements

**Alternatives considered**:
- Simple string logging: Insufficient for production debugging
- External APM only: Missing BigQuery-specific context
- No correlation tracking: Difficult to trace request flows

## Technical Dependencies Validated

### google-cloud-bigquery Library
- Version: Latest stable (3.x)
- Authentication: ADC support confirmed
- Vector operations: VECTOR_SEARCH function available
- Query patterns: Parameterized queries for security
- Error handling: Comprehensive exception hierarchy

### Existing Vector Utilities
- Location: Confirmed in src/janapada-memory/
- Interface: Compatible with fallback requirements
- Performance: Adequate for local testing and fallback scenarios
- Dependencies: No additional requirements identified

### BigQuery Schema Requirements
- Table: source_embeddings with VECTOR(D) column
- Indexes: None required (VECTOR_SEARCH handles optimization)
- Permissions: BigQuery Data Viewer, Job User minimum required
- Dataset: Existing source_ingestion dataset from M1/M2

## Integration Patterns Identified

### Configuration Loading Pattern
```python
# Environment variables with sensible defaults
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT', 'konveyn2ai')
BIGQUERY_DATASET_ID = os.getenv('BIGQUERY_DATASET_ID', 'source_ingestion')
BIGQUERY_TABLE_PREFIX = os.getenv('BIGQUERY_TABLE_PREFIX', 'source_')
```

### BigQuery Query Pattern
```sql
SELECT chunk_id, distance
FROM VECTOR_SEARCH(
  TABLE source_ingestion.source_embeddings,
  'embedding_vector',
  @query_vector,
  top_k => 150,
  distance_type => 'COSINE'
)
ORDER BY distance ASC
```

### Fallback Activation Pattern
```python
try:
    # BigQuery similarity search
    return bigquery_similarity_search(query_vector)
except (google.cloud.exceptions.NotFound,
        google.cloud.exceptions.Forbidden,
        ConnectionError) as e:
    logger.warning("BigQuery unavailable, using fallback", extra={
        "error": str(e), "correlation_id": correlation_id
    })
    return local_similarity_search(query_vector)
```

## Research Complete
All technical unknowns resolved. No NEEDS CLARIFICATION items remain. Ready for Phase 1 design work.