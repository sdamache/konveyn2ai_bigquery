# Data Model: BigQuery Memory Adapter

## Core Entities

### VectorSearchConfig
**Purpose**: Configuration for BigQuery vector search operations
**Fields**:
- `project_id: str` - Google Cloud project ID
- `dataset_id: str` - BigQuery dataset containing embeddings
- `table_name: str` - Source embeddings table name
- `vector_column: str` - Column name containing VECTOR data
- `distance_type: str` - Distance metric (COSINE, DOT_PRODUCT)
- `top_k: int` - Number of nearest neighbors to return

**Validation Rules**:
- project_id must be valid GCP project format
- dataset_id must exist and be accessible
- table_name must exist in dataset
- vector_column must be VECTOR type
- distance_type must be supported by BigQuery VECTOR_SEARCH
- top_k must be positive integer ≤ 1000

**State Transitions**: Immutable configuration object

### VectorSearchResult
**Purpose**: Standardized result from similarity search operations
**Fields**:
- `chunk_id: str` - Unique identifier for retrieved chunk
- `distance: float` - Similarity distance score
- `metadata: Dict[str, Any]` - Additional chunk metadata
- `source: str` - Source of result (bigquery, fallback)

**Validation Rules**:
- chunk_id must be non-empty string
- distance must be non-negative float
- metadata must be serializable dict
- source must be enum value (bigquery, local)

**Relationships**: Collection returned by similarity_search operations

### BigQueryConnection
**Purpose**: Manages BigQuery client connection and authentication
**Fields**:
- `client: bigquery.Client` - Authenticated BigQuery client
- `project_id: str` - Active project ID
- `credentials: Optional[Credentials]` - Authentication credentials
- `is_available: bool` - Connection health status

**Validation Rules**:
- client must be authenticated
- project_id must match client configuration
- is_available reflects actual connectivity

**State Transitions**:
- Uninitialized → Connected → (Disconnected/Error)
- Error → Connected (retry logic)

## Interface Contracts

### VectorIndex (Existing - Must Preserve)
**Purpose**: Abstract interface for vector similarity operations
**Methods**:
- `similarity_search(query_vector: List[float], top_k: int) -> List[VectorSearchResult]`
- `add_vectors(vectors: List[Tuple[str, List[float]]]) -> None`
- `remove_vector(chunk_id: str) -> bool`

**Contract Requirements**:
- similarity_search must return results ordered by ascending distance
- top_k parameter must limit result count
- Exceptions must be consistent across implementations
- Return types must match exactly

### BigQueryVectorIndex (New Implementation)
**Purpose**: BigQuery-backed implementation of VectorIndex interface
**Additional Methods**:
- `_execute_bigquery_search(query_vector: List[float], top_k: int) -> List[VectorSearchResult]`
- `_fallback_search(query_vector: List[float], top_k: int) -> List[VectorSearchResult]`
- `check_bigquery_health() -> bool`

**Implementation Contract**:
- Must implement VectorIndex interface exactly
- Must handle BigQuery connection failures gracefully
- Must log fallback activation with structured logging
- Must maintain performance characteristics of original interface

## Data Flow

### Normal Operation Flow
1. `similarity_search()` called with query vector
2. Validate query vector dimensions and format
3. Construct parameterized BigQuery VECTOR_SEARCH query
4. Execute query against source_embeddings table
5. Parse results into VectorSearchResult objects
6. Return ordered list by distance

### Fallback Flow
1. BigQuery operation fails (connection, auth, or query error)
2. Log structured warning with error details and correlation ID
3. Delegate to local in-memory vector index
4. Execute similarity search on local fallback index
5. Mark results with source='local'
6. Return ordered list maintaining interface contract

### Configuration Flow
1. Load environment variables with defaults
2. Validate BigQuery project and dataset accessibility
3. Create authenticated BigQuery client
4. Initialize local fallback index
5. Cache configuration for operation lifecycle

## Error Handling

### BigQuery Errors
- `google.cloud.exceptions.NotFound`: Dataset/table missing → fallback
- `google.cloud.exceptions.Forbidden`: Permission denied → fallback
- `google.cloud.exceptions.BadRequest`: Query syntax error → raise
- `ConnectionError`: Network issues → fallback

### Validation Errors
- Invalid query vector dimensions → ValueError
- Missing configuration → ConfigurationError
- Authentication failure → AuthenticationError

### Fallback Errors
- Local index unavailable → RuntimeError
- No fallback configured → NotImplementedError

All errors include correlation IDs and remediation context.