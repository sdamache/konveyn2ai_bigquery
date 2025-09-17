# Data Model: BigQuery Vector Embeddings

## Core Entities

### EmbeddingRecord
**Purpose**: Represents a single embedding vector with metadata
**Storage**: BigQuery `source_embeddings` table

**Fields**:
```python
@dataclass
class EmbeddingRecord:
    chunk_id: str           # Primary key linking to source_metadata
    model: str              # Embedding model name (e.g., "text-embedding-004")  
    content_hash: str       # SHA256 hash of normalized text content
    embedding: List[float]  # 768-dimensional vector array
    created_at: datetime    # Timestamp of embedding generation
    source_type: str        # Optional: artifact type (k8s, fastapi, etc.)
    artifact_id: str        # Optional: source identifier
    partition_date: date    # Date partition for BigQuery optimization
```

**Validation Rules**:
- `chunk_id`: Must exist in `source_metadata` table
- `model`: Must be valid embedding model identifier
- `content_hash`: Must be 64-character SHA256 hex string  
- `embedding`: Must be exactly 768 float values
- `created_at`: Must be valid timestamp
- Composite unique key: `(chunk_id, model, content_hash)`

**Relationships**:
- Many-to-One with SourceMetadata via `chunk_id`
- One-to-Many with VectorSearchResults (query results)

### CacheEntry
**Purpose**: Represents cached embedding data for performance optimization
**Storage**: Local filesystem `.cache/embeddings/`

**Fields**:
```python
@dataclass 
class CacheEntry:
    model: str              # Embedding model used
    vector: List[float]     # Cached embedding vector
    created_at: datetime    # Cache creation timestamp
    content_hash: str       # SHA256 of original content
```

**Validation Rules**:
- Cache file named: `{model}_{content_hash}.json`
- `vector`: Must match model's expected dimensions
- `created_at`: ISO format timestamp
- Cache TTL: Configurable, default no expiration

### EmbeddingRequest
**Purpose**: Request payload for embedding generation
**Storage**: Transient (in-memory only)

**Fields**:
```python
@dataclass
class EmbeddingRequest:
    chunk_id: str           # Target chunk identifier
    text_content: str       # Text to embed
    source_type: str        # Source artifact type
    artifact_id: str        # Source identifier
    model: str              # Requested embedding model
```

**Validation Rules**:
- `text_content`: 1-1,000,000 characters
- `chunk_id`: Must be valid identifier format
- `model`: Must be supported model name

### EmbeddingResponse  
**Purpose**: Response from embedding generation
**Storage**: Transient (in-memory only)

**Fields**:
```python
@dataclass
class EmbeddingResponse:
    chunk_id: str           # Original chunk identifier
    embedding: List[float]  # Generated vector (None if failed)
    content_hash: str       # Content hash for caching
    model: str              # Model used for generation
    cached: bool            # Whether result came from cache
    api_latency_ms: int     # API call duration (0 if cached)
    error: str              # Error message if generation failed
```

### ProcessingStats
**Purpose**: Statistics and metrics for embedding generation runs
**Storage**: Transient (in-memory, logged)

**Fields**:
```python
@dataclass
class ProcessingStats:
    chunks_scanned: int     # Total chunks processed
    embeddings_generated: int  # Successful embeddings created
    embeddings_stored: int  # Records written to BigQuery
    api_calls: int          # Total API requests made
    cache_hits: int         # Cache lookups that succeeded
    cache_misses: int       # Cache lookups that failed
    failed_requests: int    # API calls that failed
    total_latency_ms: int   # Cumulative API latency
    processing_time_ms: int # Total pipeline runtime
```

## State Transitions

### Embedding Generation Lifecycle
```
[Pending] → [Cache Check] → [API Call] → [Stored]
    ↓           ↓              ↓
[Skipped]   [Cache Hit]   [Failed]
```

**States**:
- **Pending**: Chunk exists in source_metadata, no embedding
- **Cache Check**: Looking up content hash in local cache
- **Cache Hit**: Found valid cached embedding  
- **API Call**: Requesting embedding from Gemini API
- **Failed**: API call or storage operation failed
- **Stored**: Embedding successfully written to BigQuery
- **Skipped**: Embedding already exists (idempotent behavior)

### Cache Lifecycle
```
[Miss] → [API Call] → [Cache Write] → [Cache Hit]
```

**Cache States**:
- **Miss**: Content hash not found in cache
- **Cache Write**: Storing API response to filesystem
- **Cache Hit**: Reading existing cached embedding

## Data Flow Patterns

### Batch Processing Flow
```
1. Query source_metadata for pending chunks
2. Group chunks by content hash (deduplication)
3. Check cache for each unique content hash
4. Batch API calls for cache misses
5. Store embeddings with metadata in BigQuery
6. Update cache with new embeddings
```

### Idempotency Pattern
```
1. Calculate composite key (chunk_id, model, content_hash)
2. Check if embedding exists in BigQuery
3. Skip if exists, process if missing
4. Ensure deterministic content hashing
```

### Error Recovery Pattern
```
1. Continue processing other chunks if one fails
2. Log detailed error information  
3. Track failed chunks for retry
4. Provide summary statistics
```

## Performance Considerations

### BigQuery Optimization
- **Partitioning**: By `partition_date` for time-based queries
- **Clustering**: By `(chunk_id, model, content_hash)` for lookups
- **Data Types**: FLOAT64 arrays for vector operations
- **Query Patterns**: Optimize for similarity search and metadata joins

### Cache Optimization  
- **Directory Structure**: Flat directory with hash-based filenames
- **File Format**: JSON for readability and debugging
- **Concurrent Access**: Thread-safe cache operations
- **Storage Limits**: Configurable cache size limits

### Memory Management
- **Streaming**: Process chunks in configurable batches
- **Resource Limits**: Prevent memory overflow
- **Garbage Collection**: Regular cleanup of temporary objects
- **Vector Storage**: Efficient numpy array handling

## Integration Patterns

### BigQuery Integration
```python
# Query pattern for pending chunks
SELECT sm.chunk_id, sm.text_content, sm.source, sm.artifact_type
FROM source_metadata sm
LEFT JOIN source_embeddings se ON sm.chunk_id = se.chunk_id
WHERE se.chunk_id IS NULL

# Insert pattern for new embeddings  
INSERT INTO source_embeddings 
(chunk_id, model, content_hash, embedding, created_at, source_type, artifact_id, partition_date)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
```

### Gemini API Integration
```python
# Embedding request pattern
response = genai.embed_content(
    model="models/text-embedding-004",
    content=normalized_content,
    task_type="retrieval_document"
)
```

### Cache Integration
```python
# Cache key pattern
cache_key = f"{model}_{sha256(content).hexdigest()}"
cache_file = f".cache/embeddings/{cache_key}.json"
```

## Error Handling

### Validation Errors
- **Schema Mismatch**: Vector dimension validation
- **Data Type**: Float array validation  
- **Constraint Violation**: Unique key enforcement
- **Missing Dependencies**: Required field validation

### API Errors
- **Rate Limiting**: 429 status code handling
- **Quota Exceeded**: API quota management
- **Authentication**: Invalid API key handling
- **Network Issues**: Timeout and retry logic

### Storage Errors
- **BigQuery Quota**: Slot/query limit handling
- **Permission Issues**: IAM access validation
- **Schema Evolution**: Table version compatibility
- **Disk Space**: Cache storage limits

## Monitoring and Metrics

### Performance Metrics
- **Throughput**: Chunks processed per minute
- **Latency**: API call response times
- **Cache Efficiency**: Hit/miss ratios
- **Error Rates**: Failed operations percentage

### Cost Metrics
- **API Costs**: Gemini API usage tracking
- **Storage Costs**: BigQuery storage growth
- **Query Costs**: Vector search operation costs
- **Cache Benefits**: Cost savings from cache hits

### Operational Metrics
- **Data Quality**: Embedding validation success rates
- **System Health**: Error rates and recovery times
- **Resource Usage**: Memory and disk utilization
- **Scalability**: Performance under load