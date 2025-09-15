# BigQuery Vector Embeddings Generation - Spec

**Issue Reference:** Issue#3

## Preface — BigQuery Semantic Gap Detector (Vector Backend)

### What
Replace KonveyN2AI's vector store with BigQuery VECTOR tables and ship a baseline dataset for metadata, embeddings, and analysis outputs. Keep Svami's search interface stable.

### Why
Reproducible, judge-friendly notebook; deterministic SQL; no scope creep (static viewer later, no multi-agent sprawl).

### Constraints
- Keep `vector_index` interface stable.
- One command: `make setup && make run`.
- Works in BigQuery Studio & Colab.
- LLM only for summaries later, not for core search.

## Source Issue (verbatim)

### Overview
Compute 768-dimensional embeddings for each ingested chunk using Gemini or Vertex AI and store them in BigQuery's vector table. Ensure embedding generation is batched and cached to reduce costs.

### Tasks
1. Add a module in pipeline/embedding.py that iterates over rows in source_metadata lacking an embedding, calls the Vertex AI embedding endpoint (e.g., text-embedding-004), and collects the resulting vectors.
2. Implement batching and exponential backoff for API calls and write a local disk cache keyed by a hash of the text content.
3. Extend the BigQuery vector table creation script with a VECTOR<768> column. Insert new embeddings along with their chunk_id and source metadata.
4. Document the embedding generation process in the repo's README, including expected runtime and cost estimates.

## Acceptance Criteria (verbatim)
- Running make embeddings populates the vector table with embeddings for all existing metadata rows.
- Re-running the command does not duplicate existing embeddings (idempotent behaviour).
- Embedding generation logs the number of API calls and caches repeated invocations.
- At least one query using BigQuery's approximate_neighbors function returns the nearest neighbours for a test chunk.

## Approach

### Schema Design (Enhanced)
```sql
-- Enhanced source_embeddings table
CREATE TABLE source_embeddings (
  chunk_id STRING NOT NULL,
  model STRING NOT NULL,
  content_hash STRING NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL, -- 768-dimensional vector
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  source_type STRING,
  artifact_id STRING,
  partition_date DATE DEFAULT CURRENT_DATE(),
  PRIMARY KEY (chunk_id, model, content_hash) NOT ENFORCED
);
```

### Implementation Architecture

#### 1. Embedding Pipeline (`pipeline/embedding.py`)
- **EmbeddingCache**: Disk-based cache with SHA256 content hashing
- **EmbeddingGenerator**: Gemini API client with batching and backoff
- **EmbeddingPipeline**: Main orchestrator for end-to-end generation

#### 2. Key Features
- **Idempotency**: Upsert key `(chunk_id, model, content_hash)` - skip existing rows
- **Batching**: Configurable `EMBED_BATCH_SIZE` (default 32) with exponential backoff on 429/5xx
- **Caching**: `.cache/embeddings/{sha256}.json` with `{model, vector, created_at}`
- **Cost Optimization**: Cache hits avoid API calls, content deduplication

#### 3. Setup Path
```bash
# Environment setup
export GOOGLE_CLOUD_PROJECT=konveyn2ai
export BIGQUERY_DATASET_ID=semantic_gap_detector  
export GOOGLE_API_KEY=your_gemini_api_key

# Generate embeddings
make embeddings

# Dry run with limits
LIMIT=100 DRY_RUN=1 make embeddings
```

### Technical Implementation

#### Embedding Generation Flow
1. **Query pending chunks**: Get chunks from `source_metadata` lacking embeddings
2. **Content normalization**: SHA256 hash for deduplication  
3. **Cache lookup**: Check `.cache/embeddings/` for existing vectors
4. **API batching**: Process in configurable batch sizes with backoff
5. **Storage**: Write to `source_embeddings` with metadata

#### Vector Search Integration
- **VECTOR_SEARCH()**: BigQuery native vector similarity search
- **ML.APPROXIMATE_NEIGHBORS()**: Legacy compatibility
- **Similarity metrics**: Cosine distance with configurable thresholds

## Risks

### Technical Risks
- **API Rate Limits**: Gemini API throttling with large datasets
- **Cost Escalation**: Embedding generation costs at scale
- **Cache Invalidation**: Content changes requiring re-embedding
- **Vector Dimensionality**: Mismatch between models/expectations

### Operational Risks
- **Storage Costs**: BigQuery storage for large embedding datasets
- **Query Performance**: Vector search latency at scale
- **Cache Management**: Disk space consumption over time

### Mitigation Strategies
- Exponential backoff with jitter for API resilience
- Configurable batch sizes and limits for cost control
- Content hashing for intelligent cache invalidation
- Comprehensive logging and monitoring

## Test Plan

### Unit Tests
- Cache hit/miss behavior validation
- Content normalization and hashing
- API client error handling and retries
- BigQuery schema compliance

### Integration Tests
- End-to-end embedding generation pipeline
- Vector similarity search accuracy
- Idempotent behavior validation
- Cache persistence across runs

### Performance Tests
- Large dataset processing (1000+ chunks)
- Concurrent embedding generation
- Vector search query performance
- Memory usage profiling

### Compatibility Tests
- Multiple embedding models support
- BigQuery Studio integration
- Google Colab environment compatibility

## Expected Performance

### Cost Estimates
- **API Costs**: ~$0.025 per 1,000 chunks (768-dim embeddings)
- **Storage**: ~$0.02 per GB per month in BigQuery
- **Query Costs**: ~$5 per TB scanned for vector searches

### Performance Targets
- **Processing Rate**: 100-500 chunks per minute (with caching)
- **API Latency**: ~100-200ms per embedding request
- **Cache Hit Rate**: 70-90% on subsequent runs
- **Vector Search**: <1 second for top-k similarity queries

### Monitoring Metrics
- Total API calls and costs
- Cache hit/miss ratios
- Processing throughput
- Error rates and retry counts

## Rollback/Migration Steps

### Rollback Procedures
1. **Table Rollback**: Drop or rename `source_embeddings` table
2. **Configuration Revert**: Restore previous embedding settings
3. **Cache Cleanup**: Clear embedding cache if needed

### Migration Support
- **Model Changes**: Create versioned embedding tables
- **Dimension Updates**: Support multiple vector dimensions
- **Provider Switch**: Abstract embedding interface for flexibility

### Verification Steps
1. **Functional Testing**: Vector search returns expected results
2. **Data Integrity**: Embedding count matches source metadata
3. **Performance Validation**: Query latency within acceptable bounds
4. **Cost Monitoring**: Actual costs align with estimates