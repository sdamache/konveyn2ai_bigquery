# Quickstart: BigQuery Memory Adapter

## Prerequisites

1. **BigQuery Setup**: Ensure `make setup` has been run to create required tables
2. **Authentication**: Google Cloud credentials configured (ADC or service account)
3. **Environment**: Required environment variables set in `.env`

## Environment Configuration

Create or update `.env` file:

```bash
# BigQuery Configuration
GOOGLE_CLOUD_PROJECT=konveyn2ai
BIGQUERY_DATASET_ID=semantic_gap_detector
BIGQUERY_TABLE_PREFIX=source_

# Optional: Explicit credentials (if not using ADC)
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Quick Validation

### 1. Test BigQuery Connectivity

```bash
# Verify BigQuery table exists and is accessible
python -c "
from google.cloud import bigquery
client = bigquery.Client()
table = client.get_table('konveyn2ai.semantic_gap_detector.source_embeddings')
print(f'Table exists: {table.table_id}')
print(f'Schema: {[f.name for f in table.schema]}')
"
```

### 2. Run Contract Tests

```bash
# Contract tests MUST FAIL initially (RED phase)
pytest specs/004-bigquery-memory-adapter/contracts/vector_index_contract.py -v

# Integration tests against real BigQuery
pytest specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py -v
```

### 3. Test Vector Search Query

```bash
# Validate VECTOR_SEARCH function works
bq query --use_legacy_sql=false "
SELECT COUNT(*) as embedding_count
FROM \`konveyn2ai.semantic_gap_detector.source_embeddings\`
"
```

## Implementation Workflow

### Phase 1: Make Tests Pass (RED → GREEN)

1. **Create BigQueryVectorIndex class**:
   ```bash
   # Create implementation file
   mkdir -p src/janapada_memory
   touch src/janapada_memory/bigquery_vector_index.py
   ```

2. **Implement VectorIndex interface**:
   - `similarity_search()` method with BigQuery VECTOR_SEARCH
   - `add_vectors()` and `remove_vector()` methods
   - Configuration loading from environment

3. **Add fallback mechanism**:
   - Local vector index as backup
   - Error handling for BigQuery failures
   - Structured logging for observability

4. **Verify contract tests pass**:
   ```bash
   pytest specs/004-bigquery-memory-adapter/contracts/ -v
   ```

### Phase 2: Integration Testing

1. **Test with Svami orchestrator**:
   ```bash
   # End-to-end test: Svami → Janapada → BigQuery
   python -m src.svami_orchestrator.test_integration
   ```

2. **Performance validation**:
   ```bash
   # Verify <500ms similarity search target
   pytest specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py::TestBigQueryVectorIndexIntegration::test_performance_baseline -v
   ```

3. **Fallback testing**:
   ```bash
   # Simulate BigQuery outage
   export GOOGLE_CLOUD_PROJECT=invalid_project
   pytest specs/004-bigquery-memory-adapter/contracts/vector_index_contract.py::TestVectorIndexContract::test_bigquery_fallback_transparency -v
   ```

## Troubleshooting

### BigQuery Authentication Issues

```bash
# Check ADC status
gcloud auth application-default print-access-token

# Or set explicit credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Permission Errors

Required BigQuery IAM roles:
- `roles/bigquery.dataViewer` - Read table data
- `roles/bigquery.jobUser` - Execute queries

```bash
# Grant permissions (replace USER_EMAIL)
gcloud projects add-iam-policy-binding konveyn2ai \
    --member="user:USER_EMAIL" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding konveyn2ai \
    --member="user:USER_EMAIL" \
    --role="roles/bigquery.jobUser"
```

### Table Not Found

```bash
# Recreate BigQuery tables
make setup

# Verify table creation
bq ls konveyn2ai:semantic_gap_detector
```

### Vector Dimension Mismatch

```bash
# Check embedding dimensions in source_embeddings table
bq query --use_legacy_sql=false "
SELECT
  ARRAY_LENGTH(embedding_vector) as dimension_count,
  COUNT(*) as row_count
FROM \`konveyn2ai.semantic_gap_detector.source_embeddings\`
GROUP BY ARRAY_LENGTH(embedding_vector)
"
```

## Expected Outcomes

### Successful Implementation

1. **Contract tests pass**: All tests in `contracts/` directory green
2. **Interface preserved**: Svami orchestrator continues working unchanged
3. **Fallback functional**: System works when BigQuery unavailable
4. **Performance met**: Similarity search completes in <500ms
5. **Observability**: Structured logs with correlation IDs

### Test Results

```bash
# All tests should pass after implementation
pytest specs/004-bigquery-memory-adapter/contracts/ -v

================================ test session starts ================================
contracts/vector_index_contract.py::TestVectorIndexContract::test_similarity_search_returns_correct_type PASSED
contracts/vector_index_contract.py::TestVectorIndexContract::test_similarity_search_orders_by_distance PASSED
contracts/bigquery_integration_contract.py::TestBigQueryIntegrationContract::test_end_to_end_similarity_search PASSED
================================ 3 passed, 0 failed ================================
```

### Usage Example

```python
from src.janapada_memory.bigquery_vector_index import BigQueryVectorIndex

# Initialize BigQuery-backed vector index
vector_index = BigQueryVectorIndex()

# Perform similarity search (transparent BigQuery → fallback)
query_vector = [0.1] * 3072  # Gemini embedding
results = vector_index.similarity_search(query_vector, top_k=10)

# Results maintain interface contract
for result in results:
    print(f"Chunk: {result['chunk_id']}, Distance: {result['distance']:.4f}")
```

This implementation enables Svami to query BigQuery embeddings transparently while maintaining all existing interface contracts and fallback capabilities.