# BigQuery Semantic Gap Detector (Vector Backend) - Spec

**Issue Reference:** Issue#1.txt

## Acceptance Criteria (verbatim)
- A BigQuery dataset exists with three tables and the correct schemas.
- Environment variables are defined for BigQuery access.
- The vector index code writes and queries embeddings via BigQuery without errors.
- Running `make setup` creates the BigQuery tables if they do not exist.

## Approach

### Schema Design
```sql
-- semantic_gap_detector.source_metadata
CREATE TABLE source_metadata (
  chunk_id STRING NOT NULL,
  source STRING NOT NULL,
  artifact_type STRING NOT NULL, 
  text_content STRING NOT NULL,
  kind STRING,
  api_path STRING,
  record_name STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (chunk_id) NOT ENFORCED
);

-- semantic_gap_detector.source_embeddings  
CREATE TABLE source_embeddings (
  chunk_id STRING NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL, -- 768-dimensional vector
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (chunk_id) NOT ENFORCED
);

-- semantic_gap_detector.gap_metrics
CREATE TABLE gap_metrics (
  analysis_id STRING NOT NULL,
  chunk_id STRING NOT NULL,
  metric_type STRING NOT NULL,
  metric_value FLOAT64,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (analysis_id, chunk_id, metric_type) NOT ENFORCED
);
```

### Backend Swap Strategy
1. **Interface Preservation**: Maintain existing `vector_index` public API
2. **Client Wrapper**: Create BigQuery client wrapper in `src/janapada-memory/`
3. **Query Translation**: Map existing vector operations to BigQuery `VECTOR_SEARCH()` functions
4. **Gradual Migration**: Support both backends during transition period

### Setup Path
```bash
make setup  # Creates dataset + tables if missing
make run    # Launches with BigQuery backend
```

### Configuration Updates
```bash
# .env additions
BIGQUERY_PROJECT_ID=konveyn2ai
BIGQUERY_DATASET_ID=semantic_gap_detector
BIGQUERY_CREDENTIALS_PATH=./credentials/service-account.json
GOOGLE_CLOUD_PROJECT=konveyn2ai
```

## Risks

### Technical Risks
- **Vector Search Performance**: BigQuery `VECTOR_SEARCH()` may have latency differences vs Vertex AI
- **Quota Limits**: BigQuery slot/query limits could throttle vector operations  
- **Embedding Dimension Mismatch**: Current system uses 3072-dim, issue specifies 768-dim
- **Authentication Complexity**: Service account setup and credential management

### Operational Risks
- **Migration Downtime**: Existing vector data needs manual export/import
- **Cost Implications**: BigQuery storage/query costs vs current Vertex AI pricing
- **Dependency Chain**: Changes to core memory component affect orchestrator/prompter

### Mitigation Strategies
- Implement feature flag to toggle between backends
- Add comprehensive error handling and fallback mechanisms
- Create data validation scripts for schema compliance
- Document rollback procedures explicitly

## Test Plan

### Unit Tests
- Schema validation for all three tables
- CRUD operations on each table
- Vector similarity search accuracy
- Configuration loading and validation

### Integration Tests  
- End-to-end vector storage and retrieval
- BigQuery authentication flow
- `make setup` table creation idempotency
- Cross-component compatibility (memory → orchestrator)

### Performance Tests
- Vector search latency benchmarks
- Concurrent query handling
- Large dataset ingestion performance
- Memory usage under load

### Compatibility Tests
- BigQuery Studio notebook execution
- Google Colab environment compatibility
- Local development environment setup

## Rollback/Migration Steps

### Pre-Migration
1. **Data Export**: Extract existing vectors from current system
   ```bash
   python scripts/export_vectors.py --output=vectors_backup.json
   ```

2. **Schema Validation**: Verify BigQuery tables match expected structure
   ```bash
   python scripts/validate_schema.py --dataset=semantic_gap_detector
   ```

### Migration Execution
1. **Create BigQuery Infrastructure**
   ```bash
   make setup  # Creates dataset and tables
   ```

2. **Data Import**
   ```bash
   python scripts/import_vectors.py --source=vectors_backup.json
   ```

3. **Switch Backend**
   ```bash
   # Update .env
   VECTOR_BACKEND=bigquery  # vs vertex
   ```

### Rollback Procedure
1. **Immediate Rollback** (< 1 hour)
   ```bash
   # Revert .env changes
   VECTOR_BACKEND=vertex
   # Restart services
   make restart
   ```

2. **Full Rollback** (if data corruption)
   ```bash
   # Restore from backup
   python scripts/restore_vectors.py --source=vectors_backup.json
   # Revert all config changes
   git checkout HEAD~1 .env src/janapada-memory/
   ```

### Verification Steps
1. **Functional Testing**
   - Verify vector search returns expected results
   - Confirm all three tables are accessible
   - Test `make setup && make run` workflow

2. **Data Integrity**
   - Compare vector search results before/after migration
   - Validate chunk_id consistency across tables
   - Check for data loss or corruption

3. **Performance Validation**
   - Benchmark query response times
   - Monitor BigQuery slot usage
   - Verify system stability under normal load

### Contingency Plans
- **Partial Failure**: Maintain dual backends temporarily
- **Performance Issues**: Implement query optimization and caching
- **Data Loss**: Automated backup verification before migration
- **Authentication Failure**: Fallback to alternative credential methods