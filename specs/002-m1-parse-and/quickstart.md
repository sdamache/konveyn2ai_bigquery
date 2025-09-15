# Quickstart: M1 Multi-Source Ingestion

## Prerequisites

### Environment Setup
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export BQ_PROJECT="konveyn2ai"
export BQ_DATASET="source_ingestion"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### BigQuery Setup
```bash
# Create dataset and tables (idempotent)
make setup

# Verify tables exist
bq ls ${BQ_DATASET}
```

## Basic Usage

### 1. Kubernetes Ingestion
```bash
# From live cluster
make ingest_k8s SOURCE=cluster NAMESPACE=default

# From manifest files
make ingest_k8s SOURCE=./k8s-manifests/

# Dry run to preview
make ingest_k8s SOURCE=./k8s-manifests/ DRY_RUN=1
```

### 2. FastAPI Ingestion
```bash
# From running application
make ingest_fastapi SOURCE=http://localhost:8000

# From source directory
make ingest_fastapi SOURCE=./fastapi-project/

# OpenAPI spec only
make ingest_fastapi SOURCE=./openapi.json
```

### 3. COBOL Ingestion
```bash
# From copybook directory
make ingest_cobol SOURCE=./cobol-copybooks/

# Single copybook file
make ingest_cobol SOURCE=./EMPLOYEE.cpy

# Preview parsing
make ingest_cobol SOURCE=./cobol-copybooks/ DRY_RUN=1
```

### 4. IRS Ingestion
```bash
# From IMF layout files
make ingest_irs SOURCE=./irs-layouts/

# Single layout file
make ingest_irs SOURCE=./IMF-2023.txt

# Show field positions
make ingest_irs SOURCE=./irs-layouts/ DRY_RUN=1
```

### 5. MUMPS Ingestion
```bash
# From FileMan dictionaries
make ingest_mumps SOURCE=./fileman-dicts/

# From global definitions
make ingest_mumps SOURCE=./mumps-globals/ TYPE=globals

# Preview structure
make ingest_mumps SOURCE=./fileman-dicts/ DRY_RUN=1
```

## Validation Steps

### 1. Check Row Counts
```sql
-- Verify minimum 100 rows per source type
SELECT
  source_type,
  COUNT(*) as row_count
FROM `${BQ_PROJECT}.${BQ_DATASET}.source_metadata`
GROUP BY source_type
ORDER BY source_type;
```

### 2. Verify Data Quality
```sql
-- Check for missing required fields
SELECT
  source_type,
  COUNT(*) as total_rows,
  COUNT(CASE WHEN content_text IS NULL OR content_text = '' THEN 1 END) as empty_content,
  COUNT(CASE WHEN artifact_id IS NULL OR artifact_id = '' THEN 1 END) as missing_artifact_id,
  COUNT(CASE WHEN content_hash IS NULL OR LENGTH(content_hash) != 64 THEN 1 END) as invalid_hash
FROM `${BQ_PROJECT}.${BQ_DATASET}.source_metadata`
GROUP BY source_type;
```

### 3. Check Chunking Quality
```sql
-- Analyze chunk sizes and token distribution
SELECT
  source_type,
  AVG(content_tokens) as avg_tokens,
  MIN(content_tokens) as min_tokens,
  MAX(content_tokens) as max_tokens,
  STDDEV(content_tokens) as token_stddev
FROM `${BQ_PROJECT}.${BQ_DATASET}.source_metadata`
WHERE content_tokens IS NOT NULL
GROUP BY source_type;
```

### 4. Review Ingestion Logs
```sql
-- Check ingestion run status
SELECT
  source_type,
  status,
  files_processed,
  rows_written,
  errors_count,
  processing_duration_ms / 1000 as duration_seconds
FROM `${BQ_PROJECT}.${BQ_DATASET}.ingestion_log`
ORDER BY started_at DESC
LIMIT 20;
```

### 5. Investigate Errors
```sql
-- Review parsing errors by type
SELECT
  source_type,
  error_class,
  COUNT(*) as error_count,
  STRING_AGG(DISTINCT SUBSTR(error_msg, 1, 100), '; ' LIMIT 3) as sample_errors
FROM `${BQ_PROJECT}.${BQ_DATASET}.source_metadata_errors`
WHERE DATE(collected_at) = CURRENT_DATE()
GROUP BY source_type, error_class
ORDER BY error_count DESC;
```

## Common Patterns

### CLI Usage
```bash
# Direct Python module execution
python -m src.ingest.k8s --source ./manifests --dry-run
python -m src.ingest.fastapi --source ./app --output json
python -m src.ingest.cobol --source ./copybooks --max-rows 50

# With custom configuration
python -m src.ingest.mumps --source ./vista --config ./custom-config.json
```

### Batch Processing
```bash
# Process all source types sequentially
make ingest_all SOURCES_DIR=./sample-data/

# Parallel processing (use with caution)
make ingest_k8s SOURCE=./k8s/ &
make ingest_fastapi SOURCE=./api/ &
wait
```

### Performance Tuning
```bash
# Increase batch size for large datasets
export MAX_ROWS_PER_BATCH=1000
make ingest_cobol SOURCE=./large-copybooks/

# Enable verbose logging
export LOG_LEVEL=DEBUG
make ingest_irs SOURCE=./irs-data/

# Use local temp files for large content
export USE_TEMP_FILES=true
make ingest_mumps SOURCE=./vista-large/
```

## Troubleshooting

### Common Issues

1. **BigQuery Authentication**
   ```bash
   # Test credentials
   gcloud auth application-default login
   bq ls ${BQ_DATASET}
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall with specific versions
   pip install --force-reinstall -r requirements.txt
   ```

3. **Parser Errors**
   ```bash
   # Check sample files first
   python -m src.ingest.cobol --source ./sample.cpy --dry-run --verbose
   ```

4. **Memory Issues**
   ```bash
   # Process in smaller batches
   export MAX_ROWS_PER_BATCH=100
   make ingest_large_dataset
   ```

### Expected Outputs

After successful ingestion:
- **Kubernetes**: 100+ rows with k8s_kind, k8s_namespace fields populated
- **FastAPI**: 100+ rows with fastapi_route_path, fastapi_http_method fields populated
- **COBOL**: 100+ rows with cobol_structure_level, cobol_pic_clauses fields populated
- **IRS**: 100+ rows with irs_start_position, irs_field_length fields populated
- **MUMPS**: 100+ rows with mumps_global_name or mumps_file_no fields populated

### Performance Benchmarks

Target processing speeds:
- **Kubernetes**: 50-100 manifests/minute
- **FastAPI**: 20-30 endpoints/minute
- **COBOL**: 10-20 copybooks/minute
- **IRS**: 30-50 layouts/minute
- **MUMPS**: 15-25 dictionaries/minute

Total time for demo dataset: < 5 minutes

## Next Steps

After successful M1 ingestion:
1. Run embedding generation on `source_metadata` table
2. Implement vector search and similarity queries
3. Execute gap analysis rules on ingested content
4. Generate visualization dashboards for data quality

## Support

- Check logs: `tail -f logs/ingestion.log`
- Review errors: Query `source_metadata_errors` table
- Validate schemas: Use `--dry-run` flag before full ingestion
- Performance issues: Monitor `ingestion_log` processing times