# Quickstart: BigQuery Vector Embeddings

## Prerequisites

- Python 3.11+
- Google Cloud Project with BigQuery enabled
- Google API key for Gemini
- Service account with BigQuery permissions

## 🚀 Quick Setup (5 minutes)

### 1. Environment Configuration
```bash
# Clone repository and setup
git clone https://github.com/sdamache/konveyn2ai_bigquery.git
cd konveyn2ai_bigquery
git checkout 003-bigquery-vector-embeddings

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys and Credentials
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
export GOOGLE_CLOUD_PROJECT=your-project-id
export BIGQUERY_DATASET_ID=semantic_gap_detector  
export GOOGLE_API_KEY=your-gemini-api-key
export GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json

# Verify credentials
python -c "from google.cloud import bigquery; print('✅ BigQuery connection OK')"
```

### 3. Initialize BigQuery Infrastructure
```bash
# Create dataset and tables
make setup

# Expected output:
# ✅ BigQuery setup completed
# - Dataset: semantic_gap_detector
# - Tables: source_metadata, source_embeddings, gap_metrics
```

## 🧠 Generate Your First Embeddings

### 4. Basic Embedding Generation
```bash
# Generate embeddings for all pending chunks
make embeddings

# Expected output:
# 🧠 Generating embeddings for BigQuery vector store...
# ✅ Embedding generation completed
# 
# Embedding Generation Results:
#   Chunks scanned: 150
#   Embeddings generated: 145  
#   Embeddings stored: 145
#   Processing time: 45000ms
```

### 5. Test Vector Search
```bash
# Test similarity search functionality
python test_vector_search.py \
    --project your-project-id \
    --dataset semantic_gap_detector \
    --verbose

# Expected output:
# Vector Search Test Results:
#   Test Status: SUCCESS
#   Total embeddings: 145
#   Embedding dimensions: 768
#   VECTOR_SEARCH Results (5 found)
```

## 📊 Verify Results

### 6. Check BigQuery Data
```sql
-- Query in BigQuery Studio
SELECT 
  COUNT(*) as total_embeddings,
  COUNT(DISTINCT chunk_id) as unique_chunks,
  AVG(ARRAY_LENGTH(embedding)) as avg_dimensions
FROM `your-project.semantic_gap_detector.source_embeddings`;
```

### 7. Test Similarity Search
```sql
-- Find similar content using vector search
DECLARE query_embedding ARRAY<FLOAT64>;

-- Get a sample embedding for testing
SET query_embedding = (
  SELECT embedding 
  FROM `your-project.semantic_gap_detector.source_embeddings` 
  LIMIT 1
);

-- Find 5 most similar chunks
SELECT 
  base.chunk_id,
  base.distance,
  1.0 - base.distance as similarity_score,
  SUBSTR(sm.text_content, 1, 100) as content_preview
FROM VECTOR_SEARCH(
  TABLE `your-project.semantic_gap_detector.source_embeddings`,
  'embedding',
  query_embedding,
  top_k => 5,
  distance_type => 'COSINE'
) AS base
JOIN `your-project.semantic_gap_detector.source_metadata` sm
  ON base.chunk_id = sm.chunk_id
ORDER BY base.distance;
```

## 🔧 Advanced Usage

### Batch Processing with Limits
```bash
# Process only 100 chunks (testing)
LIMIT=100 make embeddings

# Dry run (no actual storage)
DRY_RUN=1 LIMIT=50 make embeddings

# Custom batch size
EMBED_BATCH_SIZE=16 make embeddings

# Filter specific source types
WHERE="source_type = 'k8s'" make embeddings
```

### Cache Management
```bash
# Check cache status
ls -la .cache/embeddings/
echo "Cache entries: $(ls .cache/embeddings/ | wc -l)"

# Clear cache (force regeneration)
rm -rf .cache/embeddings/

# Custom cache directory
EMBED_CACHE_DIR=/tmp/embeddings make embeddings
```

### Python API Usage
```python
from pipeline.embedding import EmbeddingPipeline

# Initialize pipeline
pipeline = EmbeddingPipeline(
    project_id="your-project-id",
    dataset_id="semantic_gap_detector",
    api_key="your-gemini-api-key"
)

# Generate embeddings
result = pipeline.generate_embeddings(limit=100)

print(f"✅ Generated {result['embeddings_generated']} embeddings")
print(f"📊 API calls: {result['generator_stats']['api_calls']}")
print(f"🎯 Cache hits: {result['generator_stats']['cache_hits']}")
```

## 📈 Performance Optimization

### Monitor Cache Efficiency
```python
from pipeline.embedding import EmbeddingCache

cache = EmbeddingCache()
cache_files = list(cache.cache_dir.glob("*.json"))
print(f"Cache entries: {len(cache_files)}")

# Calculate cache size
total_size = sum(f.stat().st_size for f in cache_files)
print(f"Cache size: {total_size / 1024 / 1024:.1f} MB")
```

### Query Performance Analysis
```sql
-- Analyze embedding distribution
SELECT 
  source_type,
  COUNT(*) as embeddings,
  AVG(ARRAY_LENGTH(embedding)) as avg_dimensions,
  COUNT(DISTINCT content_hash) as unique_content
FROM `your-project.semantic_gap_detector.source_embeddings`
GROUP BY source_type
ORDER BY embeddings DESC;

-- Check for duplicates (cache effectiveness)
SELECT 
  content_hash,
  COUNT(*) as duplicate_count,
  COUNT(DISTINCT chunk_id) as unique_chunks
FROM `your-project.semantic_gap_detector.source_embeddings`
GROUP BY content_hash
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 10;
```

## 🐛 Troubleshooting

### Common Issues

**"No pending chunks found"**
```bash
# Check if source_metadata has data
python -c "
from src.janapada_memory.bigquery_connection import BigQueryConnection
conn = BigQueryConnection('your-project', 'semantic_gap_detector')
result = list(conn.execute_query('SELECT COUNT(*) as count FROM source_metadata'))
print(f'Source metadata rows: {result[0].count}')
"
```

**"API key invalid"**
```bash
# Test Gemini API access
python -c "
import google.generativeai as genai
genai.configure(api_key='your-api-key')
response = genai.embed_content(model='models/text-embedding-004', content='test')
print(f'✅ API key works, got {len(response.embedding)} dimensions')
"
```

**"Permission denied"**
```bash
# Check BigQuery permissions
python -c "
from google.cloud import bigquery
client = bigquery.Client()
dataset = client.dataset('semantic_gap_detector')
print(f'✅ Can access dataset: {dataset.dataset_id}')
"
```

### Performance Issues

**Slow embedding generation**
- Reduce batch size: `EMBED_BATCH_SIZE=8`
- Use cache: Ensure `.cache/embeddings/` is persistent
- Check API quotas in Google Cloud Console

**High API costs**
- Monitor cache hit rate (should be >70%)
- Check for content duplication
- Use `DRY_RUN=1` for testing

**BigQuery query timeouts**
- Ensure proper clustering is enabled
- Use time-based partitioning for large datasets
- Consider materialized views for frequent queries

## 📚 Next Steps

1. **Integrate with Existing Workflows**: Add embedding generation to your data pipelines
2. **Optimize for Scale**: Configure batch sizes and caching for your dataset size
3. **Monitor Costs**: Set up BigQuery cost monitoring and alerts
4. **Extend Functionality**: Add support for multiple embedding models
5. **Production Deployment**: Set up automated embedding generation with Cloud Functions

## 🔗 Additional Resources

- [BigQuery Vector Search Documentation](https://cloud.google.com/bigquery/docs/vector-search)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Project Architecture Guide](../../ARCHITECTURE.md)
- [Performance Optimization Guide](./research.md#performance-optimization-research)

---

**💡 Pro Tip**: Start with `LIMIT=10 DRY_RUN=1` to test your setup before processing large datasets!