# BigQuery Vector Backend - Quickstart Guide

## Prerequisites

- Google Cloud Project with BigQuery API enabled
- Python 3.11 or higher
- Service account with BigQuery Admin permissions
- ~1000 existing vectors from current Vertex AI system (for migration testing)

## Setup Instructions

### 1. Environment Configuration

```bash
# Clone and navigate to project
git clone https://github.com/sdamache/konveyn2ai_bigquery.git
cd konveyn2ai_bigquery
git checkout 001-bigquery-vector-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install google-cloud-bigquery google-cloud-aiplatform pandas numpy scikit-learn
```

### 2. Google Cloud Configuration

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT=konveyn2ai
export BIGQUERY_DATASET_ID=semantic_gap_detector
export GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account.json

# Authenticate with Google Cloud (for local development)
gcloud auth application-default login
gcloud config set project konveyn2ai
```

### 3. BigQuery Dataset and Tables Setup

```bash
# Run the setup command
make setup
```

**Expected output:**
```
Creating BigQuery dataset: konveyn2ai.semantic_gap_detector
✓ Dataset created successfully
Creating table: source_metadata
✓ Table source_metadata created with partitioning
Creating table: source_embeddings  
✓ Table source_embeddings created with clustering
Creating table: gap_metrics
✓ Table gap_metrics created
Creating vector index: embedding_similarity_index
✓ Vector index created (status: CREATING)
Setup completed in 12.3 seconds
```

### 4. Verify Installation

```python
# test_setup.py
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
from src.janapada_memory.schema_manager import SchemaManager

# Test BigQuery connection
schema_manager = SchemaManager()
validation_result = schema_manager.validate_schema()
print(f"Schema validation: {validation_result['overall_status']}")

# Test vector store
vector_store = BigQueryVectorStore()
health_status = vector_store.health_check()
print(f"Vector store status: {health_status}")
```

## Basic Usage Examples

### 1. Insert Vector Embeddings

```python
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
import numpy as np

# Initialize vector store
vector_store = BigQueryVectorStore()

# Sample code chunk
chunk_data = {
    "chunk_id": "func_calculate_similarity_001", 
    "text_content": "def calculate_similarity(vector1, vector2): return cosine_similarity(vector1, vector2)",
    "source": "src/utils/similarity.py",
    "artifact_type": "code",
    "kind": "function",
    "record_name": "calculate_similarity"
}

# Generate embedding (768 dimensions)
embedding = np.random.rand(768).tolist()  # Replace with actual embedding

# Insert into BigQuery
result = vector_store.insert_embedding(
    chunk_data=chunk_data,
    embedding=embedding
)
print(f"Inserted: {result['chunk_id']}")
```

### 2. Search Similar Vectors

```python
# Text-based similarity search
query_text = "function that calculates cosine similarity between vectors"
results = vector_store.search_similar_text(
    query_text=query_text,
    limit=5,
    similarity_threshold=0.7,
    artifact_types=["code"]
)

for result in results:
    print(f"Score: {result['similarity_score']:.3f} - {result['text_content'][:100]}...")
```

### 3. Direct Vector Search

```python
# Direct vector similarity search
query_embedding = np.random.rand(768).tolist()  # Your query vector
results = vector_store.search_similar_vectors(
    query_embedding=query_embedding,
    limit=10,
    similarity_threshold=0.6
)

for result in results:
    print(f"Chunk: {result['chunk_id']} - Score: {result['similarity_score']:.3f}")
```

## Migration from Vertex AI

### 1. Export Existing Vectors

```python
# scripts/export_vertex_vectors.py
from src.janapada_memory.vector_index import VectorIndex  # Current Vertex AI implementation
import json

# Export current vectors
vector_index = VectorIndex()
existing_vectors = vector_index.export_all_vectors()

# Save to backup file
with open('vectors_backup.json', 'w') as f:
    json.dump(existing_vectors, f)

print(f"Exported {len(existing_vectors)} vectors to vectors_backup.json")
```

### 2. Dimension Reduction (3072 → 768)

```python
# scripts/reduce_dimensions.py
from sklearn.decomposition import PCA
import json
import numpy as np

# Load exported vectors
with open('vectors_backup.json', 'r') as f:
    vectors_data = json.load(f)

# Extract embeddings for PCA
embeddings_3072 = [v['embedding'] for v in vectors_data]
embeddings_array = np.array(embeddings_3072)

# Apply PCA reduction
pca = PCA(n_components=768)
embeddings_768 = pca.fit_transform(embeddings_array)

# Update vector data
for i, vector_data in enumerate(vectors_data):
    vector_data['embedding'] = embeddings_768[i].tolist()
    
# Save reduced vectors
with open('vectors_768_dim.json', 'w') as f:
    json.dump(vectors_data, f)

print(f"Variance preserved: {pca.explained_variance_ratio_.sum():.3f}")
print(f"Reduced {len(vectors_data)} vectors to 768 dimensions")
```

### 3. Import to BigQuery

```python
# scripts/import_to_bigquery.py
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
import json
from tqdm import tqdm

# Initialize BigQuery vector store
vector_store = BigQueryVectorStore()

# Load reduced vectors
with open('vectors_768_dim.json', 'r') as f:
    vectors_data = json.load(f)

# Batch import
batch_size = 100
successful_imports = 0

for i in tqdm(range(0, len(vectors_data), batch_size)):
    batch = vectors_data[i:i + batch_size]
    
    for vector_data in batch:
        try:
            result = vector_store.insert_embedding(
                chunk_data={
                    'chunk_id': vector_data['chunk_id'],
                    'text_content': vector_data['text_content'],
                    'source': vector_data['source'],
                    'artifact_type': vector_data['artifact_type'],
                    'kind': vector_data.get('kind'),
                    'api_path': vector_data.get('api_path'),
                    'record_name': vector_data.get('record_name')
                },
                embedding=vector_data['embedding']
            )
            successful_imports += 1
        except Exception as e:
            print(f"Failed to import {vector_data['chunk_id']}: {e}")

print(f"Successfully imported {successful_imports}/{len(vectors_data)} vectors")
```

## Testing the Migration

### 1. Quality Verification

```python
# scripts/verify_migration_quality.py
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
from src.janapada_memory.vector_index import VectorIndex  # Old system
import numpy as np

bigquery_store = BigQueryVectorStore()
vertex_store = VectorIndex()

# Test queries
test_queries = [
    "function that calculates similarity between vectors",
    "api endpoint for user authentication", 
    "database schema for user profiles"
]

for query in test_queries:
    # Query both systems
    bq_results = bigquery_store.search_similar_text(query, limit=5)
    vertex_results = vertex_store.search_similar_text(query, limit=5)
    
    # Compare top results
    bq_top_chunks = [r['chunk_id'] for r in bq_results[:3]]
    vertex_top_chunks = [r['chunk_id'] for r in vertex_results[:3]]
    
    overlap = len(set(bq_top_chunks) & set(vertex_top_chunks))
    print(f"Query: {query}")
    print(f"Top-3 overlap: {overlap}/3 ({overlap/3*100:.1f}%)")
    print("---")
```

### 2. Performance Benchmarks

```python
# scripts/benchmark_performance.py
import time
import numpy as np
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore

vector_store = BigQueryVectorStore()

# Performance test
test_queries = [np.random.rand(768).tolist() for _ in range(10)]
latencies = []

for query_vector in test_queries:
    start_time = time.time()
    results = vector_store.search_similar_vectors(query_vector, limit=10)
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    latencies.append(latency)

print(f"Average latency: {np.mean(latencies):.1f}ms")
print(f"95th percentile: {np.percentile(latencies, 95):.1f}ms")
print(f"Max latency: {np.max(latencies):.1f}ms")
```

## Demo Workflow

### End-to-End Demo Script

```python
# demo.py - Complete workflow demonstration
from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
from src.janapada_memory.schema_manager import SchemaManager
import numpy as np

print("🚀 KonveyN2AI BigQuery Vector Backend Demo")
print("=" * 50)

# 1. Schema validation
print("1. Validating BigQuery schema...")
schema_manager = SchemaManager()
validation = schema_manager.validate_schema()
print(f"   ✓ Schema status: {validation['overall_status']}")

# 2. Initialize vector store
print("2. Initializing vector store...")
vector_store = BigQueryVectorStore()
health = vector_store.health_check()
print(f"   ✓ Vector store: {health['status']}")

# 3. Insert sample data
print("3. Inserting sample embeddings...")
sample_data = [
    {
        "chunk_id": "demo_func_001",
        "text_content": "def calculate_cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
        "source": "demo/similarity.py", 
        "artifact_type": "code",
        "kind": "function"
    },
    {
        "chunk_id": "demo_api_001", 
        "text_content": "POST /api/v1/similarity - Calculate similarity between two vectors",
        "source": "api/endpoints.md",
        "artifact_type": "documentation",
        "kind": "api_doc"
    }
]

for data in sample_data:
    embedding = np.random.rand(768).tolist()
    result = vector_store.insert_embedding(data, embedding)
    print(f"   ✓ Inserted: {result['chunk_id']}")

# 4. Search demonstration
print("4. Performing similarity search...")
search_results = vector_store.search_similar_text(
    "similarity calculation function",
    limit=5,
    similarity_threshold=0.1
)

print(f"   Found {len(search_results)} similar items:")
for i, result in enumerate(search_results, 1):
    print(f"   {i}. Score: {result['similarity_score']:.3f} - {result['chunk_id']}")

print("\n🎉 Demo completed successfully!")
print("BigQuery vector backend is ready for production use.")
```

## Next Steps

1. **Scale Testing**: Test with larger datasets (1000+ vectors)
2. **Performance Tuning**: Optimize vector indexes based on query patterns
3. **Integration**: Connect to existing KonveyN2AI orchestrator and prompter components
4. **Monitoring**: Set up cost and performance monitoring dashboards
5. **Production Deploy**: Configure Cloud Run deployment with proper authentication

## Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
# Verify credentials
gcloud auth list
gcloud auth application-default print-access-token
```

**BigQuery Permissions:**
- Ensure service account has `BigQuery Admin` or minimum required permissions
- Check dataset exists and is accessible

**Vector Index Issues:**
```sql
-- Check index status
SELECT * FROM `konveyn2ai.semantic_gap_detector.INFORMATION_SCHEMA.VECTOR_INDEXES`;
```

**Performance Issues:**
- Monitor slot usage in BigQuery console
- Check partition pruning is working
- Verify vector index is in ACTIVE state

### Support

- Check logs: `make logs`
- Run diagnostics: `make diagnose`
- View metrics: Access BigQuery console monitoring tab

This quickstart provides a complete path from setup to production-ready BigQuery vector backend for KonveyN2AI.