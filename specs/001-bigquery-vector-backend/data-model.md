# Data Model: BigQuery Vector Backend

## Entity Relationships

```
source_metadata (1) ←→ (1) source_embeddings [chunk_id]
source_metadata (1) ←→ (*) gap_metrics [chunk_id]
source_embeddings (1) ←→ (*) gap_metrics [chunk_id]
```

## Table Schemas

### source_metadata
Primary table containing structured metadata for code artifacts.

```sql
CREATE TABLE `konveyn2ai.semantic_gap_detector.source_metadata` (
  chunk_id STRING NOT NULL,
  source STRING NOT NULL,
  artifact_type STRING NOT NULL,
  text_content STRING NOT NULL,
  kind STRING,
  api_path STRING,
  record_name STRING,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  partition_date DATE GENERATED ALWAYS AS (DATE(created_at)) STORED
)
PARTITION BY partition_date
CLUSTER BY artifact_type, source, chunk_id
OPTIONS (
  partition_expiration_days = 365,
  description = "Structured metadata for code artifacts and semantic chunks"
);
```

**Field Definitions:**
- `chunk_id`: Unique identifier linking to embeddings and metrics tables
- `source`: File path or source identifier for the artifact
- `artifact_type`: Type classification (code, documentation, api, schema)  
- `text_content`: Raw text content for the chunk
- `kind`: Specific kind within artifact type (function, class, endpoint)
- `api_path`: API endpoint path if applicable
- `record_name`: Specific record/entity name if applicable

**Validation Rules:**
- `chunk_id` must be unique across all tables
- `artifact_type` must be one of: ['code', 'documentation', 'api', 'schema', 'test']
- `text_content` minimum length: 10 characters
- `source` must be valid file path or URL format

### source_embeddings
Vector embeddings table storing 768-dimensional semantic representations.

```sql
CREATE TABLE `konveyn2ai.semantic_gap_detector.source_embeddings` (
  chunk_id STRING NOT NULL,
  embedding ARRAY<FLOAT64> NOT NULL,
  embedding_model STRING DEFAULT 'text-embedding-004',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  partition_date DATE GENERATED ALWAYS AS (DATE(created_at)) STORED
)
PARTITION BY partition_date
CLUSTER BY chunk_id
OPTIONS (
  partition_expiration_days = 365,
  description = "768-dimensional vector embeddings for semantic similarity search"
);
```

**Field Definitions:**
- `chunk_id`: Foreign key reference to source_metadata.chunk_id
- `embedding`: 768-dimensional float array representing semantic content
- `embedding_model`: Model used to generate embedding for version tracking
- `created_at`: Timestamp for embedding generation tracking

**Validation Rules:**
- `embedding` array must contain exactly 768 float64 elements
- All embedding values must be finite (not NaN or Infinity)
- `chunk_id` must exist in source_metadata table
- `embedding_model` must follow format: '{provider}-{model}-{version}'

### gap_metrics  
Analysis outputs and computed metrics for semantic gap detection.

```sql
CREATE TABLE `konveyn2ai.semantic_gap_detector.gap_metrics` (
  analysis_id STRING NOT NULL,
  chunk_id STRING NOT NULL, 
  metric_type STRING NOT NULL,
  metric_value FLOAT64,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  partition_date DATE GENERATED ALWAYS AS (DATE(created_at)) STORED
)
PARTITION BY partition_date
CLUSTER BY analysis_id, metric_type, chunk_id  
OPTIONS (
  partition_expiration_days = 365,
  description = "Computed metrics and analysis results for gap detection"
);
```

**Field Definitions:**
- `analysis_id`: Unique identifier for analysis batch/session
- `chunk_id`: Foreign key reference to source_metadata.chunk_id  
- `metric_type`: Type of analysis metric computed
- `metric_value`: Numerical result of the metric calculation
- `metadata`: JSON object containing additional analysis context

**Validation Rules:**
- `metric_type` must be one of: ['similarity_score', 'coverage_gap', 'complexity_score', 'relevance_score']
- `metric_value` must be between 0.0 and 1.0 for score types
- `analysis_id` format: '{timestamp}_{analysis_type}_{version}'
- `metadata` JSON must be valid and parseable

## Entity State Transitions

### Chunk Lifecycle
```
1. [CREATED] → source_metadata inserted with text content
2. [EMBEDDED] → source_embeddings generated for chunk_id  
3. [ANALYZED] → gap_metrics computed using embedding similarity
4. [ARCHIVED] → automatic partition expiration after 365 days
```

### Analysis Workflow States
```
PENDING → PROCESSING → COMPLETED → ARCHIVED
```

## Data Relationships & Constraints

### Primary Keys
- `source_metadata`: chunk_id (enforced at application level)
- `source_embeddings`: chunk_id (enforced at application level)
- `gap_metrics`: (analysis_id, chunk_id, metric_type) composite key

### Foreign Key Relationships
```sql
-- Application-level foreign key constraints
-- BigQuery does not enforce referential integrity

-- source_embeddings.chunk_id REFERENCES source_metadata.chunk_id
-- gap_metrics.chunk_id REFERENCES source_metadata.chunk_id
```

### Data Integrity Rules
1. Every `chunk_id` in source_embeddings must have corresponding source_metadata record
2. Vector embeddings must be generated before gap metrics can be computed
3. Analysis metrics are immutable once created (append-only pattern)
4. Partition pruning required for queries spanning >30 days

## Performance Optimization Schema

### Indexing Strategy
```sql
-- Vector index for similarity search
CREATE VECTOR INDEX embedding_similarity_index
ON `konveyn2ai.semantic_gap_detector.source_embeddings`(embedding)
OPTIONS(
  distance_type='COSINE',
  index_type='IVF',
  ivf_options='{"num_lists": 1000}'
);

-- Standard indexes for frequent queries
CREATE INDEX metadata_lookup_index 
ON `konveyn2ai.semantic_gap_detector.source_metadata`(artifact_type, source);

CREATE INDEX metrics_analysis_index
ON `konveyn2ai.semantic_gap_detector.gap_metrics`(analysis_id, metric_type);
```

### Query Patterns Optimization
1. **Similarity Search**: Optimized for vector similarity queries with COSINE distance
2. **Temporal Queries**: Partition pruning on created_at timestamps  
3. **Join Patterns**: Clustered keys optimize metadata → embeddings joins
4. **Analytical Queries**: Clustered metric_type enables efficient aggregations

## Data Validation Schema

### Application-Level Constraints
```python
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class SourceMetadata(BaseModel):
    chunk_id: str
    source: str  
    artifact_type: str
    text_content: str
    kind: Optional[str] = None
    api_path: Optional[str] = None
    record_name: Optional[str] = None
    created_at: datetime
    
    @validator('artifact_type')
    def validate_artifact_type(cls, v):
        allowed_types = ['code', 'documentation', 'api', 'schema', 'test']
        if v not in allowed_types:
            raise ValueError(f'artifact_type must be one of {allowed_types}')
        return v
    
    @validator('text_content')
    def validate_text_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('text_content must be at least 10 characters')
        return v

class SourceEmbedding(BaseModel):
    chunk_id: str
    embedding: List[float]
    embedding_model: str = 'text-embedding-004'
    created_at: datetime
    
    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        if len(v) != 768:
            raise ValueError('embedding must contain exactly 768 dimensions')
        if not all(isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x) for x in v):
            raise ValueError('embedding must contain finite numeric values')
        return v

class GapMetric(BaseModel):
    analysis_id: str
    chunk_id: str
    metric_type: str  
    metric_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        allowed_types = ['similarity_score', 'coverage_gap', 'complexity_score', 'relevance_score']
        if v not in allowed_types:
            raise ValueError(f'metric_type must be one of {allowed_types}')
        return v
    
    @validator('metric_value')
    def validate_metric_value(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('metric_value must be between 0.0 and 1.0')
        return v
```

## Migration Strategy

### From Existing System
1. **Extract** current 3072-dim embeddings from Vertex AI
2. **Transform** to 768-dim using PCA reduction  
3. **Load** into BigQuery tables with validation
4. **Verify** data integrity and semantic quality preservation

### Schema Evolution
```sql
-- Future schema changes (example)
-- Add new fields without breaking existing queries
ALTER TABLE `konveyn2ai.semantic_gap_detector.source_metadata`
ADD COLUMN language STRING,
ADD COLUMN complexity_score FLOAT64;

-- Modify with backward compatibility
CREATE OR REPLACE VIEW source_metadata_v2 AS
SELECT 
  *,
  COALESCE(language, 'unknown') as language_normalized,
  COALESCE(complexity_score, 0.0) as complexity_normalized
FROM `konveyn2ai.semantic_gap_detector.source_metadata`;
```

This data model provides a robust foundation for the BigQuery vector backend while maintaining flexibility for future enhancements and optimizations.