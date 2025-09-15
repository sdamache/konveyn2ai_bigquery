-- BigQuery Schema Contracts for Vector Embeddings
-- Project: KonveyN2AI BigQuery Vector Backend
-- Dataset: semantic_gap_detector

-- Enhanced source_embeddings table with vector support
CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.${DATASET_ID}.source_embeddings` (
  -- Primary identifiers
  chunk_id STRING NOT NULL 
    OPTIONS(description="Unique chunk identifier linking to source_metadata"),
  
  -- Embedding metadata  
  model STRING NOT NULL
    OPTIONS(description="Embedding model name (e.g., text-embedding-004)"),
  
  content_hash STRING NOT NULL
    OPTIONS(description="SHA256 hash of normalized text content for deduplication"),
  
  -- Vector data (768-dimensional for text-embedding-004)
  embedding ARRAY<FLOAT64> NOT NULL
    OPTIONS(description="768-dimensional embedding vector array"),
  
  -- Timestamps and partitioning
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    OPTIONS(description="Timestamp when embedding was generated"),
  
  partition_date DATE DEFAULT CURRENT_DATE()
    OPTIONS(description="Date partition for query optimization"),
  
  -- Optional metadata for analytics
  source_type STRING
    OPTIONS(description="Type of source artifact (k8s, fastapi, cobol, etc.)"),
  
  artifact_id STRING  
    OPTIONS(description="Source identifier or filename"),
  
  -- Constraints
  CONSTRAINT pk_source_embeddings 
    PRIMARY KEY (chunk_id, model, content_hash) NOT ENFORCED
)
PARTITION BY partition_date
CLUSTER BY chunk_id, model, content_hash
OPTIONS(
  description="Stores 768-dimensional embeddings for semantic search with BigQuery VECTOR operations",
  labels=[("component", "vector-backend"), ("env", "production")]
);

-- Create vector index for similarity search optimization
CREATE VECTOR INDEX IF NOT EXISTS embedding_vector_index
ON `${PROJECT_ID}.${DATASET_ID}.source_embeddings`(embedding)
OPTIONS(
  index_type='IVF',
  distance_type='COSINE',
  ivf_options='{"num_lists": 1000}'
);

-- Validate embedding dimensions constraint
CREATE OR REPLACE FUNCTION `${PROJECT_ID}.${DATASET_ID}.validate_embedding_dimensions`(embedding ARRAY<FLOAT64>)
RETURNS BOOL
LANGUAGE js AS """
  if (!embedding || embedding.length !== 768) {
    return false;
  }
  return embedding.every(val => typeof val === 'number' && !isNaN(val));
""";

-- View for embedding statistics and monitoring
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.embedding_stats` AS
SELECT 
  COUNT(*) as total_embeddings,
  COUNT(DISTINCT chunk_id) as unique_chunks,
  COUNT(DISTINCT model) as unique_models,
  COUNT(DISTINCT source_type) as source_types,
  MIN(created_at) as first_embedding,
  MAX(created_at) as last_embedding,
  COUNTIF(ARRAY_LENGTH(embedding) = 768) as valid_dimensions,
  COUNTIF(ARRAY_LENGTH(embedding) != 768) as invalid_dimensions,
  AVG(ARRAY_LENGTH(embedding)) as avg_dimensions,
  -- Storage metrics
  SUM(
    -- Approximate size: 768 floats * 8 bytes + metadata overhead
    768 * 8 + LENGTH(chunk_id) + LENGTH(model) + LENGTH(IFNULL(source_type, '')) + 100
  ) / 1024 / 1024 as estimated_size_mb
FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings`;

-- View for cache efficiency analysis
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.content_deduplication_stats` AS
WITH content_groups AS (
  SELECT 
    content_hash,
    COUNT(*) as duplicate_count,
    COUNT(DISTINCT chunk_id) as unique_chunks,
    ARRAY_AGG(DISTINCT model LIMIT 5) as models_used,
    MIN(created_at) as first_seen,
    MAX(created_at) as last_seen
  FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
  GROUP BY content_hash
)
SELECT 
  COUNT(*) as total_content_hashes,
  COUNT(*) - COUNTIF(duplicate_count = 1) as deduplicated_hashes,
  SUM(duplicate_count - 1) as api_calls_saved,
  ROUND(100.0 * (SUM(duplicate_count - 1) / SUM(duplicate_count)), 2) as deduplication_rate_percent,
  AVG(duplicate_count) as avg_duplicates_per_hash,
  MAX(duplicate_count) as max_duplicates_per_hash
FROM content_groups;

-- Materialized view for frequently accessed embedding metadata
CREATE MATERIALIZED VIEW IF NOT EXISTS `${PROJECT_ID}.${DATASET_ID}.embedding_metadata_mv`
PARTITION BY partition_date
CLUSTER BY chunk_id, source_type
AS
SELECT 
  e.chunk_id,
  e.model,
  e.content_hash,
  e.created_at,
  e.partition_date,
  e.source_type,
  e.artifact_id,
  -- Join with source metadata for context
  sm.source,
  sm.artifact_type,
  sm.text_content,
  sm.kind,
  sm.api_path,
  sm.record_name,
  -- Embedding vector length validation
  ARRAY_LENGTH(e.embedding) as embedding_dimensions,
  `${PROJECT_ID}.${DATASET_ID}.validate_embedding_dimensions`(e.embedding) as valid_embedding
FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings` e
JOIN `${PROJECT_ID}.${DATASET_ID}.source_metadata` sm
  ON e.chunk_id = sm.chunk_id;

-- Function for vector similarity search with metadata
CREATE OR REPLACE FUNCTION `${PROJECT_ID}.${DATASET_ID}.find_similar_embeddings`(
  query_vector ARRAY<FLOAT64>,
  top_k INT64,
  min_similarity FLOAT64
)
RETURNS TABLE<
  chunk_id STRING,
  similarity_score FLOAT64,
  text_content STRING,
  source_type STRING,
  model STRING
>
LANGUAGE SQL AS """
  SELECT 
    base.chunk_id,
    1.0 - base.distance as similarity_score,
    SUBSTR(mv.text_content, 1, 200) as text_content,
    mv.source_type,
    base.model
  FROM VECTOR_SEARCH(
    TABLE `${PROJECT_ID}.${DATASET_ID}.source_embeddings`,
    'embedding',
    query_vector,
    top_k => top_k,
    distance_type => 'COSINE'
  ) AS base
  JOIN `${PROJECT_ID}.${DATASET_ID}.embedding_metadata_mv` mv
    ON base.chunk_id = mv.chunk_id
  WHERE (1.0 - base.distance) >= min_similarity
  ORDER BY base.distance ASC;
""";

-- Stored procedure for batch embedding validation
CREATE OR REPLACE PROCEDURE `${PROJECT_ID}.${DATASET_ID}.validate_embeddings_batch`(
  start_date DATE,
  end_date DATE,
  OUT validation_summary STRING
)
BEGIN
  DECLARE total_embeddings INT64;
  DECLARE invalid_embeddings INT64;
  DECLARE duplicate_embeddings INT64;
  DECLARE orphaned_embeddings INT64;
  
  -- Count total embeddings in date range
  SET total_embeddings = (
    SELECT COUNT(*)
    FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
    WHERE partition_date BETWEEN start_date AND end_date
  );
  
  -- Count invalid embeddings (wrong dimensions)
  SET invalid_embeddings = (
    SELECT COUNT(*)
    FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
    WHERE partition_date BETWEEN start_date AND end_date
      AND NOT `${PROJECT_ID}.${DATASET_ID}.validate_embedding_dimensions`(embedding)
  );
  
  -- Count duplicate embeddings (same chunk_id, model, content_hash)
  SET duplicate_embeddings = (
    SELECT SUM(cnt - 1)
    FROM (
      SELECT COUNT(*) as cnt
      FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
      WHERE partition_date BETWEEN start_date AND end_date
      GROUP BY chunk_id, model, content_hash
      HAVING COUNT(*) > 1
    )
  );
  
  -- Count orphaned embeddings (no corresponding source_metadata)
  SET orphaned_embeddings = (
    SELECT COUNT(*)
    FROM `${PROJECT_ID}.${DATASET_ID}.source_embeddings` e
    LEFT JOIN `${PROJECT_ID}.${DATASET_ID}.source_metadata` sm
      ON e.chunk_id = sm.chunk_id
    WHERE e.partition_date BETWEEN start_date AND end_date
      AND sm.chunk_id IS NULL
  );
  
  -- Generate summary report
  SET validation_summary = FORMAT("""
Embedding Validation Report (%t to %t):
- Total embeddings: %d
- Invalid dimensions: %d (%.2f%%)
- Duplicate entries: %d (%.2f%%)
- Orphaned embeddings: %d (%.2f%%)
- Health score: %.1f%%
""", 
    start_date, end_date,
    total_embeddings,
    invalid_embeddings, 100.0 * invalid_embeddings / GREATEST(total_embeddings, 1),
    IFNULL(duplicate_embeddings, 0), 100.0 * IFNULL(duplicate_embeddings, 0) / GREATEST(total_embeddings, 1),
    orphaned_embeddings, 100.0 * orphaned_embeddings / GREATEST(total_embeddings, 1),
    100.0 * (1.0 - (invalid_embeddings + IFNULL(duplicate_embeddings, 0) + orphaned_embeddings) / GREATEST(total_embeddings, 1))
  );
END;

-- Row-level security policy for multi-tenant scenarios (optional)
-- CREATE ROW ACCESS POLICY IF NOT EXISTS tenant_isolation_policy
-- ON `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
-- GRANT TO ('user:authorized@company.com')
-- FILTER USING (artifact_id LIKE CONCAT(SESSION_USER(), '%'));

-- Grants for service accounts (template - customize per environment)
-- GRANT `roles/bigquery.dataEditor` ON TABLE `${PROJECT_ID}.${DATASET_ID}.source_embeddings` 
-- TO 'serviceAccount:embedding-pipeline@${PROJECT_ID}.iam.gserviceaccount.com';

-- Data retention policy (optional - customize based on requirements)
-- ALTER TABLE `${PROJECT_ID}.${DATASET_ID}.source_embeddings`
-- SET OPTIONS (
--   partition_expiration_days = 365,
--   description = "Embeddings older than 1 year will be automatically deleted"
-- );