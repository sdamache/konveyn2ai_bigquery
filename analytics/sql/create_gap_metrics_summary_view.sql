-- Create gap_metrics_summary view for notebook compatibility
-- This view aggregates data from gap_metrics and source_metadata tables
-- to provide the interface expected by issue6_visualization_notebook.ipynb

CREATE OR REPLACE VIEW `{{ project_id }}.{{ dataset }}.gap_metrics_summary` AS
WITH gap_aggregates AS (
  SELECT
    gm.analysis_id,
    sm.artifact_type,
    gm.metric_type as rule_name,
    COUNTIF(gm.metric_value >= 0.5) as count_passed,
    COUNTIF(gm.metric_value < 0.5) as count_failed,
    COUNT(*) as total_chunks,
    AVG(gm.metric_value) as avg_confidence,
    MIN(gm.metric_value) as min_confidence,
    MAX(gm.metric_value) as max_confidence,
    STDDEV(gm.metric_value) as confidence_stddev,
    ARRAY_AGG(gm.chunk_id LIMIT 3) as sample_chunks_array
  FROM `{{ project_id }}.{{ dataset }}.gap_metrics` gm
  JOIN `{{ project_id }}.{{ dataset }}.source_metadata` sm
    ON gm.chunk_id = sm.chunk_id
  GROUP BY gm.analysis_id, sm.artifact_type, gm.metric_type
)
SELECT
  artifact_type,
  rule_name,
  count_passed,
  count_failed,
  total_chunks,
  avg_confidence,
  min_confidence,
  max_confidence,
  confidence_stddev,
  CASE
    WHEN total_chunks > 0
    THEN CAST((count_passed * 100.0) / total_chunks AS NUMERIC)
    ELSE 0.0
  END as pass_rate_percent,
  ARRAY_TO_STRING(sample_chunks_array, ',') as sample_chunks
FROM gap_aggregates
ORDER BY artifact_type, rule_name;