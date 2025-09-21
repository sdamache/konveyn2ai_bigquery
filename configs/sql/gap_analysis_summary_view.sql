-- Gap Metrics Summary View for Issue #6 Notebook
-- Transforms the actual gap_metrics output to the format expected by the visualization notebook

CREATE OR REPLACE VIEW `{project}.{dataset}.gap_metrics_summary` AS
WITH latest_analysis AS (
  -- Get the most recent analysis run
  SELECT MAX(analysis_id) as latest_analysis_id
  FROM `{project}.{dataset}.gap_metrics`
),
aggregated_metrics AS (
  SELECT
    JSON_VALUE(metadata, '$.artifact_type') AS artifact_type,
    metric_type AS rule_name,
    COUNTIF(CAST(JSON_VALUE(metadata, '$.passed') AS BOOL)) AS count_passed,
    COUNTIF(NOT CAST(JSON_VALUE(metadata, '$.passed') AS BOOL)) AS count_failed,
    AVG(metric_value) AS avg_confidence,
    COUNT(*) AS total_chunks,
    -- Additional metrics for richer analysis
    MIN(metric_value) AS min_confidence,
    MAX(metric_value) AS max_confidence,
    STDDEV(metric_value) AS confidence_stddev,
    STRING_AGG(DISTINCT chunk_id ORDER BY chunk_id LIMIT 5) AS sample_chunks
  FROM `{project}.{dataset}.gap_metrics` gm
  CROSS JOIN latest_analysis la
  WHERE gm.analysis_id = la.latest_analysis_id
    AND JSON_VALUE(metadata, '$.artifact_type') IS NOT NULL
  GROUP BY artifact_type, rule_name
)
SELECT
  artifact_type,
  rule_name,
  count_passed,
  count_failed,
  total_chunks,
  ROUND(avg_confidence, 3) AS avg_confidence,
  ROUND(min_confidence, 3) AS min_confidence,
  ROUND(max_confidence, 3) AS max_confidence,
  ROUND(confidence_stddev, 3) AS confidence_stddev,
  ROUND(SAFE_DIVIDE(count_passed, (count_passed + count_failed)) * 100, 1) AS pass_rate_percent,
  sample_chunks
FROM aggregated_metrics
ORDER BY artifact_type, rule_name;