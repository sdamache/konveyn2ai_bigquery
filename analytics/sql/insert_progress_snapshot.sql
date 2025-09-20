-- Inserts a new documentation_progress_snapshots row using aggregated metrics.
-- Parameters required when running with bq CLI:
--   @snapshot_date (DATE): logical date the metrics represent
--   @run_id (STRING): identifier for this metrics recomputation run
-- Optional parameters:
--   @metrics_version (STRING)

INSERT INTO `{{ project_id }}.{{ dataset }}.documentation_progress_snapshots`
(
  snapshot_date,
  snapshot_timestamp,
  run_id,
  metrics_version,
  total_chunks_processed,
  total_chunks_documented,
  coverage_percentage,
  total_gaps_detected,
  total_gaps_open,
  total_gaps_closed,
  mean_confidence_score,
  severity_distribution,
  gaps_closed_by_rule,
  metadata
)
WITH chunk_metrics AS (
  SELECT
    COUNT(*) AS total_chunks_processed,
    COALESCE(SUM(CASE WHEN documented THEN 1 ELSE 0 END), 0) AS total_chunks_documented,
    AVG(confidence_score) AS mean_confidence_score
  FROM `{{ project_id }}.{{ dataset }}.documentation_chunk_metrics`
  WHERE DATE(processed_at) <= @snapshot_date
),
latest_gap_state AS (
  SELECT
    gap_id,
    status,
    severity,
    rule_id,
    rule_name,
    detected_at,
    status_at,
    closed_at
  FROM (
    SELECT
      gap_id,
      status,
      severity,
      rule_id,
      rule_name,
      detected_at,
      status_at,
      closed_at,
      ROW_NUMBER() OVER (PARTITION BY gap_id ORDER BY status_at DESC) AS rn
    FROM `{{ project_id }}.{{ dataset }}.documentation_gap_history`
    WHERE DATE(status_at) <= @snapshot_date
  )
  WHERE rn = 1
),
gap_rollup AS (
  SELECT
    COUNT(*) AS total_gaps_detected,
    COUNTIF(status = 'OPEN') AS total_gaps_open,
    COUNTIF(status = 'CLOSED') AS total_gaps_closed
  FROM latest_gap_state
),
severity_distribution AS (
  SELECT
    severity,
    COUNT(*) AS gaps_detected,
    COUNTIF(status = 'OPEN') AS gaps_open,
    COUNTIF(status = 'CLOSED') AS gaps_closed
  FROM latest_gap_state
  GROUP BY severity
),
rule_closures AS (
  SELECT
    rule_id,
    ANY_VALUE(rule_name) AS rule_name,
    COUNTIF(status = 'CLOSED' AND DATE(closed_at) = @snapshot_date) AS gaps_closed
  FROM `{{ project_id }}.{{ dataset }}.documentation_gap_history`
  WHERE DATE(closed_at) BETWEEN @snapshot_date - 7 AND @snapshot_date
  GROUP BY rule_id
)
SELECT
  @snapshot_date AS snapshot_date,
  CURRENT_TIMESTAMP() AS snapshot_timestamp,
  @run_id AS run_id,
  IFNULL(@metrics_version, 'v1') AS metrics_version,
  cm.total_chunks_processed,
  cm.total_chunks_documented,
  CAST(
    IFNULL(
      SAFE_MULTIPLY(
        SAFE_DIVIDE(cm.total_chunks_documented, NULLIF(cm.total_chunks_processed, 0)),
        100
      ),
      0
    )
  AS NUMERIC) AS coverage_percentage,
  gr.total_gaps_detected,
  gr.total_gaps_open,
  gr.total_gaps_closed,
  SAFE_CAST(IFNULL(cm.mean_confidence_score, 0) AS NUMERIC) AS mean_confidence_score,
  ARRAY(
    SELECT AS STRUCT sd.severity, sd.gaps_detected, sd.gaps_open, sd.gaps_closed
    FROM severity_distribution sd
    ORDER BY sd.severity
  ),
  ARRAY(
    SELECT AS STRUCT rc.rule_id, rc.rule_name, rc.gaps_closed
    FROM rule_closures rc
    WHERE rc.gaps_closed > 0
    ORDER BY rc.gaps_closed DESC
  ),
  TO_JSON(
    STRUCT(
      @snapshot_date AS snapshot_date,
      @run_id AS run_id,
      IFNULL(@metrics_version, 'v1') AS metrics_version,
      CURRENT_TIMESTAMP() AS inserted_at,
      '{{ dataset }}' AS source_dataset
    )
  ) AS metadata
FROM chunk_metrics cm
CROSS JOIN gap_rollup gr;
