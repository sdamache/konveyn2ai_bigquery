-- Create table storing daily documentation coverage snapshots.
-- To apply: bq query --nouse_legacy_sql --project_id=$PROJECT_ID --dataset_id=$DATASET < analytics/sql/create_progress_snapshot_table.sql

CREATE TABLE IF NOT EXISTS `{{ project_id }}.{{ dataset }}.documentation_progress_snapshots`
(
  snapshot_date DATE NOT NULL OPTIONS(description="Calendar date the metrics represent"),
  snapshot_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp when the snapshot was recorded"),
  run_id STRING NOT NULL OPTIONS(description="Identifier for the metrics recomputation run"),
  metrics_version STRING NOT NULL OPTIONS(description="Semantic version for the snapshot calculation logic"),
  total_chunks_processed INT64 NOT NULL OPTIONS(description="Number of documentation chunks evaluated in this run"),
  total_chunks_documented INT64 NOT NULL OPTIONS(description="Number of chunks considered fully documented"),
  coverage_percentage NUMERIC NOT NULL OPTIONS(description="Documentation coverage percentage (documented / processed * 100)"),
  total_gaps_detected INT64 NOT NULL OPTIONS(description="Count of documentation gaps detected in this run"),
  total_gaps_open INT64 NOT NULL OPTIONS(description="Count of gaps still open after remediation"),
  total_gaps_closed INT64 NOT NULL OPTIONS(description="Count of gaps closed during this run"),
  mean_confidence_score NUMERIC OPTIONS(description="Average confidence score from the evaluator"),
  severity_distribution ARRAY<STRUCT<
    severity STRING,
    gaps_detected INT64,
    gaps_open INT64,
    gaps_closed INT64
  >> OPTIONS(description="Aggregated gap counts per severity level"),
  gaps_closed_by_rule ARRAY<STRUCT<
    rule_id STRING,
    rule_name STRING,
    gaps_closed INT64
  >> OPTIONS(description="Gap closure counts grouped by documentation rule"),
  metadata JSON OPTIONS(description="Unstructured metadata (e.g. source dataset, git hash, operator)")
)
PARTITION BY snapshot_date
CLUSTER BY run_id
OPTIONS(
  description="Daily documentation coverage metrics with severity and rule level breakdowns"
);
