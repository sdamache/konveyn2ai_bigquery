-- Semantic candidate view for hybrid gap analysis scoring.
--
-- Replace {{PROJECT_ID}} and {{DATASET_ID}} with your BigQuery identifiers when
-- materialising the view.
--
-- Example:
--   bq query --use_legacy_sql=false --destination_table \
--     konveyn2ai.semantic_gap_detector.semantic_candidate_view \
--     --replace --nouse_cache < configs/sql/gap_analysis/semantic_candidates.sql
--
-- The view surfaces the latest semantic neighbor rows produced by
-- ``src.gap_analysis.semantic_candidates.SemanticCandidateBuilder`` so SQL rules
-- can join on consistent column names without hardcoding table paths.

SELECT
  rule_name,
  artifact_type,
  chunk_id,
  similarity_score,
  weight,
  neighbor_rank,
  generated_at,
  source,
  probe_artifact_type,
  partition_date
FROM `{{PROJECT_ID}}.{{DATASET_ID}}.semantic_candidates`;
