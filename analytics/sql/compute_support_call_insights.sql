-- Builds or refreshes support_call_insights entries for a specific date.
-- Required parameters when executing with the BigQuery CLI:
--   @snapshot_date (DATE): the business date to aggregate.
-- Optional parameters:
--   @metrics_version (STRING): semantic identifier for transformation logic (reserved for future use).

INSERT INTO `{{ project_id }}.{{ dataset }}.support_call_insights`
(
  insight_date,
  category,
  total_calls,
  escalated_calls,
  unresolved_calls,
  followups_required,
  avg_sentiment,
  avg_duration_minutes,
  csat_response_rate,
  csat_mean,
  top_subtopics,
  leading_drivers,
  action_recommendation
)
WITH calls AS (
  SELECT
    call_id,
    COALESCE(topic, 'Uncategorized') AS category,
    subtopic,
    call_timestamp,
    call_duration_minutes,
    sentiment_score,
    resolution_status,
    escalation_flag,
    followup_required,
    csat_score,
    tags
  FROM `{{ project_id }}.{{ dataset }}.support_call_logs_raw`
  WHERE DATE(call_timestamp) = @snapshot_date
),
category_rollup AS (
  SELECT
    @snapshot_date AS insight_date,
    category,
    COUNT(*) AS total_calls,
    COUNTIF(COALESCE(escalation_flag, FALSE) OR LOWER(resolution_status) = 'escalated') AS escalated_calls,
    COUNTIF(LOWER(resolution_status) NOT IN ('resolved', 'closed', 'completed')) AS unresolved_calls,
    COUNTIF(COALESCE(followup_required, FALSE)) AS followups_required,
    AVG(sentiment_score) AS avg_sentiment,
    AVG(call_duration_minutes) AS avg_duration_minutes,
    SAFE_MULTIPLY(SAFE_DIVIDE(COUNTIF(csat_score IS NOT NULL), COUNT(*)), 100) AS csat_response_rate,
    AVG(csat_score) AS csat_mean
  FROM calls
  GROUP BY category
)
SELECT
  insight_date,
  category,
  total_calls,
  escalated_calls,
  unresolved_calls,
  followups_required,
  SAFE_CAST(IFNULL(avg_sentiment, 0) AS NUMERIC) AS avg_sentiment,
  SAFE_CAST(IFNULL(avg_duration_minutes, 0) AS NUMERIC) AS avg_duration_minutes,
  SAFE_CAST(IFNULL(csat_response_rate, 0) AS NUMERIC) AS csat_response_rate,
  SAFE_CAST(csat_mean AS NUMERIC) AS csat_mean,
  ARRAY(
    SELECT subtopic
    FROM (
      SELECT subtopic, COUNT(*) AS cnt
      FROM calls c2
      WHERE c2.category = category AND subtopic IS NOT NULL
      GROUP BY subtopic
      ORDER BY cnt DESC
      LIMIT 5
    )
  ) AS top_subtopics,
  ARRAY(
    SELECT tag
    FROM (
      SELECT tag, COUNT(*) AS cnt
      FROM calls c3, UNNEST(IFNULL(c3.tags, [])) AS tag
      WHERE c3.category = category AND tag IS NOT NULL
      GROUP BY tag
      ORDER BY cnt DESC
      LIMIT 5
    )
  ) AS leading_drivers,
  CASE
    WHEN total_calls = 0 THEN 'No calls captured. Monitor data pipeline.'
    WHEN SAFE_DIVIDE(escalated_calls, total_calls) >= 0.25 THEN CONCAT('Prioritise escalation playbook for ', category, ' (', escalated_calls, ' escalations).')
    WHEN SAFE_DIVIDE(unresolved_calls, total_calls) >= 0.2 THEN CONCAT('Launch enablement sprint to improve first-touch resolution for ', category, '.')
    WHEN SAFE_CAST(IFNULL(avg_sentiment, 0) AS NUMERIC) < 0 THEN CONCAT('Partner with product to address negative sentiment in ', category, '.')
    WHEN SAFE_CAST(csat_mean AS NUMERIC) IS NOT NULL AND SAFE_CAST(csat_mean AS NUMERIC) < 3 THEN CONCAT('Investigate low CSAT drivers for ', category, '.')
    ELSE CONCAT('Maintain monitoring cadence for ', category, '.')
  END AS action_recommendation
FROM category_rollup;
