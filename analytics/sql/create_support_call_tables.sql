-- Create raw and aggregated tables for support call insight pipeline.
-- Render placeholders with project/dataset values before running with bq CLI.

CREATE TABLE IF NOT EXISTS `{{ project_id }}.{{ dataset }}.support_call_logs_raw`
(
  call_id STRING NOT NULL OPTIONS(description="Unique identifier for the support interaction"),
  customer_id STRING OPTIONS(description="Customer identifier"),
  customer_segment STRING OPTIONS(description="Optional segment or persona"),
  call_timestamp TIMESTAMP NOT NULL OPTIONS(description="Timestamp for the support engagement"),
  call_duration_minutes NUMERIC OPTIONS(description="Length of the call in minutes"),
  channel STRING OPTIONS(description="Source channel, e.g. phone, chat, email"),
  agent_id STRING OPTIONS(description="Support agent identifier"),
  agent_region STRING OPTIONS(description="Agent location or region"),
  topic STRING OPTIONS(description="Primary topic label for the call"),
  subtopic STRING OPTIONS(description="Secondary topic classification"),
  sentiment_score NUMERIC OPTIONS(description="Sentiment score in range [-1, 1]"),
  resolution_status STRING OPTIONS(description="Outcome status (Resolved, Escalated, Pending, Returned)"),
  escalation_flag BOOL OPTIONS(description="TRUE if escalated beyond front-line"),
  followup_required BOOL OPTIONS(description="TRUE if additional follow-up is required"),
  csat_score NUMERIC OPTIONS(description="Customer satisfaction score if captured"),
  transcript STRING OPTIONS(description="Raw transcript or free-form notes"),
  tags ARRAY<STRING> OPTIONS(description="Additional free-form tags")
)
PARTITION BY DATE(call_timestamp)
CLUSTER BY topic;

CREATE TABLE IF NOT EXISTS `{{ project_id }}.{{ dataset }}.support_call_insights`
(
  insight_date DATE NOT NULL OPTIONS(description="Date representing the aggregated insights"),
  category STRING NOT NULL OPTIONS(description="Support topic category"),
  total_calls INT64 NOT NULL OPTIONS(description="Number of calls captured for the category"),
  escalated_calls INT64 NOT NULL OPTIONS(description="Calls escalated beyond tier 1"),
  unresolved_calls INT64 NOT NULL OPTIONS(description="Calls not resolved at first touch"),
  followups_required INT64 NOT NULL OPTIONS(description="Calls requiring follow-up"),
  avg_sentiment NUMERIC OPTIONS(description="Average sentiment score across calls"),
  avg_duration_minutes NUMERIC OPTIONS(description="Average call duration in minutes"),
  csat_response_rate NUMERIC OPTIONS(description="Percentage of calls that captured a CSAT score"),
  csat_mean NUMERIC OPTIONS(description="Average CSAT score"),
  top_subtopics ARRAY<STRING> OPTIONS(description="Top subtopics contributing to the category"),
  leading_drivers ARRAY<STRING> OPTIONS(description="Key drivers derived from call tags"),
  action_recommendation STRING OPTIONS(description="Suggested next best action for leadership")
)
PARTITION BY insight_date
CLUSTER BY category;
