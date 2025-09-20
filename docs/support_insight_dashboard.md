# Support Call Insight Dashboard

This guide explains how to derive executive-ready insights from raw support call logs and surface them in the Streamlit dashboard.

## 1. BigQuery Tables

### Raw call log table

Create the raw event store that captures an entry per interaction:

```bash
python3 - <<'PY'
from pathlib import Path
sql = Path("analytics/sql/create_support_call_tables.sql").read_text()
sql = sql.replace("{{ project_id }}", "konveyn2ai").replace("{{ dataset }}", "documentation_ops")
Path("/tmp/create_support_call_tables.sql").write_text(sql)
PY

bq query --project_id=konveyn2ai --nouse_legacy_sql < /tmp/create_support_call_tables.sql
```

`support_call_logs_raw` expects the following fields (use whatever ETL pipeline is convenient):

| Column | Type | Notes |
|--------|------|-------|
| `call_id` | STRING | Unique identifier for the conversation |
| `customer_id` | STRING | Optional reference for the customer |
| `customer_segment` | STRING | Persona or segment label |
| `call_timestamp` | TIMESTAMP | Interaction timestamp (partition key) |
| `call_duration_minutes` | NUMERIC | Length of the call |
| `channel` | STRING | Phone, Chat, Email, etc. |
| `agent_id` | STRING | Handling agent identifier |
| `topic` / `subtopic` | STRING | Manual or model-derived classification |
| `sentiment_score` | NUMERIC | Range [-1, 1] at the conversation level |
| `resolution_status` | STRING | Resolved, Escalated, Pending, Returned |
| `escalation_flag` | BOOL | TRUE if escalated beyond tier-one |
| `followup_required` | BOOL | TRUE if follow-up ticket required |
| `csat_score` | NUMERIC | Optional survey score |
| `transcript` | STRING | Free-form notes or transcript |
| `tags` | ARRAY<STRING> | Additional labels (error codes, product areas) |

### Aggregated insight table

`support_call_insights` stores the daily/category aggregates used by the dashboard. The table is created by the SQL above and is partitioned by `insight_date`.

## 2. Seed sample data (optional)

For local testing without a full ETL, insert a few representative rows:

```sql
INSERT INTO `konveyn2ai.documentation_ops.support_call_logs_raw`
  (call_id, customer_id, customer_segment, call_timestamp, call_duration_minutes, channel,
   agent_id, agent_region, topic, subtopic, sentiment_score, resolution_status,
   escalation_flag, followup_required, csat_score, tags)
VALUES
  ('call-1001', 'cust-01', 'Enterprise', TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY), 18,
   'Phone', 'agent-5', 'US-East', 'Billing', 'Invoice Delay', -0.45, 'Escalated', TRUE, TRUE, 2,
   ['billing', 'invoice']),
  ('call-1002', 'cust-02', 'SMB', TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY), 9,
   'Chat', 'agent-8', 'EU-West', 'Product Feedback', 'Feature Request', 0.2, 'Resolved', FALSE, FALSE, 4,
   ['ux', 'feature-request']);
```

## 3. Generate insight rows

Use the helper script (inside the virtual environment, if you created one):

```bash
python scripts/compute_support_call_insights.py \
  --project-id konveyn2ai \
  --dataset documentation_ops \
  --snapshot-date $(date +%F) \
  --delete-existing --verbose
```

Flags:
- `--delete-existing` removes previous aggregates for that day (pass `--no-delete-existing` if you add that flag later).
- `--dry-run` prints the rendered SQL so you can review before writing.

## 4. Streamlit dashboard

Launch the consolidated dashboard (documentation coverage + support insights):

```bash
streamlit run analytics/dashboard/coverage_dashboard.py -- \
  --project-id konveyn2ai \
  --dataset documentation_ops
```

Use the **Support Call Insights** tab to explore:
- Daily call volume by category
- Sentiment and CSAT trends
- Recommended executive actions, backed by top subtopics and driver tags

If you do not see any data, confirm that:
1. `support_call_logs_raw` has rows for the selected date range
2. `scripts/compute_support_call_insights.py` has been executed for those dates
3. The Streamlit sidebar window includes the target dates (adjust the date range picker)

## 5. Adding the pipeline to automation

A Makefile target is available:

```bash
make support-insights \
  PROJECT_ID=konveyn2ai \
  DATASET=documentation_ops \
  SUPPORT_SNAPSHOT_DATE=$(date +%F)
```

Add this target to your recurring jobs (Cloud Scheduler, Cloud Build, or any orchestrator) once your raw call ingestion runs on a schedule.

## 6. Extending the transformation

The default SQL groups by `topic` and surfaces:
- Total calls, escalations, unresolved and follow-up counts
- Average sentiment and duration
- CSAT response rate and mean score
- Top subtopics and tag drivers (limited to 5 each)
- An automatically generated leadership action recommendation

To enrich the insights:
- Add additional features to `support_call_logs_raw` (e.g. NPS, product area)
- Extend the CASE statement in `analytics/sql/compute_support_call_insights.sql`
- Join with CRM or revenue data before aggregation to prioritise high-value segments

Re-run the script after making SQL changes, and restart the Streamlit app to load the new metadata.
