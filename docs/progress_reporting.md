# Documentation Coverage Progress Reporting

This guide explains how to create and maintain daily documentation coverage snapshots, monitor trends, and export reports for stakeholders.

## 1. BigQuery Progress Snapshot Table

1. Update the placeholders in `analytics/sql/create_progress_snapshot_table.sql` with your BigQuery project and dataset identifiers (for example by running `envsubst`).
2. Apply the definition:

```bash
env PROJECT_ID=konveyn2ai DATASET=semantic_gap_detector \
  envsubst < analytics/sql/create_progress_snapshot_table.sql > /tmp/create_progress.sql

bq query --project_id=konveyn2ai --nouse_legacy_sql < /tmp/create_progress.sql
```

The table stores daily totals for:
- Chunks processed vs. documented (coverage percentage)
- Gap counts (detected, open, closed)
- Mean confidence score
- Severity distributions (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, etc.)
- Rule-level closure counts

The insert query template lives at `analytics/sql/insert_progress_snapshot.sql`. The `scripts/compute_progress_snapshot.py` helper renders the template with the correct project/dataset and injects the `snapshot_date`, `run_id`, and `metrics_version` parameters.

## 2. Snapshot Job / Automation

### Ad-hoc run via Makefile

```bash
make progress-snapshot \
  PROJECT_ID=my-project \
  DATASET=semantic_gap_detector \
  SNAPSHOT_DATE=$(date +%F)
```

Add `EXTRA_ARGS="--dry-run --verbose"` to inspect the rendered SQL without writing data.

### Daily automation (Cloud Scheduler)

```bash
gcloud scheduler jobs create pubsub coverage-snapshot \
  --project=my-project \
  --schedule="0 2 * * *" \
  --topic=metrics-jobs \
  --message-body='{"command": "python scripts/compute_progress_snapshot.py"}'
```

Alternatively schedule a Cloud Run job or Cloud Build step that executes the same script/Make target after the metrics pipeline finishes.

## 3. Analytics Dashboard

Run the Streamlit dashboard locally:

```bash
streamlit run analytics/dashboard/coverage_dashboard.py \
  -- \
  --project-id my-project \
  --dataset semantic_gap_detector
```

Dashboard sections:
- **Summary metrics**: latest coverage, open gaps, confidence
- **Coverage trend**: line chart over time
- **Gap severity heat map**: open gaps by severity vs. date
- **Gaps closed per rule**: bar chart for the latest snapshot

Use the sidebar to change the dataset, narrow the date window, or refresh cached query results.

## 4. Progress Report Exporter

Generate Markdown/CSV summaries for stakeholders:

```bash
python scripts/export_progress_report.py \
  --project-id my-project \
  --dataset semantic_gap_detector \
  --start-date 2025-01-01 \
  --end-date 2025-01-31 \
  --output-dir reports --formats md,csv --verbose
```

Outputs land in `reports/` with timestamped filenames. The Markdown report captures:
- Coverage deltas and key highlights
- Snapshot-by-snapshot tabular metrics
- Latest severity distribution and rule closures

CSV exports mirror the raw progress table for easy import into spreadsheets.

## 5. Suggested Validation Steps

- Run `make progress-snapshot EXTRA_ARGS="--dry-run --verbose"` to confirm the rendered SQL.
- After inserting a real snapshot, open the Streamlit dashboard and verify new points appear.
- Use the report exporter to generate a Markdown report and review the highlights section for accuracy.

