#!/usr/bin/env python3
"""Aggregate raw support call logs into executive insight metrics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from google.cloud import bigquery


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate support call logs into the support_call_insights table.",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai"),
        help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("SUPPORT_DATASET", "documentation_ops"),
        help="BigQuery dataset that contains support call tables",
    )
    parser.add_argument(
        "--template-path",
        default=str(
            Path(__file__).resolve().parent.parent
            / "analytics"
            / "sql"
            / "compute_support_call_insights.sql"
        ),
        help="Path to the SQL template used for inserting insights",
    )
    parser.add_argument(
        "--snapshot-date",
        type=lambda value: dt.date.fromisoformat(value),
        default=dt.date.today(),
        help="Logical date for the insight aggregation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--metrics-version",
        default="v1",
        help="Semantic version identifier for the insight aggregation logic",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing insight rows for the snapshot date before inserting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the SQL without executing it",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rendered SQL and query job statistics",
    )
    return parser.parse_args(argv)


def render_sql_template(template_path: Path, replacements: dict[str, str]) -> str:
    sql_template = template_path.read_text(encoding="utf-8")
    rendered = sql_template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    template_path = Path(args.template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"SQL template not found: {template_path}")

    replacements = {
        "{{ project_id }}": args.project_id,
        "{{ dataset }}": args.dataset,
    }

    rendered_sql = render_sql_template(template_path, replacements)

    if args.verbose or args.dry_run:
        print("\n-- Rendered support call insight statement --\n")
        print(rendered_sql)
        print("\n-- End statement --\n")

    if args.dry_run:
        return 0

    client = bigquery.Client(project=args.project_id)

    query_parameters = [
        bigquery.ScalarQueryParameter("snapshot_date", "DATE", args.snapshot_date.isoformat()),
        bigquery.ScalarQueryParameter("metrics_version", "STRING", args.metrics_version),
    ]

    if args.delete_existing:
        delete_sql = (
            f"DELETE FROM `{args.project_id}.{args.dataset}.support_call_insights` "
            "WHERE insight_date = @snapshot_date"
        )
        delete_job = client.query(
            delete_sql,
            job_config=bigquery.QueryJobConfig(query_parameters=query_parameters[:1]),
        )
        delete_job.result()

    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    query_job = client.query(rendered_sql, job_config=job_config)
    result = query_job.result()

    if args.verbose:
        stats: dict[str, Any] = {
            "job_id": query_job.job_id,
            "bytes_processed": query_job.total_bytes_processed,
            "slot_millis": query_job.slot_millis,
            "rows_affected": result.total_rows,
        }
        print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
