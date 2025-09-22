#!/usr/bin/env python3
"""Compute documentation coverage metrics and persist a progress snapshot in BigQuery."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from google.cloud import bigquery


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute documentation coverage metrics and insert a snapshot row into BigQuery",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai"),
        help="Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("BIGQUERY_DATASET_ID", "semantic_gap_detector"),
        help="BigQuery dataset containing documentation metrics",
    )
    parser.add_argument(
        "--template-path",
        default=str(
            Path(__file__).resolve().parent.parent
            / "analytics"
            / "sql"
            / "insert_progress_snapshot.sql"
        ),
        help="Path to the SQL template used for inserting the snapshot",
    )
    parser.add_argument(
        "--snapshot-date",
        type=lambda value: dt.date.fromisoformat(value),
        default=dt.date.today(),
        help="Logical date for the snapshot (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--run-id",
        help="Identifier for this metrics recomputation run (defaults to generated UUID)",
    )
    parser.add_argument(
        "--metrics-version",
        default="v1",
        help="Semantic version identifier for the snapshot logic",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the query without executing it",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rendered query and response metadata",
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

    run_id = (
        args.run_id
        or f"progress-{args.snapshot_date.isoformat()}-{uuid.uuid4().hex[:8]}"
    )

    replacements = {
        "{{ project_id }}": args.project_id,
        "{{ dataset }}": args.dataset,
    }

    rendered_sql = render_sql_template(template_path, replacements)

    if args.verbose or args.dry_run:
        print("\n-- Rendered snapshot insert statement --\n")
        print(rendered_sql)
        print("\n-- End statement --\n")

    if args.dry_run:
        return 0

    client = bigquery.Client(project=args.project_id)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "snapshot_date", "DATE", args.snapshot_date.isoformat()
            ),
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
            bigquery.ScalarQueryParameter(
                "metrics_version", "STRING", args.metrics_version
            ),
        ]
    )

    query_job = client.query(rendered_sql, job_config=job_config)
    result = query_job.result()

    if args.verbose:
        print("Query job completed")
        stats: dict[str, Any] = {
            "job_id": query_job.job_id,
            "slot_millis": query_job.slot_millis,
            "bytes_processed": query_job.total_bytes_processed,
            "rows_affected": result.total_rows,
        }
        print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
