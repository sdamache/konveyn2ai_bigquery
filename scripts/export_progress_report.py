#!/usr/bin/env python3
"""Export documentation coverage progress reports as Markdown and CSV."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from google.api_core import exceptions as gcloud_exceptions
from google.cloud import bigquery

DEFAULT_PROJECT = "konveyn2ai"
DEFAULT_DATASET = "documentation_ops"
DEFAULT_TABLE = "documentation_progress_snapshots"
DEFAULT_OUTPUT_DIR = Path("reports")


@dataclass
class ReportContext:
    progress: pd.DataFrame
    severity: pd.DataFrame | None
    rule_closures: pd.DataFrame | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export documentation coverage progress snapshots to markdown and CSV formats.",
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT),
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DOCUMENTATION_DATASET", DEFAULT_DATASET),
        help="Dataset containing the progress table",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Progress table name",
    )
    parser.add_argument(
        "--start-date",
        type=lambda value: dt.date.fromisoformat(value),
        default=None,
        help="Optional start date (YYYY-MM-DD). Defaults to the earliest snapshot",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: dt.date.fromisoformat(value),
        default=None,
        help="Optional end date (YYYY-MM-DD). Defaults to the latest snapshot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reports will be written",
    )
    parser.add_argument(
        "--formats",
        default="md,csv",
        help="Comma separated list of output formats (md, csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostics",
    )
    return parser.parse_args(argv)


def build_client(project_id: str) -> bigquery.Client:
    return bigquery.Client(project=project_id)


def load_progress_data(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    table: str,
    start_date: dt.date | None,
    end_date: dt.date | None,
) -> pd.DataFrame:
    conditions: list[str] = []
    parameters: list[bigquery.ScalarQueryParameter] = []
    if start_date:
        conditions.append("snapshot_date >= @start_date")
        parameters.append(
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date.isoformat())
        )
    if end_date:
        conditions.append("snapshot_date <= @end_date")
        parameters.append(
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date.isoformat())
        )
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            snapshot_date,
            coverage_percentage,
            total_chunks_processed,
            total_chunks_documented,
            total_gaps_detected,
            total_gaps_open,
            total_gaps_closed,
            mean_confidence_score,
            run_id,
            metrics_version
        FROM `{project_id}.{dataset}.{table}`
        {where_clause}
        ORDER BY snapshot_date
    """
    job_config = bigquery.QueryJobConfig(query_parameters=parameters)
    df = client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return df
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    numeric_columns = [
        "coverage_percentage",
        "total_chunks_processed",
        "total_chunks_documented",
        "total_gaps_detected",
        "total_gaps_open",
        "total_gaps_closed",
        "mean_confidence_score",
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_latest_severity(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    table: str,
) -> pd.DataFrame | None:
    query = f"""
        WITH latest AS (
            SELECT MAX(snapshot_date) AS snapshot_date
            FROM `{project_id}.{dataset}.{table}`
        )
        SELECT
            latest.snapshot_date,
            bucket.severity AS severity,
            bucket.gaps_open,
            bucket.gaps_closed,
            bucket.gaps_detected
        FROM `{project_id}.{dataset}.{table}` progress
        JOIN latest ON progress.snapshot_date = latest.snapshot_date
        CROSS JOIN UNNEST(progress.severity_distribution) AS bucket
        ORDER BY bucket.severity
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return None
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    for column in ["gaps_open", "gaps_closed", "gaps_detected"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def load_latest_rule_closures(
    client: bigquery.Client,
    project_id: str,
    dataset: str,
    table: str,
) -> pd.DataFrame | None:
    query = f"""
        WITH latest AS (
            SELECT MAX(snapshot_date) AS snapshot_date
            FROM `{project_id}.{dataset}.{table}`
        )
        SELECT
            bucket.rule_id,
            bucket.rule_name,
            bucket.gaps_closed,
            latest.snapshot_date
        FROM `{project_id}.{dataset}.{table}` progress
        JOIN latest ON progress.snapshot_date = latest.snapshot_date
        CROSS JOIN UNNEST(progress.gaps_closed_by_rule) AS bucket
        WHERE bucket.gaps_closed > 0
        ORDER BY bucket.gaps_closed DESC, bucket.rule_name
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return None
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["gaps_closed"] = pd.to_numeric(df["gaps_closed"], errors="coerce").fillna(0)
    return df


def compute_highlights(progress: pd.DataFrame) -> dict[str, float | None]:
    latest = progress.iloc[-1]
    earliest = progress.iloc[0]
    return {
        "coverage_delta": float(latest["coverage_percentage"] - earliest["coverage_percentage"])
        if pd.notna(latest["coverage_percentage"]) and pd.notna(earliest["coverage_percentage"])
        else None,
        "gaps_closed": float(progress["total_gaps_closed"].sum()),
        "gaps_open_latest": float(latest["total_gaps_open"]),
        "mean_confidence": float(latest["mean_confidence_score"])
        if pd.notna(latest["mean_confidence_score"])
        else None,
    }


def render_markdown(
    output_path: Path,
    progress: pd.DataFrame,
    severity: pd.DataFrame | None,
    rule_closures: pd.DataFrame | None,
) -> None:
    start = progress["snapshot_date"].min().date()
    end = progress["snapshot_date"].max().date()
    highlights = compute_highlights(progress)

    lines = [
        f"# Documentation Coverage Progress Report ({start} → {end})",
        "",
        "## Highlights",
    ]
    coverage_delta = highlights["coverage_delta"]
    if coverage_delta is not None:
        lines.append(f"- Coverage changed by **{coverage_delta:+.2f} ppts** across the period.")
    lines.append(
        f"- Cumulative gaps closed: **{int(highlights['gaps_closed'])}** (latest open gaps: {int(highlights['gaps_open_latest'])})."
    )
    if highlights["mean_confidence"] is not None:
        lines.append(
            f"- Latest mean confidence score: **{highlights['mean_confidence']:.2f}**."
        )

    lines.extend(
        [
            "",
            "## Snapshot Summary",
        ]
    )
    summary_df = progress.copy()
    summary_df["snapshot_date"] = summary_df["snapshot_date"].dt.date
    summary_df["coverage_percentage"] = summary_df["coverage_percentage"].map(lambda v: f"{v:.2f}")
    summary_df["mean_confidence_score"] = summary_df["mean_confidence_score"].map(
        lambda v: "n/a" if pd.isna(v) else f"{v:.2f}"
    )
    table_columns = [
        "snapshot_date",
        "coverage_percentage",
        "total_chunks_processed",
        "total_chunks_documented",
        "total_gaps_detected",
        "total_gaps_closed",
        "total_gaps_open",
        "mean_confidence_score",
    ]
    table_df = summary_df[table_columns]
    lines.append(table_df.to_markdown(index=False))

    if severity is not None and not severity.empty:
        lines.extend(
            [
                "",
                f"## Severity Distribution (Snapshot {severity['snapshot_date'].iloc[0].date()})",
            ]
        )
        severity_table = severity[["severity", "gaps_detected", "gaps_closed", "gaps_open"]]
        lines.append(severity_table.to_markdown(index=False))

    if rule_closures is not None and not rule_closures.empty:
        lines.extend(
            [
                "",
                f"## Gaps Closed per Rule (Snapshot {rule_closures['snapshot_date'].iloc[0].date()})",
            ]
        )
        rules_table = rule_closures[["rule_name", "gaps_closed"]]
        lines.append(rules_table.to_markdown(index=False))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(output_path: Path, progress: pd.DataFrame) -> None:
    csv_df = progress.copy()
    csv_df["snapshot_date"] = csv_df["snapshot_date"].dt.date
    csv_df.to_csv(output_path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    client = build_client(args.project_id)

    try:
        progress_df = load_progress_data(
            client,
            args.project_id,
            args.dataset,
            args.table,
            args.start_date,
            args.end_date,
        )
    except gcloud_exceptions.GoogleAPIError as error:
        raise SystemExit(f"Failed to read progress snapshots: {error}") from error

    if progress_df.empty:
        raise SystemExit("No progress snapshots were found for the requested window.")

    severity_df = load_latest_severity(client, args.project_id, args.dataset, args.table)
    rule_df = load_latest_rule_closures(client, args.project_id, args.dataset, args.table)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    formats = {fmt.strip().lower() for fmt in args.formats.split(',') if fmt.strip()}
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"progress_report_{progress_df['snapshot_date'].min().date()}_{progress_df['snapshot_date'].max().date()}_{timestamp}"

    if "md" in formats:
        markdown_path = args.output_dir / f"{base_name}.md"
        render_markdown(markdown_path, progress_df, severity_df, rule_df)
        if args.verbose:
            print(f"Wrote markdown report to {markdown_path}")

    if "csv" in formats:
        csv_path = args.output_dir / f"{base_name}.csv"
        write_csv(csv_path, progress_df)
        if args.verbose:
            print(f"Wrote CSV export to {csv_path}")

    if not formats & {"md", "csv"}:
        raise SystemExit("No supported output formats were specified (md, csv).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
