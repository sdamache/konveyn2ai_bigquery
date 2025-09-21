"""CLI entry point for running the hybrid gap metrics job."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from google.cloud import bigquery

from src.gap_analysis.metrics_runner import GapMetricsRunner, GapMetricRecord
from src.janapada_memory.config.bigquery_config import BigQueryConfig
from src.janapada_memory.connections.bigquery_connection import (
    BigQueryConnectionManager,
)

logger = logging.getLogger(__name__)


def _generate_analysis_id() -> str:
    now = datetime.now(timezone.utc)
    return f"gap_analysis_{now.strftime('%Y%m%d_%H%M%S')}"


def _write_records(
    connection: BigQueryConnectionManager,
    records: Iterable[GapMetricRecord],
    dry_run: bool = False,
) -> int:
    """Persist runner output into the existing gap_metrics table."""

    records = list(records)
    if not records:
        logger.info("GapMetricsRunner returned no records; nothing to persist")
        return 0

    if dry_run:
        logger.info("Dry run enabled; skipping BigQuery write")
        return len(records)

    table_ref = (
        f"{connection.config.project_id}.{connection.config.dataset_id}.gap_metrics"
    )
    payload: List[dict] = []

    for record in records:
        metadata = {
            "rule_name": record.rule_name,
            "ruleset_version": record.ruleset_version,
            "artifact_type": record.artifact_type,
            "passed": record.passed,
            "severity": record.severity,
            "base_confidence": record.base_confidence,
            "final_confidence": record.final_confidence,
            "semantic_similarity": record.semantic_similarity,
            "semantic_weight": record.semantic_weight,
            "suggested_fix": record.suggested_fix,
            "semantic_breakdown": record.metadata.get("confidence_breakdown", {}),
            "component_scores": record.metadata.get("component_scores", {}),
        }

        payload.append(
            {
                "analysis_id": record.analysis_id,
                "chunk_id": record.chunk_id,
                "metric_type": record.rule_name,
                "metric_value": record.final_confidence,
                "metadata": metadata,
                "created_at": record.created_at.isoformat(),
                "partition_date": record.partition_date,
            }
        )

    client = connection.client
    job = client.load_table_from_json(
        payload,
        table_ref,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()

    logger.info("Persisted %s hybrid metrics rows into %s", len(payload), table_ref)
    return len(payload)


def run_metrics_job(
    project_id: str,
    dataset_id: str,
    analysis_id: Optional[str] = None,
    ruleset_version: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> List[GapMetricRecord]:
    """Execute the hybrid gap analysis job and optionally persist results."""

    analysis_id = analysis_id or _generate_analysis_id()
    connection = BigQueryConnectionManager(
        config=BigQueryConfig(project_id=project_id, dataset_id=dataset_id)
    )

    runner = GapMetricsRunner(connection=connection)
    records = runner.run(
        analysis_id=analysis_id,
        ruleset_version=ruleset_version,
        limit=limit,
        dry_run=dry_run,
    )

    _write_records(connection, records, dry_run=dry_run)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the hybrid gap analysis metrics job",
    )
    parser.add_argument("--project", required=True, help="BigQuery project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument(
        "--analysis-id",
        help="Optional analysis identifier; generated automatically when omitted",
    )
    parser.add_argument(
        "--ruleset-version",
        help="Optional ruleset version override for logging and metadata",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of rows returned by each rule's evaluation SQL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip BigQuery persistence and just print results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON to stdout (one object per line)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    records = run_metrics_job(
        project_id=args.project,
        dataset_id=args.dataset,
        analysis_id=args.analysis_id,
        ruleset_version=args.ruleset_version,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    logger.info("Hybrid metrics produced %s rows", len(records))

    if args.json:
        for record in records:
            print(json.dumps(asdict(record), default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
