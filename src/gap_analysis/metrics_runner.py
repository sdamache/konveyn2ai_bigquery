"""Gap metrics runner that blends deterministic SQL checks with semantic support.

This module executes Issue #5 rule SQL against BigQuery, enriches each result with
semantic neighbor scores, and feeds the combined signal through the
``ConfidenceCalculator`` so downstream consumers receive a single confidence
value that already reflects hybrid scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from google.cloud import bigquery

from .confidence_calculator import (
    ConfidenceCalculator,
    ConfidenceResult,
    ConfidenceWeights,
    FieldAnalysis,
    PenaltyAssessment,
)
from .rule_loader import RuleLoader
from .semantic_support import SemanticSupportFetcher
from src.janapada_memory.config.bigquery_config import BigQueryConfig
from src.janapada_memory.connections.bigquery_connection import (
    BigQueryConnectionManager,
)

logger = logging.getLogger(__name__)


@dataclass
class GapMetricRecord:
    """Structured result emitted by the gap metrics runner."""

    analysis_id: str
    rule_name: str
    ruleset_version: str
    chunk_id: str
    artifact_type: str
    passed: bool
    severity: int
    base_confidence: float
    final_confidence: float
    semantic_similarity: float
    semantic_weight: float
    suggested_fix: str
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def partition_date(self) -> str:
        """Return BigQuery-compatible partition date string."""
        return self.created_at.date().isoformat()

    def to_bigquery_row(self) -> Dict[str, Any]:
        """Serialise record into a BigQuery load_row compatible structure."""
        return {
            "analysis_id": self.analysis_id,
            "rule_name": self.rule_name,
            "ruleset_version": self.ruleset_version,
            "chunk_id": self.chunk_id,
            "artifact_type": self.artifact_type,
            "passed": self.passed,
            "severity": self.severity,
            "base_confidence": round(self.base_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "semantic_weight": round(self.semantic_weight, 4),
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "partition_date": self.partition_date,
        }


class GapMetricsRunner:
    """Evaluate deterministic rules and blend semantic support into scoring."""

    def __init__(
        self,
        connection: Optional[BigQueryConnectionManager] = None,
        rules: Optional[Sequence[Dict[str, Any]]] = None,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> None:
        self.connection = connection or BigQueryConnectionManager(
            config=BigQueryConfig(
                project_id=project_id or "konveyn2ai",
                dataset_id=dataset_id or "semantic_gap_detector",
            )
        )
        self.project_id = self.connection.config.project_id
        self.dataset_id = self.connection.config.dataset_id

        if rules is not None:
            self.rules = list(rules)
        else:
            self.rules = RuleLoader().load()

        self.semantic_fetcher = SemanticSupportFetcher(self.connection)

    # ------------------------------------------------------------------
    def run(
        self,
        analysis_id: str,
        ruleset_version: Optional[str] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[GapMetricRecord]:
        """Execute all configured rules and return enriched gap metrics."""

        records: List[GapMetricRecord] = []
        for rule in self.rules:
            rule_version = ruleset_version or rule.get("version", "unversioned")
            rule_records = self._evaluate_rule(
                analysis_id=analysis_id,
                rule=rule,
                ruleset_version=rule_version,
                limit=limit,
            )
            records.extend(rule_records)

        if dry_run:
            logger.info(
                "Dry-run mode enabled; produced %s records without persistence",
                len(records),
            )

        return records

    # ------------------------------------------------------------------
    def _evaluate_rule(
        self,
        analysis_id: str,
        rule: Dict[str, Any],
        ruleset_version: str,
        limit: Optional[int] = None,
    ) -> List[GapMetricRecord]:
        sql = rule["evaluation_sql"].strip()
        if limit is not None:
            sql = f"WITH evaluation AS ({sql}) SELECT * FROM evaluation LIMIT {int(limit)}"

        job_config = bigquery.QueryJobConfig(
            default_dataset=bigquery.DatasetReference(self.project_id, self.dataset_id),
            use_query_cache=True,
        )

        logger.debug(
            "Executing evaluation SQL for rule %s (limit=%s)",
            rule["rule_name"],
            limit,
        )
        rows = list(self.connection.execute_query(sql, job_config=job_config))
        if not rows:
            return []

        chunk_metadata = self._fetch_chunk_metadata(row.chunk_id for row in rows)

        semantic_weight = float(
            rule.get("semantic_probe", {}).get("weight", 0.0) or 0.0
        )
        semantic_map: Dict[str, float]
        if semantic_weight > 0:
            semantic_map = self.semantic_fetcher.fetch_rule_similarity(
                rule["rule_name"]
            )
        else:
            semantic_map = {}

        calculator = ConfidenceCalculator(
            weights=ConfidenceWeights(
                field_completeness=1.0,
                content_quality=0.0,
                semantic_support_weight=semantic_weight,
            )
        )

        min_length = int(
            rule.get("quality_thresholds", {}).get("min_content_length", 0) or 0
        )

        records: List[GapMetricRecord] = []
        for row in rows:
            chunk_id = getattr(row, "chunk_id", None)
            if not chunk_id:
                continue

            base_confidence = float(getattr(row, "confidence", 0.0) or 0.0)
            passed = bool(getattr(row, "passed", False))

            metadata = chunk_metadata.get(chunk_id, {})
            artifact_type = metadata.get(
                "artifact_type", rule.get("artifact_type", "unknown")
            )
            text_content = metadata.get("text_content", "")

            content_quality = calculator.analyze_content_quality(
                text_content=text_content,
                artifact_type=artifact_type,
                min_length=min_length,
            )

            field_analysis = FieldAnalysis(
                required_fields_present=int(round(base_confidence * 100)),
                total_required_fields=100,
                optional_fields_present=0,
                total_optional_fields=0,
                critical_missing_fields=0 if passed else 1,
                field_values={},
            )

            penalties = PenaltyAssessment(
                critical_violations=[] if passed else [rule["rule_name"]],
                security_violations=[],
                compliance_violations=[],
                format_violations=[],
            )

            semantic_similarity = float(semantic_map.get(chunk_id, 0.0) or 0.0)
            confidence_result: ConfidenceResult = calculator.calculate_confidence(
                field_analysis=field_analysis,
                content_quality=content_quality,
                penalty_assessment=penalties,
                artifact_type=artifact_type,
                semantic_similarity=semantic_similarity,
            )

            metadata_payload: Dict[str, Any] = {
                "base_confidence": base_confidence,
                "semantic_similarity": semantic_similarity,
                "confidence_breakdown": confidence_result.breakdown,
                "component_scores": {
                    component.value: score
                    for component, score in confidence_result.component_scores.items()
                },
            }
            if "source" in metadata:
                metadata_payload["source"] = metadata["source"]

            suggested_fix = ""
            if not passed:
                suggested_fix = rule.get("suggested_fix_template", "")

            record = GapMetricRecord(
                analysis_id=analysis_id,
                rule_name=rule["rule_name"],
                ruleset_version=ruleset_version,
                chunk_id=chunk_id,
                artifact_type=artifact_type,
                passed=passed,
                severity=int(rule.get("severity", 1)),
                base_confidence=base_confidence,
                final_confidence=confidence_result.final_confidence,
                semantic_similarity=semantic_similarity,
                semantic_weight=semantic_weight,
                suggested_fix=suggested_fix,
                metadata=metadata_payload,
            )

            records.append(record)

        return records

    # ------------------------------------------------------------------
    def _fetch_chunk_metadata(
        self, chunk_ids: Iterable[str]
    ) -> Dict[str, Dict[str, Any]]:
        unique_ids = sorted({chunk_id for chunk_id in chunk_ids if chunk_id})
        if not unique_ids:
            return {}

        query = f"""
        SELECT
          chunk_id,
          artifact_type,
          text_content,
          source
        FROM `{self.project_id}.{self.dataset_id}.source_metadata`
        WHERE chunk_id IN UNNEST(@chunk_ids)
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("chunk_ids", "STRING", unique_ids)
            ],
            use_query_cache=True,
        )

        results = self.connection.execute_query(query, job_config=job_config)
        metadata_map: Dict[str, Dict[str, Any]] = {}
        for row in results:
            metadata_map[row.chunk_id] = {
                "artifact_type": row.artifact_type,
                "text_content": row.text_content or "",
                "source": getattr(row, "source", None),
            }
        return metadata_map


__all__ = ["GapMetricsRunner", "GapMetricRecord"]
