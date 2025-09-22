"""Unit tests for the hybrid gap metrics runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List
from unittest.mock import patch

import pytest

from src.gap_analysis.metrics_runner import GapMetricsRunner


class FakeConnection:
    """Minimal BigQuery connection stub for unit testing."""

    def __init__(self, responses: List[Iterable]) -> None:
        self.config = SimpleNamespace(
            project_id="test-project", dataset_id="test-dataset"
        )
        self._responses = list(responses)

    def execute_query(
        self, query: str, job_config=None
    ):  # pragma: no cover - simple stub
        if not self._responses:
            raise AssertionError("No more fake responses configured")
        return self._responses.pop(0)


@pytest.mark.parametrize(
    "semantic_score, expected_confidence", [(0.8, 0.8), (0.0, 0.6)]
)
def test_metrics_runner_blends_semantic_similarity(semantic_score, expected_confidence):
    """Semantic support should adjust the final confidence when a probe weight is present."""

    rule = {
        "rule_name": "fastapi_endpoint_documentation",
        "artifact_type": "fastapi",
        "version": "1.0.0",
        "evaluation_sql": "SELECT 'chunk-1' AS chunk_id, TRUE AS passed, 0.6 AS confidence",
        "severity": 3,
        "suggested_fix_template": "Add FastAPI endpoint documentation.",
        "semantic_probe": {"weight": 0.25},
        "quality_thresholds": {"min_content_length": 10},
    }

    evaluation_row = SimpleNamespace(chunk_id="chunk-1", passed=True, confidence=0.6)
    metadata_row = SimpleNamespace(
        chunk_id="chunk-1",
        artifact_type="fastapi",
        text_content="FastAPI endpoint with thorough documentation and examples.",
        source="repository",
    )

    connection = FakeConnection([iter([evaluation_row]), iter([metadata_row])])

    with patch(
        "src.gap_analysis.metrics_runner.SemanticSupportFetcher.fetch_rule_similarity",
        return_value={"chunk-1": semantic_score},
    ) as mock_fetch:
        runner = GapMetricsRunner(connection=connection, rules=[rule])
        records = runner.run(analysis_id="analysis-123")

    assert (
        mock_fetch.called
    ), "Semantic support fetcher should be invoked when a weight is set"
    assert len(records) == 1

    record = records[0]
    assert record.chunk_id == "chunk-1"
    assert pytest.approx(record.base_confidence, abs=1e-6) == 0.6
    assert pytest.approx(record.final_confidence, abs=1e-6) == pytest.approx(
        expected_confidence, abs=1e-6
    )
    assert record.metadata["semantic_similarity"] == pytest.approx(
        semantic_score, abs=1e-6
    )
    assert (
        "semantic_support" in record.metadata["component_scores"]
    ), "Semantic component should be captured in breakdown"

    # When semantic_score is zero, fetch_rule_similarity still called but contribution drops to baseline confidence.
    if semantic_score == 0.0:
        assert record.final_confidence == pytest.approx(0.6, abs=1e-6)
