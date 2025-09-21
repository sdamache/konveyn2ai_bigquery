"""Helpers for retrieving semantic support metrics from BigQuery."""

from __future__ import annotations

from typing import Dict

from google.cloud import bigquery

from src.janapada_memory.connections.bigquery_connection import (
    BigQueryConnectionManager,
)
from src.janapada_memory.config.bigquery_config import BigQueryConfig


class SemanticSupportFetcher:
    """Fetch semantic similarity scores from the semantic_candidates table."""

    def __init__(self, connection: BigQueryConnectionManager) -> None:
        self.connection = connection
        self.project_id = connection.config.project_id
        self.dataset_id = connection.config.dataset_id

    def fetch_rule_similarity(self, rule_name: str) -> Dict[str, float]:
        """Return the maximum similarity per chunk for a given rule."""
        query = f"""
        SELECT
          chunk_id,
          MAX(similarity_score) AS similarity_score
        FROM `{self.project_id}.{self.dataset_id}.semantic_candidates`
        WHERE rule_name = @rule_name
        GROUP BY chunk_id
        """
        params = [bigquery.ScalarQueryParameter("rule_name", "STRING", rule_name)]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = self.connection.execute_query(query, job_config=job_config)
        return {row.chunk_id: float(row.similarity_score) for row in results}

    def fetch_similarity(self, rule_name: str, chunk_id: str) -> float:
        """Return the maximum similarity for a single rule/chunk combination."""
        query = f"""
        SELECT
          MAX(similarity_score) AS similarity_score
        FROM `{self.project_id}.{self.dataset_id}.semantic_candidates`
        WHERE rule_name = @rule_name AND chunk_id = @chunk_id
        """
        params = [
            bigquery.ScalarQueryParameter("rule_name", "STRING", rule_name),
            bigquery.ScalarQueryParameter("chunk_id", "STRING", chunk_id),
        ]
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = list(self.connection.execute_query(query, job_config=job_config))
        if not results or results[0].similarity_score is None:
            return 0.0
        return float(results[0].similarity_score)


def fetch_semantic_support(
    project_id: str,
    dataset_id: str,
    rule_name: str,
    chunk_id: str,
) -> float:
    """Convenience helper to fetch semantic support for ad-hoc calculations."""

    connection = BigQueryConnectionManager(
        config=BigQueryConfig(project_id=project_id, dataset_id=dataset_id)
    )
    fetcher = SemanticSupportFetcher(connection)
    return fetcher.fetch_similarity(rule_name, chunk_id)
