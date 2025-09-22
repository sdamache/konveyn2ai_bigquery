"""Semantic candidate builder for rules with semantic probes.

This module materialises semantic probe embeddings and semantic neighbor rows
so the deterministic gap analysis pipeline can reference them during rule
execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from google.cloud import bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel

from .rule_loader import RuleLoader
from .rule_validator import RuleConfigValidator
from src.janapada_memory.connections.bigquery_connection import (
    BigQueryConnectionManager,
)
from src.janapada_memory.config.bigquery_config import BigQueryConfig
from src.janapada_memory.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


@dataclass
class SemanticProbe:
    rule_name: str
    artifact_type: str
    query_text: str
    top_k: int
    similarity_threshold: float
    weight: float
    vector_table: str


class SemanticCandidateBuilder:
    """Compute semantic neighbors for rules that declare semantic probes."""

    def __init__(
        self,
        connection: BigQueryConnectionManager,
        embedding_model: str = "text-embedding-004",
    ) -> None:
        self.connection = connection
        self.embedding_model = embedding_model
        self.project_id = connection.config.project_id
        self.dataset_id = connection.config.dataset_id
        self.bq_client = connection.client

        vertexai.init(
            project=self.project_id,
            location=connection.config.location,
        )
        self.vertex_model = TextEmbeddingModel.from_pretrained(self.embedding_model)

        # Ensure supporting tables exist with up-to-date schema.
        schema_manager = SchemaManager(connection=self.connection)
        schema_manager.create_dataset()
        schema_manager.create_tables(
            tables=["semantic_probe_embeddings", "semantic_candidates"],
            force_recreate=False,
        )

    # ------------------------------------------------------------------
    def refresh_candidates(self, rules: Iterable[Dict[str, Any]]) -> None:
        """Generate embeddings and neighbor rows for semantic probes."""

        probes = []
        for rule in rules:
            probe_config = rule.get("semantic_probe")
            if not isinstance(probe_config, dict):
                continue

            probes.append(
                SemanticProbe(
                    rule_name=rule["rule_name"],
                    artifact_type=rule["artifact_type"],
                    query_text=probe_config["query_text"].strip(),
                    top_k=int(probe_config.get("top_k", 5)),
                    similarity_threshold=float(
                        probe_config.get("similarity_threshold", 0.5)
                    ),
                    weight=float(probe_config.get("weight", 0.0)),
                    vector_table=str(
                        probe_config.get("vector_table", "source_embeddings")
                    ),
                )
            )

        if not probes:
            logger.info("No semantic probes defined; skipping candidate refresh")
            return

        embeddings = self._upsert_probe_embeddings(probes)
        self._rebuild_candidate_table(probes, embeddings)

    # ------------------------------------------------------------------
    def _embedding_for_text(self, text: str) -> List[float]:
        response = self.vertex_model.get_embeddings([text])
        if not response:
            raise RuntimeError("Vertex AI returned empty embedding response")
        return response[0].values

    def _upsert_probe_embeddings(
        self, probes: List[SemanticProbe]
    ) -> Dict[Tuple[str, str], List[float]]:
        table_id = f"{self.project_id}.{self.dataset_id}.semantic_probe_embeddings"
        now_dt = datetime.now(timezone.utc)
        partition_date = now_dt.date().isoformat()

        embedding_map: Dict[Tuple[str, str], List[float]] = {}
        rows: List[Dict[str, Any]] = []

        for probe in probes:
            embedding = self._embedding_for_text(probe.query_text)
            embedding_map[(probe.rule_name, probe.artifact_type)] = embedding

            # Remove any existing record for this probe to avoid duplicates.
            delete_query = f"DELETE FROM `{table_id}` WHERE rule_name = @rule_name AND artifact_type = @artifact_type"
            delete_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "rule_name", "STRING", probe.rule_name
                    ),
                    bigquery.ScalarQueryParameter(
                        "artifact_type", "STRING", probe.artifact_type
                    ),
                ]
            )
            self.connection.execute_query(delete_query, job_config=delete_config)

            rows.append(
                {
                    "rule_name": probe.rule_name,
                    "artifact_type": probe.artifact_type,
                    "query_text": probe.query_text,
                    "embedding": embedding,
                    "stored_at": now_dt.isoformat(),
                    "partition_date": partition_date,
                }
            )

        if rows:
            load_job = self.bq_client.load_table_from_json(
                rows,
                table_id,
                job_config=bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
                ),
            )
            load_job.result()
            logger.info("Stored %s semantic probe embeddings", len(rows))

        return embedding_map

    def _rebuild_candidate_table(
        self,
        probes: List[SemanticProbe],
        embedding_map: Dict[Tuple[str, str], List[float]],
    ) -> None:
        table_id = f"{self.project_id}.{self.dataset_id}.semantic_candidates"

        # Truncate existing candidate rows.
        self.connection.execute_query(f"TRUNCATE TABLE `{table_id}`")

        inserts: List[Dict[str, Any]] = []

        for probe in probes:
            embedding = embedding_map[(probe.rule_name, probe.artifact_type)]

            query = f"""
            WITH candidates AS (
              SELECT
                vs.base.chunk_id AS chunk_id,
                vs.distance AS distance,
                sm.source AS source,
                sm.artifact_type AS neighbor_artifact_type
              FROM VECTOR_SEARCH(
                TABLE `{self.project_id}.{self.dataset_id}.{probe.vector_table}`,
                'embedding_vector',
                (SELECT @query_embedding AS embedding_vector),
                top_k => @top_k,
                distance_type => 'COSINE'
              ) AS vs
              JOIN `{self.project_id}.{self.dataset_id}.source_metadata` sm
                ON vs.base.chunk_id = sm.chunk_id
            )
            SELECT
              chunk_id,
              source,
              neighbor_artifact_type,
              (1 - distance) AS similarity_score
            FROM candidates
            WHERE (1 - distance) >= @similarity_threshold
            ORDER BY similarity_score DESC
            LIMIT @top_k
            """

            params = [
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embedding),
                bigquery.ScalarQueryParameter("top_k", "INT64", probe.top_k),
                bigquery.ScalarQueryParameter(
                    "similarity_threshold", "FLOAT64", probe.similarity_threshold
                ),
            ]

            results = self.connection.execute_query(
                query, job_config=bigquery.QueryJobConfig(query_parameters=params)
            )

            for rank, row in enumerate(results, start=1):
                generated_at = datetime.now(timezone.utc)
                inserts.append(
                    {
                        "rule_name": probe.rule_name,
                        "artifact_type": probe.artifact_type,
                        "chunk_id": row.chunk_id,
                        "source": row.source,
                        "probe_artifact_type": row.neighbor_artifact_type,
                        "similarity_score": float(row.similarity_score),
                        "weight": probe.weight,
                        "neighbor_rank": rank,
                        "generated_at": generated_at.isoformat(),
                        "partition_date": generated_at.date().isoformat(),
                    }
                )

        if inserts:
            load_job = self.bq_client.load_table_from_json(
                inserts,
                table_id,
                job_config=bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
                ),
            )
            load_job.result()
            logger.info("Inserted %s semantic candidate rows", len(inserts))
        else:
            logger.info("No semantic candidates generated for current probes")


def build_semantic_candidates(
    project_id: str,
    dataset_id: str,
    embedding_model: str = "text-embedding-004",
) -> None:
    connection = BigQueryConnectionManager(
        config=BigQueryConfig(project_id=project_id, dataset_id=dataset_id)
    )

    loader = RuleLoader()
    rules = loader.load()
    validator = RuleConfigValidator()
    for rule in rules:
        validator.validate(rule)

    builder = SemanticCandidateBuilder(
        connection=connection,
        embedding_model=embedding_model,
    )
    builder.refresh_candidates(rules)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Populate semantic probe embeddings and candidate neighbors",
    )
    parser.add_argument("--project", required=True, help="BigQuery project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument(
        "--model",
        default="text-embedding-004",
        help="Vertex AI text embedding model name",
    )

    args = parser.parse_args()
    build_semantic_candidates(
        project_id=args.project,
        dataset_id=args.dataset,
        embedding_model=args.model,
    )
