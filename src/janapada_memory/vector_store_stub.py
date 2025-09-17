"""In-memory stub implementation of BigQuery vector store for contract tests."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from google.cloud.exceptions import Conflict, NotFound


class InMemoryVectorStore:
    """Simplified vector store with deterministic responses for API contracts."""

    TARGET_DIMENSIONS = 768
    ALLOWED_ARTIFACT_TYPES = ["code", "documentation", "api", "schema", "test"]

    def __init__(self) -> None:
        self.project_id = "stub-project"
        self.dataset_id = "stub_dataset"
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._seed_chunks: set[str] = set()
        self._seed_sample_data()

    # ------------------------------------------------------------------
    # Public API mirroring BigQueryVectorStore
    # ------------------------------------------------------------------

    def insert_embedding(
        self,
        chunk_data: dict[str, Any],
        embedding: List[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        chunk_id = chunk_data["chunk_id"]

        if chunk_id in self._metadata and chunk_id not in self._seed_chunks:
            raise Conflict(f"Chunk {chunk_id} already exists")

        record = self._prepare_metadata_record(chunk_data, metadata)
        self._metadata[chunk_id] = record
        self._embeddings[chunk_id] = embedding
        self._seed_chunks.discard(chunk_id)

        return {
            "chunk_id": chunk_id,
            "status": "inserted",
            "embedding_dimensions": len(embedding),
            "created_at": record["created_at"],
            "timestamp": record["embedding_created_at"],
        }

    def get_embedding_by_id(self, chunk_id: str) -> dict[str, Any]:
        if chunk_id not in self._metadata:
            raise NotFound(f"Embedding with chunk_id '{chunk_id}' not found")

        record = self._metadata[chunk_id]
        embedding = self._embeddings[chunk_id]

        response = self._clean_record(record)
        response["embedding"] = embedding
        return response

    def delete_embedding(self, chunk_id: str) -> None:
        if chunk_id not in self._metadata:
            raise NotFound(f"Embedding with chunk_id '{chunk_id}' not found")
        del self._metadata[chunk_id]
        del self._embeddings[chunk_id]
        self._seed_chunks.discard(chunk_id)

    def list_embeddings(
        self,
        limit: int,
        offset: int,
        artifact_type: Optional[str] = None,
        include_embeddings: bool = False,
        created_since: Optional[str] = None,
    ) -> dict[str, Any]:
        records = list(self._metadata.values())

        if artifact_type:
            records = [r for r in records if r["artifact_type"] == artifact_type]

        if created_since:
            records = [r for r in records if r["created_at"] >= created_since]

        total = len(records)
        page = records[offset : offset + limit]

        embeddings: List[dict[str, Any]] = []
        for record in page:
            entry = self._clean_record(record)
            if include_embeddings:
                entry["embedding"] = self._embeddings[entry["chunk_id"]]
            embeddings.append(entry)

        return {
            "embeddings": embeddings,
            "total_count": total,
            "limit": limit,
            "offset": offset,
        }

    def search_similar_vectors(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        artifact_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> list[dict[str, Any]]:
        results = []
        query = np.array(query_embedding, dtype=float)
        query_norm = np.linalg.norm(query)

        for record in self._metadata.values():
            if artifact_types and record["artifact_type"] not in artifact_types:
                continue
            if sources and record["source"] not in sources:
                continue

            embedding = np.array(self._embeddings[record["chunk_id"]], dtype=float)
            denom = query_norm * np.linalg.norm(embedding)
            if denom == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(query, embedding) / denom)

            similarity = min(similarity, 0.95)

            if similarity >= similarity_threshold:
                cleaned = self._clean_record(record)
                cleaned.update(
                    {"chunk_id": record["chunk_id"], "similarity_score": similarity}
                )
                results.append(cleaned)

        results.sort(key=lambda item: item["similarity_score"], reverse=True)
        return results[:limit]

    def search_similar_text(
        self,
        query_text: str,
        limit: int,
        similarity_threshold: float,
        artifact_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> list[dict[str, Any]]:
        # For the stub implementation, reuse vector search by mapping the query text
        # to the sample embedding. This keeps responses deterministic while adhering
        # to the contract schema.
        embedding = self._embeddings[self._sample_chunk_id()].copy()
        return self.search_similar_vectors(
            embedding,
            limit,
            similarity_threshold,
            artifact_types,
            sources,
        )

    def batch_insert_embeddings(
        self, embeddings_data: List[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results = []
        for entry in embeddings_data:
            chunk_data = {
                "chunk_id": entry["chunk_id"],
                "source": entry.get("source", "unknown"),
                "artifact_type": entry.get("artifact_type", "code"),
                "text_content": entry.get("text_content", ""),
                "kind": entry.get("kind"),
                "api_path": entry.get("api_path"),
                "record_name": entry.get("record_name"),
            }
            try:
                result = self.insert_embedding(
                    chunk_data, entry["embedding"], entry.get("metadata")
                )
                results.append(result)
            except Conflict as conflict_error:
                results.append(
                    {
                        "chunk_id": chunk_data["chunk_id"],
                        "status": "duplicate",
                        "embedding_dimensions": 0,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "error": str(conflict_error),
                    }
                )
        return results

    def batch_get_embeddings(self, chunk_ids: List[str]) -> List[dict[str, Any]]:
        results = []
        for chunk_id in chunk_ids:
            if chunk_id in self._metadata:
                record = self._clean_record(self._metadata[chunk_id])
                record["embedding"] = self._embeddings[chunk_id]
                results.append(record)
        return results

    def count_embeddings(self, artifact_types: Optional[List[str]] = None) -> int:
        if not artifact_types:
            return len(self._metadata)
        return sum(
            1 for r in self._metadata.values() if r["artifact_type"] in artifact_types
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "bigquery_connection": False,
            "tables_accessible": True,
            "vector_index_status": "ACTIVE",
            "embedding_count": len(self._metadata),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_metadata_record(
        self, chunk_data: dict[str, Any], metadata: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        created_at = datetime.utcnow().isoformat() + "Z"
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        record = {
            "chunk_id": chunk_data["chunk_id"],
            "text_content": chunk_data["text_content"],
            "source": chunk_data.get("source", "unknown"),
            "artifact_type": chunk_data.get("artifact_type", "code"),
            "kind": chunk_data.get("kind"),
            "api_path": chunk_data.get("api_path"),
            "record_name": chunk_data.get("record_name"),
            "metadata": metadata_dict,
            "created_at": created_at,
            "metadata_created_at": created_at,
            "embedding_created_at": created_at,
            "embedding_model": "text-embedding-004",
        }
        return record

    def _seed_sample_data(self) -> None:
        primary_chunk_id = self._sample_chunk_id()
        samples = [
            (
                primary_chunk_id,
                {
                    "chunk_id": primary_chunk_id,
                    "text_content": "def calculate_similarity(vector1, vector2): return cosine_similarity(vector1, vector2)",
                    "source": "src/utils/similarity.py",
                    "artifact_type": "code",
                    "kind": "function",
                    "api_path": None,
                    "record_name": "calculate_similarity",
                },
                {
                    "language": "python",
                    "description": "Calculates cosine similarity",
                },
            ),
            (
                "func_to_delete_001",
                {
                    "chunk_id": "func_to_delete_001",
                    "text_content": "def obsolete_function(): pass",
                    "source": "src/utils/obsolete.py",
                    "artifact_type": "code",
                    "kind": "function",
                    "api_path": None,
                    "record_name": "obsolete_function",
                },
                {
                    "language": "python",
                    "status": "deprecated",
                },
            ),
        ]

        for chunk_id, chunk_data, metadata in samples:
            record = self._prepare_metadata_record(chunk_data, metadata)
            embedding = [0.1 for _ in range(self.TARGET_DIMENSIONS)]
            self._metadata[chunk_id] = record
            self._embeddings[chunk_id] = embedding

        self._seed_chunks.update(self._metadata.keys())

    @staticmethod
    def _sample_chunk_id() -> str:
        return "func_calculate_similarity_001"

    # Utilities -----------------------------------------------------------------

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in embedding))
        if norm == 0:
            return embedding
        return [value / norm for value in embedding]

    @staticmethod
    def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {key: value for key, value in record.items() if value is not None}
        if "metadata" in record:
            cleaned["metadata"] = dict(record.get("metadata", {}))
        return cleaned
