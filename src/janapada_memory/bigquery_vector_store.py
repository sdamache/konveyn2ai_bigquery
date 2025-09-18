"""
BigQuery Vector Store

Core vector storage operations using BigQuery with VECTOR_SEARCH capabilities.
Handles embedding insertion, retrieval, similarity search, and batch operations.
"""

import json
import logging
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from .connections.bigquery_connection import BigQueryConnectionManager
from .dimension_reducer import DimensionReducer

logger = logging.getLogger(__name__)


class BigQueryVectorStore:
    """BigQuery-based vector storage with similarity search capabilities."""

    def __init__(
        self,
        connection: Optional[BigQueryConnectionManager] = None,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        embedding_model: str = "text-embedding-004",
        target_dimensions: int = 768,
        dimension_reducer: Optional[DimensionReducer] = None,
    ):
        """
        Initialize BigQuery vector store.

        Args:
            connection: BigQuery connection instance
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            embedding_model: Default embedding model name
            target_dimensions: Target embedding dimensions
            dimension_reducer: Optional dimension reducer for legacy embeddings
        """
        if connection:
            self.connection = connection
        else:
            from .config import BigQueryConfig
            config = BigQueryConfig(
                project_id=project_id or "konveyn2ai",
                dataset_id=dataset_id or "semantic_gap_detector"
            )
            self.connection = BigQueryConnectionManager(config=config)

        self.project_id = self.connection.config.project_id
        self.dataset_id = self.connection.config.dataset_id
        self.embedding_model = embedding_model
        self.target_dimensions = target_dimensions
        self.dimension_reducer = dimension_reducer

        # Table references
        self.metadata_table = f"`{self.project_id}.{self.dataset_id}.source_metadata`"
        self.embeddings_table = (
            f"`{self.project_id}.{self.dataset_id}.source_embeddings`"
        )
        self.metrics_table = f"`{self.project_id}.{self.dataset_id}.gap_metrics`"

        logger.info(
            f"BigQuery vector store initialized: {self.project_id}.{self.dataset_id}"
        )

    def insert_embedding(
        self,
        chunk_data: dict[str, Any],
        embedding: list[float],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Insert embedding with associated metadata.

        Args:
            chunk_data: Chunk metadata (chunk_id, source, artifact_type, text_content, etc.)
            embedding: Vector embedding
            metadata: Additional metadata

        Returns:
            Insertion result
        """
        chunk_id = chunk_data["chunk_id"]

        # Validate and potentially reduce dimensions
        processed_embedding = self._process_embedding(embedding)

        # Prepare metadata record
        metadata_record = self._prepare_metadata_record(chunk_data, metadata)

        # Prepare embedding record
        embedding_record = {
            "chunk_id": chunk_id,
            "embedding": processed_embedding,
            "embedding_model": self.embedding_model,
            "created_at": datetime.now(),
            "partition_date": date.today(),
        }

        try:
            # Insert metadata
            metadata_errors = self.connection.insert_rows(
                self.connection.client.dataset(self.dataset_id).table(
                    "source_metadata"
                ),
                [metadata_record],
            )

            if metadata_errors:
                logger.error(f"Metadata insertion errors: {metadata_errors}")
                raise ValueError(f"Failed to insert metadata: {metadata_errors}")

            # Insert embedding
            embedding_errors = self.connection.insert_rows(
                self.connection.client.dataset(self.dataset_id).table(
                    "source_embeddings"
                ),
                [embedding_record],
            )

            if embedding_errors:
                logger.error(f"Embedding insertion errors: {embedding_errors}")
                raise ValueError(f"Failed to insert embedding: {embedding_errors}")

            logger.debug(f"Successfully inserted embedding for chunk: {chunk_id}")

            return {
                "chunk_id": chunk_id,
                "status": "inserted",
                "embedding_dimensions": len(processed_embedding),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to insert embedding for chunk {chunk_id}: {e}")
            raise

    def get_embedding_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve embedding and metadata by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Combined chunk data and embedding
        """
        try:
            query = f"""
            SELECT
                m.chunk_id,
                m.source,
                m.artifact_type,
                m.text_content,
                m.kind,
                m.api_path,
                m.record_name,
                m.metadata,
                m.created_at as metadata_created_at,
                e.embedding,
                e.embedding_model,
                e.created_at as embedding_created_at
            FROM {self.metadata_table} m
            JOIN {self.embeddings_table} e ON m.chunk_id = e.chunk_id
            WHERE m.chunk_id = @chunk_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("chunk_id", "STRING", chunk_id)
                ]
            )

            results = list(self.connection.execute_query(query, job_config=job_config))

            if not results:
                logger.warning(f"Chunk not found: {chunk_id}")
                raise NotFound(f"Chunk with ID '{chunk_id}' not found")

            row = results[0]

            result = {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "artifact_type": row.artifact_type,
                "text_content": row.text_content,
                "kind": row.kind,
                "api_path": row.api_path,
                "record_name": row.record_name,
                "metadata": json.loads(row.metadata) if row.metadata else {},
                "embedding": row.embedding,
                "embedding_model": row.embedding_model,
                "metadata_created_at": row.metadata_created_at.isoformat(),
                "embedding_created_at": row.embedding_created_at.isoformat(),
            }

            return result

        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to get embedding for chunk {chunk_id}: {e}")
            raise

    def list_embeddings(
        self,
        limit: int = 100,
        offset: int = 0,
        artifact_types: Optional[list[str]] = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        List embeddings with pagination and filtering.

        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            artifact_types: Filter by artifact types
            include_embeddings: Whether to include embedding vectors

        Returns:
            Paginated embedding list
        """
        try:
            # Build WHERE clause
            where_conditions = []
            query_params = []

            if artifact_types:
                placeholders = ", ".join(
                    [f"@artifact_type_{i}" for i in range(len(artifact_types))]
                )
                where_conditions.append(f"m.artifact_type IN ({placeholders})")

                for i, artifact_type in enumerate(artifact_types):
                    query_params.append(
                        bigquery.ScalarQueryParameter(
                            f"artifact_type_{i}", "STRING", artifact_type
                        )
                    )

            where_clause = (
                f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            )

            # Select fields
            embedding_field = "e.embedding," if include_embeddings else ""

            # Build query
            query = f"""
            SELECT
                m.chunk_id,
                m.source,
                m.artifact_type,
                m.text_content,
                m.kind,
                m.api_path,
                m.record_name,
                m.metadata,
                m.created_at as metadata_created_at,
                {embedding_field}
                e.embedding_model,
                e.created_at as embedding_created_at
            FROM {self.metadata_table} m
            JOIN {self.embeddings_table} e ON m.chunk_id = e.chunk_id
            {where_clause}
            ORDER BY m.created_at DESC
            LIMIT @limit
            OFFSET @offset
            """

            # Add pagination parameters
            query_params.extend(
                [
                    bigquery.ScalarQueryParameter("limit", "INT64", limit),
                    bigquery.ScalarQueryParameter("offset", "INT64", offset),
                ]
            )

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            results = list(self.connection.execute_query(query, job_config=job_config))

            # Format results
            embeddings = []
            for row in results:
                embedding_data = {
                    "chunk_id": row.chunk_id,
                    "source": row.source,
                    "artifact_type": row.artifact_type,
                    "text_content": row.text_content,
                    "kind": row.kind,
                    "api_path": row.api_path,
                    "record_name": row.record_name,
                    "metadata": json.loads(row.metadata) if row.metadata else {},
                    "embedding_model": row.embedding_model,
                    "metadata_created_at": row.metadata_created_at.isoformat(),
                    "embedding_created_at": row.embedding_created_at.isoformat(),
                }

                if include_embeddings:
                    embedding_data["embedding"] = row.embedding

                embeddings.append(embedding_data)

            # Get total count for pagination
            count_query = f"""
            SELECT COUNT(*) as total_count
            FROM {self.metadata_table} m
            JOIN {self.embeddings_table} e ON m.chunk_id = e.chunk_id
            {where_clause}
            """

            count_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    p for p in query_params if p.name.startswith("artifact_type_")
                ]
            )
            count_result = list(
                self.connection.execute_query(count_query, job_config=count_job_config)
            )
            total_count = count_result[0].total_count if count_result else 0

            return {
                "embeddings": embeddings,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total_count": total_count,
                    "has_next": offset + limit < total_count,
                    "has_previous": offset > 0,
                },
                "filters": {"artifact_types": artifact_types},
            }

        except Exception as e:
            logger.error(f"Failed to list embeddings: {e}")
            raise

    def search_similar_vectors(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        artifact_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors using BigQuery VECTOR_SEARCH.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            artifact_types: Filter by artifact types

        Returns:
            List of similar vectors with similarity scores
        """
        try:
            # Process query embedding
            processed_query = self._process_embedding(query_embedding)

            # Build WHERE clause for artifact type filtering
            artifact_filter = ""
            query_params = [
                bigquery.ArrayQueryParameter(
                    "query_embedding", "FLOAT64", processed_query
                ),
                bigquery.ScalarQueryParameter(
                    "similarity_threshold", "FLOAT64", similarity_threshold
                ),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]

            if artifact_types:
                placeholders = ", ".join(
                    [f"@artifact_type_{i}" for i in range(len(artifact_types))]
                )
                artifact_filter = f"AND m.artifact_type IN ({placeholders})"

                for i, artifact_type in enumerate(artifact_types):
                    query_params.append(
                        bigquery.ScalarQueryParameter(
                            f"artifact_type_{i}", "STRING", artifact_type
                        )
                    )

            # BigQuery vector similarity search
            query = f"""
            WITH vector_search AS (
                SELECT
                    base.chunk_id,
                    distance
                FROM VECTOR_SEARCH(
                    TABLE {self.embeddings_table},
                    'embedding',
                    (SELECT @query_embedding as embedding),
                    top_k => @limit,
                    distance_type => 'COSINE'
                ) AS base
                WHERE (1 - distance) >= @similarity_threshold
            )
            SELECT
                vs.chunk_id,
                (1 - vs.distance) as similarity_score,
                m.source,
                m.artifact_type,
                m.text_content,
                m.kind,
                m.api_path,
                m.record_name,
                m.metadata,
                m.created_at
            FROM vector_search vs
            JOIN {self.metadata_table} m ON vs.chunk_id = m.chunk_id
            {artifact_filter}
            ORDER BY similarity_score DESC
            """

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            results = list(self.connection.execute_query(query, job_config=job_config))

            # Format results
            similar_vectors = []
            for row in results:
                similar_vectors.append(
                    {
                        "chunk_id": row.chunk_id,
                        "similarity_score": float(row.similarity_score),
                        "source": row.source,
                        "artifact_type": row.artifact_type,
                        "text_content": row.text_content,
                        "kind": row.kind,
                        "api_path": row.api_path,
                        "record_name": row.record_name,
                        "metadata": json.loads(row.metadata) if row.metadata else {},
                        "created_at": row.created_at.isoformat(),
                    }
                )

            logger.info(f"Vector search returned {len(similar_vectors)} results")
            return similar_vectors

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise

    def search_similar_text(
        self,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        artifact_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar text using text embedding and vector search.

        Args:
            query_text: Query text to search for
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            artifact_types: Filter by artifact types

        Returns:
            List of similar text chunks with similarity scores
        """
        # Note: This is a placeholder implementation
        # In practice, you would generate embeddings for query_text using your embedding model
        # For now, we'll create a dummy embedding

        logger.warning(
            "search_similar_text using dummy embedding - implement actual text embedding generation"
        )

        # Generate dummy embedding (replace with actual embedding generation)
        query_embedding = np.random.rand(self.target_dimensions).tolist()

        # Use vector search
        return self.search_similar_vectors(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            artifact_types=artifact_types,
        )

    def delete_embedding(self, chunk_id: str) -> bool:
        """
        Delete embedding and associated metadata.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if deletion was successful
        """
        try:
            # Delete from both tables
            delete_metadata_query = f"""
            DELETE FROM {self.metadata_table}
            WHERE chunk_id = @chunk_id
            """

            delete_embedding_query = f"""
            DELETE FROM {self.embeddings_table}
            WHERE chunk_id = @chunk_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("chunk_id", "STRING", chunk_id)
                ]
            )

            # Execute deletions
            self.connection.execute_query(delete_metadata_query, job_config=job_config)
            self.connection.execute_query(delete_embedding_query, job_config=job_config)

            logger.info(f"Deleted embedding for chunk: {chunk_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete embedding for chunk {chunk_id}: {e}")
            raise

    def batch_insert_embeddings(
        self, embeddings_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Insert multiple embeddings in batch.

        Args:
            embeddings_data: List of embedding data dictionaries

        Returns:
            List of insertion results
        """
        try:
            metadata_records = []
            embedding_records = []

            # Prepare batch data
            for data in embeddings_data:
                chunk_id = data["chunk_id"]
                embedding = self._process_embedding(data["embedding"])

                metadata_record = self._prepare_metadata_record(
                    data, data.get("metadata")
                )
                metadata_records.append(metadata_record)

                embedding_record = {
                    "chunk_id": chunk_id,
                    "embedding": embedding,
                    "embedding_model": data.get(
                        "embedding_model", self.embedding_model
                    ),
                    "created_at": datetime.now(),
                    "partition_date": date.today(),
                }
                embedding_records.append(embedding_record)

            # Batch insert metadata
            metadata_errors = self.connection.insert_rows(
                self.connection.client.dataset(self.dataset_id).table(
                    "source_metadata"
                ),
                metadata_records,
            )

            if metadata_errors:
                logger.error(f"Batch metadata insertion errors: {metadata_errors}")
                raise ValueError(f"Failed to batch insert metadata: {metadata_errors}")

            # Batch insert embeddings
            embedding_errors = self.connection.insert_rows(
                self.connection.client.dataset(self.dataset_id).table(
                    "source_embeddings"
                ),
                embedding_records,
            )

            if embedding_errors:
                logger.error(f"Batch embedding insertion errors: {embedding_errors}")
                raise ValueError(
                    f"Failed to batch insert embeddings: {embedding_errors}"
                )

            # Return results
            results = []
            for data in embeddings_data:
                results.append(
                    {
                        "chunk_id": data["chunk_id"],
                        "status": "inserted",
                        "embedding_dimensions": len(
                            self._process_embedding(data["embedding"])
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            logger.info(f"Batch inserted {len(results)} embeddings")
            return results

        except Exception as e:
            logger.error(f"Batch embedding insertion failed: {e}")
            raise

    def batch_get_embeddings(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve multiple embeddings by chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            List of embedding data
        """
        try:
            if not chunk_ids:
                return []

            # Build query with IN clause
            placeholders = ", ".join([f"@chunk_id_{i}" for i in range(len(chunk_ids))])
            query = f"""
            SELECT
                m.chunk_id,
                m.source,
                m.artifact_type,
                m.text_content,
                m.kind,
                m.api_path,
                m.record_name,
                m.metadata,
                m.created_at as metadata_created_at,
                e.embedding,
                e.embedding_model,
                e.created_at as embedding_created_at
            FROM {self.metadata_table} m
            JOIN {self.embeddings_table} e ON m.chunk_id = e.chunk_id
            WHERE m.chunk_id IN ({placeholders})
            ORDER BY m.created_at DESC
            """

            query_params = [
                bigquery.ScalarQueryParameter(f"chunk_id_{i}", "STRING", chunk_id)
                for i, chunk_id in enumerate(chunk_ids)
            ]

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            results = list(self.connection.execute_query(query, job_config=job_config))

            # Format results
            embeddings = []
            for row in results:
                embeddings.append(
                    {
                        "chunk_id": row.chunk_id,
                        "source": row.source,
                        "artifact_type": row.artifact_type,
                        "text_content": row.text_content,
                        "kind": row.kind,
                        "api_path": row.api_path,
                        "record_name": row.record_name,
                        "metadata": json.loads(row.metadata) if row.metadata else {},
                        "embedding": row.embedding,
                        "embedding_model": row.embedding_model,
                        "metadata_created_at": row.metadata_created_at.isoformat(),
                        "embedding_created_at": row.embedding_created_at.isoformat(),
                    }
                )

            logger.info(f"Batch retrieved {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Batch get embeddings failed: {e}")
            raise

    def count_embeddings(self, artifact_types: Optional[list[str]] = None) -> int:
        """
        Count total number of embeddings.

        Args:
            artifact_types: Optional filter by artifact types

        Returns:
            Total count of embeddings
        """
        try:
            where_conditions = []
            query_params = []

            if artifact_types:
                placeholders = ", ".join(
                    [f"@artifact_type_{i}" for i in range(len(artifact_types))]
                )
                where_conditions.append(f"m.artifact_type IN ({placeholders})")

                for i, artifact_type in enumerate(artifact_types):
                    query_params.append(
                        bigquery.ScalarQueryParameter(
                            f"artifact_type_{i}", "STRING", artifact_type
                        )
                    )

            where_clause = (
                f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
            )

            query = f"""
            SELECT COUNT(*) as total_count
            FROM {self.metadata_table} m
            JOIN {self.embeddings_table} e ON m.chunk_id = e.chunk_id
            {where_clause}
            """

            job_config = bigquery.QueryJobConfig(query_parameters=query_params)
            results = list(self.connection.execute_query(query, job_config=job_config))

            return results[0].total_count if results else 0

        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on vector store.

        Returns:
            Health status information
        """
        health_status = {
            "status": "unknown",
            "bigquery_connection": False,
            "tables_accessible": False,
            "vector_index_status": "unknown",
            "embedding_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check BigQuery connection
            connection_health = self.connection.health_check()
            health_status["bigquery_connection"] = connection_health[
                "bigquery_connection"
            ]

            # Check table accessibility
            try:
                embedding_count = self.count_embeddings()
                health_status["tables_accessible"] = True
                health_status["embedding_count"] = embedding_count
            except Exception as e:
                logger.warning(f"Table access check failed: {e}")
                health_status["tables_accessible"] = False

            # Check vector index status (if available)
            try:
                index_query = f"""
                SELECT COUNT(*) as index_count
                FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
                WHERE table_name = 'source_embeddings' AND status = 'ACTIVE'
                """
                index_results = list(self.connection.execute_query(index_query))
                index_count = index_results[0].index_count if index_results else 0
                health_status["vector_index_status"] = (
                    "active" if index_count > 0 else "inactive"
                )
            except Exception as e:
                logger.debug(f"Vector index status check failed: {e}")
                health_status["vector_index_status"] = "unknown"

            # Determine overall health
            if (
                health_status["bigquery_connection"]
                and health_status["tables_accessible"]
            ):
                health_status["status"] = "healthy"
            else:
                health_status["status"] = "unhealthy"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def _process_embedding(self, embedding: list[float]) -> list[float]:
        """Process embedding (dimension reduction if needed)."""
        if len(embedding) == self.target_dimensions:
            return embedding

        if self.dimension_reducer and self.dimension_reducer.is_fitted:
            # Reduce dimensions using fitted PCA
            reduced = self.dimension_reducer.transform([embedding])
            return reduced[0].tolist()

        if len(embedding) > self.target_dimensions:
            # Simple truncation as fallback
            logger.warning(
                f"Truncating embedding from {len(embedding)} to {self.target_dimensions} dimensions"
            )
            return embedding[: self.target_dimensions]

        # Pad with zeros if too short
        if len(embedding) < self.target_dimensions:
            logger.warning(
                f"Padding embedding from {len(embedding)} to {self.target_dimensions} dimensions"
            )
            return embedding + [0.0] * (self.target_dimensions - len(embedding))

        return embedding

    def _prepare_metadata_record(
        self, chunk_data: dict[str, Any], metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Prepare metadata record for insertion."""
        record = {
            "chunk_id": chunk_data["chunk_id"],
            "source": chunk_data["source"],
            "artifact_type": chunk_data["artifact_type"],
            "text_content": chunk_data["text_content"],
            "kind": chunk_data.get("kind"),
            "api_path": chunk_data.get("api_path"),
            "record_name": chunk_data.get("record_name"),
            "metadata": json.dumps(metadata) if metadata else None,
            "created_at": datetime.now(),
            "partition_date": date.today(),
        }

        return record
