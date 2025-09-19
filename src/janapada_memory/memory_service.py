"""
Janapada Memory Service - Unified BigQuery Vector Search Interface.

This module provides the main service interface that integrates all components
and implements the similarity_search method expected by the orchestrator.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .bigquery_vector_index import BigQueryVectorIndex, SearchMode
from .connections.bigquery_connection import BigQueryConnectionManager
from .models.vector_search_config import VectorSearchConfig

logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Custom exception for memory service errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class JanapadaMemoryService:
    """
    Unified memory service implementing BigQuery vector search with fallback.

    This is the main service interface that the orchestrator uses for semantic search.
    It integrates BigQueryVectorIndex, provides the similarity_search method required
    by issue #4 acceptance criteria, and handles configuration and lifecycle management.
    """

    def __init__(
        self,
        connection: Optional[BigQueryConnectionManager] = None,
        config: Optional[VectorSearchConfig] = None,
        enable_fallback: bool = True,
        fallback_max_entries: int = 1000,
    ):
        """
        Initialize Janapada memory service.

        Args:
            connection: BigQuery connection manager
            config: Vector search configuration
            enable_fallback: Whether to enable local fallback
            fallback_max_entries: Maximum entries in fallback index
        """
        self.connection = connection or BigQueryConnectionManager()
        self.config = config or VectorSearchConfig.from_env()
        self.enable_fallback = enable_fallback

        # Initialize the vector index
        self.vector_index = BigQueryVectorIndex(
            connection=self.connection,
            config=self.config,
            enable_fallback=enable_fallback,
            fallback_max_entries=fallback_max_entries,
            search_mode=SearchMode.BIGQUERY_WITH_FALLBACK,
        )

        # Service metadata
        self.service_id = self.vector_index.index_id
        self.start_time = time.time()

        self.logger = logging.getLogger(f"{__name__}.{self.service_id[:8]}")

        self.logger.info(
            "Janapada memory service initialized",
            extra={
                "service_id": self.service_id,
                "project_id": self.connection.config.project_id,
                "dataset_id": self.connection.config.dataset_id,
                "enable_fallback": enable_fallback,
                "fallback_max_entries": fallback_max_entries,
            },
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        artifact_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using BigQuery with automatic fallback.

        This method implements the similarity_search interface required by issue #4:
        - Uses BigQuery VECTOR_SEARCH with approximate_neighbors
        - Falls back to local in-memory vector index when BigQuery is unavailable
        - Returns ordered results with similarity scores

        Args:
            query_embedding: Query vector embedding
            k: Number of results to return (1-1000)
            artifact_types: Optional filter by artifact types

        Returns:
            List of similar vectors as dictionaries with chunk_id, similarity_score,
            source, artifact_type, text_content, and metadata

        Raises:
            MemoryServiceError: If search fails completely
        """
        try:
            # Validate parameters
            if not query_embedding:
                raise MemoryServiceError("Query embedding cannot be empty")

            if k < 1 or k > 1000:
                raise MemoryServiceError(
                    f"Parameter k must be between 1 and 1000, got {k}"
                )

            # Perform search using the vector index
            results = self.vector_index.similarity_search(
                query_embedding=query_embedding,
                k=k,
                artifact_types=artifact_types,
            )

            # Convert VectorSearchResult objects to dictionaries for orchestrator
            search_results = []
            for result in results:
                search_results.append(
                    {
                        "chunk_id": result.chunk_id,
                        "similarity_score": result.similarity_score,
                        "distance": result.distance,
                        "source": result.source,
                        "artifact_type": result.artifact_type,
                        "text_content": result.text_content,
                        "metadata": result.metadata,
                    }
                )

            self.logger.debug(
                f"Similarity search completed: {len(search_results)} results for k={k}"
            )

            return search_results

        except Exception as e:
            error_msg = f"Similarity search failed: {e}"
            self.logger.error(error_msg)
            raise MemoryServiceError(error_msg, e)

    def add_vectors(
        self,
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add vectors to the memory service.

        Args:
            vectors: List of vector data dictionaries

        Returns:
            Addition result summary
        """
        try:
            return self.vector_index.add_vectors(vectors)
        except Exception as e:
            error_msg = f"Vector addition failed: {e}"
            self.logger.error(error_msg)
            raise MemoryServiceError(error_msg, e)

    def remove_vector(self, chunk_id: str) -> bool:
        """
        Remove a vector from the memory service.

        Args:
            chunk_id: ID of vector to remove

        Returns:
            True if vector was removed, False if not found
        """
        try:
            return self.vector_index.remove_vector(chunk_id)
        except Exception as e:
            error_msg = f"Vector removal failed: {e}"
            self.logger.error(error_msg)
            raise MemoryServiceError(error_msg, e)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the memory service.

        Returns:
            Health status information including BigQuery connectivity,
            fallback index status, and service statistics
        """
        try:
            index_health = self.vector_index.health_check()

            # Add service-level information
            service_health = {
                **index_health,
                "service_id": self.service_id,
                "service_type": "janapada_memory",
                "uptime_seconds": time.time() - self.start_time,
                "api_version": "1.0.0",
                "bigquery_integration": {
                    "enabled": True,
                    "project_id": self.connection.config.project_id,
                    "dataset_id": self.connection.config.dataset_id,
                    "table_name": self.config.table_name,
                },
                "fallback_integration": {
                    "enabled": self.enable_fallback,
                    "type": "local_vector_index",
                },
            }

            return service_health

        except Exception as e:
            return {
                "status": "unhealthy",
                "service_id": self.service_id,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with service performance and usage statistics
        """
        try:
            index_stats = self.vector_index.get_stats()

            return {
                **index_stats,
                "service_id": self.service_id,
                "uptime_seconds": time.time() - self.start_time,
                "service_start_time": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(self.start_time)
                ),
            }

        except Exception as e:
            return {
                "error": str(e),
                "service_id": self.service_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

    def configure_search_mode(self, mode: SearchMode) -> None:
        """
        Configure the search execution mode.

        Args:
            mode: Search mode (bigquery_only, fallback_only, bigquery_with_fallback)
        """
        self.vector_index.search_mode = mode

        self.logger.info(
            f"Search mode configured to: {mode.value}",
            extra={"service_id": self.service_id, "search_mode": mode.value},
        )

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate service configuration and dependencies.

        Returns:
            Validation results including BigQuery connectivity and table access
        """
        validation_results = {
            "configuration_valid": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        try:
            # Check BigQuery connection
            health = self.connection.check_health()
            validation_results["checks"]["bigquery_connection"] = {
                "status": "pass" if health.is_healthy else "fail",
                "response_time_ms": health.response_time_ms,
                "error": health.error_message,
            }

            if not health.is_healthy:
                validation_results["errors"].append(
                    f"BigQuery connection failed: {health.error_message}"
                )
                validation_results["configuration_valid"] = False

            # Check dataset access
            dataset_accessible = self.connection.check_dataset_access()
            validation_results["checks"]["dataset_access"] = {
                "status": "pass" if dataset_accessible else "fail",
            }

            if not dataset_accessible:
                validation_results["errors"].append("BigQuery dataset not accessible")
                validation_results["configuration_valid"] = False

            # Check table access
            table_accessible = self.connection.check_table_access("source_embeddings")
            validation_results["checks"]["embeddings_table"] = {
                "status": "pass" if table_accessible else "fail",
            }

            if not table_accessible:
                validation_results["warnings"].append(
                    "source_embeddings table not found - may need setup"
                )

            # Check fallback index
            if self.enable_fallback:
                fallback_health = self.vector_index.local_index.health_check()
                validation_results["checks"]["fallback_index"] = {
                    "status": (
                        "pass" if fallback_health["status"] == "healthy" else "fail"
                    ),
                    "vector_count": fallback_health.get("vector_count", 0),
                }

        except Exception as e:
            validation_results["errors"].append(f"Configuration validation failed: {e}")
            validation_results["configuration_valid"] = False

        return validation_results

    def close(self) -> None:
        """Close the memory service and cleanup resources."""
        self.logger.info(f"Closing Janapada memory service {self.service_id[:8]}...")

        try:
            self.vector_index.close()
            self.logger.info("Memory service closed successfully")
        except Exception as e:
            self.logger.error(f"Error during service shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"JanapadaMemoryService("
            f"service_id='{self.service_id[:8]}...', "
            f"project='{self.connection.config.project_id}', "
            f"dataset='{self.connection.config.dataset_id}', "
            f"fallback={self.enable_fallback}"
            f")"
        )


# Factory function for easy service creation
def create_memory_service(
    project_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    enable_fallback: bool = True,
    **kwargs,
) -> JanapadaMemoryService:
    """
    Factory function to create a configured memory service.

    Args:
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        enable_fallback: Whether to enable local fallback
        **kwargs: Additional configuration options

    Returns:
        Configured JanapadaMemoryService instance
    """
    # Create configuration
    if project_id or dataset_id:
        from .config import BigQueryConfig

        config = BigQueryConfig(
            project_id=project_id or "konveyn2ai",
            dataset_id=dataset_id or "semantic_gap_detector",
        )
        connection = BigQueryConnectionManager(config=config)
    else:
        connection = None

    return JanapadaMemoryService(
        connection=connection, enable_fallback=enable_fallback, **kwargs
    )
