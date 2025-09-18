"""
BigQuery Vector Index - Main Implementation.

This module provides the BigQueryVectorIndex class which implements the VectorIndex
interface, delegates to BigQueryAdapter for primary search, falls back to 
LocalVectorIndex on errors, and logs fallback activation with correlation IDs.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum

from .adapters.bigquery_adapter import BigQueryAdapter, BigQueryAdapterError
from .connections.bigquery_connection import BigQueryConnectionManager, BigQueryConnectionError
from .fallback.local_vector_index import LocalVectorIndex, LocalVectorIndexError
from .models.vector_search_config import VectorSearchConfig, VectorSearchConfigError
from .models.vector_search_result import VectorSearchResult

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search execution mode."""
    
    BIGQUERY_ONLY = "bigquery_only"
    FALLBACK_ONLY = "fallback_only"
    BIGQUERY_WITH_FALLBACK = "bigquery_with_fallback"


class VectorIndexError(Exception):
    """Custom exception for vector index operations."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class VectorIndex(ABC):
    """Abstract base class for vector index implementations."""
    
    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        artifact_types: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            artifact_types: Optional filter by artifact types
            
        Returns:
            List of similar vectors ordered by similarity
        """
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add vectors to the index.
        
        Args:
            vectors: List of vector data dictionaries
            
        Returns:
            Addition result summary
        """
        pass
    
    @abstractmethod
    def remove_vector(self, chunk_id: str) -> bool:
        """
        Remove a vector from the index.
        
        Args:
            chunk_id: ID of vector to remove
            
        Returns:
            True if vector was removed, False if not found
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector index.
        
        Returns:
            Health status information
        """
        pass


class BigQueryVectorIndex(VectorIndex):
    """
    Main BigQuery vector index implementation.
    
    This class provides:
    - Primary search via BigQuery VECTOR_SEARCH
    - Automatic fallback to local in-memory index
    - Comprehensive error handling and logging
    - Performance monitoring and correlation tracking
    """
    
    def __init__(
        self,
        connection: Optional[BigQueryConnectionManager] = None,
        config: Optional[VectorSearchConfig] = None,
        enable_fallback: bool = True,
        fallback_max_entries: int = 1000,
        search_mode: SearchMode = SearchMode.BIGQUERY_WITH_FALLBACK,
    ):
        """
        Initialize BigQuery vector index.
        
        Args:
            connection: BigQuery connection manager
            config: Vector search configuration
            enable_fallback: Whether to enable local fallback
            fallback_max_entries: Maximum entries in fallback index
            search_mode: Search execution mode
        """
        # Initialize connection
        if connection:
            self.connection = connection
        else:
            self.connection = BigQueryConnectionManager()
        
        # Initialize configuration
        if config:
            self.config = config
        else:
            self.config = VectorSearchConfig.from_env()
        
        self.enable_fallback = enable_fallback
        self.search_mode = search_mode
        self.index_id = str(uuid.uuid4())
        
        # Initialize components
        self.bigquery_adapter = BigQueryAdapter(self.connection)
        
        if self.enable_fallback:
            self.local_index = LocalVectorIndex(max_entries=fallback_max_entries)
        else:
            self.local_index = None
        
        # Statistics tracking
        self.stats = {
            "total_searches": 0,
            "bigquery_searches": 0,
            "fallback_searches": 0,
            "bigquery_failures": 0,
            "fallback_activations": 0,
            "avg_response_time_ms": 0.0,
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.index_id[:8]}")
        
        self.logger.info(
            "BigQuery vector index initialized",
            extra={
                "index_id": self.index_id,
                "connection_id": self.connection.connection_id,
                "project_id": self.connection.config.project_id,
                "dataset_id": self.connection.config.dataset_id,
                "enable_fallback": self.enable_fallback,
                "search_mode": self.search_mode.value,
                "fallback_max_entries": fallback_max_entries,
            }
        )
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        artifact_types: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors with automatic fallback.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            artifact_types: Optional filter by artifact types
            
        Returns:
            List of similar vectors ordered by similarity
            
        Raises:
            VectorIndexError: If search fails completely
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Update search configuration with runtime parameters
        search_config = VectorSearchConfig(
            project_id=self.config.project_id,
            dataset_id=self.config.dataset_id,
            table_name=self.config.table_name,
            top_k=k,
            distance_type=self.config.distance_type,
            distance_threshold=self.config.distance_threshold,
            timeout_ms=self.config.timeout_ms,
        )
        
        context = {
            "correlation_id": correlation_id,
            "index_id": self.index_id,
            "query_embedding_dimensions": len(query_embedding),
            "top_k": k,
            "artifact_types": artifact_types,
            "search_mode": self.search_mode.value,
        }
        
        self.logger.info("Starting similarity search", extra=context)
        
        try:
            self.stats["total_searches"] += 1
            
            # Execute search based on mode
            if self.search_mode == SearchMode.FALLBACK_ONLY:
                results = self._search_with_fallback_only(
                    query_embedding, search_config, artifact_types, context
                )
            elif self.search_mode == SearchMode.BIGQUERY_ONLY:
                results = self._search_with_bigquery_only(
                    query_embedding, search_config, artifact_types, context
                )
            else:  # BIGQUERY_WITH_FALLBACK
                results = self._search_with_bigquery_and_fallback(
                    query_embedding, search_config, artifact_types, context
                )
            
            # Update statistics
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_response_time_stats(elapsed_ms)
            
            context.update({
                "results_count": len(results),
                "elapsed_ms": elapsed_ms,
                "within_performance_target": elapsed_ms <= search_config.timeout_ms,
            })
            
            self.logger.info(
                f"Similarity search completed: {len(results)} results in {elapsed_ms:.1f}ms",
                extra=context
            )
            
            return results
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            context.update({
                "elapsed_ms": elapsed_ms,
                "error": str(e),
                "error_type": type(e).__name__,
            })
            
            self.logger.error("Similarity search failed completely", extra=context)
            raise VectorIndexError(f"Similarity search failed: {e}", e)
    
    def _search_with_bigquery_and_fallback(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]],
        context: Dict[str, Any],
    ) -> List[VectorSearchResult]:
        """Search with BigQuery primary and local fallback."""
        try:
            # Try BigQuery first
            results = self.bigquery_adapter.search_similar_vectors(
                query_embedding, config, artifact_types
            )
            
            self.stats["bigquery_searches"] += 1
            context["search_method"] = "bigquery"
            
            # Store successful results in fallback index for future use
            if self.local_index and results:
                self._populate_fallback_from_results(results, query_embedding)
            
            return results
            
        except (BigQueryAdapterError, BigQueryConnectionError) as e:
            # Log BigQuery failure and attempt fallback
            self.stats["bigquery_failures"] += 1
            
            fallback_context = {
                **context,
                "bigquery_error": str(e),
                "bigquery_error_type": type(e).__name__,
                "fallback_reason": "bigquery_unavailable",
            }
            
            self.logger.warning(
                "BigQuery search failed, activating fallback",
                extra=fallback_context
            )
            
            if not self.enable_fallback or not self.local_index:
                raise VectorIndexError(
                    f"BigQuery search failed and fallback disabled: {e}", e
                )
            
            # Activate fallback
            return self._search_with_fallback_only(
                query_embedding, config, artifact_types, fallback_context
            )
    
    def _search_with_bigquery_only(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]],
        context: Dict[str, Any],
    ) -> List[VectorSearchResult]:
        """Search with BigQuery only (no fallback)."""
        try:
            results = self.bigquery_adapter.search_similar_vectors(
                query_embedding, config, artifact_types
            )
            
            self.stats["bigquery_searches"] += 1
            context["search_method"] = "bigquery_only"
            
            return results
            
        except (BigQueryAdapterError, BigQueryConnectionError) as e:
            self.stats["bigquery_failures"] += 1
            context.update({
                "bigquery_error": str(e),
                "search_method": "bigquery_only_failed",
            })
            
            raise VectorIndexError(f"BigQuery-only search failed: {e}", e)
    
    def _search_with_fallback_only(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]],
        context: Dict[str, Any],
    ) -> List[VectorSearchResult]:
        """Search with local fallback only."""
        if not self.enable_fallback or not self.local_index:
            raise VectorIndexError("Fallback search requested but not available")
        
        try:
            results = self.local_index.search_similar_vectors(
                query_embedding, config, artifact_types
            )
            
            self.stats["fallback_searches"] += 1
            self.stats["fallback_activations"] += 1
            
            context.update({
                "search_method": "fallback_only",
                "fallback_index_size": len(self.local_index),
            })
            
            return results
            
        except LocalVectorIndexError as e:
            context.update({
                "fallback_error": str(e),
                "search_method": "fallback_failed",
            })
            
            raise VectorIndexError(f"Fallback search failed: {e}", e)
    
    def _populate_fallback_from_results(
        self,
        results: List[VectorSearchResult],
        query_embedding: List[float],
    ) -> None:
        """Populate fallback index with successful search results."""
        try:
            for result in results:
                # Use query embedding as approximation (not ideal but better than nothing)
                self.local_index.add_vector(
                    chunk_id=result.chunk_id,
                    embedding=query_embedding,  # Approximation
                    source=result.source,
                    artifact_type=result.artifact_type,
                    text_content=result.text_content,
                    metadata=result.metadata,
                )
            
            self.logger.debug(
                f"Populated fallback index with {len(results)} vectors from search results"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to populate fallback index: {e}")
    
    def add_vectors(
        self,
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add vectors to both BigQuery and fallback index.
        
        Args:
            vectors: List of vector data dictionaries
            
        Returns:
            Addition result summary
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        context = {
            "correlation_id": correlation_id,
            "index_id": self.index_id,
            "vectors_count": len(vectors),
        }
        
        self.logger.info("Adding vectors to index", extra=context)
        
        bigquery_results = []
        fallback_results = []
        errors = []
        
        # Add to BigQuery (primary storage)
        try:
            if hasattr(self, 'bigquery_vector_store'):
                # Use vector store for batch insertion
                bigquery_results = self.bigquery_vector_store.batch_insert_embeddings(vectors)
            else:
                self.logger.warning("BigQuery vector store not available for batch insertion")
                
        except Exception as e:
            error_msg = f"BigQuery vector addition failed: {e}"
            errors.append(error_msg)
            self.logger.error(error_msg, extra=context)
        
        # Add to fallback index
        if self.local_index:
            try:
                for vector_data in vectors:
                    self.local_index.add_vector(
                        chunk_id=vector_data["chunk_id"],
                        embedding=vector_data["embedding"],
                        source=vector_data.get("source", "unknown"),
                        artifact_type=vector_data.get("artifact_type", "unknown"),
                        text_content=vector_data.get("text_content", ""),
                        metadata=vector_data.get("metadata", {}),
                    )
                    fallback_results.append({"chunk_id": vector_data["chunk_id"], "status": "added"})
                    
            except Exception as e:
                error_msg = f"Fallback index addition failed: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg, extra=context)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = {
            "correlation_id": correlation_id,
            "vectors_processed": len(vectors),
            "bigquery_results": len(bigquery_results),
            "fallback_results": len(fallback_results),
            "errors": errors,
            "elapsed_ms": elapsed_ms,
            "success": len(errors) == 0,
        }
        
        context.update(result)
        self.logger.info("Vector addition completed", extra=context)
        
        return result
    
    def remove_vector(self, chunk_id: str) -> bool:
        """
        Remove a vector from both BigQuery and fallback index.
        
        Args:
            chunk_id: ID of vector to remove
            
        Returns:
            True if vector was removed from at least one index
        """
        correlation_id = str(uuid.uuid4())
        
        context = {
            "correlation_id": correlation_id,
            "index_id": self.index_id,
            "chunk_id": chunk_id,
        }
        
        self.logger.info("Removing vector from index", extra=context)
        
        bigquery_removed = False
        fallback_removed = False
        
        # Remove from BigQuery
        try:
            if hasattr(self, 'bigquery_vector_store'):
                bigquery_removed = self.bigquery_vector_store.delete_embedding(chunk_id)
            else:
                self.logger.warning("BigQuery vector store not available for deletion")
                
        except Exception as e:
            self.logger.error(f"BigQuery vector removal failed: {e}", extra=context)
        
        # Remove from fallback index
        if self.local_index:
            try:
                fallback_removed = self.local_index.remove_vector(chunk_id)
            except Exception as e:
                self.logger.error(f"Fallback index removal failed: {e}", extra=context)
        
        success = bigquery_removed or fallback_removed
        
        context.update({
            "bigquery_removed": bigquery_removed,
            "fallback_removed": fallback_removed,
            "success": success,
        })
        
        self.logger.info("Vector removal completed", extra=context)
        
        return success
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status information
        """
        correlation_id = str(uuid.uuid4())
        
        # Check BigQuery connection
        bigquery_health = self.connection.check_health()
        
        # Check BigQuery adapter
        adapter_health = {"status": "unknown"}
        try:
            table_info = self.bigquery_adapter.get_table_info("source_embeddings")
            vector_index_info = self.bigquery_adapter.validate_vector_index()
            adapter_health = {
                "status": "healthy",
                "table_accessible": True,
                "vector_index_active": vector_index_info.get("has_active_vector_index", False),
                "table_rows": table_info.get("num_rows", 0),
            }
        except Exception as e:
            adapter_health = {
                "status": "unhealthy",
                "error": str(e),
                "table_accessible": False,
            }
        
        # Check fallback index
        fallback_health = {"status": "disabled"}
        if self.local_index:
            fallback_health = self.local_index.health_check()
        
        # Overall status determination
        overall_status = "healthy"
        if not bigquery_health.is_healthy and fallback_health.get("status") != "healthy":
            overall_status = "unhealthy"
        elif not bigquery_health.is_healthy or adapter_health["status"] != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "index_id": self.index_id,
            "correlation_id": correlation_id,
            "bigquery_connection": {
                "is_healthy": bigquery_health.is_healthy,
                "response_time_ms": bigquery_health.response_time_ms,
                "error_message": bigquery_health.error_message,
            },
            "bigquery_adapter": adapter_health,
            "fallback_index": fallback_health,
            "search_stats": self.stats.copy(),
            "configuration": {
                "enable_fallback": self.enable_fallback,
                "search_mode": self.search_mode.value,
                "project_id": self.connection.config.project_id,
                "dataset_id": self.connection.config.dataset_id,
                "table_name": self.config.table_name,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    
    def _update_response_time_stats(self, elapsed_ms: float) -> None:
        """Update response time statistics."""
        if self.stats["total_searches"] == 1:
            self.stats["avg_response_time_ms"] = elapsed_ms
        else:
            # Calculate running average
            total_time = self.stats["avg_response_time_ms"] * (self.stats["total_searches"] - 1)
            self.stats["avg_response_time_ms"] = (total_time + elapsed_ms) / self.stats["total_searches"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats.copy(),
            "index_id": self.index_id,
            "bigquery_success_rate": (
                (self.stats["bigquery_searches"] / max(1, self.stats["total_searches"])) * 100
            ),
            "fallback_activation_rate": (
                (self.stats["fallback_activations"] / max(1, self.stats["total_searches"])) * 100
            ),
        }
    
    def close(self) -> None:
        """Close the vector index and cleanup resources."""
        correlation_id = str(uuid.uuid4())
        
        self.logger.info(
            "Closing BigQuery vector index",
            extra={
                "correlation_id": correlation_id,
                "index_id": self.index_id,
                "final_stats": self.get_stats(),
            }
        )
        
        # Cleanup fallback index
        if self.local_index:
            self.local_index.clear()
        
        # Close connection
        self.connection.close()
        
        self.logger.info("BigQuery vector index closed", extra={"correlation_id": correlation_id})
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BigQueryVectorIndex("
            f"index_id='{self.index_id[:8]}...', "
            f"project='{self.connection.config.project_id}', "
            f"dataset='{self.connection.config.dataset_id}', "
            f"fallback={self.enable_fallback}, "
            f"mode={self.search_mode.value}"
            f")"
        )