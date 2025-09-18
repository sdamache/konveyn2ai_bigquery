"""
BigQuery Adapter for Vector Search Operations.

This module provides the BigQueryAdapter class which constructs parameterized
VECTOR_SEARCH queries, handles vector encoding for BigQuery, and parses result
rows into VectorSearchResult objects.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Forbidden, BadRequest

from ..connections.bigquery_connection import BigQueryConnectionManager
from ..models.vector_search_config import VectorSearchConfig
from ..models.vector_search_result import VectorSearchResult

logger = logging.getLogger(__name__)


class BigQueryAdapterError(Exception):
    """Custom exception for BigQuery adapter errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class BigQueryAdapter:
    """
    BigQuery adapter for vector search operations.
    
    This class handles:
    - Construction of parameterized VECTOR_SEARCH queries
    - Vector encoding for BigQuery compatibility
    - Result parsing into VectorSearchResult objects
    - Error handling and query optimization
    """
    
    def __init__(self, connection: BigQueryConnectionManager):
        """
        Initialize BigQuery adapter.
        
        Args:
            connection: BigQuery connection manager
        """
        self.connection = connection
        self.logger = logging.getLogger(f"{__name__}.{connection.connection_id[:8]}")
        
        self.logger.info(
            "BigQuery adapter initialized",
            extra={
                "connection_id": connection.connection_id,
                "project_id": connection.config.project_id,
                "dataset_id": connection.config.dataset_id,
            }
        )
    
    def search_similar_vectors(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors using BigQuery VECTOR_SEARCH.
        
        Args:
            query_embedding: Query vector embedding
            config: Vector search configuration
            artifact_types: Optional filter by artifact types
            
        Returns:
            List of similar vectors with similarity scores
            
        Raises:
            BigQueryAdapterError: If search fails
        """
        with self.connection.query_context("vector_similarity_search") as context:
            try:
                # Validate query embedding
                self._validate_query_embedding(query_embedding)
                
                # Build and execute query
                query, query_params = self._build_vector_search_query(
                    query_embedding, config, artifact_types
                )
                
                context.update({
                    "query_embedding_dimensions": len(query_embedding),
                    "top_k": config.top_k,
                    "distance_type": config.distance_type.value,
                    "artifact_types": artifact_types,
                })
                
                # Execute query with timeout
                job_config = bigquery.QueryJobConfig(
                    query_parameters=query_params,
                    use_query_cache=True,
                    job_timeout_ms=config.timeout_ms,  # BigQuery expects timeout in milliseconds
                )
                
                start_time = time.time()
                results = list(self.connection.execute_query(query, job_config=job_config))
                query_time_ms = (time.time() - start_time) * 1000
                
                # Parse results
                vector_results = self._parse_search_results(results)
                
                context.update({
                    "query_time_ms": query_time_ms,
                    "results_count": len(vector_results),
                    "within_performance_target": query_time_ms <= config.timeout_ms,
                })
                
                self.logger.info(
                    f"Vector search completed: {len(vector_results)} results in {query_time_ms:.1f}ms",
                    extra=context
                )
                
                return vector_results
                
            except (NotFound, Forbidden, BadRequest) as e:
                context["bigquery_error"] = str(e)
                error_msg = f"BigQuery vector search failed: {e}"
                self.logger.error(error_msg, extra=context)
                raise BigQueryAdapterError(error_msg, e)
                
            except Exception as e:
                context["unexpected_error"] = str(e)
                error_msg = f"Unexpected error in vector search: {e}"
                self.logger.error(error_msg, extra=context)
                raise BigQueryAdapterError(error_msg, e)
    
    def _validate_query_embedding(self, query_embedding: List[float]) -> None:
        """
        Validate query embedding format and dimensions.
        
        Args:
            query_embedding: Query vector to validate
            
        Raises:
            BigQueryAdapterError: If embedding is invalid
        """
        if not query_embedding:
            raise BigQueryAdapterError("Query embedding cannot be empty")
            
        if not isinstance(query_embedding, list):
            raise BigQueryAdapterError(f"Query embedding must be a list, got {type(query_embedding)}")
            
        if not all(isinstance(x, (int, float)) for x in query_embedding):
            raise BigQueryAdapterError("Query embedding must contain only numbers")
            
        # Check for common dimension sizes
        valid_dimensions = {768, 1536, 3072}  # Common embedding model dimensions
        if len(query_embedding) not in valid_dimensions:
            self.logger.warning(
                f"Query embedding has {len(query_embedding)} dimensions, "
                f"expected one of {valid_dimensions}"
            )
    
    def _build_vector_search_query(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]] = None,
    ) -> tuple[str, List[bigquery.ScalarQueryParameter]]:
        """
        Build parameterized VECTOR_SEARCH query.
        
        Args:
            query_embedding: Query vector
            config: Search configuration
            artifact_types: Optional artifact type filter
            
        Returns:
            Tuple of (query_string, query_parameters)
        """
        # Build artifact type filter
        artifact_filter = ""
        query_params = [
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("top_k", "INT64", config.top_k),
        ]
        
        if artifact_types:
            placeholders = ", ".join([f"@artifact_type_{i}" for i in range(len(artifact_types))])
            artifact_filter = f"AND m.artifact_type IN ({placeholders})"
            
            for i, artifact_type in enumerate(artifact_types):
                query_params.append(
                    bigquery.ScalarQueryParameter(f"artifact_type_{i}", "STRING", artifact_type)
                )
        
        # Build distance threshold filter
        threshold_filter = ""
        if config.distance_threshold is not None:
            if config.distance_type.value == "COSINE":
                # For cosine distance, threshold is on similarity score (1 - distance)
                threshold_filter = "AND (1 - vs.distance) >= @similarity_threshold"
                query_params.append(
                    bigquery.ScalarQueryParameter(
                        "similarity_threshold", "FLOAT64", config.distance_threshold
                    )
                )
            else:
                # For other distance types, threshold is on raw distance
                threshold_filter = "AND vs.distance <= @distance_threshold"
                query_params.append(
                    bigquery.ScalarQueryParameter(
                        "distance_threshold", "FLOAT64", config.distance_threshold
                    )
                )
        
        # Construct the main query
        query = f"""
        WITH vector_search AS (
            SELECT
                base.chunk_id,
                distance
            FROM VECTOR_SEARCH(
                TABLE `{config.full_table_reference}`,
                'embedding',
                (SELECT @query_embedding as embedding),
                top_k => @top_k,
                distance_type => '{config.distance_type.value}'
            ) AS base
            {threshold_filter.replace('vs.', 'base.')}
        )
        SELECT
            vs.chunk_id,
            vs.distance,
            CASE 
                WHEN '{config.distance_type.value}' = 'COSINE' THEN (1 - vs.distance)
                ELSE vs.distance
            END as similarity_score,
            m.source,
            m.artifact_type,
            m.text_content,
            m.kind,
            m.api_path,
            m.record_name,
            m.metadata,
            m.created_at
        FROM vector_search vs
        JOIN `{self.connection.config.project_id}.{self.connection.config.dataset_id}.source_metadata` m 
        ON vs.chunk_id = m.chunk_id
        {artifact_filter}
        ORDER BY vs.distance ASC
        """
        
        return query.strip(), query_params
    
    def _parse_search_results(self, results: List[Any]) -> List[VectorSearchResult]:
        """
        Parse BigQuery results into VectorSearchResult objects.
        
        Args:
            results: Raw BigQuery query results
            
        Returns:
            List of parsed VectorSearchResult objects
        """
        vector_results = []
        
        for row in results:
            try:
                # Parse metadata JSON if present
                metadata = {}
                if row.metadata:
                    try:
                        metadata = json.loads(row.metadata)
                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse metadata for chunk {row.chunk_id}: {e}"
                        )
                        metadata = {"parsing_error": str(e)}
                
                # Create VectorSearchResult
                result = VectorSearchResult(
                    chunk_id=row.chunk_id,
                    distance=float(row.distance),
                    similarity_score=float(row.similarity_score),
                    source=row.source,
                    artifact_type=row.artifact_type,
                    text_content=row.text_content,
                    metadata={
                        "kind": row.kind,
                        "api_path": row.api_path,
                        "record_name": row.record_name,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        **metadata,
                    }
                )
                
                vector_results.append(result)
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to parse result row for chunk {getattr(row, 'chunk_id', 'unknown')}: {e}"
                )
                continue
        
        return vector_results
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a BigQuery table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
            
        Raises:
            BigQueryAdapterError: If table access fails
        """
        try:
            table_ref = f"{self.connection.config.project_id}.{self.connection.config.dataset_id}.{table_name}"
            table = self.connection.client.get_table(table_ref)
            
            return {
                "table_id": table.table_id,
                "full_table_id": table.full_table_id,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "schema_fields": len(table.schema),
                "location": table.location,
            }
            
        except NotFound:
            raise BigQueryAdapterError(f"Table {table_name} not found")
        except Exception as e:
            raise BigQueryAdapterError(f"Failed to get table info: {e}", e)
    
    def validate_vector_index(self, table_name: str = "source_embeddings") -> Dict[str, Any]:
        """
        Validate that vector index exists and is properly configured.
        
        Args:
            table_name: Name of the embeddings table
            
        Returns:
            Dictionary with index validation results
        """
        try:
            # Check if table has vector column
            table_info = self.get_table_info(table_name)
            
            # Query for vector index information
            index_query = f"""
            SELECT 
                index_name,
                index_status,
                covering_columns,
                index_usage_bytes
            FROM `{self.connection.config.project_id}.{self.connection.config.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
            WHERE table_name = @table_name AND index_status = 'ACTIVE'
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("table_name", "STRING", table_name)
                ]
            )
            
            index_results = list(self.connection.execute_query(index_query, job_config=job_config))
            
            return {
                "table_exists": True,
                "table_info": table_info,
                "vector_indexes": [
                    {
                        "index_name": row.index_name,
                        "status": row.index_status,
                        "covering_columns": row.covering_columns,
                        "usage_bytes": row.index_usage_bytes,
                    }
                    for row in index_results
                ],
                "has_active_vector_index": len(index_results) > 0,
            }
            
        except BigQueryAdapterError:
            return {
                "table_exists": False,
                "error": f"Table {table_name} not accessible",
                "has_active_vector_index": False,
            }
        except Exception as e:
            self.logger.warning(f"Vector index validation failed: {e}")
            return {
                "table_exists": True,
                "error": f"Index validation failed: {e}",
                "has_active_vector_index": False,
            }