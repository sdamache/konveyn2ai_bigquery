"""
Local Vector Index Fallback Implementation.

This module provides a local in-memory vector index for fallback when BigQuery
is unavailable. It implements approximate similarity search using cosine similarity
and maintains result ordering contract.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np
from dataclasses import dataclass, field

from ..models.vector_search_config import VectorSearchConfig
from ..models.vector_search_result import VectorSearchResult

logger = logging.getLogger(__name__)


@dataclass
class LocalVectorEntry:
    """Local vector storage entry."""
    
    chunk_id: str
    embedding: List[float]
    source: str
    artifact_type: str
    text_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class LocalVectorIndexError(Exception):
    """Custom exception for local vector index errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class LocalVectorIndex:
    """
    Local in-memory vector index for fallback functionality.
    
    This class provides:
    - In-memory vector storage and retrieval
    - Cosine similarity search with approximate results
    - Fallback compatibility with BigQuery interface
    - Result ordering and filtering
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize local vector index.
        
        Args:
            max_entries: Maximum number of vectors to store in memory
        """
        self.max_entries = max_entries
        self.vectors: Dict[str, LocalVectorEntry] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        self._dirty = True  # Flag to rebuild matrix when needed
        
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        
        self.logger.info(
            f"Local vector index initialized with max_entries={max_entries}"
        )
    
    def add_vector(
        self,
        chunk_id: str,
        embedding: List[float],
        source: str,
        artifact_type: str,
        text_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a vector to the local index.
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding: Vector embedding
            source: Source file/location
            artifact_type: Type of artifact
            text_content: Text content of the chunk
            metadata: Additional metadata
        """
        if len(self.vectors) >= self.max_entries and chunk_id not in self.vectors:
            # Simple LRU-like eviction - remove oldest entry
            oldest_id = next(iter(self.vectors))
            self.remove_vector(oldest_id)
            self.logger.debug(f"Evicted vector {oldest_id} to make space")
        
        entry = LocalVectorEntry(
            chunk_id=chunk_id,
            embedding=embedding,
            source=source,
            artifact_type=artifact_type,
            text_content=text_content,
            metadata=metadata or {},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        
        self.vectors[chunk_id] = entry
        self._dirty = True
        
        self.logger.debug(f"Added vector {chunk_id} to local index")
    
    def remove_vector(self, chunk_id: str) -> bool:
        """
        Remove a vector from the local index.
        
        Args:
            chunk_id: ID of vector to remove
            
        Returns:
            True if vector was removed, False if not found
        """
        if chunk_id in self.vectors:
            del self.vectors[chunk_id]
            self._dirty = True
            self.logger.debug(f"Removed vector {chunk_id} from local index")
            return True
        return False
    
    def search_similar_vectors(
        self,
        query_embedding: List[float],
        config: VectorSearchConfig,
        artifact_types: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_embedding: Query vector
            config: Search configuration
            artifact_types: Optional filter by artifact types
            
        Returns:
            List of similar vectors ordered by similarity
            
        Raises:
            LocalVectorIndexError: If search fails
        """
        start_time = time.time()
        
        try:
            # Filter vectors by artifact type if specified
            candidate_vectors = {}
            for chunk_id, entry in self.vectors.items():
                if artifact_types and entry.artifact_type not in artifact_types:
                    continue
                candidate_vectors[chunk_id] = entry
            
            if not candidate_vectors:
                self.logger.info("No candidate vectors found for search")
                return []
            
            # Rebuild embeddings matrix if needed
            self._ensure_embeddings_matrix(candidate_vectors)
            
            # Compute similarities
            query_array = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query_array)
            
            if query_norm == 0:
                raise LocalVectorIndexError("Query embedding has zero norm")
            
            # Normalize query vector
            query_normalized = query_array / query_norm
            
            # Compute cosine similarities
            similarities = np.dot(self.candidate_embeddings, query_normalized)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:config.top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                chunk_id = self.candidate_chunk_ids[idx]
                entry = candidate_vectors[chunk_id]
                similarity = float(similarities[idx])
                
                # Apply distance threshold if specified
                if config.distance_threshold is not None:
                    if config.distance_type.value == "COSINE":
                        # For cosine, we use similarity score directly
                        if similarity < config.distance_threshold:
                            continue
                    else:
                        # For other distance types, convert similarity to distance
                        distance = 1.0 - similarity
                        if distance > config.distance_threshold:
                            continue
                
                result = VectorSearchResult(
                    chunk_id=chunk_id,
                    distance=1.0 - similarity,  # Convert similarity to distance
                    similarity_score=similarity,
                    source="local",  # This indicates the search source (local vs bigquery)
                    artifact_type=entry.artifact_type,
                    text_content=entry.text_content,
                    metadata={
                        **entry.metadata,
                        "fallback_search": True,
                        "local_index_size": len(candidate_vectors),
                        "file_source": entry.source,  # Store file source in metadata
                    }
                )
                
                results.append(result)
            
            search_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Local vector search completed: {len(results)} results in {search_time_ms:.1f}ms "
                f"(searched {len(candidate_vectors)} vectors)"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Local vector search failed: {e}"
            self.logger.error(error_msg)
            raise LocalVectorIndexError(error_msg, e)
    
    def _ensure_embeddings_matrix(self, candidate_vectors: Dict[str, LocalVectorEntry]) -> None:
        """
        Ensure embeddings matrix is built for the candidate vectors.
        
        Args:
            candidate_vectors: Vectors to include in the matrix
        """
        if not self._dirty and len(candidate_vectors) == len(self.chunk_ids):
            return  # Matrix is up to date
        
        self.candidate_chunk_ids = list(candidate_vectors.keys())
        embeddings_list = [candidate_vectors[chunk_id].embedding for chunk_id in self.candidate_chunk_ids]
        
        # Convert to numpy array and normalize
        self.candidate_embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.candidate_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.candidate_embeddings = self.candidate_embeddings / norms
        
        self._dirty = False
        
        self.logger.debug(
            f"Built embeddings matrix: {self.candidate_embeddings.shape} "
            f"for {len(candidate_vectors)} vectors"
        )
    
    def get_vector(self, chunk_id: str) -> Optional[LocalVectorEntry]:
        """
        Get a specific vector by chunk ID.
        
        Args:
            chunk_id: ID of vector to retrieve
            
        Returns:
            Vector entry if found, None otherwise
        """
        return self.vectors.get(chunk_id)
    
    def list_vectors(
        self, 
        artifact_types: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LocalVectorEntry]:
        """
        List vectors with optional filtering and pagination.
        
        Args:
            artifact_types: Optional filter by artifact types
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of vector entries
        """
        filtered_vectors = []
        
        for entry in self.vectors.values():
            if artifact_types and entry.artifact_type not in artifact_types:
                continue
            filtered_vectors.append(entry)
        
        # Sort by created_at (most recent first)
        filtered_vectors.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        
        return filtered_vectors[start_idx:end_idx]
    
    def count_vectors(self, artifact_types: Optional[List[str]] = None) -> int:
        """
        Count vectors with optional filtering.
        
        Args:
            artifact_types: Optional filter by artifact types
            
        Returns:
            Number of matching vectors
        """
        if not artifact_types:
            return len(self.vectors)
        
        count = 0
        for entry in self.vectors.values():
            if entry.artifact_type in artifact_types:
                count += 1
        
        return count
    
    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.vectors.clear()
        self.embeddings_matrix = None
        self.chunk_ids.clear()
        self._dirty = True
        
        self.logger.info("Local vector index cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the local vector index.
        
        Returns:
            Health status information
        """
        try:
            # Basic statistics
            vector_count = len(self.vectors)
            artifact_types = list(set(entry.artifact_type for entry in self.vectors.values()))
            
            # Memory usage estimation (rough)
            if self.vectors:
                sample_entry = next(iter(self.vectors.values()))
                estimated_bytes_per_vector = (
                    len(sample_entry.embedding) * 4 +  # float32 embedding
                    len(sample_entry.text_content.encode('utf-8')) +  # text content
                    100  # metadata and other fields
                )
                estimated_memory_mb = (vector_count * estimated_bytes_per_vector) / (1024 * 1024)
            else:
                estimated_memory_mb = 0
            
            return {
                "status": "healthy",
                "vector_count": vector_count,
                "max_entries": self.max_entries,
                "utilization_percent": (vector_count / self.max_entries) * 100,
                "artifact_types": artifact_types,
                "estimated_memory_mb": round(estimated_memory_mb, 2),
                "matrix_built": not self._dirty,
                "search_capability": vector_count > 0,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "vector_count": len(self.vectors) if hasattr(self, 'vectors') else 0,
            }
    
    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return len(self.vectors)
    
    def __contains__(self, chunk_id: str) -> bool:
        """Check if a chunk ID exists in the index."""
        return chunk_id in self.vectors
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LocalVectorIndex("
            f"size={len(self.vectors)}, "
            f"max_entries={self.max_entries}, "
            f"dirty={self._dirty}"
            f")"
        )