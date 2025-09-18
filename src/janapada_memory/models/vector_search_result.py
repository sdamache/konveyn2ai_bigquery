"""
Vector search result data model for BigQuery Memory Adapter.

This module provides the VectorSearchResult dataclass which represents
individual search results from vector similarity searches, supporting
both BigQuery and local fallback sources.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class VectorSearchResult:
    """
    Represents a single result from vector similarity search.

    This dataclass encapsulates all information about a search result,
    including the chunk identifier, similarity distance, metadata,
    and the source of the result (bigquery or local).

    Attributes:
        chunk_id: Unique identifier for the text chunk
        distance: Similarity distance (lower is more similar)
        metadata: Additional metadata about the chunk
        source: Origin of the result ("bigquery" or "local")
        content: Optional actual text content of the chunk
        embedding_vector: Optional embedding vector (for debugging)
        correlation_id: Optional correlation ID for tracking
        fallback_reason: Optional reason if this is a fallback result
    """

    chunk_id: str
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "bigquery"
    content: Optional[str] = None
    embedding_vector: Optional[list] = None
    correlation_id: Optional[str] = None
    fallback_reason: Optional[str] = None

    def __post_init__(self):
        """Validate result data after initialization."""
        self._validate_chunk_id()
        self._validate_distance()
        self._validate_source()
        self._validate_metadata()

    def _validate_chunk_id(self) -> None:
        """Validate chunk_id is not empty."""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.chunk_id, str):
            raise TypeError(
                f"chunk_id must be string, got {type(self.chunk_id).__name__}"
            )

    def _validate_distance(self) -> None:
        """Validate distance is a valid number."""
        if not isinstance(self.distance, (int, float)):
            raise TypeError(
                f"distance must be numeric, got {type(self.distance).__name__}"
            )

        # Distance should be non-negative for most metrics
        # (though DOT_PRODUCT can be negative)
        if self.distance != self.distance:  # Check for NaN
            raise ValueError("distance cannot be NaN")

    def _validate_source(self) -> None:
        """Validate source is either 'bigquery' or 'local'."""
        valid_sources = {"bigquery", "local"}
        if self.source not in valid_sources:
            raise ValueError(
                f"source must be one of {valid_sources}, got {self.source}"
            )

    def _validate_metadata(self) -> None:
        """Validate metadata is a dictionary."""
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be dict, got {type(self.metadata).__name__}"
            )

    def __lt__(self, other: "VectorSearchResult") -> bool:
        """
        Implement comparison for distance ordering.

        Results are ordered by ascending distance (smaller distance = more similar).
        This allows easy sorting of results.
        """
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return self.distance < other.distance

    def __le__(self, other: "VectorSearchResult") -> bool:
        """Less than or equal comparison based on distance."""
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return self.distance <= other.distance

    def __gt__(self, other: "VectorSearchResult") -> bool:
        """Greater than comparison based on distance."""
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return self.distance > other.distance

    def __ge__(self, other: "VectorSearchResult") -> bool:
        """Greater than or equal comparison based on distance."""
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return self.distance >= other.distance

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on chunk_id and source.

        Two results are considered equal if they reference the same chunk
        from the same source, regardless of distance.
        """
        if not isinstance(other, VectorSearchResult):
            return NotImplemented
        return self.chunk_id == other.chunk_id and self.source == other.source

    def __hash__(self) -> int:
        """Hash based on chunk_id and source for use in sets/dicts."""
        return hash((self.chunk_id, self.source))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns:
            Dictionary representation of the result
        """
        result = {
            "chunk_id": self.chunk_id,
            "distance": self.distance,
            "metadata": self.metadata,
            "source": self.source,
        }

        # Include optional fields only if they have values
        if self.content is not None:
            result["content"] = self.content
        if self.correlation_id is not None:
            result["correlation_id"] = self.correlation_id
        if self.fallback_reason is not None:
            result["fallback_reason"] = self.fallback_reason
        if self.embedding_vector is not None:
            # Don't include full vector in dict by default (too large)
            result["has_embedding"] = True

        return result

    def to_json(self) -> str:
        """
        Convert result to JSON string.

        Returns:
            JSON string representation of the result
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_bigquery_row(cls, row: Dict[str, Any]) -> "VectorSearchResult":
        """
        Create VectorSearchResult from BigQuery query row.

        Args:
            row: Dictionary containing BigQuery result row

        Returns:
            VectorSearchResult instance
        """
        # Extract standard fields
        chunk_id = row.get("chunk_id", row.get("id", ""))
        distance = float(row.get("distance", 0.0))

        # Extract metadata fields
        metadata = {}
        for key, value in row.items():
            if key not in {"chunk_id", "id", "distance", "embedding_vector"}:
                metadata[key] = value

        # Get content if available
        content = row.get("content", row.get("text", None))

        return cls(
            chunk_id=chunk_id,
            distance=distance,
            metadata=metadata,
            source="bigquery",
            content=content,
        )

    @classmethod
    def from_local_result(
        cls,
        chunk_id: str,
        distance: float,
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
        fallback_reason: Optional[str] = None,
    ) -> "VectorSearchResult":
        """
        Create VectorSearchResult from local vector search.

        Args:
            chunk_id: Chunk identifier
            distance: Similarity distance
            metadata: Optional metadata dictionary
            content: Optional chunk content
            fallback_reason: Optional reason for fallback

        Returns:
            VectorSearchResult instance marked as local source
        """
        return cls(
            chunk_id=chunk_id,
            distance=distance,
            metadata=metadata or {},
            source="local",
            content=content,
            fallback_reason=fallback_reason,
        )

    def with_correlation_id(self, correlation_id: str) -> "VectorSearchResult":
        """
        Create a new result with correlation ID added.

        Args:
            correlation_id: Correlation ID for tracking

        Returns:
            New VectorSearchResult with correlation ID
        """
        return VectorSearchResult(
            chunk_id=self.chunk_id,
            distance=self.distance,
            metadata=self.metadata,
            source=self.source,
            content=self.content,
            embedding_vector=self.embedding_vector,
            correlation_id=correlation_id,
            fallback_reason=self.fallback_reason,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"VectorSearchResult("
            f"chunk_id='{self.chunk_id}', "
            f"distance={self.distance:.4f}, "
            f"source='{self.source}', "
            f"metadata_keys={list(self.metadata.keys())}"
            f")"
        )
