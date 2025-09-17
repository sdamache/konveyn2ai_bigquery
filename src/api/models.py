"""
API Models - Pydantic models for request/response validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# Request Models
class EmbeddingInsertRequest(BaseModel):
    """Request model for embedding insertion."""

    chunk_id: Optional[str] = Field(
        None, description="Unique chunk identifier"
    )
    source: Optional[str] = Field(
        None, description="Source file or location"
    )
    artifact_type: Optional[str] = Field(
        None, description="Type of artifact (code, documentation, etc.)"
    )
    text_content: Optional[str] = Field(
        None, description="Text content of the chunk"
    )
    kind: Optional[str] = Field(
        None, description="Kind of artifact (function, class, etc.)"
    )
    api_path: Optional[str] = Field(None, description="API path if applicable")
    record_name: Optional[str] = Field(None, description="Record name")
    embedding: Optional[list[float]] = Field(
        None, description="Vector embedding"
    )
    metadata: Optional[Any] = Field(
        None, description="Additional metadata"
    )


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""

    query_embedding: Optional[list[float]] = Field(
        None, description="Query vector"
    )
    query_text: Optional[str] = Field(
        None, description="Text query for similarity search"
    )
    limit: Optional[int] = Field(
        None, description="Maximum number of results"
    )
    similarity_threshold: Optional[float] = Field(
        None, description="Minimum similarity score"
    )
    artifact_types: Optional[list[str]] = Field(
        None, description="Filter by artifact types"
    )
    sources: Optional[list[str]] = Field(
        None, description="Filter by source identifiers"
    )


class TextSearchRequest(BaseModel):
    """Request model for text-based similarity search."""

    query_text: Optional[str] = Field(None, description="Query text")
    limit: Optional[int] = Field(
        None, description="Maximum number of results"
    )
    similarity_threshold: Optional[float] = Field(
        None, description="Minimum similarity score"
    )
    artifact_types: Optional[list[str]] = Field(
        None, description="Filter by artifact types"
    )
    sources: Optional[list[str]] = Field(None, description="Filter by sources")


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding operations."""

    embeddings: Optional[list[EmbeddingInsertRequest]] = Field(
        None, description="List of embeddings to insert"
    )


# Response Models
class EmbeddingResponse(BaseModel):
    """Response model for embedding data."""

    chunk_id: str
    source: str
    artifact_type: str
    text_content: str
    kind: Optional[str] = None
    api_path: Optional[str] = None
    record_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str
    metadata_created_at: str
    embedding_created_at: str
    embedding: Optional[list[float]] = None


class EmbeddingInsertResponse(BaseModel):
    """Response model for embedding insertion."""

    chunk_id: str
    status: str
    embedding_dimensions: int
    created_at: str
    timestamp: Optional[str] = None


class EmbeddingListResponse(BaseModel):
    """Response model for embedding list."""

    embeddings: list[EmbeddingResponse]
    total_count: int
    limit: int
    offset: int
    next_offset: Optional[int] = None


class SimilarityResult(BaseModel):
    """Response model for similarity search result."""

    chunk_id: str
    similarity_score: float
    source: str
    artifact_type: str
    text_content: str
    kind: Optional[str] = None
    api_path: Optional[str] = None
    record_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class VectorSearchResponse(BaseModel):
    """Response model for vector search."""

    results: list[SimilarityResult]
    query_embedding: list[float]
    total_results: int
    search_time_ms: int


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding operations."""

    results: list[EmbeddingInsertResponse]
    summary: dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str
    bigquery_connection: bool
    tables_accessible: bool
    vector_index_status: str
    embedding_count: int
    timestamp: str


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    message: str
    details: Optional[dict[str, Any]] = None
    timestamp: str
