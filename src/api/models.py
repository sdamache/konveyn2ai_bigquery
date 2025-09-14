"""
API Models - Pydantic models for request/response validation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


# Request Models
class EmbeddingInsertRequest(BaseModel):
    """Request model for embedding insertion."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Source file or location")
    artifact_type: str = Field(..., description="Type of artifact (code, documentation, etc.)")
    text_content: str = Field(..., description="Text content of the chunk")
    kind: Optional[str] = Field(None, description="Kind of artifact (function, class, etc.)")
    api_path: Optional[str] = Field(None, description="API path if applicable")
    record_name: Optional[str] = Field(None, description="Record name")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('embedding')
    def validate_embedding(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numbers")
        return v

class EmbeddingListRequest(BaseModel):
    """Request model for listing embeddings."""
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    artifact_types: Optional[List[str]] = Field(None, description="Filter by artifact types")
    include_embeddings: bool = Field(False, description="Whether to include embedding vectors")

class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""
    query_embedding: List[float] = Field(..., description="Query vector")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    artifact_types: Optional[List[str]] = Field(None, description="Filter by artifact types")
    
    @validator('query_embedding')
    def validate_query_embedding(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Query embedding cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Query embedding must contain only numbers")
        return v

class TextSearchRequest(BaseModel):
    """Request model for text-based similarity search."""
    query_text: str = Field(..., description="Query text")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    artifact_types: Optional[List[str]] = Field(None, description="Filter by artifact types")

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding operations."""
    embeddings: List[EmbeddingInsertRequest] = Field(..., description="List of embeddings to insert")
    
    @validator('embeddings')
    def validate_embeddings_list(cls, v):
        if not v:
            raise ValueError("Embeddings list cannot be empty")
        if len(v) > 100:
            raise ValueError("Cannot insert more than 100 embeddings at once")
        return v


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
    metadata: Dict[str, Any] = {}
    embedding_model: str
    metadata_created_at: str
    embedding_created_at: str
    embedding: Optional[List[float]] = None

class EmbeddingInsertResponse(BaseModel):
    """Response model for embedding insertion."""
    chunk_id: str
    status: str
    embedding_dimensions: int
    timestamp: str

class PaginationResponse(BaseModel):
    """Response model for pagination info."""
    limit: int
    offset: int
    total_count: int
    has_next: bool
    has_previous: bool

class EmbeddingListResponse(BaseModel):
    """Response model for embedding list."""
    embeddings: List[EmbeddingResponse]
    pagination: PaginationResponse
    filters: Dict[str, Any]

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
    metadata: Dict[str, Any] = {}
    created_at: str

class VectorSearchResponse(BaseModel):
    """Response model for vector search."""
    results: List[SimilarityResult]
    query_info: Dict[str, Any]

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding operations."""
    results: List[EmbeddingInsertResponse]
    summary: Dict[str, Any]

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
    details: Optional[Dict[str, Any]] = None
    timestamp: str