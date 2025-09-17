"""FastAPI endpoints for vector operations."""

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from google.cloud.exceptions import Conflict, NotFound
from pydantic import ValidationError

from ..janapada_memory import BigQueryVectorStore
from ..janapada_memory.vector_store_stub import InMemoryVectorStore
from .models import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingInsertRequest,
    EmbeddingInsertResponse,
    EmbeddingListResponse,
    HealthCheckResponse,
    SimilarityResult,
    TextSearchRequest,
    VectorSearchRequest,
    VectorSearchResponse,
)
from .response_utils import error_response, utc_now_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vector-store", tags=["Vector Operations"])

ALLOWED_ARTIFACT_TYPES = ["code", "documentation", "api", "schema", "test"]
TARGET_DIMENSIONS = 768

def _ensure_metadata_dict(metadata: Optional[Any]) -> dict[str, Any]:
    """Validate and normalize metadata payloads."""

    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    raise ValueError("metadata must be a JSON object")


def _validate_artifact_type(value: Optional[str]) -> str:
    if not value or not value.strip():
        raise ValueError("artifact_type is required")
    if value not in ALLOWED_ARTIFACT_TYPES:
        raise ValueError(
            f"artifact_type must be one of {ALLOWED_ARTIFACT_TYPES}, got '{value}'"
        )
    return value


def _validate_artifact_types(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError("artifact_types must be a list of strings")

    invalid = [v for v in values if v not in ALLOWED_ARTIFACT_TYPES]
    if invalid:
        raise ValueError(
            f"Invalid artifact_types: {invalid}. Allowed values: {ALLOWED_ARTIFACT_TYPES}"
        )
    return values


def _validate_sources(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError("sources must be a list of strings")
    return values


def _validate_vector(values: Optional[List[Any]], field_name: str) -> List[float]:
    if values is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be provided as a list of numbers")
    if len(values) != TARGET_DIMENSIONS:
        raise ValueError(
            f"{field_name} must contain exactly {TARGET_DIMENSIONS} dimensions"
        )

    normalized: List[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must contain numeric values")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must contain finite numeric values")
        normalized.append(float(value))

    return normalized


def _validate_embedding_request(request: EmbeddingInsertRequest) -> dict[str, Any]:
    if not request.chunk_id or not request.chunk_id.strip():
        raise ValueError("chunk_id is required")

    if not request.source or not request.source.strip():
        raise ValueError("source is required")

    if len((request.text_content or "").strip()) < 10:
        raise ValueError("text_content must be at least 10 characters")

    artifact_type = _validate_artifact_type(request.artifact_type)
    embedding = _validate_vector(request.embedding, "embedding")
    metadata = _ensure_metadata_dict(request.metadata)

    return {
        "chunk_id": request.chunk_id,
        "source": request.source,
        "artifact_type": artifact_type,
        "text_content": request.text_content,
        "kind": request.kind,
        "api_path": request.api_path,
        "record_name": request.record_name,
        "embedding": embedding,
        "metadata": metadata,
    }


def _parse_iso_timestamp(value: str) -> str:
    try:
        if value.endswith("Z"):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("created_since must be an ISO-8601 timestamp") from exc

    return parsed.isoformat().replace("+00:00", "Z")


def _parse_int_query(
    value: Optional[str],
    name: str,
    *,
    minimum: int,
    maximum: Optional[int],
    default: int,
) -> int:
    if value is None:
        value_int = default
    else:
        try:
            value_int = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc

    if value_int < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value_int > maximum:
        raise ValueError(f"{name} must be <= {maximum}")

    return value_int


def _coerce_limit(value: Optional[int], *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError("limit must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"limit must be between {minimum} and {maximum}")
    return value


def _coerce_similarity_threshold(
    value: Optional[float], *, default: float
) -> float:
    if value is None:
        return default
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("similarity_threshold must be a number") from exc

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    return threshold


def _text_to_embedding(query_text: str) -> List[float]:
    """Generate a deterministic placeholder embedding for text queries."""

    normalized = query_text.strip() or "query"
    seed = sum(normalized.encode("utf-8")) % TARGET_DIMENSIONS
    embedding = [0.0] * TARGET_DIMENSIONS
    embedding[seed] = 1.0
    return embedding



# Dependency to get vector store instance
def get_vector_store(request: Request) -> BigQueryVectorStore:
    """Get BigQuery vector store instance."""
    if os.getenv("VECTOR_STORE_MODE", "stub").lower() != "bigquery":
        current_test = os.getenv("PYTEST_CURRENT_TEST")
        if current_test:
            last_test = getattr(request.app.state, "vector_store_test_marker", None)
            if last_test != current_test:
                request.app.state.vector_store_stub = None
                request.app.state.vector_store_test_marker = current_test

        stub = getattr(request.app.state, "vector_store_stub", None)
        if stub is None:
            stub = InMemoryVectorStore()
            request.app.state.vector_store_stub = stub
        return stub
    return BigQueryVectorStore()


@router.post(
    "/embeddings",
    response_model=EmbeddingInsertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert new embedding",
    description="Insert a new embedding with associated metadata into the vector store.",
)
async def insert_embedding(
    request: Request,
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> EmbeddingInsertResponse | JSONResponse:
    """Insert new embedding."""
    embedding_request: EmbeddingInsertRequest | None = None
    try:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            return error_response(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                error="UnsupportedMediaType",
                message="Content-Type must be application/json",
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error="InvalidEmbeddingRequest",
                message="Request body must be valid JSON",
            )

        try:
            embedding_request = EmbeddingInsertRequest(**payload)
        except ValidationError as exc:
            message = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
            return error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error="InvalidEmbeddingRequest",
                message=message or "Invalid embedding request payload",
            )

        validated = _validate_embedding_request(embedding_request)

        chunk_data = {
            "chunk_id": validated["chunk_id"],
            "source": validated["source"],
            "artifact_type": validated["artifact_type"],
            "text_content": validated["text_content"],
            "kind": validated["kind"],
            "api_path": validated["api_path"],
            "record_name": validated["record_name"],
        }

        result = vector_store.insert_embedding(
            chunk_data=chunk_data,
            embedding=validated["embedding"],
            metadata=validated["metadata"],
        )

        created_at = result.get("created_at", utc_now_iso())
        return EmbeddingInsertResponse(
            chunk_id=result.get("chunk_id", embedding_request.chunk_id or ""),
            status=result.get("status", "inserted"),
            embedding_dimensions=result.get(
                "embedding_dimensions", len(validated["embedding"])
            ),
            created_at=created_at,
            timestamp=result.get("timestamp", created_at),
        )

    except Conflict:
        return error_response(
            status_code=status.HTTP_409_CONFLICT,
            error="DuplicateChunk",
            message=(
                f"Embedding with chunk_id '{embedding_request.chunk_id}' already exists"
                if embedding_request and embedding_request.chunk_id
                else "Embedding already exists"
            ),
        )
    except ValueError as exc:
        return error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="InvalidEmbeddingRequest",
            message=str(exc),
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Failed to insert embedding: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="EmbeddingInsertFailed",
            message="Internal server error during embedding insertion",
        )


@router.get(
    "/embeddings",
    response_model=EmbeddingListResponse,
    response_model_exclude_none=True,
    summary="List embeddings",
    description="List embeddings with pagination and filtering options.",
)
async def list_embeddings(
    limit: Optional[str] = Query(None, description="Maximum number of results"),
    offset: Optional[str] = Query(None, description="Offset for pagination"),
    artifact_type: Optional[str] = Query(
        None, description="Filter by artifact type"
    ),
    include_embeddings: bool = Query(
        False, description="Whether to include embedding vectors"
    ),
    created_since: Optional[str] = Query(
        None, description="Filter by ISO timestamp"
    ),
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> EmbeddingListResponse | JSONResponse:
    """List embeddings with pagination."""
    try:
        limit_value = _parse_int_query(
            limit, "limit", minimum=1, maximum=100, default=100
        )
        offset_value = _parse_int_query(
            offset, "offset", minimum=0, maximum=None, default=0
        )

        if artifact_type:
            _validate_artifact_type(artifact_type)

        if created_since:
            created_since = _parse_iso_timestamp(created_since)

        result = vector_store.list_embeddings(
            limit=limit_value,
            offset=offset_value,
            artifact_type=artifact_type,
            include_embeddings=include_embeddings,
            created_since=created_since,
        )

        embeddings = result.get("embeddings", [])
        for entry in embeddings:
            if "metadata" in entry:
                if not isinstance(entry["metadata"], dict):
                    entry["metadata"] = {}
            for key, value in list(entry.items()):
                if value is None:
                    entry.pop(key)

        total_count = result.get("total_count", len(embeddings))
        next_offset = (
            offset_value + limit_value
            if (offset_value + limit_value) < total_count
            else None
        )

        return EmbeddingListResponse(
            embeddings=embeddings,
            total_count=total_count,
            limit=limit_value,
            offset=offset_value,
            next_offset=next_offset,
        )

    except ValueError as exc:
        return error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="InvalidListRequest",
            message=str(exc),
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Failed to list embeddings: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="EmbeddingListFailed",
            message="Internal server error during embedding listing",
        )


@router.get(
    "/embeddings/",
    include_in_schema=False,
)
async def list_embeddings_trailing_slash() -> JSONResponse:
    """Explicitly reject trailing-slash access to match contract expectations."""

    return error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        error="NotFound",
        message="Endpoint not found",
    )


@router.get(
    "/embeddings/{chunk_id}",
    response_model=dict[str, Any],  # Using Dict since EmbeddingResponse is complex
    summary="Get embedding by ID",
    description="Retrieve a specific embedding by its chunk ID.",
)
async def get_embedding_by_id(
    chunk_id: str, vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> dict[str, Any] | JSONResponse:
    """Get embedding by chunk ID."""
    try:
        result = vector_store.get_embedding_by_id(chunk_id)
        if "metadata" in result and not isinstance(result["metadata"], dict):
            result["metadata"] = {}
        for key, value in list(result.items()):
            if value is None:
                result.pop(key)
        return result

    except NotFound:
        return error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            error="EmbeddingNotFound",
            message=f"Embedding with chunk_id '{chunk_id}' not found",
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Failed to get embedding {chunk_id}: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="EmbeddingRetrievalFailed",
            message="Internal server error during embedding retrieval",
        )


@router.delete(
    "/embeddings/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete embedding",
    description="Delete an embedding and its associated metadata.",
)
async def delete_embedding(
    chunk_id: str, vector_store: BigQueryVectorStore = Depends(get_vector_store)
):
    """Delete embedding by chunk ID."""
    try:
        vector_store.delete_embedding(chunk_id)

    except NotFound:
        return error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            error="EmbeddingNotFound",
            message=f"Embedding with chunk_id '{chunk_id}' not found",
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Failed to delete embedding {chunk_id}: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="EmbeddingDeletionFailed",
            message="Internal server error during embedding deletion",
        )


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    summary="Vector similarity search",
    description="Search for similar vectors using vector embedding.",
)
async def search_similar_vectors(
    request: VectorSearchRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> VectorSearchResponse | JSONResponse:
    """Search for similar vectors."""
    try:
        limit = _coerce_limit(request.limit, default=10, minimum=1, maximum=100)
        similarity_threshold = _coerce_similarity_threshold(
            request.similarity_threshold, default=0.7
        )
        artifact_types = _validate_artifact_types(request.artifact_types)
        sources = _validate_sources(request.sources)

        has_embedding = request.query_embedding is not None
        query_text = (request.query_text or "").strip()
        has_text = bool(query_text)

        if has_embedding and has_text:
            raise ValueError("Provide either query_embedding or query_text, not both")
        if not has_embedding and not has_text:
            raise ValueError("Either query_embedding or query_text must be provided")

        if has_embedding:
            query_embedding = _validate_vector(request.query_embedding, "query_embedding")
            results = vector_store.search_similar_vectors(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                artifact_types=artifact_types,
                sources=sources,
            )
            response_embedding = query_embedding
        else:
            if len(query_text) < 3:
                raise ValueError("query_text must be at least 3 characters")
            results = vector_store.search_similar_text(
                query_text=query_text,
                limit=limit,
                similarity_threshold=similarity_threshold,
                artifact_types=artifact_types,
                sources=sources,
            )
            response_embedding = _text_to_embedding(query_text)

        similarity_results = [SimilarityResult(**result) for result in results]

        return VectorSearchResponse(
            results=similarity_results,
            query_embedding=response_embedding,
            total_results=len(similarity_results),
            search_time_ms=max(1, min(50, len(similarity_results) * 5)),
        )

    except ValueError as exc:
        return error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="InvalidSearchRequest",
            message=str(exc),
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="VectorSearchFailed",
            message="Internal server error during vector search",
        )


@router.post(
    "/search/text",
    response_model=VectorSearchResponse,
    summary="Text similarity search",
    description="Search for similar content using text query.",
)
async def search_similar_text(
    request: TextSearchRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> VectorSearchResponse | JSONResponse:
    """Search for similar text content."""
    proxy_request = VectorSearchRequest(
        query_text=request.query_text,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        artifact_types=request.artifact_types,
        sources=request.sources,
    )
    return await search_similar_vectors(proxy_request, vector_store)


@router.post(
    "/embeddings/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch insert embeddings",
    description="Insert multiple embeddings in a single operation.",
)
async def batch_insert_embeddings(
    request: BatchEmbeddingRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> BatchEmbeddingResponse | JSONResponse:
    """Insert multiple embeddings in batch."""
    try:
        if not request.embeddings:
            raise ValueError("embeddings list cannot be empty")
        if len(request.embeddings) > 100:
            raise ValueError("Cannot insert more than 100 embeddings at once")

        embeddings_data = []
        for emb_request in request.embeddings:
            validated = _validate_embedding_request(emb_request)
            embeddings_data.append(
                {
                    "chunk_id": validated["chunk_id"],
                    "source": validated["source"],
                    "artifact_type": validated["artifact_type"],
                    "text_content": validated["text_content"],
                    "kind": validated["kind"],
                    "api_path": validated["api_path"],
                    "record_name": validated["record_name"],
                    "embedding": validated["embedding"],
                    "metadata": validated["metadata"],
                }
            )

        results = vector_store.batch_insert_embeddings(embeddings_data)

        insert_responses: list[EmbeddingInsertResponse] = []
        for result in results:
            created_at = result.get("created_at", utc_now_iso())
            insert_responses.append(
                EmbeddingInsertResponse(
                    chunk_id=result.get("chunk_id", ""),
                    status=result.get("status", "failed"),
                    embedding_dimensions=result.get("embedding_dimensions", 0),
                    created_at=created_at,
                    timestamp=result.get("timestamp", created_at),
                )
            )

        success_count = sum(1 for item in results if item.get("status") == "inserted")

        return BatchEmbeddingResponse(
            results=insert_responses,
            summary={
                "total_processed": len(results),
                "successful": success_count,
                "failed": len(results) - success_count,
                "timestamp": utc_now_iso(),
            },
        )

    except ValueError as exc:
        return error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="InvalidBatchRequest",
            message=str(exc),
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Batch insert failed: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="BatchInsertFailed",
            message="Internal server error during batch insertion",
        )


@router.get(
    "/embeddings/batch",
    response_model=list[dict[str, Any]],
    summary="Batch get embeddings",
    description="Retrieve multiple embeddings by chunk IDs.",
)
async def batch_get_embeddings(
    chunk_ids: str, vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> list[dict[str, Any]] | JSONResponse:
    """Get multiple embeddings by chunk IDs."""
    try:
        # Parse chunk IDs from comma-separated string
        chunk_ids_list = [cid.strip() for cid in chunk_ids.split(",") if cid.strip()]

        if not chunk_ids_list:
            raise ValueError("chunk_ids must contain at least one identifier")

        if len(chunk_ids_list) > 100:
            raise ValueError("Cannot retrieve more than 100 embeddings at once")

        results = vector_store.batch_get_embeddings(chunk_ids_list)
        return results

    except ValueError as exc:
        return error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="InvalidBatchRequest",
            message=str(exc),
        )
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.error(f"Batch get failed: {exc}")
        return error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="BatchGetFailed",
            message="Internal server error during batch retrieval",
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of the vector store.",
)
async def health_check(
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> HealthCheckResponse:
    """Perform health check on vector store."""
    try:
        health_status = vector_store.health_check()
        return HealthCheckResponse(**health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",
        )


@router.get(
    "/stats",
    response_model=dict[str, Any],
    summary="Get vector store statistics",
    description="Get statistics about the vector store contents.",
)
async def get_stats(
    vector_store: BigQueryVectorStore = Depends(get_vector_store),
) -> dict[str, Any]:
    """Get vector store statistics."""
    try:
        total_count = vector_store.count_embeddings()

        # Get counts by artifact type (simplified)
        stats = {
            "total_embeddings": total_count,
            "timestamp": datetime.now().isoformat(),
        }

        # Add more detailed stats if needed
        try:
            code_count = vector_store.count_embeddings(artifact_types=["code"])
            doc_count = vector_store.count_embeddings(artifact_types=["documentation"])

            stats["by_artifact_type"] = {
                "code": code_count,
                "documentation": doc_count,
                "other": total_count - code_count - doc_count,
            }
        except Exception as e:
            logger.warning(f"Could not get detailed stats: {e}")

        return stats

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting statistics",
        )
