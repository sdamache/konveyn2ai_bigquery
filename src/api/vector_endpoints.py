"""
FastAPI endpoints for vector operations.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from google.cloud.exceptions import NotFound, Conflict

from .models import (
    EmbeddingInsertRequest, EmbeddingInsertResponse,
    EmbeddingListRequest, EmbeddingListResponse, 
    VectorSearchRequest, VectorSearchResponse,
    TextSearchRequest, SimilarityResult,
    BatchEmbeddingRequest, BatchEmbeddingResponse,
    HealthCheckResponse, ErrorResponse
)
from ..janapada_memory import BigQueryVectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vector-store", tags=["Vector Operations"])

# Dependency to get vector store instance
def get_vector_store() -> BigQueryVectorStore:
    """Get BigQuery vector store instance."""
    return BigQueryVectorStore()


@router.post(
    "/embeddings",
    response_model=EmbeddingInsertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert new embedding",
    description="Insert a new embedding with associated metadata into the vector store."
)
async def insert_embedding(
    request: EmbeddingInsertRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> EmbeddingInsertResponse:
    """Insert new embedding."""
    try:
        # Convert request to dict for vector store
        chunk_data = {
            "chunk_id": request.chunk_id,
            "source": request.source,
            "artifact_type": request.artifact_type,
            "text_content": request.text_content,
            "kind": request.kind,
            "api_path": request.api_path,
            "record_name": request.record_name
        }
        
        result = vector_store.insert_embedding(
            chunk_data=chunk_data,
            embedding=request.embedding,
            metadata=request.metadata
        )
        
        return EmbeddingInsertResponse(**result)
        
    except Conflict as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Embedding with chunk_id '{request.chunk_id}' already exists"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to insert embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during embedding insertion"
        )


@router.get(
    "/embeddings",
    response_model=EmbeddingListResponse,
    summary="List embeddings",
    description="List embeddings with pagination and filtering options."
)
async def list_embeddings(
    limit: int = 100,
    offset: int = 0,
    artifact_types: str = None,
    include_embeddings: bool = False,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> EmbeddingListResponse:
    """List embeddings with pagination."""
    try:
        # Parse artifact_types if provided
        artifact_types_list = None
        if artifact_types:
            artifact_types_list = [t.strip() for t in artifact_types.split(",")]
        
        result = vector_store.list_embeddings(
            limit=limit,
            offset=offset,
            artifact_types=artifact_types_list,
            include_embeddings=include_embeddings
        )
        
        return EmbeddingListResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to list embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during embedding listing"
        )


@router.get(
    "/embeddings/{chunk_id}",
    response_model=Dict[str, Any],  # Using Dict since EmbeddingResponse is complex
    summary="Get embedding by ID",
    description="Retrieve a specific embedding by its chunk ID."
)
async def get_embedding_by_id(
    chunk_id: str,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> Dict[str, Any]:
    """Get embedding by chunk ID."""
    try:
        result = vector_store.get_embedding_by_id(chunk_id)
        return result
        
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding with chunk_id '{chunk_id}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to get embedding {chunk_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during embedding retrieval"
        )


@router.delete(
    "/embeddings/{chunk_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete embedding",
    description="Delete an embedding and its associated metadata."
)
async def delete_embedding(
    chunk_id: str,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
):
    """Delete embedding by chunk ID."""
    try:
        vector_store.delete_embedding(chunk_id)
        
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding with chunk_id '{chunk_id}' not found"
        )
    except Exception as e:
        logger.error(f"Failed to delete embedding {chunk_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during embedding deletion"
        )


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    summary="Vector similarity search",
    description="Search for similar vectors using vector embedding."
)
async def search_similar_vectors(
    request: VectorSearchRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> VectorSearchResponse:
    """Search for similar vectors."""
    try:
        results = vector_store.search_similar_vectors(
            query_embedding=request.query_embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            artifact_types=request.artifact_types
        )
        
        similarity_results = [SimilarityResult(**result) for result in results]
        
        return VectorSearchResponse(
            results=similarity_results,
            query_info={
                "dimensions": len(request.query_embedding),
                "similarity_threshold": request.similarity_threshold,
                "limit": request.limit,
                "artifact_types": request.artifact_types,
                "results_count": len(similarity_results)
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during vector search"
        )


@router.post(
    "/search/text",
    response_model=VectorSearchResponse,
    summary="Text similarity search",
    description="Search for similar content using text query."
)
async def search_similar_text(
    request: TextSearchRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> VectorSearchResponse:
    """Search for similar text content."""
    try:
        results = vector_store.search_similar_text(
            query_text=request.query_text,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            artifact_types=request.artifact_types
        )
        
        similarity_results = [SimilarityResult(**result) for result in results]
        
        return VectorSearchResponse(
            results=similarity_results,
            query_info={
                "query_text": request.query_text,
                "similarity_threshold": request.similarity_threshold,
                "limit": request.limit,
                "artifact_types": request.artifact_types,
                "results_count": len(similarity_results)
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during text search"
        )


@router.post(
    "/embeddings/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch insert embeddings",
    description="Insert multiple embeddings in a single operation."
)
async def batch_insert_embeddings(
    request: BatchEmbeddingRequest,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> BatchEmbeddingResponse:
    """Insert multiple embeddings in batch."""
    try:
        # Convert requests to format expected by vector store
        embeddings_data = []
        for emb_request in request.embeddings:
            emb_data = {
                "chunk_id": emb_request.chunk_id,
                "source": emb_request.source,
                "artifact_type": emb_request.artifact_type,
                "text_content": emb_request.text_content,
                "kind": emb_request.kind,
                "api_path": emb_request.api_path,
                "record_name": emb_request.record_name,
                "embedding": emb_request.embedding,
                "metadata": emb_request.metadata
            }
            embeddings_data.append(emb_data)
        
        results = vector_store.batch_insert_embeddings(embeddings_data)
        
        insert_responses = [EmbeddingInsertResponse(**result) for result in results]
        
        return BatchEmbeddingResponse(
            results=insert_responses,
            summary={
                "total_processed": len(results),
                "successful": len([r for r in results if r["status"] == "inserted"]),
                "failed": len([r for r in results if r["status"] != "inserted"]),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch insert failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch insertion"
        )


@router.get(
    "/embeddings/batch",
    response_model=List[Dict[str, Any]],
    summary="Batch get embeddings",
    description="Retrieve multiple embeddings by chunk IDs."
)
async def batch_get_embeddings(
    chunk_ids: str,
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> List[Dict[str, Any]]:
    """Get multiple embeddings by chunk IDs."""
    try:
        # Parse chunk IDs from comma-separated string
        chunk_ids_list = [cid.strip() for cid in chunk_ids.split(",")]
        
        if len(chunk_ids_list) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot retrieve more than 100 embeddings at once"
            )
        
        results = vector_store.batch_get_embeddings(chunk_ids_list)
        return results
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch get failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch retrieval"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of the vector store."
)
async def health_check(
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> HealthCheckResponse:
    """Perform health check on vector store."""
    try:
        health_status = vector_store.health_check()
        return HealthCheckResponse(**health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable"
        )


@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="Get vector store statistics",
    description="Get statistics about the vector store contents."
)
async def get_stats(
    vector_store: BigQueryVectorStore = Depends(get_vector_store)
) -> Dict[str, Any]:
    """Get vector store statistics."""
    try:
        total_count = vector_store.count_embeddings()
        
        # Get counts by artifact type (simplified)
        stats = {
            "total_embeddings": total_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add more detailed stats if needed
        try:
            code_count = vector_store.count_embeddings(artifact_types=["code"])
            doc_count = vector_store.count_embeddings(artifact_types=["documentation"])
            
            stats["by_artifact_type"] = {
                "code": code_count,
                "documentation": doc_count,
                "other": total_count - code_count - doc_count
            }
        except Exception as e:
            logger.warning(f"Could not get detailed stats: {e}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting statistics"
        )