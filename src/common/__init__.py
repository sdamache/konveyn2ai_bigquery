"""
Common utilities and models for KonveyN2AI multi-agent system.

This module provides shared functionality across all three components:
- Amatya Role Prompter
- Janapada Memory
- Svami Orchestrator
- M1 Multi-Source Ingestion (NEW)

Includes JSON-RPC protocol implementation, shared data models,
configuration management, inter-service communication utilities,
and M1 ingestion utilities for BigQuery data processing.
"""

# Existing multi-agent system imports
from .agent_manifest import (
    AgentCapability,
    AgentDiscovery,
    AgentManifest,
    AgentManifestGenerator,
    MethodSchema,
    ParameterSchema,
)
from .models import (
    AdviceRequest,
    AnswerResponse,
    Document,
    DocumentChunk,
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcRequest,
    JsonRpcResponse,
    QueryRequest,
    SearchQuery,
    SearchRequest,
    SearchResult,
    SearchType,
    Snippet,
)

# M1 ingestion utilities (T019-T022)
from .chunking import ContentChunker, ChunkConfig, ChunkResult, create_chunker
from .ids import (
    ArtifactIDGenerator, ContentHashGenerator, ULIDGenerator,
    create_artifact_id_generator, create_content_hash_generator, generate_run_id
)
from .normalize import ContentNormalizer, create_normalizer, normalize_for_hashing
from .bq_writer import BigQueryWriter, WriteResult, BatchConfig, create_bigquery_writer

__version__ = "1.0.0"
__all__ = [
    # Existing multi-agent system exports
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "JsonRpcErrorCode",
    "Snippet",
    "SearchRequest",
    "SearchQuery",
    "SearchResult",
    "SearchType",
    "DocumentChunk",
    "Document",
    "AdviceRequest",
    "QueryRequest",
    "AnswerResponse",
    "AgentCapability",
    "AgentDiscovery",
    "AgentManifest",
    "AgentManifestGenerator",
    "MethodSchema",
    "ParameterSchema",

    # M1 ingestion utilities (T019-T022)
    "ContentChunker", "ChunkConfig", "ChunkResult", "create_chunker",
    "ArtifactIDGenerator", "ContentHashGenerator", "ULIDGenerator",
    "create_artifact_id_generator", "create_content_hash_generator", "generate_run_id",
    "ContentNormalizer", "create_normalizer", "normalize_for_hashing",
    "BigQueryWriter", "WriteResult", "BatchConfig", "create_bigquery_writer"
]
