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
from .bq_writer import BatchConfig, BigQueryWriter, WriteResult, create_bigquery_writer

# Datetime utilities
from .datetime_utils import (
    datetime_to_iso,
    json_dumps_safe,
    now_iso,
    prepare_metadata_for_json,
    safe_datetime_serializer,
)

# M1 ingestion utilities (T019-T022)
from .chunking import ChunkConfig, ChunkResult, ContentChunker, create_chunker
from .ids import (
    ArtifactIDGenerator,
    ContentHashGenerator,
    ULIDGenerator,
    create_artifact_id_generator,
    create_content_hash_generator,
    generate_run_id,
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
from .normalize import ContentNormalizer, create_normalizer, normalize_for_hashing

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
    "ContentChunker",
    "ChunkConfig",
    "ChunkResult",
    "create_chunker",
    "ArtifactIDGenerator",
    "ContentHashGenerator",
    "ULIDGenerator",
    "create_artifact_id_generator",
    "create_content_hash_generator",
    "generate_run_id",
    "ContentNormalizer",
    "create_normalizer",
    "normalize_for_hashing",
    "BigQueryWriter",
    "WriteResult",
    "BatchConfig",
    "create_bigquery_writer",
    # Datetime utilities
    "now_iso",
    "datetime_to_iso",
    "safe_datetime_serializer",
    "json_dumps_safe",
    "prepare_metadata_for_json",
]
