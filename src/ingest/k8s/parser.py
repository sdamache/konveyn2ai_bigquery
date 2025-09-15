"""
Kubernetes Parser Implementation for T023
Implements KubernetesParserImpl class extending the contract from parser-interfaces.py
Uses kr8s + PyYAML for Kubernetes API integration and manifest parsing
"""

import asyncio
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import kr8s
    from kr8s.asyncapi import Api

    KR8S_AVAILABLE = True
except ImportError:
    KR8S_AVAILABLE = False
    # Only log warning when live cluster functionality is specifically requested
    # The warning will be shown in extract_live_resources() if needed


# Contract interfaces (standardized import)
# Import common utilities
from src.common.chunking import ChunkConfig, ChunkingStrategy, ContentChunker
from src.common.ids import ArtifactIDGenerator
from src.common.normalize import ContentNormalizer
from src.common.parser_interfaces import (
    ChunkMetadata,
    ErrorClass,
    KubernetesParser,
    ParseError,
    ParseResult,
    SourceType,
)


class KubernetesParserImpl(KubernetesParser):
    """
    Implementation of Kubernetes parser with kr8s and PyYAML support

    Features:
    - YAML parsing with multi-document support
    - Kubernetes resource validation and type detection
    - Live cluster resource extraction via kr8s
    - Namespace handling and resource identification
    - Content chunking optimized for K8s manifests
    - Metadata extraction for labels, annotations, specs
    - Robust error handling with ParseError objects
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(version)

        # Initialize utilities
        self.chunker = ContentChunker(
            ChunkConfig(
                max_tokens=1000,
                overlap_pct=0.15,
                strategy=ChunkingStrategy.SEMANTIC_BLOCKS,
                preserve_boundaries=True,
            )
        )
        self.id_generator = ArtifactIDGenerator("kubernetes")
        self.normalizer = ContentNormalizer(preserve_semantics=True)

        # Kubernetes resource types that we support
        self.supported_kinds = {
            "Pod",
            "Service",
            "Deployment",
            "ConfigMap",
            "Secret",
            "PersistentVolume",
            "PersistentVolumeClaim",
            "StatefulSet",
            "DaemonSet",
            "ReplicaSet",
            "Job",
            "CronJob",
            "Ingress",
            "NetworkPolicy",
            "ServiceAccount",
            "Role",
            "RoleBinding",
            "ClusterRole",
            "ClusterRoleBinding",
            "HorizontalPodAutoscaler",
            "Namespace",
            "Node",
            "CustomResourceDefinition",
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _get_source_type(self) -> SourceType:
        """Return the source type this parser handles"""
        return SourceType.KUBERNETES

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single Kubernetes manifest file"""
        start_time = time.time()
        chunks = []
        errors = []

        try:
            # Read file content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse the manifest content
            file_chunks = self.parse_manifest(content)
            # Update source_uri for file chunks
            for chunk in file_chunks:
                if not chunk.source_uri:
                    chunk.source_uri = file_path
            chunks.extend(file_chunks)

        except Exception as e:
            error = ParseError(
                source_type=self.source_type,
                source_uri=file_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Failed to parse file: {str(e)}",
                sample_text=content[:200] if "content" in locals() else None,
                stack_trace=traceback.format_exc(),
            )
            errors.append(error)
            self.logger.error(f"Error parsing file {file_path}: {e}")

        duration_ms = int((time.time() - start_time) * 1000)

        return ParseResult(
            chunks=chunks,
            errors=errors,
            files_processed=1,
            processing_duration_ms=duration_ms,
        )

    def parse_directory(self, directory_path: str) -> ParseResult:
        """Parse all Kubernetes manifest files in a directory"""
        start_time = time.time()
        chunks = []
        errors = []
        files_processed = 0

        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")

            # Find all YAML/JSON files
            patterns = ["*.yaml", "*.yml", "*.json"]
            manifest_files = []
            for pattern in patterns:
                manifest_files.extend(directory.rglob(pattern))

            for file_path in manifest_files:
                try:
                    result = self.parse_file(str(file_path))
                    chunks.extend(result.chunks)
                    errors.extend(result.errors)
                    files_processed += 1

                except Exception as e:
                    error = ParseError(
                        source_type=self.source_type,
                        source_uri=str(file_path),
                        error_class=ErrorClass.PARSING,
                        error_msg=f"Failed to parse file: {str(e)}",
                        stack_trace=traceback.format_exc(),
                    )
                    errors.append(error)
                    self.logger.error(f"Error parsing file {file_path}: {e}")

        except Exception as e:
            error = ParseError(
                source_type=self.source_type,
                source_uri=directory_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Failed to parse directory: {str(e)}",
                stack_trace=traceback.format_exc(),
            )
            errors.append(error)
            self.logger.error(f"Error parsing directory {directory_path}: {e}")

        duration_ms = int((time.time() - start_time) * 1000)

        return ParseResult(
            chunks=chunks,
            errors=errors,
            files_processed=files_processed,
            processing_duration_ms=duration_ms,
        )

    def validate_content(self, content: str) -> bool:
        """Validate if content is a valid Kubernetes manifest"""
        try:
            # Try to parse as YAML
            documents = list(yaml.safe_load_all(content))

            # Check if any document looks like a Kubernetes resource
            for doc in documents:
                if not isinstance(doc, dict):
                    continue

                # Must have apiVersion and kind
                if "apiVersion" in doc and "kind" in doc:
                    return True

            return False

        except yaml.YAMLError:
            return False
        except Exception:
            return False

    def parse_manifest(self, manifest_content: str) -> list[ChunkMetadata]:
        """Parse individual Kubernetes manifest with multi-document support"""
        chunks = []

        try:
            # Normalize content
            normalized_content = self.normalizer.normalize_content(
                manifest_content, source_type="yaml"
            )

            # Parse YAML documents
            documents = list(yaml.safe_load_all(normalized_content))

            for doc_index, document in enumerate(documents):
                # Skip None and non-dict documents
                if document is None or not isinstance(document, dict):
                    continue

                # Skip empty documents
                if not document:
                    continue

                # Validate Kubernetes resource structure
                if not self._is_valid_k8s_resource(document):
                    continue

                # Extract resource metadata
                kind = document.get("kind", "Unknown")
                api_version = document.get("apiVersion", "")
                metadata = document.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                name = metadata.get("name", f"unnamed-{doc_index}")
                namespace = metadata.get("namespace", "default")

                # Generate artifact ID
                artifact_id = self.id_generator.generate_artifact_id(
                    source_path=f"{namespace}/{kind}/{name}",
                    metadata={
                        "namespace": namespace,
                        "kind": kind,
                        "name": name,
                        "apiVersion": api_version,
                    },
                )

                # Convert document back to YAML for chunking
                doc_yaml = yaml.dump(document, default_flow_style=False)

                # Chunk the document content
                chunk_texts = self._chunk_k8s_manifest(doc_yaml, document)

                # Create chunks
                for chunk_index, chunk_text in enumerate(chunk_texts):
                    # Use base artifact_id for single chunks, add suffix for multiple chunks
                    if len(chunk_texts) == 1:
                        chunk_artifact_id = artifact_id
                    else:
                        chunk_artifact_id = f"{artifact_id}#chunk-{chunk_index}"

                    chunk_metadata = ChunkMetadata(
                        source_type=self.source_type,
                        artifact_id=chunk_artifact_id,
                        parent_id=artifact_id,
                        parent_type=kind,
                        content_text=chunk_text,
                        content_tokens=self._estimate_tokens(chunk_text),
                        content_hash=self.generate_content_hash(chunk_text),
                        source_uri="",  # Will be updated by caller
                        collected_at=datetime.utcnow(),
                        source_metadata={
                            "kind": kind,
                            "api_version": api_version,  # Use snake_case for consistency
                            "namespace": namespace,
                            "resource_name": name,  # Use resource_name as expected by tests
                            "labels": metadata.get("labels", {}),
                            "annotations": metadata.get("annotations", {}),
                            "resource_version": metadata.get(
                                "resourceVersion"
                            ),  # snake_case
                            "uid": metadata.get("uid"),
                            "chunk_index": chunk_index,
                            "total_chunks": len(chunk_texts),
                            "document_index": doc_index,
                        },
                    )
                    chunks.append(chunk_metadata)

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in manifest: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing manifest: {e}")

        return chunks

    def extract_live_resources(self, namespace: Optional[str] = None) -> ParseResult:
        """Extract resources from live Kubernetes cluster using kr8s"""
        start_time = time.time()
        chunks = []
        errors = []

        if not KR8S_AVAILABLE:
            error = ParseError(
                source_type=self.source_type,
                source_uri="live-cluster",
                error_class=ErrorClass.VALIDATION,
                error_msg="kr8s library not available - cannot extract live resources",
            )
            errors.append(error)
            return ParseResult(
                chunks=[], errors=errors, files_processed=0, processing_duration_ms=0
            )

        try:
            # Run async extraction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chunks, resource_errors = loop.run_until_complete(
                    self._extract_live_resources_async(namespace)
                )
                errors.extend(resource_errors)
            finally:
                loop.close()

        except Exception as e:
            error = ParseError(
                source_type=self.source_type,
                source_uri="live-cluster",
                error_class=ErrorClass.INGESTION,
                error_msg=f"Failed to extract live resources: {str(e)}",
                stack_trace=traceback.format_exc(),
            )
            errors.append(error)
            self.logger.error(f"Error extracting live resources: {e}")

        duration_ms = int((time.time() - start_time) * 1000)

        return ParseResult(
            chunks=chunks,
            errors=errors,
            files_processed=len(chunks),
            processing_duration_ms=duration_ms,
        )

    async def _extract_live_resources_async(
        self, namespace: Optional[str] = None
    ) -> tuple[list[ChunkMetadata], list[ParseError]]:
        """Async implementation of live resource extraction"""
        chunks = []
        errors = []

        try:
            # Create kr8s API client
            async with Api() as api:
                # Get resources by kind
                for kind in self.supported_kinds:
                    try:
                        if namespace:
                            resources = await api.get(kind, namespace=namespace)
                        else:
                            resources = await api.get(kind)

                        if not isinstance(resources, list):
                            resources = [resources] if resources else []

                        for resource in resources:
                            try:
                                # Convert resource to YAML
                                resource_dict = (
                                    resource.raw
                                    if hasattr(resource, "raw")
                                    else dict(resource)
                                )
                                resource_yaml = yaml.dump(
                                    resource_dict, default_flow_style=False
                                )

                                # Parse as manifest
                                resource_chunks = self.parse_manifest(resource_yaml)
                                # Update source_uri for live resources
                                live_uri = f"live://{kind}/{resource_dict.get('metadata', {}).get('name', 'unknown')}"
                                for chunk in resource_chunks:
                                    if not chunk.source_uri:
                                        chunk.source_uri = live_uri
                                chunks.extend(resource_chunks)

                            except Exception as e:
                                error = ParseError(
                                    source_type=self.source_type,
                                    source_uri=f"live://{kind}",
                                    error_class=ErrorClass.INGESTION,
                                    error_msg=f"Failed to process live resource: {str(e)}",
                                )
                                errors.append(error)

                    except Exception as e:
                        error = ParseError(
                            source_type=self.source_type,
                            source_uri=f"live://{kind}",
                            error_class=ErrorClass.INGESTION,
                            error_msg=f"Failed to get resources of kind {kind}: {str(e)}",
                        )
                        errors.append(error)

        except Exception as e:
            error = ParseError(
                source_type=self.source_type,
                source_uri="live-cluster",
                error_class=ErrorClass.INGESTION,
                error_msg=f"Failed to connect to cluster: {str(e)}",
            )
            errors.append(error)

        return chunks, errors

    def _is_valid_k8s_resource(self, document: dict[str, Any]) -> bool:
        """Check if document is a valid Kubernetes resource"""
        if not isinstance(document, dict):
            return False

        # Must have apiVersion and kind
        if "apiVersion" not in document or "kind" not in document:
            return False

        # Kind should be recognized (but we're lenient for custom resources)
        kind = document.get("kind", "")
        if kind in self.supported_kinds:
            return True

        # Allow custom resources if they have proper structure
        if "metadata" in document:
            metadata = document["metadata"]
            if isinstance(metadata, dict) and "name" in metadata:
                return True

        return False

    def _chunk_k8s_manifest(
        self, manifest_yaml: str, document: dict[str, Any]
    ) -> list[str]:
        """Chunk Kubernetes manifest using semantic boundaries"""
        # For small manifests, return as single chunk
        if len(manifest_yaml) < 500:
            return [manifest_yaml]

        # Split by major sections while preserving YAML structure
        chunks = []
        lines = manifest_yaml.split("\n")

        current_chunk = []
        current_section = None
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Track indent level
            if stripped:
                line_indent = len(line) - len(line.lstrip())

                # Major section boundaries (top-level keys)
                if line_indent == 0 and ":" in line and not line.startswith(" "):
                    # Start new chunk for major sections
                    if current_chunk and len("\n".join(current_chunk)) > 200:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []

                    current_section = stripped.split(":")[0]

                current_chunk.append(line)
            else:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # If we only got one chunk, use traditional chunking
        if len(chunks) == 1:
            chunk_results = self.chunker.chunk_content(
                manifest_yaml, source_type="kubernetes"
            )
            return [chunk_result.content for chunk_result in chunk_results]

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)

    def generate_artifact_id(self, source_path: str, **kwargs) -> str:
        """Generate Kubernetes-specific artifact ID"""
        # Support both metadata dict and direct kwargs
        metadata = kwargs.get("metadata", {})

        # Check for direct kwargs first, then fallback to metadata dict
        # Handle None vs empty string properly
        namespace = kwargs.get("namespace")
        if namespace is None:
            namespace = metadata.get("namespace", "default")

        kind = kwargs.get("kind")
        if kind is None:
            kind = metadata.get("kind", "Unknown")

        name = kwargs.get("name")
        if name is None:
            name = metadata.get("name", "unnamed")

        return f"k8s://{namespace}/{kind}/{name}"


# CLI support functions
def create_k8s_parser() -> KubernetesParserImpl:
    """Factory function for creating Kubernetes parser"""
    return KubernetesParserImpl()


def main():
    """CLI entry point for Kubernetes parser"""
    import argparse

    parser = argparse.ArgumentParser(description="Kubernetes manifest parser")
    parser.add_argument("--source", required=True, help="Source file or directory")
    parser.add_argument("--live", action="store_true", help="Extract from live cluster")
    parser.add_argument(
        "--namespace", help="Kubernetes namespace (for live extraction)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be processed"
    )
    parser.add_argument(
        "--output", default="console", choices=["console", "json", "bigquery"]
    )
    parser.add_argument(
        "--version", action="version", version="Kubernetes Parser 1.0.0"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create parser
    k8s_parser = create_k8s_parser()

    # Parse based on mode
    if args.live:
        result = k8s_parser.extract_live_resources(args.namespace)
    elif os.path.isfile(args.source):
        result = k8s_parser.parse_file(args.source)
    elif os.path.isdir(args.source):
        result = k8s_parser.parse_directory(args.source)
    else:
        print(f"Error: Source '{args.source}' not found")
        return 1

    # Output results
    if args.output == "console":
        print(f"Processed {result.files_processed} files")
        print(f"Generated {len(result.chunks)} chunks")
        print(f"Encountered {len(result.errors)} errors")
        print(f"Duration: {result.processing_duration_ms}ms")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error.error_msg}")

    elif args.output == "json":
        import json

        output = {
            "chunks_count": len(result.chunks),
            "errors_count": len(result.errors),
            "files_processed": result.files_processed,
            "processing_duration_ms": result.processing_duration_ms,
        }
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
