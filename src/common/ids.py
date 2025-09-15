"""
Deterministic ID generation using SHA256 and semantic paths
T020: Implements consistent, reproducible artifact and content IDs for idempotent ingestion
"""

import hashlib
import os
import re
from datetime import datetime
from typing import Any

import ulid


class ArtifactIDGenerator:
    """Generates deterministic artifact IDs based on source type and semantic paths"""

    # Source-specific ID patterns for consistency
    ID_PATTERNS = {
        "kubernetes": "k8s://{namespace}/{kind}/{name}",
        "fastapi": "py://{src_path}#{start_line}-{end_line}",
        "cobol": "cobol://{file}/01-{structure_name}",
        "irs": "irs://{record_type}/{layout_version}/{section}",
        "mumps": "mumps://{global_name}/{node_path}",
    }

    def __init__(self, source_type: str):
        self.source_type = source_type.lower()

    def generate_artifact_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """
        Generate deterministic artifact ID based on source type and metadata

        Args:
            source_path: Path to source file/resource
            metadata: Source-specific metadata for ID generation

        Returns:
            Deterministic artifact ID string
        """
        if self.source_type == "kubernetes":
            return self._generate_k8s_id(source_path, metadata)
        elif self.source_type == "fastapi":
            return self._generate_fastapi_id(source_path, metadata)
        elif self.source_type == "cobol":
            return self._generate_cobol_id(source_path, metadata)
        elif self.source_type == "irs":
            return self._generate_irs_id(source_path, metadata)
        elif self.source_type == "mumps":
            return self._generate_mumps_id(source_path, metadata)
        else:
            return self._generate_generic_id(source_path, metadata)

    def _generate_k8s_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate Kubernetes artifact ID: k8s://{namespace}/{kind}/{name}"""
        namespace = metadata.get("namespace", "default")
        kind = metadata.get("kind", "unknown")  # Keep original capitalization
        name = metadata.get("name", self._extract_filename(source_path))

        # Sanitize components
        namespace = self._sanitize_path_component(namespace)
        kind = self._sanitize_path_component(kind)
        name = self._sanitize_path_component(name)

        return f"k8s://{namespace}/{kind}/{name}"

    def _generate_fastapi_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate FastAPI artifact ID: py://{src_path}#{start_line}-{end_line}"""
        # Normalize source path
        normalized_path = self._normalize_file_path(source_path)

        start_line = metadata.get("start_line", 1)
        end_line = metadata.get("end_line", start_line)

        return f"py://{normalized_path}#{start_line}-{end_line}"

    def _generate_cobol_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate COBOL artifact ID: cobol://{file}/01-{structure_name}"""
        file_name = self._extract_filename(source_path)
        structure_name = metadata.get(
            "structure_name", metadata.get("record_name", "main")
        )

        # Sanitize components
        file_name = self._sanitize_path_component(file_name)
        structure_name = self._sanitize_path_component(structure_name)

        return f"cobol://{file_name}/01-{structure_name}"

    def _generate_irs_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate IRS artifact ID: irs://{record_type}/{layout_version}/{section}"""
        record_type = metadata.get("record_type", "IMF")
        layout_version = metadata.get("layout_version", "2024")
        section = metadata.get("section", metadata.get("section_name", "main"))

        # Sanitize components
        record_type = self._sanitize_path_component(record_type)
        layout_version = self._sanitize_path_component(layout_version)
        section = self._sanitize_path_component(section)

        return f"irs://{record_type}/{layout_version}/{section}"

    def _generate_mumps_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate MUMPS artifact ID: mumps://{global_name}/{node_path}"""
        global_name = metadata.get("global_name", metadata.get("file_name", "UNKNOWN"))
        node_path = metadata.get("node_path", metadata.get("field_path", "ROOT"))

        # Sanitize components
        global_name = self._sanitize_path_component(global_name)
        node_path = self._sanitize_path_component(node_path)

        return f"mumps://{global_name}/{node_path}"

    def _generate_generic_id(self, source_path: str, metadata: dict[str, Any]) -> str:
        """Generate generic artifact ID for unknown source types"""
        normalized_path = self._normalize_file_path(source_path)
        section = metadata.get("section", metadata.get("name", "main"))
        section = self._sanitize_path_component(section)

        return f"{self.source_type}://{normalized_path}/{section}"

    def _extract_filename(self, path: str) -> str:
        """Extract filename without extension from path"""
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

    def _normalize_file_path(self, path: str) -> str:
        """Normalize file path for consistent IDs"""
        # Convert to forward slashes and remove redundant path components
        normalized = os.path.normpath(path).replace("\\", "/")

        # Remove leading paths that might vary between environments
        if "/src/" in normalized:
            normalized = normalized.split("/src/", 1)[1]
        elif "/app/" in normalized:
            normalized = normalized.split("/app/", 1)[1]

        return normalized

    def _sanitize_path_component(self, component: str) -> str:
        """Sanitize path component for URL safety"""
        if not component:
            return "unknown"

        # Replace problematic characters with safe alternatives
        sanitized = re.sub(r"[^\w\-\.]", "_", str(component))

        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        return sanitized or "unknown"


class ContentHashGenerator:
    """Generates deterministic content hashes for idempotent ingestion"""

    def __init__(
        self, normalize_whitespace: bool = True, ignore_comments: bool = False
    ):
        self.normalize_whitespace = normalize_whitespace
        self.ignore_comments = ignore_comments

    def generate_content_hash(self, content: str, source_type: str = "generic") -> str:
        """
        Generate SHA256 hash of normalized content

        Args:
            content: Raw content to hash
            source_type: Source type for content-specific normalization

        Returns:
            Hex-encoded SHA256 hash
        """
        normalized = self._normalize_content(content, source_type)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _normalize_content(self, content: str, source_type: str) -> str:
        """Normalize content for consistent hashing"""
        normalized = content

        # Source-specific normalization
        if source_type.lower() in ["kubernetes", "yaml"]:
            normalized = self._normalize_yaml_content(normalized)
        elif source_type.lower() in ["fastapi", "python"]:
            normalized = self._normalize_python_content(normalized)
        elif source_type.lower() in ["cobol"]:
            normalized = self._normalize_cobol_content(normalized)
        elif source_type.lower() in ["irs"]:
            normalized = self._normalize_irs_content(normalized)
        elif source_type.lower() in ["mumps"]:
            normalized = self._normalize_mumps_content(normalized)

        # General normalization
        if self.normalize_whitespace:
            normalized = self._normalize_whitespace_general(normalized)

        if self.ignore_comments:
            normalized = self._remove_comments(normalized, source_type)

        return normalized.strip()

    def _normalize_yaml_content(self, content: str) -> str:
        """Normalize YAML content for consistent hashing"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace but preserve indentation
            line = line.rstrip()

            # Skip empty lines for normalization
            if line.strip():
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_python_content(self, content: str) -> str:
        """Normalize Python content for consistent hashing"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace but preserve indentation
            line = line.rstrip()

            # Skip empty lines and comments if configured
            if line.strip() and not (
                self.ignore_comments and line.strip().startswith("#")
            ):
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_cobol_content(self, content: str) -> str:
        """Normalize COBOL content for consistent hashing"""
        # COBOL is position-sensitive, so minimal normalization
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Only remove trailing whitespace
            line = line.rstrip()
            if line:  # Keep structure intact
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_irs_content(self, content: str) -> str:
        """Normalize IRS content for consistent hashing"""
        # IRS layouts are position-sensitive
        return content.strip()

    def _normalize_mumps_content(self, content: str) -> str:
        """Normalize MUMPS content for consistent hashing"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            if line.strip():
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_whitespace_general(self, content: str) -> str:
        """General whitespace normalization"""
        # Normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.split("\n")]

        # Remove consecutive empty lines
        normalized_lines = []
        prev_empty = False

        for line in lines:
            is_empty = not line.strip()
            if not (is_empty and prev_empty):
                normalized_lines.append(line)
            prev_empty = is_empty

        return "\n".join(normalized_lines)

    def _remove_comments(self, content: str, source_type: str) -> str:
        """Remove comments based on source type"""
        if source_type.lower() in ["python", "fastapi"]:
            # Remove Python comments
            lines = content.split("\n")
            non_comment_lines = []

            for line in lines:
                # Simple comment removal (doesn't handle strings with #)
                comment_idx = line.find("#")
                if comment_idx >= 0:
                    line = line[:comment_idx].rstrip()
                if line or not line.strip():  # Keep empty lines for structure
                    non_comment_lines.append(line)

            return "\n".join(non_comment_lines)

        elif source_type.lower() in ["yaml", "kubernetes"]:
            # Remove YAML comments
            lines = content.split("\n")
            non_comment_lines = []

            for line in lines:
                # Simple YAML comment removal
                comment_idx = line.find("#")
                if comment_idx >= 0:
                    line = line[:comment_idx].rstrip()
                if line or not line.strip():
                    non_comment_lines.append(line)

            return "\n".join(non_comment_lines)

        return content


class ULIDGenerator:
    """Generate ULIDs for run tracking and temporal ordering"""

    @staticmethod
    def generate() -> str:
        """Generate a new ULID"""
        return str(ulid.new())

    @staticmethod
    def generate_with_timestamp(timestamp: datetime) -> str:
        """Generate ULID with specific timestamp"""
        return str(ulid.from_timestamp(timestamp))

    @staticmethod
    def extract_timestamp(ulid_str: str) -> datetime:
        """Extract timestamp from ULID"""
        ulid_obj = ulid.parse(ulid_str)
        return ulid_obj.timestamp().datetime


# Factory functions for easy usage
def create_artifact_id_generator(source_type: str) -> ArtifactIDGenerator:
    """Create artifact ID generator for specific source type"""
    return ArtifactIDGenerator(source_type)


def create_content_hash_generator(
    normalize_whitespace: bool = True, ignore_comments: bool = False
) -> ContentHashGenerator:
    """Create content hash generator with specific options"""
    return ContentHashGenerator(normalize_whitespace, ignore_comments)


def generate_run_id() -> str:
    """Generate a run ID for tracking ingestion sessions"""
    return ULIDGenerator.generate()


# Validation and testing utilities
def validate_artifact_id(artifact_id: str, source_type: str) -> dict[str, bool]:
    """Validate artifact ID format for source type"""
    expected_prefix = f"{source_type.lower()}://"

    return {
        "has_correct_prefix": artifact_id.startswith(expected_prefix),
        "has_valid_format": len(artifact_id) > len(expected_prefix),
        "no_invalid_chars": not re.search(r'[<>:"|?*]', artifact_id),
        "reasonable_length": 10 <= len(artifact_id) <= 500,
    }


def validate_content_hash(content_hash: str) -> dict[str, bool]:
    """Validate content hash format"""
    return {
        "is_hex": re.match(r"^[a-f0-9]+$", content_hash) is not None,
        "correct_length": len(content_hash) == 64,  # SHA256 = 64 hex chars
        "not_empty": len(content_hash) > 0,
    }


def test_determinism(
    content: str, source_type: str, iterations: int = 5
) -> dict[str, Any]:
    """Test that ID and hash generation is deterministic"""
    generator = create_content_hash_generator()

    hashes = []
    for _ in range(iterations):
        hash_result = generator.generate_content_hash(content, source_type)
        hashes.append(hash_result)

    return {
        "all_hashes_identical": len(set(hashes)) == 1,
        "sample_hash": hashes[0] if hashes else None,
        "iterations_tested": iterations,
    }
