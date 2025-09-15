"""
Parser Interface Contracts for M1 Multi-Source Ingestion
Defines standardized interfaces for all source parsers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
from enum import Enum


class SourceType(Enum):
    """Supported source types for ingestion"""
    KUBERNETES = "kubernetes"
    FASTAPI = "fastapi"
    COBOL = "cobol"
    IRS = "irs"
    MUMPS = "mumps"


class ErrorClass(Enum):
    """Error classification categories"""
    PARSING = "parsing"
    VALIDATION = "validation"
    INGESTION = "ingestion"


@dataclass
class ChunkMetadata:
    """Common metadata for all chunk types"""
    source_type: SourceType
    artifact_id: str
    parent_id: Optional[str] = None
    parent_type: Optional[str] = None
    content_text: str = ""
    content_tokens: Optional[int] = None
    content_hash: str = ""
    source_uri: str = ""
    repo_ref: Optional[str] = None
    collected_at: Optional[datetime] = None

    # Source-specific metadata stored as dict
    source_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}
        if self.collected_at is None:
            self.collected_at = datetime.utcnow()


@dataclass
class ParseError:
    """Represents a parsing error during ingestion"""
    source_type: SourceType
    source_uri: str
    error_class: ErrorClass
    error_msg: str
    sample_text: Optional[str] = None
    stack_trace: Optional[str] = None
    collected_at: Optional[datetime] = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.utcnow()


@dataclass
class ParseResult:
    """Result of parsing operation containing chunks and errors"""
    chunks: List[ChunkMetadata]
    errors: List[ParseError]
    files_processed: int
    processing_duration_ms: int


class BaseParser(ABC):
    """Abstract base class for all source parsers"""

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.source_type = self._get_source_type()

    @abstractmethod
    def _get_source_type(self) -> SourceType:
        """Return the source type this parser handles"""
        pass

    @abstractmethod
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single file and return chunks with metadata"""
        pass

    @abstractmethod
    def parse_directory(self, directory_path: str) -> ParseResult:
        """Parse all files in a directory"""
        pass

    @abstractmethod
    def validate_content(self, content: str) -> bool:
        """Validate if content is parseable by this parser"""
        pass

    def generate_artifact_id(self, source_path: str, **kwargs) -> str:
        """Generate deterministic artifact ID"""
        # Default implementation, can be overridden
        return f"{self.source_type.value}://{source_path}"

    def generate_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of normalized content"""
        import hashlib
        normalized = content.strip().encode('utf-8')
        return hashlib.sha256(normalized).hexdigest()

    def chunk_content(self, content: str, max_tokens: int = 1000, overlap_pct: float = 0.15) -> List[str]:
        """Split content into overlapping chunks"""
        # Basic implementation - parsers should override with source-specific logic
        words = content.split()
        chunk_size = max(1, int(max_tokens * 0.75))  # Rough token estimation
        overlap_size = int(chunk_size * overlap_pct)

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break
            start = end - overlap_size

        return chunks


class KubernetesParser(BaseParser):
    """Parser for Kubernetes YAML/JSON manifests"""

    def _get_source_type(self) -> SourceType:
        return SourceType.KUBERNETES

    @abstractmethod
    def parse_manifest(self, manifest_content: str) -> List[ChunkMetadata]:
        """Parse individual Kubernetes manifest"""
        pass

    @abstractmethod
    def extract_live_resources(self, namespace: Optional[str] = None) -> ParseResult:
        """Extract resources from live cluster"""
        pass


class FastAPIParser(BaseParser):
    """Parser for FastAPI applications and OpenAPI specs"""

    def _get_source_type(self) -> SourceType:
        return SourceType.FASTAPI

    @abstractmethod
    def parse_openapi_spec(self, spec_content: str) -> List[ChunkMetadata]:
        """Parse OpenAPI specification JSON"""
        pass

    @abstractmethod
    def parse_source_code(self, python_file: str) -> List[ChunkMetadata]:
        """Parse Python source code for FastAPI routes and models"""
        pass


class COBOLParser(BaseParser):
    """Parser for COBOL copybooks"""

    def _get_source_type(self) -> SourceType:
        return SourceType.COBOL

    @abstractmethod
    def parse_copybook(self, copybook_content: str) -> List[ChunkMetadata]:
        """Parse COBOL copybook with level structures"""
        pass

    @abstractmethod
    def extract_pic_clauses(self, structure: Dict[str, Any]) -> Dict[str, str]:
        """Extract PIC clauses from COBOL structure"""
        pass


class IRSParser(BaseParser):
    """Parser for IRS IMF record layouts"""

    def _get_source_type(self) -> SourceType:
        return SourceType.IRS

    @abstractmethod
    def parse_imf_layout(self, layout_content: str) -> List[ChunkMetadata]:
        """Parse IRS IMF fixed-width record layout"""
        pass

    @abstractmethod
    def extract_field_positions(self, layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract field positions and lengths"""
        pass


class MUMPSParser(BaseParser):
    """Parser for MUMPS/VistA FileMan dictionaries"""

    def _get_source_type(self) -> SourceType:
        return SourceType.MUMPS

    @abstractmethod
    def parse_fileman_dict(self, dict_content: str) -> List[ChunkMetadata]:
        """Parse FileMan data dictionary"""
        pass

    @abstractmethod
    def parse_global_definition(self, global_content: str) -> List[ChunkMetadata]:
        """Parse MUMPS global variable definitions"""
        pass


# CLI Interface Contract
class CLIInterface:
    """Standardized CLI interface for all parsers"""

    @staticmethod
    def create_argument_parser(parser_class: str) -> 'argparse.ArgumentParser':
        """Create standardized argument parser"""
        import argparse
        parser = argparse.ArgumentParser(description=f"{parser_class} ingestion")
        parser.add_argument('--source', required=True, help='Source file or directory')
        parser.add_argument('--dry-run', action='store_true', help='Show what would be processed')
        parser.add_argument('--output', default='console', choices=['console', 'json', 'bigquery'])
        parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
        parser.add_argument('--max-rows', type=int, default=None, help='Limit number of rows')
        return parser

    @staticmethod
    def format_output(result: ParseResult, format_type: str = 'console') -> str:
        """Format parser result for output"""
        if format_type == 'json':
            import json
            # Convert to JSON-serializable format
            return json.dumps({
                'chunks_count': len(result.chunks),
                'errors_count': len(result.errors),
                'files_processed': result.files_processed,
                'processing_duration_ms': result.processing_duration_ms
            }, indent=2)
        elif format_type == 'console':
            return (
                f"Processed {result.files_processed} files\n"
                f"Generated {len(result.chunks)} chunks\n"
                f"Encountered {len(result.errors)} errors\n"
                f"Duration: {result.processing_duration_ms}ms"
            )
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# BigQuery Writer Contract
class BigQueryWriter:
    """Contract for writing parsed data to BigQuery"""

    @abstractmethod
    def write_chunks(self, chunks: List[ChunkMetadata], table_name: str) -> int:
        """Write chunks to BigQuery source_metadata table"""
        pass

    @abstractmethod
    def write_errors(self, errors: List[ParseError], table_name: str) -> int:
        """Write errors to BigQuery source_metadata_errors table"""
        pass

    @abstractmethod
    def log_ingestion_run(self, run_info: Dict[str, Any], table_name: str) -> str:
        """Log ingestion run details, return run_id"""
        pass

    @abstractmethod
    def create_tables_if_not_exist(self, dataset_name: str) -> bool:
        """Create BigQuery tables from DDL schema"""
        pass