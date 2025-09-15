"""
MUMPS/VistA FileMan Dictionary Parser Implementation
T027: Implements MUMPS parser with custom regex patterns and pyGTM validation

This module implements the MUMPSParser contract from parser-interfaces.py,
providing functionality to parse FileMan data dictionaries and global variable
definitions from VistA/MUMPS systems.

Key Features:
- FileMan data dictionary parsing with field definitions
- Global variable structure analysis (^GLOBAL(subscript) patterns)
- Field number and data type extraction
- Cross-reference and index detection
- Node path navigation for hierarchical data
- VistA-specific metadata extraction
- Content chunking optimized for MUMPS hierarchical structures
- Custom regex patterns for MUMPS syntax
- Artifact ID generation: mumps://{global_name}/{node_path}
"""

import os
import re
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator

# Import contract interfaces
import sys
import importlib.util

# Import parser interface contract
try:
    project_root = Path(__file__).parent.parent.parent.parent
    specs_path = project_root / "specs" / "002-m1-parse-and" / "contracts" / "parser-interfaces.py"
    spec = importlib.util.spec_from_file_location("parser_interfaces", specs_path)

    # Check if parser_interfaces already exists in sys.modules
    if "parser_interfaces" in sys.modules:
        parser_interfaces = sys.modules["parser_interfaces"]
    else:
        parser_interfaces = importlib.util.module_from_spec(spec)
        sys.modules["parser_interfaces"] = parser_interfaces  # Add to sys.modules
        spec.loader.exec_module(parser_interfaces)

    BaseParser = parser_interfaces.BaseParser
    MUMPSParser = parser_interfaces.MUMPSParser
    ChunkMetadata = parser_interfaces.ChunkMetadata
    ParseResult = parser_interfaces.ParseResult
    ParseError = parser_interfaces.ParseError
    SourceType = parser_interfaces.SourceType
    ErrorClass = parser_interfaces.ErrorClass
    CONTRACT_IMPORTS_SUCCESS = True
except (ImportError, FileNotFoundError) as e:
    CONTRACT_IMPORTS_SUCCESS = False
    # Fallback for standalone usage
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from enum import Enum

    class SourceType(Enum):
        MUMPS = "mumps"

    class ErrorClass(Enum):
        PARSING = "parsing"
        VALIDATION = "validation"
        INGESTION = "ingestion"

    @dataclass
    class ChunkMetadata:
        source_type: SourceType
        artifact_id: str
        content_text: str
        content_hash: str
        source_uri: str
        parent_id: Optional[str] = None
        parent_type: Optional[str] = None
        content_tokens: Optional[int] = None
        repo_ref: Optional[str] = None
        collected_at: Optional[datetime] = None
        source_metadata: Dict[str, Any] = None

    @dataclass
    class ParseError:
        source_type: SourceType
        source_uri: str
        error_class: ErrorClass
        error_msg: str
        sample_text: Optional[str] = None
        stack_trace: Optional[str] = None
        collected_at: Optional[datetime] = None

    @dataclass
    class ParseResult:
        chunks: List[ChunkMetadata]
        errors: List[ParseError]
        files_processed: int
        processing_duration_ms: int

    class BaseParser(ABC):
        def __init__(self, version: str = "1.0.0"):
            self.version = version
            self.source_type = self._get_source_type()

        @abstractmethod
        def _get_source_type(self) -> SourceType:
            pass

        def generate_artifact_id(self, source_path: str, **kwargs) -> str:
            return f"{self.source_type.value}://{source_path}"

        def generate_content_hash(self, content: str) -> str:
            import hashlib
            return hashlib.sha256(content.strip().encode('utf-8')).hexdigest()

    class MUMPSParser(BaseParser):
        @abstractmethod
        def parse_fileman_dict(self, dict_content: str) -> List[ChunkMetadata]:
            pass

        @abstractmethod
        def parse_global_definition(self, global_content: str) -> List[ChunkMetadata]:
            pass

# Import common utilities
try:
    from src.common.chunking import create_chunker, ChunkConfig, ChunkingStrategy
    from src.common.ids import ArtifactIDGenerator, ContentHashGenerator
    from src.common.normalize import ContentNormalizer
    from src.common.bq_writer import BigQueryWriter
except ImportError:
    # Fallback implementations for testing
    class ContentNormalizer:
        def normalize_content(self, content: str, source_type: str, options=None):
            return content.strip()

    class ArtifactIDGenerator:
        def __init__(self, source_type: str):
            self.source_type = source_type

        def generate_artifact_id(self, source_path: str, metadata: Dict[str, Any]) -> str:
            global_name = metadata.get("global_name", "UNKNOWN")
            node_path = metadata.get("node_path", "ROOT")
            return f"mumps://{global_name}/{node_path}"

    class ContentHashGenerator:
        def generate_content_hash(self, content: str, source_type: str = "mumps") -> str:
            import hashlib
            return hashlib.sha256(content.strip().encode('utf-8')).hexdigest()

    def create_chunker(source_type: str, max_tokens: int = 1000, overlap_pct: float = 0.15):
        return None


class MUMPSParserImpl(MUMPSParser):
    """
    Concrete implementation of MUMPS/VistA FileMan dictionary parser

    This parser handles:
    - FileMan data dictionary entries (^DD, ^DDD)
    - Global variable definitions (^GLOBAL)
    - Field definitions with input transforms
    - Cross-references and indices
    - Node hierarchies and subscript structures
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(version)

        # Initialize utilities
        self.normalizer = ContentNormalizer()
        self.id_generator = ArtifactIDGenerator("mumps")
        self.hash_generator = ContentHashGenerator()
        self.chunker = create_chunker("mumps", max_tokens=1000, overlap_pct=0.15)

        # MUMPS-specific regex patterns
        self._setup_regex_patterns()

        # pyGTM validation (fallback to regex if not available)
        self.has_pygtm = self._check_pygtm_availability()

    def _get_source_type(self) -> SourceType:
        """Return the source type this parser handles"""
        return SourceType.MUMPS

    def _check_pygtm_availability(self) -> bool:
        """Check if pyGTM is available for validation"""
        try:
            import pygtm
            return True
        except ImportError:
            return False

    def _setup_regex_patterns(self):
        """Setup regex patterns for MUMPS/VistA parsing"""

        # FileMan Data Dictionary patterns
        self.patterns = {
            # ^DD(file,field,attribute) patterns
            'dd_entry': re.compile(r'^\^DD\(([^,]+),([^,]+),([^)]+)\)=(.*)$', re.MULTILINE),

            # ^DDD(file,field,attribute) patterns
            'ddd_entry': re.compile(r'^\^DDD\(([^,]+),([^,]+),([^)]+)\)=(.*)$', re.MULTILINE),

            # Global variable patterns: ^GLOBAL(subscript1,subscript2,...)=value
            'global_ref': re.compile(r'^\^([A-Z0-9_]+)\(([^)]*)\)=(.*)$', re.MULTILINE),

            # Field number patterns (e.g., .01, .02, 1, 2.1)
            'field_number': re.compile(r'^\.?\d+(\.\d+)?$'),

            # File number patterns
            'file_number': re.compile(r'^\d+(\.\d+)?$'),

            # Input transform patterns
            'input_transform': re.compile(r'K:.*?\sX|S\s.*?=.*?|D\s\^.*?'),

            # Cross-reference patterns
            'xref_pattern': re.compile(r'^\^[A-Z0-9_]+\([^,]*,"([A-Z0-9_]+)"'),

            # Data type patterns in field definitions
            'data_type': re.compile(r'"[^"]*\^([A-Z]+)[^"]*"'),

            # Global name extraction
            'global_name': re.compile(r'^\^([A-Z0-9_]+)'),

            # Node path extraction
            'node_path': re.compile(r'\(([^)]*)\)'),

            # Comments (lines starting with ;)
            'comment': re.compile(r'^\s*;.*$', re.MULTILINE),

            # FileMan DD header
            'dd_header': re.compile(r'^\^DD\((\d+),0\)="([^"]*)"'),

            # Global header
            'global_header': re.compile(r'^\^([A-Z0-9_]+)\(([^,]*),0\)="([^"]*)"')
        }

    def validate_content(self, content: str) -> bool:
        """
        Validate if content is parseable MUMPS/VistA content

        Args:
            content: Raw content to validate

        Returns:
            True if content appears to be valid MUMPS, False otherwise
        """
        if not content or not content.strip():
            return False

        # Use pyGTM if available
        if self.has_pygtm:
            return self._validate_with_pygtm(content)

        # Fallback to regex validation
        return self._validate_with_regex(content)

    def _validate_with_pygtm(self, content: str) -> bool:
        """Validate content using pyGTM (if available)"""
        try:
            import pygtm
            # Basic validation using pyGTM
            # This is a placeholder - actual pyGTM validation would be more sophisticated
            return True
        except (ImportError, Exception):
            return self._validate_with_regex(content)

    def _validate_with_regex(self, content: str) -> bool:
        """Validate content using custom regex patterns"""
        lines = content.strip().split('\n')
        mumps_lines = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # Check for MUMPS patterns
            if (self.patterns['dd_entry'].match(line) or
                self.patterns['ddd_entry'].match(line) or
                self.patterns['global_ref'].match(line)):
                mumps_lines += 1

        # Consider valid if at least 50% of non-comment lines are MUMPS
        total_lines = len([l for l in lines if l.strip() and not l.strip().startswith(';')])
        if total_lines == 0:
            return False

        return (mumps_lines / total_lines) >= 0.5

    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse a single MUMPS file

        Args:
            file_path: Path to the MUMPS file

        Returns:
            ParseResult with chunks and any errors
        """
        start_time = time.time()
        chunks = []
        errors = []

        try:
            if not os.path.exists(file_path):
                errors.append(ParseError(
                    source_type=SourceType.MUMPS,
                    source_uri=file_path,
                    error_class=ErrorClass.INGESTION,
                    error_msg=f"File not found: {file_path}",
                    collected_at=datetime.now(UTC)
                ))
                return ParseResult(chunks=[], errors=errors, files_processed=0,
                                 processing_duration_ms=int((time.time() - start_time) * 1000))

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not self.validate_content(content):
                errors.append(ParseError(
                    source_type=SourceType.MUMPS,
                    source_uri=file_path,
                    error_class=ErrorClass.VALIDATION,
                    error_msg="Content does not appear to be valid MUMPS/VistA format",
                    sample_text=content[:200],
                    collected_at=datetime.now(UTC)
                ))
                return ParseResult(chunks=chunks, errors=errors, files_processed=1,
                                 processing_duration_ms=int((time.time() - start_time) * 1000))

            # Determine content type and parse accordingly
            if self._is_fileman_dict(content):
                chunks.extend(self.parse_fileman_dict(content))
            elif self._is_global_definition(content):
                chunks.extend(self.parse_global_definition(content))
            else:
                # Try to parse as mixed content
                chunks.extend(self._parse_mixed_content(content))

            # Update source_uri for all chunks
            for chunk in chunks:
                chunk.source_uri = file_path

        except Exception as e:
            errors.append(ParseError(
                source_type=SourceType.MUMPS,
                source_uri=file_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Error parsing file: {str(e)}",
                stack_trace=traceback.format_exc(),
                collected_at=datetime.now(UTC)
            ))

        processing_time = int((time.time() - start_time) * 1000)
        return ParseResult(chunks=chunks, errors=errors, files_processed=1,
                         processing_duration_ms=processing_time)

    def parse_directory(self, directory_path: str) -> ParseResult:
        """
        Parse all MUMPS files in a directory

        Args:
            directory_path: Path to directory containing MUMPS files

        Returns:
            ParseResult with all chunks and errors from directory
        """
        start_time = time.time()
        all_chunks = []
        all_errors = []
        files_processed = 0

        if not os.path.exists(directory_path):
            all_errors.append(ParseError(
                source_type=SourceType.MUMPS,
                source_uri=directory_path,
                error_class=ErrorClass.INGESTION,
                error_msg=f"Directory not found: {directory_path}",
                collected_at=datetime.now(UTC)
            ))
            return ParseResult(chunks=[], errors=all_errors, files_processed=0,
                             processing_duration_ms=int((time.time() - start_time) * 1000))

        # Find MUMPS files (common extensions: .m, .ro, .gbl, .txt)
        mumps_extensions = {'.m', '.ro', '.gbl', '.txt', '.mumps'}

        for file_path in Path(directory_path).rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in mumps_extensions:
                result = self.parse_file(str(file_path))
                all_chunks.extend(result.chunks)
                all_errors.extend(result.errors)
                files_processed += result.files_processed

        processing_time = int((time.time() - start_time) * 1000)
        return ParseResult(chunks=all_chunks, errors=all_errors,
                         files_processed=files_processed, processing_duration_ms=processing_time)

    def parse_fileman_dict(self, dict_content: str) -> List[ChunkMetadata]:
        """
        Parse FileMan data dictionary content

        Args:
            dict_content: Raw FileMan dictionary content

        Returns:
            List of ChunkMetadata objects for dictionary entries
        """
        chunks = []
        content = self.normalizer.normalize_content(dict_content, "mumps", {})

        # Extract file-level information first
        file_info = self._extract_file_info(content)

        # Parse DD entries (^DD patterns)
        dd_chunks = self._parse_dd_entries(content, file_info)
        chunks.extend(dd_chunks)

        # Parse DDD entries (^DDD patterns)
        ddd_chunks = self._parse_ddd_entries(content, file_info)
        chunks.extend(ddd_chunks)

        return chunks

    def parse_global_definition(self, global_content: str) -> List[ChunkMetadata]:
        """
        Parse MUMPS global variable definitions

        Args:
            global_content: Raw global definition content

        Returns:
            List of ChunkMetadata objects for global entries
        """
        chunks = []
        content = self.normalizer.normalize_content(global_content, "mumps", {})

        # Extract global-level information
        global_info = self._extract_global_info(content)

        # Parse global entries
        for match in self.patterns['global_ref'].finditer(content):
            global_name = match.group(1)
            subscripts = match.group(2)
            value = match.group(3)

            # Create node path from subscripts
            node_path = self._create_node_path(subscripts)

            # Generate chunk for this global entry
            chunk_content = match.group(0)

            metadata = {
                'global_name': global_name,
                'node_path': node_path,
                'subscripts': subscripts,
                'value': value,
                'file_no': global_info.get('file_no'),
                'field_no': None,
                'xrefs': self._extract_xrefs(chunk_content),
                'input_transform': None
            }

            artifact_id = self.id_generator.generate_artifact_id(
                f"{global_name}.gbl", metadata
            )

            chunk = ChunkMetadata(
                source_type=SourceType.MUMPS,
                artifact_id=artifact_id,
                content_text=chunk_content,
                content_hash=self.hash_generator.generate_content_hash(chunk_content, "mumps"),
                source_uri="",  # Will be set by caller
                content_tokens=self._estimate_tokens(chunk_content),
                collected_at=datetime.now(UTC),
                source_metadata=metadata
            )

            chunks.append(chunk)

        return chunks

    def _is_fileman_dict(self, content: str) -> bool:
        """Check if content appears to be FileMan dictionary"""
        return (self.patterns['dd_entry'].search(content) is not None or
                self.patterns['ddd_entry'].search(content) is not None)

    def _is_global_definition(self, content: str) -> bool:
        """Check if content appears to be global definitions"""
        return self.patterns['global_ref'].search(content) is not None

    def _parse_mixed_content(self, content: str) -> List[ChunkMetadata]:
        """Parse content that contains both dictionary and global definitions"""
        chunks = []

        # Split content into sections
        lines = content.split('\n')
        dict_lines = []
        global_lines = []

        for line in lines:
            if self.patterns['dd_entry'].match(line) or self.patterns['ddd_entry'].match(line):
                dict_lines.append(line)
            elif self.patterns['global_ref'].match(line):
                global_lines.append(line)

        # Parse dictionary sections
        if dict_lines:
            dict_content = '\n'.join(dict_lines)
            chunks.extend(self.parse_fileman_dict(dict_content))

        # Parse global sections
        if global_lines:
            global_content = '\n'.join(global_lines)
            chunks.extend(self.parse_global_definition(global_content))

        return chunks

    def _extract_file_info(self, content: str) -> Dict[str, Any]:
        """Extract file-level information from dictionary content"""
        file_info = {}

        # Look for file header: ^DD(file,0)="name^file^..."
        header_match = self.patterns['dd_header'].search(content)
        if header_match:
            file_info['file_no'] = int(header_match.group(1))
            file_info['file_name'] = header_match.group(2)

        return file_info

    def _extract_global_info(self, content: str) -> Dict[str, Any]:
        """Extract global-level information"""
        global_info = {}

        # Look for global header: ^GLOBAL(file,0)="name^file^..."
        header_match = self.patterns['global_header'].search(content)
        if header_match:
            global_info['global_name'] = header_match.group(1)
            global_info['file_no'] = header_match.group(2)
            global_info['description'] = header_match.group(3)

        return global_info

    def _parse_dd_entries(self, content: str, file_info: Dict[str, Any]) -> List[ChunkMetadata]:
        """Parse ^DD entries from content"""
        chunks = []

        for match in self.patterns['dd_entry'].finditer(content):
            file_num = match.group(1)
            field_num = match.group(2)
            attribute = match.group(3)
            value = match.group(4)

            # Create chunk for this DD entry
            chunk_content = match.group(0)

            metadata = {
                'global_name': 'DD',
                'node_path': f"{file_num},{field_num},{attribute}",
                'file_no': file_num,
                'field_no': field_num,
                'attribute': attribute,
                'value': value,
                'xrefs': self._extract_xrefs(chunk_content),
                'input_transform': self._extract_input_transform(value)
            }

            artifact_id = self.id_generator.generate_artifact_id(
                f"DD_{file_num}.m", metadata
            )

            chunk = ChunkMetadata(
                source_type=SourceType.MUMPS,
                artifact_id=artifact_id,
                content_text=chunk_content,
                content_hash=self.hash_generator.generate_content_hash(chunk_content, "mumps"),
                source_uri="",
                content_tokens=self._estimate_tokens(chunk_content),
                collected_at=datetime.now(UTC),
                source_metadata=metadata
            )

            chunks.append(chunk)

        return chunks

    def _parse_ddd_entries(self, content: str, file_info: Dict[str, Any]) -> List[ChunkMetadata]:
        """Parse ^DDD entries from content"""
        chunks = []

        for match in self.patterns['ddd_entry'].finditer(content):
            file_num = match.group(1)
            field_num = match.group(2)
            attribute = match.group(3)
            value = match.group(4)

            chunk_content = match.group(0)

            metadata = {
                'global_name': 'DDD',
                'node_path': f"{file_num},{field_num},{attribute}",
                'file_no': file_num,
                'field_no': field_num,
                'attribute': attribute,
                'value': value,
                'xrefs': self._extract_xrefs(chunk_content),
                'input_transform': self._extract_input_transform(value)
            }

            artifact_id = self.id_generator.generate_artifact_id(
                f"DDD_{file_num}.m", metadata
            )

            chunk = ChunkMetadata(
                source_type=SourceType.MUMPS,
                artifact_id=artifact_id,
                content_text=chunk_content,
                content_hash=self.hash_generator.generate_content_hash(chunk_content, "mumps"),
                source_uri="",
                content_tokens=self._estimate_tokens(chunk_content),
                collected_at=datetime.now(UTC),
                source_metadata=metadata
            )

            chunks.append(chunk)

        return chunks

    def _create_node_path(self, subscripts: str) -> str:
        """Create hierarchical node path from subscripts"""
        if not subscripts:
            return "ROOT"

        # Clean up subscripts and create path
        cleaned = subscripts.replace('"', '').replace("'", '')
        parts = [part.strip() for part in cleaned.split(',')]
        return '/'.join(parts)

    def _extract_xrefs(self, content: str) -> Dict[str, List[str]]:
        """Extract cross-references from content"""
        xrefs = {}

        xref_matches = self.patterns['xref_pattern'].findall(content)
        for xref in xref_matches:
            if 'indices' not in xrefs:
                xrefs['indices'] = []
            xrefs['indices'].append(xref)

        return xrefs

    def _extract_input_transform(self, value: str) -> Optional[str]:
        """Extract input transform from field definition"""
        if not value:
            return None

        transform_match = self.patterns['input_transform'].search(value)
        return transform_match.group(0) if transform_match else None

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        # Simple estimation: ~4 characters per token
        return max(1, len(content.strip()) // 4)

    def generate_artifact_id(self, source_path: str, **kwargs) -> str:
        """Generate MUMPS-specific artifact ID"""
        global_name = kwargs.get('global_name', 'UNKNOWN')
        node_path = kwargs.get('node_path', 'ROOT')
        return f"mumps://{global_name}/{node_path}"

    def chunk_content(self, content: str, max_tokens: int = 1000, overlap_pct: float = 0.15) -> List[str]:
        """Chunk MUMPS content using hierarchical strategy"""
        if self.chunker:
            chunks = self.chunker.chunk_content(content, "mumps")
            return [chunk.content for chunk in chunks]

        # Fallback chunking
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            if current_size + line_size > max_tokens * 4 and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks


# Factory function for easy usage
def create_mumps_parser() -> MUMPSParserImpl:
    """Create a MUMPS parser instance"""
    return MUMPSParserImpl()


# Testing and validation functions
def validate_mumps_parser(parser: MUMPSParserImpl) -> Dict[str, bool]:
    """Validate that MUMPS parser implementation is correct"""
    results = {}

    # Test basic functionality
    results['has_source_type'] = parser._get_source_type() == SourceType.MUMPS
    results['has_validate_method'] = hasattr(parser, 'validate_content')
    results['has_parse_fileman_method'] = hasattr(parser, 'parse_fileman_dict')
    results['has_parse_global_method'] = hasattr(parser, 'parse_global_definition')

    # Test validation
    sample_dd = '^DD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>30 X"'
    results['validates_mumps_content'] = parser.validate_content(sample_dd)
    results['rejects_invalid_content'] = not parser.validate_content("This is not MUMPS")

    return results


if __name__ == "__main__":
    # Basic testing
    parser = create_mumps_parser()

    # Test validation
    test_dd = '^DD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>30!($L(X)<3) X"'
    print(f"Validates DD entry: {parser.validate_content(test_dd)}")

    test_global = '^VA(200,123,0)="DOCTOR,JANE^123456789^MD^F"'
    print(f"Validates global entry: {parser.validate_content(test_global)}")

    # Test parsing
    chunks = parser.parse_fileman_dict(test_dd)
    print(f"Generated {len(chunks)} chunks from DD entry")

    if chunks:
        chunk = chunks[0]
        print(f"Sample chunk artifact_id: {chunk.artifact_id}")
        print(f"Sample chunk metadata: {chunk.source_metadata}")

    # Validation
    validation_results = validate_mumps_parser(parser)
    print(f"Parser validation: {validation_results}")