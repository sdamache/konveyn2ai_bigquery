"""
COBOL Copybook Parser Implementation
T025: COBOL parser using python-cobol + regex for copybook parsing with level structures
"""

import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Import common utilities
from src.common.chunking import ChunkConfig, ChunkingStrategy, ContentChunker
from src.common.ids import ArtifactIDGenerator, ContentHashGenerator
from src.common.normalize import ContentNormalizer

# Contract interfaces (standardized import)
from src.common.parser_interfaces import (
    ChunkMetadata,
    COBOLParser,
    ErrorClass,
    ParseError,
    ParseResult,
    SourceType,
)


class COBOLParserImpl(COBOLParser):
    """
    COBOL Copybook Parser Implementation

    Parses COBOL copybooks with level structures (01-88 levels), PIC clauses,
    OCCURS, REDEFINES, and VALUE clauses. Generates chunks with hierarchical
    metadata for BigQuery ingestion.
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(version)

        # Initialize utilities
        self.chunker = ContentChunker(
            ChunkConfig(
                strategy=ChunkingStrategy.FIXED_WIDTH,
                max_tokens=1000,
                overlap_pct=0.10,  # Smaller overlap for COBOL records
                preserve_boundaries=True,
            )
        )

        self.id_generator = ArtifactIDGenerator("cobol")
        self.hash_generator = ContentHashGenerator(
            normalize_whitespace=False
        )  # COBOL is position-sensitive
        self.normalizer = ContentNormalizer(
            preserve_semantics=True, aggressive_whitespace=False
        )

        # COBOL parsing patterns
        self._setup_cobol_patterns()

    def _setup_cobol_patterns(self):
        """Setup regex patterns for COBOL parsing"""
        # Level number pattern (01-88)
        self.level_pattern = re.compile(
            r"^\s*(\d{2})\s+([A-Z0-9_-]+)(?:\s+(.*?))?\.?\s*$", re.MULTILINE
        )

        # Structure definition pattern
        self.structure_pattern = re.compile(r"^\s*01\s+([A-Z0-9_-]+)", re.MULTILINE)

        # PIC clause patterns
        self.pic_pattern = re.compile(
            r"PIC(?:TURE)?\s+([A-Z0-9\(\)V\+\-\.\$,]+)", re.IGNORECASE
        )

        # OCCURS clause pattern
        self.occurs_pattern = re.compile(
            r"OCCURS\s+(\d+)(?:\s+TO\s+(\d+))?\s+TIMES", re.IGNORECASE
        )

        # REDEFINES clause pattern
        self.redefines_pattern = re.compile(r"REDEFINES\s+([A-Z0-9_-]+)", re.IGNORECASE)

        # VALUE clause pattern
        self.value_pattern = re.compile(r"VALUE\s+(?:IS\s+)?([^\.]+)", re.IGNORECASE)

        # USAGE clause pattern
        self.usage_pattern = re.compile(
            r"(?:USAGE\s+(?:IS\s+)?|COMP-?|BINARY|PACKED-DECIMAL)\s*([A-Z0-9\-]+)?",
            re.IGNORECASE,
        )

        # 88-level condition pattern
        self.condition_pattern = re.compile(
            r"^\s*88\s+([A-Z0-9_-]+)\s+VALUE\s+(.+?)\.?\s*$", re.MULTILINE
        )

    def _get_source_type(self):
        """Return COBOL source type"""
        return SourceType.COBOL

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single COBOL copybook file"""
        start_time = time.perf_counter()
        chunks = []
        errors = []

        try:
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                errors.append(
                    ParseError(
                        source_type=self.source_type,
                        source_uri=file_path,
                        error_class=ErrorClass.PARSING,
                        error_msg=f"File not found: {file_path}",
                        collected_at=datetime.now(timezone.utc),
                    )
                )
                return ParseResult(
                    chunks=chunks,
                    errors=errors,
                    files_processed=0,
                    processing_duration_ms=max(
                        1, int((time.perf_counter() - start_time) * 1000)
                    ),
                )

            # Read file content
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding (COBOL often uses EBCDIC or cp1252)
                try:
                    with open(file_path, encoding="cp1252") as f:
                        content = f.read()
                except Exception as e:
                    errors.append(
                        ParseError(
                            source_type=self.source_type,
                            source_uri=file_path,
                            error_class=ErrorClass.PARSING,
                            error_msg=f"Could not read file with UTF-8 or CP1252 encoding: {e}",
                            collected_at=datetime.now(timezone.utc),
                        )
                    )
                    return ParseResult(
                        chunks=chunks,
                        errors=errors,
                        files_processed=0,
                        processing_duration_ms=max(
                            1, int((time.perf_counter() - start_time) * 1000)
                        ),
                    )

            # Validate content
            if not self.validate_content(content):
                errors.append(
                    ParseError(
                        source_type=self.source_type,
                        source_uri=file_path,
                        error_class=ErrorClass.VALIDATION,
                        error_msg="Content does not appear to be a valid COBOL copybook",
                        sample_text=content[:200] if content else None,
                        collected_at=datetime.now(timezone.utc),
                    )
                )
                return ParseResult(
                    chunks=chunks,
                    errors=errors,
                    files_processed=0,
                    processing_duration_ms=max(
                        1, int((time.perf_counter() - start_time) * 1000)
                    ),
                )

            # Parse copybook content
            try:
                file_chunks = self.parse_copybook(content)
                # Update source_uri for all chunks
                for chunk in file_chunks:
                    chunk.source_uri = file_path
                chunks.extend(file_chunks)

            except Exception as e:
                errors.append(
                    ParseError(
                        source_type=self.source_type,
                        source_uri=file_path,
                        error_class=ErrorClass.PARSING,
                        error_msg=f"Error parsing copybook: {e}",
                        sample_text=content[:500] if content else None,
                        stack_trace=traceback.format_exc(),
                        collected_at=datetime.now(timezone.utc),
                    )
                )

        except Exception as e:
            errors.append(
                ParseError(
                    source_type=self.source_type,
                    source_uri=file_path,
                    error_class=ErrorClass.PARSING,
                    error_msg=f"Unexpected error processing file: {e}",
                    stack_trace=traceback.format_exc(),
                    collected_at=datetime.utcnow(),
                )
            )

        return ParseResult(
            chunks=chunks,
            errors=errors,
            files_processed=1,
            processing_duration_ms=max(
                1, int((time.perf_counter() - start_time) * 1000)
            ),
        )

    def parse_directory(self, directory_path: str) -> ParseResult:
        """Parse all COBOL copybook files in a directory"""
        start_time = time.perf_counter()
        all_chunks = []
        all_errors = []
        files_processed = 0

        try:
            # COBOL copybook file extensions
            cobol_extensions = {".cpy", ".cob", ".copybook", ".copy", ".inc"}

            directory = Path(directory_path)
            if not directory.exists():
                all_errors.append(
                    ParseError(
                        source_type=self.source_type,
                        source_uri=directory_path,
                        error_class=ErrorClass.PARSING,
                        error_msg=f"Directory not found: {directory_path}",
                        collected_at=datetime.now(timezone.utc),
                    )
                )
                return ParseResult(
                    chunks=all_chunks,
                    errors=all_errors,
                    files_processed=0,
                    processing_duration_ms=max(
                        1, int((time.perf_counter() - start_time) * 1000)
                    ),
                )

            # Find all COBOL files
            cobol_files = []
            for ext in cobol_extensions:
                cobol_files.extend(directory.glob(f"*{ext}"))
                cobol_files.extend(directory.glob(f"**/*{ext}"))  # Recursive

            # Parse each file
            for file_path in cobol_files:
                try:
                    result = self.parse_file(str(file_path))
                    all_chunks.extend(result.chunks)
                    all_errors.extend(result.errors)
                    files_processed += result.files_processed
                except Exception as e:
                    all_errors.append(
                        ParseError(
                            source_type=self.source_type,
                            source_uri=str(file_path),
                            error_class=ErrorClass.PARSING,
                            error_msg=f"Error processing file {file_path}: {e}",
                            stack_trace=traceback.format_exc(),
                            collected_at=datetime.now(timezone.utc),
                        )
                    )

        except Exception as e:
            all_errors.append(
                ParseError(
                    source_type=self.source_type,
                    source_uri=directory_path,
                    error_class=ErrorClass.PARSING,
                    error_msg=f"Error scanning directory: {e}",
                    stack_trace=traceback.format_exc(),
                    collected_at=datetime.utcnow(),
                )
            )

        return ParseResult(
            chunks=all_chunks,
            errors=all_errors,
            files_processed=files_processed,
            processing_duration_ms=max(
                1, int((time.perf_counter() - start_time) * 1000)
            ),
        )

    def validate_content(self, content: str) -> bool:
        """Validate if content is a parseable COBOL copybook"""
        if not content or not content.strip():
            return False

        # Check for basic COBOL structure indicators
        indicators = [
            # 01-level structure
            re.search(r"^\s*01\s+[A-Z0-9_-]+", content, re.MULTILINE),
            # Level numbers (05, 10, 15, etc.)
            re.search(
                r"^\s*(?:05|10|15|20|25|30|35|40|45|49|77|88)\s+[A-Z0-9_-]+",
                content,
                re.MULTILINE,
            ),
            # PIC clauses
            re.search(
                r"PIC(?:TURE)?\s+[A-Z0-9\(\)V\+\-\.\$,]+", content, re.IGNORECASE
            ),
        ]

        # At least one indicator should be present
        return any(indicators)

    def parse_copybook(self, copybook_content: str) -> list[ChunkMetadata]:
        """Parse COBOL copybook with level structures"""
        chunks = []

        # Normalize content for consistent parsing
        normalized_content = self.normalizer.normalize_content(
            copybook_content,
            "cobol",
            {
                "remove_comments": False,  # Keep COBOL comments for context
                "normalize_line_endings": True,
                "trim_whitespace": True,
            },
        )

        # Validate COBOL syntax
        self._validate_cobol_syntax(normalized_content)

        # Find all 01-level structures
        structures = self._extract_01_structures(normalized_content)

        if not structures:
            raise ValueError("No 01-level structures found in copybook")

        # Process each structure
        for structure_name, structure_content, structure_start in structures:
            try:
                chunk = self._parse_single_structure(
                    structure_name,
                    structure_content,
                    normalized_content,
                    structure_start,
                )
                chunks.append(chunk)
            except Exception as e:
                # Log error but continue processing other structures
                raise ValueError(f"Error parsing structure {structure_name}: {e}")

        return chunks

    def _validate_cobol_syntax(self, content: str) -> None:
        """Validate COBOL syntax and raise errors for invalid content"""
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("*"):  # Skip empty lines and comments
                continue

            # Check for basic syntax errors
            try:
                # Check for malformed PIC clauses
                if "PIC" in line.upper():
                    pic_matches = re.findall(
                        r"PIC(?:TURE)?\s+([A-Z0-9\(\)V\+\-\.\$,]*)", line, re.IGNORECASE
                    )
                    for pic_clause in pic_matches:
                        if pic_clause:
                            # Check for unmatched parentheses
                            if pic_clause.count("(") != pic_clause.count(")"):
                                raise SyntaxError(
                                    f"Unmatched parentheses in PIC clause at line {line_num}: {pic_clause}"
                                )

                            # Check for valid PIC clause format
                            if not re.match(r"^[A-Z0-9\(\)V\+\-\.\$,]+$", pic_clause):
                                raise SyntaxError(
                                    f"Invalid PIC clause format at line {line_num}: {pic_clause}"
                                )

                # Check for malformed level structures
                level_match = re.match(r"^\s*(\d{2})\s+([A-Z0-9_-]+)", line)
                if level_match:
                    level = int(level_match.group(1))
                    field_name = level_match.group(2)

                    # Validate level number
                    if level < 1 or level > 88:
                        raise SyntaxError(
                            f"Invalid level number {level} at line {line_num}"
                        )

                    # Validate field name
                    if not re.match(r"^[A-Z0-9_-]+$", field_name):
                        raise SyntaxError(
                            f"Invalid field name '{field_name}' at line {line_num}"
                        )

                # Check for invalid standalone clauses
                if re.search(
                    r"^\s*(?:VALUE|PIC|OCCURS|REDEFINES|USAGE)\s+", line, re.IGNORECASE
                ):
                    if not re.match(r"^\s*\d{2}\s+[A-Z0-9_-]+", line):
                        raise SyntaxError(
                            f"Invalid standalone clause at line {line_num}: {line}"
                        )

            except SyntaxError:
                raise
            except Exception as e:
                raise SyntaxError(f"Syntax validation error at line {line_num}: {e}")

    def _extract_01_structures(self, content: str) -> list[tuple[str, str, int]]:
        """Extract all 01-level structures from copybook"""
        structures = []
        lines = content.split("\n")
        current_structure = None
        current_content = []
        structure_start = 0

        for i, line in enumerate(lines):
            # Check for 01-level start
            match = re.match(r"^\s*01\s+([A-Z0-9_-]+)", line)
            if match:
                # Save previous structure if exists
                if current_structure:
                    structures.append(
                        (current_structure, "\n".join(current_content), structure_start)
                    )

                # Start new structure
                current_structure = match.group(1)
                current_content = [line]
                structure_start = i

            elif current_structure:
                # Check if this line belongs to current structure
                # Lines with level numbers 02-88, or continuation lines
                if (
                    re.match(r"^\s*(?:0[2-9]|[1-8][0-9]|88)\s+", line)
                    or re.match(
                        r"^\s*[A-Z0-9_-]+\s+", line
                    )  # Continuation without level
                    or line.strip() == ""  # Empty lines
                    or re.match(r"^\s*\*", line)
                ):  # Comment lines
                    current_content.append(line)
                else:
                    # This might be start of next structure or end of current
                    next_01_match = re.match(r"^\s*01\s+([A-Z0-9_-]+)", line)
                    if next_01_match:
                        # Save current and start new
                        structures.append(
                            (
                                current_structure,
                                "\n".join(current_content),
                                structure_start,
                            )
                        )
                        current_structure = next_01_match.group(1)
                        current_content = [line]
                        structure_start = i
                    else:
                        # End of structure
                        break

        # Save last structure
        if current_structure:
            structures.append(
                (current_structure, "\n".join(current_content), structure_start)
            )

        return structures

    def _parse_single_structure(
        self,
        structure_name: str,
        structure_content: str,
        full_content: str,
        start_line: int,
    ) -> ChunkMetadata:
        """Parse a single 01-level structure"""

        # Extract all fields and their metadata
        fields_info = self._extract_field_information(structure_content)

        # Extract COBOL-specific metadata
        cobol_metadata = {
            "structure_level": "01",
            "structure_name": structure_name,
            "field_names": list(fields_info.keys()),
            "pic_clauses": self.extract_pic_clauses({"fields": fields_info}),
            "occurs_count": self._extract_occurs_info(structure_content),
            "redefines": self._extract_redefines_info(structure_content),
            "usage": self._extract_usage_info(structure_content),
            "values": self._extract_value_clauses(structure_content),
            "conditions": self._extract_88_conditions(structure_content),
            "field_count": len(fields_info),
            "max_level": self._get_max_level(structure_content),
            "hierarchy": self._build_field_hierarchy(fields_info),
        }

        # Generate artifact ID
        artifact_id = self.id_generator.generate_artifact_id(
            "", {"structure_name": structure_name}
        )

        # Generate content hash
        content_hash = self.hash_generator.generate_content_hash(
            structure_content, "cobol"
        )

        # Chunk the content
        chunk_results = self.chunker.chunk_content(structure_content, "cobol")
        content_text = chunk_results[0].content if chunk_results else structure_content
        content_tokens = chunk_results[0].token_count if chunk_results else None

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=content_text,
            content_tokens=content_tokens,
            content_hash=content_hash,
            source_uri="",  # Will be set by caller
            collected_at=datetime.now(timezone.utc),
            source_metadata=cobol_metadata,
        )

    def _extract_field_information(self, content: str) -> dict[str, dict[str, Any]]:
        """Extract field information from structure content"""
        fields = {}
        lines = content.split("\n")

        for line in lines:
            # Match level number and field name
            match = re.match(r"^\s*(\d{2})\s+([A-Z0-9_-]+)(?:\s+(.*?))?\.?\s*$", line)
            if match:
                level = match.group(1)
                field_name = match.group(2)
                rest = match.group(3) or ""

                fields[field_name] = {
                    "level": level,
                    "definition_line": line.strip(),
                    "pic": self._extract_pic_from_line(rest),
                    "occurs": self._extract_occurs_from_line(rest),
                    "redefines": self._extract_redefines_from_line(rest),
                    "usage": self._extract_usage_from_line(rest),
                    "value": self._extract_value_from_line(rest),
                }

        return fields

    def _extract_pic_from_line(self, line: str) -> Optional[str]:
        """Extract PIC clause from a line"""
        match = self.pic_pattern.search(line)
        return match.group(1) if match else None

    def _extract_occurs_from_line(self, line: str) -> Optional[int]:
        """Extract OCCURS count from a line"""
        match = self.occurs_pattern.search(line)
        return int(match.group(1)) if match else None

    def _extract_redefines_from_line(self, line: str) -> Optional[str]:
        """Extract REDEFINES field name from a line"""
        match = self.redefines_pattern.search(line)
        return match.group(1) if match else None

    def _extract_usage_from_line(self, line: str) -> Optional[str]:
        """Extract USAGE clause from a line"""
        # Check for COMP, COMP-3, BINARY, etc.
        if "COMP-3" in line.upper():
            return "COMP-3"
        elif "COMP" in line.upper():
            return "COMP"
        elif "BINARY" in line.upper():
            return "BINARY"
        elif "PACKED-DECIMAL" in line.upper():
            return "PACKED-DECIMAL"

        match = self.usage_pattern.search(line)
        return match.group(1) if match and match.group(1) else None

    def _extract_value_from_line(self, line: str) -> Optional[str]:
        """Extract VALUE clause from a line"""
        match = self.value_pattern.search(line)
        return match.group(1).strip(" \"'") if match else None

    def extract_pic_clauses(self, structure: dict[str, Any]) -> dict[str, str]:
        """Extract PIC clauses from COBOL structure"""
        pic_clauses = {}

        if "fields" in structure:
            for field_name, field_info in structure["fields"].items():
                if (
                    isinstance(field_info, dict)
                    and "pic" in field_info
                    and field_info["pic"]
                ):
                    pic_clauses[field_name] = field_info["pic"]

        return pic_clauses

    def _extract_occurs_info(self, content: str) -> dict[str, int]:
        """Extract OCCURS clause information"""
        occurs_info = {}
        lines = content.split("\n")

        for line in lines:
            field_match = re.match(r"^\s*\d{2}\s+([A-Z0-9_-]+)", line)
            occurs_match = self.occurs_pattern.search(line)

            if field_match and occurs_match:
                field_name = field_match.group(1)
                occurs_count = int(occurs_match.group(1))
                occurs_info[field_name] = occurs_count

        return occurs_info

    def _extract_redefines_info(self, content: str) -> dict[str, str]:
        """Extract REDEFINES clause information"""
        redefines_info = {}
        lines = content.split("\n")

        for line in lines:
            field_match = re.match(r"^\s*\d{2}\s+([A-Z0-9_-]+)", line)
            redefines_match = self.redefines_pattern.search(line)

            if field_match and redefines_match:
                field_name = field_match.group(1)
                redefines_field = redefines_match.group(1)
                redefines_info[field_name] = redefines_field

        return redefines_info

    def _extract_usage_info(self, content: str) -> dict[str, str]:
        """Extract USAGE clause information"""
        usage_info = {}
        lines = content.split("\n")

        for line in lines:
            field_match = re.match(r"^\s*\d{2}\s+([A-Z0-9_-]+)", line)
            if field_match:
                field_name = field_match.group(1)
                usage = self._extract_usage_from_line(line)
                if usage:
                    usage_info[field_name] = usage

        return usage_info

    def _extract_value_clauses(self, content: str) -> dict[str, str]:
        """Extract VALUE clauses from content"""
        value_info = {}
        lines = content.split("\n")

        for line in lines:
            field_match = re.match(r"^\s*\d{2}\s+([A-Z0-9_-]+)", line)
            if field_match:
                field_name = field_match.group(1)
                value = self._extract_value_from_line(line)
                if value:
                    value_info[field_name] = value

        return value_info

    def _extract_88_conditions(self, content: str) -> dict[str, list[str]]:
        """Extract 88-level condition names and values"""
        conditions = {}
        lines = content.split("\n")
        current_field = None

        for line in lines:
            # Check for parent field (not 88-level)
            field_match = re.match(r"^\s*(?:0[1-7]|[1-7][0-9])\s+([A-Z0-9_-]+)", line)
            if field_match:
                current_field = field_match.group(1)

            # Check for 88-level condition
            condition_match = self.condition_pattern.match(line)
            if condition_match and current_field:
                condition_name = condition_match.group(1)
                condition_value = condition_match.group(2).strip(" \"'.")

                if current_field not in conditions:
                    conditions[current_field] = []
                conditions[current_field].append(f"{condition_name}={condition_value}")

        return conditions

    def _get_max_level(self, content: str) -> int:
        """Get the maximum level number in the structure"""
        max_level = 1
        for match in re.finditer(r"^\s*(\d{2})\s+", content, re.MULTILINE):
            level = int(match.group(1))
            max_level = max(max_level, level)
        return max_level

    def _build_field_hierarchy(
        self, fields_info: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Build hierarchical representation of fields"""
        hierarchy = {}

        # Sort fields by level
        sorted_fields = sorted(fields_info.items(), key=lambda x: int(x[1]["level"]))

        level_stack = []

        for field_name, field_info in sorted_fields:
            level = int(field_info["level"])

            # Pop levels that are same or higher
            while level_stack and level_stack[-1]["level"] >= level:
                level_stack.pop()

            # Create field entry
            field_entry = {
                "name": field_name,
                "level": level,
                "pic": field_info.get("pic"),
                "children": {},
            }

            # Add to appropriate parent
            if not level_stack:
                # Top level
                hierarchy[field_name] = field_entry
            else:
                # Add to parent's children
                level_stack[-1]["children"][field_name] = field_entry

            # Push to stack for potential children
            level_stack.append(field_entry)

        return hierarchy

    def generate_artifact_id(self, source_path: str, **kwargs) -> str:
        """Generate deterministic artifact ID for COBOL structures"""
        structure_name = kwargs.get("structure_name", "main")
        return f"cobol://{source_path}/01-{structure_name}"
