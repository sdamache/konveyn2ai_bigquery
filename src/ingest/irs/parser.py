"""
IRS IMF Parser Implementation
T026: Implements IRS Individual Master File (IMF) fixed-width record layout parsing
"""

import re
import struct
import os
import time
import sys
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
from pathlib import Path

# Contract interfaces (standardized import)
from src.common.parser_interfaces import (
    IRSParser,
    BaseParser,
    SourceType,
    ChunkMetadata,
    ParseResult,
    ParseError,
    ErrorClass,
)

# Import common utilities
from src.common.chunking import ContentChunker, ChunkConfig, ChunkingStrategy
from src.common.ids import ArtifactIDGenerator, ContentHashGenerator
from src.common.normalize import ContentNormalizer


class IRSParserImpl(IRSParser):
    """Implementation of IRS IMF layout parser using regex and struct module"""

    def __init__(self, version: str = "1.0.0"):
        super().__init__(version)

        # Initialize utilities
        self.id_generator = ArtifactIDGenerator("irs")
        self.hash_generator = ContentHashGenerator(normalize_whitespace=True, ignore_comments=False)
        self.normalizer = ContentNormalizer(preserve_semantics=True, aggressive_whitespace=False)
        self.chunker = ContentChunker(ChunkConfig(
            max_tokens=1000,
            overlap_pct=0.15,
            strategy=ChunkingStrategy.FIXED_WIDTH
        ))

        # IRS-specific regex patterns
        self.patterns = {
            # Header patterns
            'layout_header': re.compile(r'IRS\s+.*Master\s+File.*Record\s+Layout', re.IGNORECASE),
            'record_type': re.compile(r'Record\s+Type:\s*(.+?)\(Type\s+(\d+)\)', re.IGNORECASE),
            'layout_version': re.compile(r'Layout\s+Version:\s*([^\s\n]+)', re.IGNORECASE),
            'tax_year': re.compile(r'Tax\s+Year\s+(\d{4})', re.IGNORECASE),

            # Field definition patterns - handles various formats:
            # 001-009  Social Security Number  PIC 9(09)  NUMERIC  Required  Section: IDENTITY
            # 001-009    9   N   Social Security Number
            'field_definition': re.compile(
                r'(\d{3})-(\d{3})\s+' +                    # Position range (001-009)
                r'([A-Za-z][A-Za-z0-9\s\(\)\/\-\.]+?)\s+' +  # Field name (must start with letter)
                r'PIC\s+([A-Z0-9\(\)V\-S]+)\s+' +        # PIC clause
                r'(NUMERIC|ALPHA|PACKED|FILLER|ALPHANUMERIC)\s+' + # Data type
                r'(Required|Optional)\s+' +               # Requirement flag
                r'Section:\s*([A-Z_][A-Z0-9_]*)',         # Section
                re.IGNORECASE | re.MULTILINE
            ),

            # Section headers
            'section_header': re.compile(r'^SECTION\s+([A-Z]):\s*([A-Z\s_]+)', re.IGNORECASE | re.MULTILINE),
            'field_definitions_start': re.compile(r'Field\s+(?:Positions\s+and\s+)?Definitions:', re.IGNORECASE),

            # Record length and other metadata
            'record_length': re.compile(r'RECORD\s+LENGTH:\s*(\d+)\s*bytes', re.IGNORECASE),
        }

    def _get_source_type(self) -> SourceType:
        """Return the IRS source type"""
        return SourceType.IRS

    def validate_content(self, content: str) -> bool:
        """Validate if content is a valid IRS IMF layout"""
        try:
            # Must contain IRS header pattern
            if not self.patterns['layout_header'].search(content):
                return False

            # Must have field definitions section
            if not self.patterns['field_definitions_start'].search(content):
                return False

            # Must have at least one field definition
            field_matches = self.patterns['field_definition'].findall(content)
            if not field_matches:
                return False

            # Validate field position format
            for match in field_matches:
                start_pos, end_pos = match[0], match[1]
                if not (start_pos.isdigit() and end_pos.isdigit()):
                    return False
                if int(start_pos) > int(end_pos):
                    return False

            return True

        except Exception:
            return False

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single IRS IMF layout file"""
        start_time = time.time()

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Validate content
            if not self.validate_content(content):
                error = ParseError(
                    source_type=SourceType.IRS,
                    source_uri=file_path,
                    error_class=ErrorClass.VALIDATION,
                    error_msg="Invalid IRS IMF layout format",
                    sample_text=content[:200] if content else "Empty file",
                    collected_at=datetime.utcnow()
                )
                return ParseResult(
                    chunks=[],
                    errors=[error],
                    files_processed=1,
                    processing_duration_ms=int((time.time() - start_time) * 1000)
                )

            # Parse layout content
            chunks = self.parse_imf_layout(content)

            # Set source URI for all chunks
            for chunk in chunks:
                chunk.source_uri = file_path

            duration_ms = max(1, int((time.time() - start_time) * 1000))
            return ParseResult(
                chunks=chunks,
                errors=[],
                files_processed=1,
                processing_duration_ms=duration_ms
            )

        except Exception as e:
            error = ParseError(
                source_type=SourceType.IRS,
                source_uri=file_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Error parsing file: {str(e)}",
                sample_text=None,
                stack_trace=str(e),
                collected_at=datetime.utcnow()
            )
            return ParseResult(
                chunks=[],
                errors=[error],
                files_processed=1,
                processing_duration_ms=int((time.time() - start_time) * 1000)
            )

    def parse_directory(self, directory_path: str) -> ParseResult:
        """Parse all IRS layout files in a directory"""
        start_time = time.time()

        all_chunks = []
        all_errors = []
        files_processed = 0

        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                error = ParseError(
                    source_type=SourceType.IRS,
                    source_uri=directory_path,
                    error_class=ErrorClass.PARSING,
                    error_msg="Directory does not exist or is not a directory",
                    collected_at=datetime.utcnow()
                )
                return ParseResult(
                    chunks=[],
                    errors=[error],
                    files_processed=0,
                    processing_duration_ms=int((time.time() - start_time) * 1000)
                )

            # Find all text files that might contain IRS layouts
            pattern_extensions = ['*.txt', '*.dat', '*.layout', '*.imf']
            irs_files = []

            for pattern in pattern_extensions:
                irs_files.extend(directory.glob(pattern))

            # Parse each file
            for file_path in irs_files:
                if file_path.is_file():
                    result = self.parse_file(str(file_path))
                    all_chunks.extend(result.chunks)
                    all_errors.extend(result.errors)
                    files_processed += result.files_processed

            return ParseResult(
                chunks=all_chunks,
                errors=all_errors,
                files_processed=files_processed,
                processing_duration_ms=int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            error = ParseError(
                source_type=SourceType.IRS,
                source_uri=directory_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Error parsing directory: {str(e)}",
                stack_trace=str(e),
                collected_at=datetime.utcnow()
            )
            return ParseResult(
                chunks=all_chunks,
                errors=[error],
                files_processed=files_processed,
                processing_duration_ms=int((time.time() - start_time) * 1000)
            )

    def parse_imf_layout(self, layout_content: str) -> List[ChunkMetadata]:
        """Parse IRS IMF fixed-width record layout"""
        # Validate content first - raise ValueError if invalid
        if not self.validate_content(layout_content):
            raise ValueError("Invalid IRS IMF layout format")

        try:
            # Extract metadata from header
            metadata = self._extract_layout_metadata(layout_content)

            # Extract field positions
            field_positions = self.extract_field_positions({'layout_content': layout_content})

            # Validate we got field positions
            if not field_positions:
                raise ValueError("No valid field positions found in layout")

            # Group fields by section
            sections = self._group_fields_by_section(field_positions)

            # Create chunks for each section
            chunks = []
            for section_name, section_fields in sections.items():
                chunk = self._create_section_chunk(
                    section_name=section_name,
                    fields=section_fields,
                    layout_metadata=metadata,
                    original_content=layout_content
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            # If parsing fails, raise ValueError
            raise ValueError(f"Failed to parse IRS IMF layout: {str(e)}")

    def extract_field_positions(self, layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract field positions and lengths from IRS layout"""
        layout_content = layout.get('layout_content', '')
        field_positions = []

        # Find all field definitions
        field_matches = self.patterns['field_definition'].findall(layout_content)

        for match in field_matches:
            start_pos_str, end_pos_str, field_name, pic_clause, data_type_long, required, section = match

            # Parse positions
            start_position = int(start_pos_str)
            end_position = int(end_pos_str)
            calculated_length = end_position - start_position + 1

            # Use calculated length
            field_length = calculated_length

            # Determine data type
            data_type = self._determine_data_type(None, data_type_long, pic_clause)

            # Clean field name
            field_name = field_name.strip()

            # Determine section (use provided or infer)
            section_name = section.strip() if section else self._infer_section_from_field(field_name)

            field_info = {
                'start_position': start_position,
                'end_position': end_position,
                'field_length': field_length,
                'field_name': field_name,
                'data_type': data_type,
                'section': section_name,
                'required': required == 'Required' if required else True,
                'pic_clause': pic_clause.strip() if pic_clause else None
            }

            field_positions.append(field_info)

        # Sort by start position
        field_positions.sort(key=lambda x: x['start_position'])

        return field_positions

    def _extract_layout_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from IRS layout header"""
        metadata = {}

        # Extract record type - REQUIRED
        record_type_match = self.patterns['record_type'].search(content)
        if record_type_match:
            metadata['record_type'] = record_type_match.group(2).strip()
            metadata['record_description'] = record_type_match.group(1).strip()
        else:
            raise ValueError("Missing required Record Type in IRS layout")

        # Extract layout version - REQUIRED (must be explicit)
        version_match = self.patterns['layout_version'].search(content)
        if version_match:
            metadata['layout_version'] = version_match.group(1).strip()
        else:
            raise ValueError("Missing required Layout Version in IRS layout")

        # Extract record length (optional)
        length_match = self.patterns['record_length'].search(content)
        if length_match:
            metadata['record_length'] = int(length_match.group(1))

        return metadata

    def _determine_data_type(self, short_type: str, long_type: str, pic_clause: str) -> str:
        """Determine field data type from various indicators"""
        # Priority: short_type > long_type > pic_clause analysis

        if short_type:
            type_map = {
                'N': 'NUMERIC',
                'A': 'ALPHA',
                'X': 'ALPHANUMERIC'
            }
            return type_map.get(short_type.upper(), 'ALPHANUMERIC')

        if long_type:
            return long_type.upper()

        if pic_clause:
            # Analyze PIC clause
            pic_upper = pic_clause.upper()
            if '9' in pic_upper or 'S9' in pic_upper:
                return 'NUMERIC'
            elif 'X' in pic_upper:
                return 'ALPHANUMERIC'
            elif 'A' in pic_upper:
                return 'ALPHA'

        # Default
        return 'ALPHANUMERIC'

    def _infer_section_from_field(self, field_name: str) -> str:
        """Infer section name from field name"""
        field_lower = field_name.lower()

        # Common section patterns
        if any(term in field_lower for term in ['ssn', 'social security', 'name', 'taxpayer']):
            return 'IDENTITY'
        elif any(term in field_lower for term in ['tax', 'income', 'liability', 'withh', 'agi']):
            return 'TAX_INFO'
        elif any(term in field_lower for term in ['address', 'city', 'state', 'zip']):
            return 'ADDRESS'
        elif any(term in field_lower for term in ['filing', 'status', 'exemption']):
            return 'STATUS'
        elif any(term in field_lower for term in ['process', 'date', 'transaction', 'cycle']):
            return 'PROCESSING'
        elif any(term in field_lower for term in ['business', 'entity', 'ein', 'employer']):
            return 'BUSINESS'
        elif any(term in field_lower for term in ['reserved', 'future', 'filler']):
            return 'RESERVED'
        else:
            return 'GENERAL'

    def _group_fields_by_section(self, field_positions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group field positions by section"""
        sections = {}

        for field in field_positions:
            section = field['section']
            if section not in sections:
                sections[section] = []
            sections[section].append(field)

        return sections

    def _create_section_chunk(self, section_name: str, fields: List[Dict[str, Any]],
                             layout_metadata: Dict[str, Any], original_content: str) -> ChunkMetadata:
        """Create a chunk for a section of fields"""

        # Create content text for this section
        content_lines = [f"Section: {section_name}"]
        content_lines.append("")

        for field in fields:
            line = f"{field['start_position']:03d}-{field['end_position']:03d} "
            line += f"{field['field_length']:3d} "
            line += f"{field['data_type']:12s} "
            line += f"{field['field_name']}"
            if field.get('pic_clause'):
                line += f" PIC {field['pic_clause']}"
            if not field.get('required', True):
                line += " (Optional)"
            content_lines.append(line)

        content_text = '\n'.join(content_lines)

        # Generate artifact ID
        artifact_id = self.id_generator.generate_artifact_id(
            "",
            {
                "record_type": layout_metadata.get('record_type', '01'),
                "layout_version": layout_metadata.get('layout_version', '2024.1'),
                "section": section_name
            }
        )

        # Generate content hash
        content_hash = self.hash_generator.generate_content_hash(content_text, "irs")

        # Calculate tokens
        content_tokens = self.chunker._estimate_tokens(content_text)

        # Prepare source metadata
        source_metadata = {
            'record_type': layout_metadata.get('record_type', '01'),
            'layout_version': layout_metadata.get('layout_version', '2024.1'),
            'section': section_name,
            'field_count': len(fields),
            'start_position': min(f['start_position'] for f in fields) if fields else 0,
            'field_length': sum(f['field_length'] for f in fields),
            'data_type': 'MIXED' if len(set(f['data_type'] for f in fields)) > 1 else fields[0]['data_type'] if fields else 'UNKNOWN',
            'fields': fields
        }

        return ChunkMetadata(
            source_type=SourceType.IRS,
            artifact_id=artifact_id,
            content_text=content_text,
            content_tokens=content_tokens,
            content_hash=content_hash,
            source_uri="",  # Will be set by parse_file
            collected_at=datetime.utcnow(),
            source_metadata=source_metadata
        )

    def generate_artifact_id(self, source_path: str, **kwargs) -> str:
        """Generate IRS-specific artifact ID"""
        record_type = kwargs.get('record_type', '01')
        layout_version = kwargs.get('layout_version', '2024.1')
        section = kwargs.get('section', 'main')

        return f"irs://{record_type}/{layout_version}/{section}"
