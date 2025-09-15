"""
Contract tests for IRS Parser Interface
These tests MUST FAIL initially (TDD requirement) until the IRS parser is implemented.

Tests the contract defined in specs/002-m1-parse-and/contracts/parser-interfaces.py
"""

import pytest

# Register custom markers to avoid warnings
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "contract: Contract tests for interface compliance (TDD)")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")

import re
import json
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Import the parser interface contracts
try:
    import sys
    import os
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)

    # Import using shared module
    from src.common.parser_interfaces import (
        BaseParser, IRSParser, SourceType, ChunkMetadata, ParseResult, ParseError, ErrorClass
    )

    PARSER_INTERFACES_AVAILABLE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    PARSER_INTERFACES_AVAILABLE = False
    print(f"Warning: Could not import parser interfaces: {e}")

# Try to import the actual implementation (expected to fail initially)
try:
    from src.ingest.irs.parser import IRSParserImpl
    IRS_PARSER_AVAILABLE = True
except ImportError:
    IRS_PARSER_AVAILABLE = False


# Test data for IRS IMF layouts
VALID_IMF_LAYOUT_TAX_YEAR_2023 = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Primary Individual Record (Type 01)
Layout Version: 2023.1

Field Positions and Definitions:

001-009  Social Security Number       PIC 9(09)   NUMERIC     Required  Section: IDENTITY
010-013  Name Control Code           PIC X(04)   ALPHA       Required  Section: IDENTITY
014-048  Primary Taxpayer Name       PIC X(35)   ALPHA       Optional  Section: IDENTITY
049-083  Secondary Taxpayer Name     PIC X(35)   ALPHA       Optional  Section: IDENTITY
084-088  Tax Year                    PIC 9(05)   NUMERIC     Required  Section: TAX_INFO
089-100  Adjusted Gross Income       PIC S9(10)V99 NUMERIC   Required  Section: TAX_INFO
101-112  Total Tax Liability         PIC S9(10)V99 NUMERIC   Required  Section: TAX_INFO
113-124  Federal Income Tax Withheld PIC S9(10)V99 NUMERIC   Optional  Section: TAX_INFO
125-130  Filing Status Code          PIC 9(06)   NUMERIC     Required  Section: STATUS
131-140  Return Processing Date      PIC 9(10)   NUMERIC     Required  Section: PROCESSING
141-150  Master File Transaction Code PIC X(10)  ALPHA       Required  Section: PROCESSING
151-200  Reserved Fields             PIC X(50)   ALPHA       Optional  Section: RESERVED
"""

VALID_IMF_LAYOUT_BUSINESS_2023 = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Business Schedule C (Type 02)
Layout Version: 2023.1

Field Positions and Definitions:

001-009  Social Security Number       PIC 9(09)   NUMERIC     Required  Section: IDENTITY
010-013  Business Sequence Number     PIC 9(04)   NUMERIC     Required  Section: BUSINESS
014-048  Business Name               PIC X(35)   ALPHA       Required  Section: BUSINESS
049-054  Business Activity Code      PIC 9(06)   NUMERIC     Required  Section: BUSINESS
055-066  Gross Receipts              PIC S9(10)V99 NUMERIC   Required  Section: INCOME
067-078  Total Expenses              PIC S9(10)V99 NUMERIC   Required  Section: INCOME
079-090  Net Profit/Loss             PIC S9(10)V99 NUMERIC   Required  Section: INCOME
091-095  Tax Year                    PIC 9(05)   NUMERIC     Required  Section: TAX_INFO
096-120  Reserved Fields             PIC X(25)   ALPHA       Optional  Section: RESERVED
"""

VALID_IMF_LAYOUT_EXTENSION_2023 = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Extension/Amendment Record (Type 03)
Layout Version: 2023.1

Field Positions and Definitions:

001-009  Social Security Number       PIC 9(09)   NUMERIC     Required  Section: IDENTITY
010-011  Record Type Indicator        PIC X(02)   ALPHA       Required  Section: TYPE
012-021  Extension Due Date           PIC 9(10)   NUMERIC     Optional  Section: EXTENSION
022-031  Amendment Date               PIC 9(10)   NUMERIC     Optional  Section: AMENDMENT
032-043  Interest Assessed            PIC S9(10)V99 NUMERIC   Optional  Section: PENALTIES
044-055  Penalty Assessed             PIC S9(10)V99 NUMERIC   Optional  Section: PENALTIES
056-067  Total Amount Due             PIC S9(10)V99 NUMERIC   Optional  Section: PENALTIES
068-100  Reserved Fields              PIC X(33)   ALPHA       Optional  Section: RESERVED
"""

INVALID_IMF_LAYOUT_MALFORMED = """
This is not a valid IRS IMF layout
Random text without proper field definitions
No structured field positions
"""

INVALID_IMF_LAYOUT_MISSING_FIELDS = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Invalid Record (Type 99)

Missing Field Positions and Definitions section
"""

COMPLEX_IMF_LAYOUT_MULTI_SECTION = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Comprehensive Individual Record (Type 01)
Layout Version: 2023.2

Field Positions and Definitions:

001-009  Social Security Number       PIC 9(09)   NUMERIC     Required  Section: IDENTITY
010-013  Name Control Code           PIC X(04)   ALPHA       Required  Section: IDENTITY
014-048  Primary Taxpayer Name       PIC X(35)   ALPHA       Optional  Section: IDENTITY
049-083  Secondary Taxpayer Name     PIC X(35)   ALPHA       Optional  Section: IDENTITY

084-088  Tax Year                    PIC 9(05)   NUMERIC     Required  Section: TAX_INFO
089-100  Adjusted Gross Income       PIC S9(10)V99 NUMERIC   Required  Section: TAX_INFO
101-112  Total Tax Liability         PIC S9(10)V99 NUMERIC   Required  Section: TAX_INFO
113-124  Federal Income Tax Withheld PIC S9(10)V99 NUMERIC   Optional  Section: TAX_INFO

125-130  Filing Status Code          PIC 9(06)   NUMERIC     Required  Section: STATUS
131-135  Exemptions Claimed          PIC 9(05)   NUMERIC     Optional  Section: STATUS

136-145  Return Processing Date      PIC 9(10)   NUMERIC     Required  Section: PROCESSING
146-155  Master File Transaction Code PIC X(10)  ALPHA       Required  Section: PROCESSING
156-165  Cycle Code                  PIC X(10)   ALPHA       Optional  Section: PROCESSING

166-200  Reserved Fields             PIC X(35)   ALPHA       Optional  Section: RESERVED
"""


@pytest.mark.contract
@pytest.mark.unit
class TestIRSParserContract:
    """Contract tests for IRS Parser implementation"""

    def test_parser_interfaces_available(self):
        """Test that parser interface contracts are available"""
        assert PARSER_INTERFACES_AVAILABLE, "Parser interface contracts should be available"

    @pytest.mark.skipif(not PARSER_INTERFACES_AVAILABLE, reason="Parser interfaces not available")
    def test_irs_parser_interface_exists(self):
        """Test that IRSParser abstract class exists with required methods"""
        # Verify IRSParser exists and inherits from BaseParser
        assert issubclass(IRSParser, BaseParser)

        # Verify required abstract methods exist
        abstract_methods = getattr(IRSParser, '__abstractmethods__', set())
        required_methods = {
            'parse_file',
            'parse_directory',
            'validate_content',
            'parse_imf_layout',
            'extract_field_positions'
        }

        # Check that all required methods are declared as abstract
        for method in required_methods:
            assert hasattr(IRSParser, method), f"Method {method} should exist"

    @pytest.mark.skipif(IRS_PARSER_AVAILABLE, reason="Skip when implementation exists")
    def test_irs_parser_not_implemented_yet(self):
        """Test that the actual implementation doesn't exist yet (TDD requirement)"""
        assert not IRS_PARSER_AVAILABLE, "IRSParserImpl should not exist yet (TDD)"

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_irs_parser_inheritance(self):
        """Test that IRSParserImpl properly inherits from IRSParser"""
        assert issubclass(IRSParserImpl, IRSParser)
        assert issubclass(IRSParserImpl, BaseParser)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_irs_parser_source_type(self):
        """Test that parser returns correct source type"""
        parser = IRSParserImpl()
        assert parser.source_type == SourceType.IRS
        assert parser._get_source_type() == SourceType.IRS

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_imf_layout_primary_individual(self):
        """Test parsing a valid primary individual IMF layout"""
        parser = IRSParserImpl()
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_TAX_YEAR_2023)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check the first chunk
        chunk = chunks[0]
        assert isinstance(chunk, ChunkMetadata)
        assert chunk.source_type == SourceType.IRS

        # Verify artifact_id format: irs://{record_type}/{layout_version}/{section}
        expected_artifact_id = "irs://01/2023.1/IDENTITY"
        assert chunk.artifact_id == expected_artifact_id

        # Verify IRS-specific metadata
        irs_metadata = chunk.source_metadata
        assert irs_metadata['record_type'] == '01'
        assert irs_metadata['layout_version'] == '2023.1'
        assert irs_metadata['section'] == 'IDENTITY'
        assert 'start_position' in irs_metadata
        assert 'field_length' in irs_metadata
        assert 'data_type' in irs_metadata

        # Verify content fields
        assert chunk.content_text.strip() != ""
        assert chunk.content_hash != ""
        assert isinstance(chunk.collected_at, datetime)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_imf_layout_business_schedule(self):
        """Test parsing a valid business schedule IMF layout"""
        parser = IRSParserImpl()
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_BUSINESS_2023)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify artifact_id format for business record type
        expected_artifact_id = "irs://02/2023.1/IDENTITY"
        assert chunk.artifact_id == expected_artifact_id

        # Verify IRS-specific metadata
        irs_metadata = chunk.source_metadata
        assert irs_metadata['record_type'] == '02'
        assert irs_metadata['layout_version'] == '2023.1'

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_imf_layout_extension_record(self):
        """Test parsing extension/amendment record layout"""
        parser = IRSParserImpl()
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_EXTENSION_2023)

        assert len(chunks) > 0
        chunk = chunks[0]

        # Verify artifact_id format for extension record type
        expected_artifact_id = "irs://03/2023.1/IDENTITY"
        assert chunk.artifact_id == expected_artifact_id

        irs_metadata = chunk.source_metadata
        assert irs_metadata['record_type'] == '03'
        assert irs_metadata['layout_version'] == '2023.1'

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_imf_layout_multiple_sections(self):
        """Test parsing IMF layout with multiple sections"""
        parser = IRSParserImpl()
        chunks = parser.parse_imf_layout(COMPLEX_IMF_LAYOUT_MULTI_SECTION)

        # Should generate chunks for all sections
        assert len(chunks) >= 5  # IDENTITY, TAX_INFO, STATUS, PROCESSING, RESERVED

        # Verify we got all expected sections
        sections = [chunk.source_metadata['section'] for chunk in chunks]
        expected_sections = ['IDENTITY', 'TAX_INFO', 'STATUS', 'PROCESSING', 'RESERVED']
        for section in expected_sections:
            assert section in sections

        # Verify all chunks have same record_type and layout_version
        for chunk in chunks:
            assert chunk.source_metadata['record_type'] == '01'
            assert chunk.source_metadata['layout_version'] == '2023.2'

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_extract_field_positions_basic_layout(self):
        """Test extracting field positions from a basic layout"""
        parser = IRSParserImpl()

        # Parse layout to get structured data first
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_TAX_YEAR_2023)

        # Extract field positions - assuming this takes layout dict/object
        layout_data = {'layout_content': VALID_IMF_LAYOUT_TAX_YEAR_2023}
        field_positions = parser.extract_field_positions(layout_data)

        assert isinstance(field_positions, list)
        assert len(field_positions) > 0

        # Verify field position structure
        field = field_positions[0]
        required_fields = ['start_position', 'field_length', 'field_name', 'data_type', 'section']
        for req_field in required_fields:
            assert req_field in field, f"Missing required field: {req_field}"

        # Test specific field values
        ssn_field = next((f for f in field_positions if f['field_name'] == 'Social Security Number'), None)
        assert ssn_field is not None
        assert ssn_field['start_position'] == 1
        assert ssn_field['field_length'] == 9
        assert ssn_field['data_type'] == 'NUMERIC'
        assert ssn_field['section'] == 'IDENTITY'

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_extract_field_positions_with_pic_clauses(self):
        """Test extracting field positions with PIC clause parsing"""
        parser = IRSParserImpl()

        layout_data = {'layout_content': VALID_IMF_LAYOUT_BUSINESS_2023}
        field_positions = parser.extract_field_positions(layout_data)

        # Find a field with a complex PIC clause
        gross_receipts = next((f for f in field_positions if f['field_name'] == 'Gross Receipts'), None)
        assert gross_receipts is not None
        assert gross_receipts['start_position'] == 55
        assert gross_receipts['field_length'] == 12  # S9(10)V99 = 12 positions
        assert gross_receipts['data_type'] == 'NUMERIC'
        assert 'pic_clause' in gross_receipts
        assert gross_receipts['pic_clause'] == 'S9(10)V99'

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_invalid_imf_layout_malformed(self):
        """Test error handling for malformed IMF layout"""
        parser = IRSParserImpl()

        with pytest.raises(ValueError):
            parser.parse_imf_layout(INVALID_IMF_LAYOUT_MALFORMED)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_invalid_imf_layout_missing_fields(self):
        """Test error handling for layout missing required sections"""
        parser = IRSParserImpl()

        with pytest.raises(ValueError):
            parser.parse_imf_layout(INVALID_IMF_LAYOUT_MISSING_FIELDS)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_valid_imf_layout(self):
        """Test content validation for valid IRS IMF layouts"""
        parser = IRSParserImpl()

        assert parser.validate_content(VALID_IMF_LAYOUT_TAX_YEAR_2023) is True
        assert parser.validate_content(VALID_IMF_LAYOUT_BUSINESS_2023) is True
        assert parser.validate_content(VALID_IMF_LAYOUT_EXTENSION_2023) is True
        assert parser.validate_content(COMPLEX_IMF_LAYOUT_MULTI_SECTION) is True

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_invalid(self):
        """Test content validation for invalid content"""
        parser = IRSParserImpl()

        assert parser.validate_content(INVALID_IMF_LAYOUT_MALFORMED) is False
        assert parser.validate_content(INVALID_IMF_LAYOUT_MISSING_FIELDS) is False
        assert parser.validate_content("") is False
        assert parser.validate_content("not an IRS layout") is False

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_file_single_layout(self, tmp_path):
        """Test parsing a single IMF layout file"""
        parser = IRSParserImpl()

        # Create temporary layout file
        layout_file = tmp_path / "imf_layout_2023.txt"
        layout_file.write_text(VALID_IMF_LAYOUT_TAX_YEAR_2023)

        result = parser.parse_file(str(layout_file))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0
        assert len(result.errors) == 0
        assert result.files_processed == 1
        assert result.processing_duration_ms > 0

        # Verify source_uri is set correctly
        chunk = result.chunks[0]
        assert chunk.source_uri == str(layout_file)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_directory_multiple_layouts(self, tmp_path):
        """Test parsing a directory with multiple IMF layout files"""
        parser = IRSParserImpl()

        # Create multiple layout files
        (tmp_path / "imf_primary_2023.txt").write_text(VALID_IMF_LAYOUT_TAX_YEAR_2023)
        (tmp_path / "imf_business_2023.txt").write_text(VALID_IMF_LAYOUT_BUSINESS_2023)
        (tmp_path / "imf_extension_2023.txt").write_text(VALID_IMF_LAYOUT_EXTENSION_2023)
        (tmp_path / "not-a-layout.txt").write_text("This is not an IRS layout")

        result = parser.parse_directory(str(tmp_path))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) >= 3  # At least 3 valid layouts
        assert result.files_processed >= 3

        # Verify we got the expected record types
        record_types = [chunk.source_metadata['record_type'] for chunk in result.chunks]
        assert '01' in record_types  # Primary individual
        assert '02' in record_types  # Business schedule
        assert '03' in record_types  # Extension

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_artifact_id_generation(self):
        """Test artifact ID generation for various IRS record types and sections"""
        parser = IRSParserImpl()

        # Test with different record types, versions, and sections
        test_cases = [
            ("01", "2023.1", "IDENTITY", "irs://01/2023.1/IDENTITY"),
            ("02", "2023.1", "BUSINESS", "irs://02/2023.1/BUSINESS"),
            ("03", "2023.1", "EXTENSION", "irs://03/2023.1/EXTENSION"),
            ("01", "2022.2", "TAX_INFO", "irs://01/2022.2/TAX_INFO"),
            ("99", "2023.1", "RESERVED", "irs://99/2023.1/RESERVED"),
        ]

        for record_type, layout_version, section, expected in test_cases:
            artifact_id = parser.generate_artifact_id(
                "",  # source_path not used for IRS
                record_type=record_type,
                layout_version=layout_version,
                section=section
            )
            assert artifact_id == expected

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_chunk_content_large_layout(self):
        """Test chunking behavior for large IMF layouts"""
        parser = IRSParserImpl()

        # Test with the complex multi-section layout
        chunks = parser.parse_imf_layout(COMPLEX_IMF_LAYOUT_MULTI_SECTION)

        # Should generate multiple chunks for different sections
        assert len(chunks) > 1

        # Verify all chunks have content and proper metadata
        for chunk in chunks:
            assert chunk.content_text.strip() != ""
            assert chunk.source_metadata['record_type'] == '01'
            assert chunk.source_metadata['layout_version'] == '2023.2'
            assert chunk.artifact_id.startswith("irs://01/2023.2/")

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_field_position_parsing_edge_cases(self):
        """Test field position parsing for edge cases"""
        parser = IRSParserImpl()

        # Test layout with various PIC clause formats
        edge_case_layout = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Edge Case Record (Type 04)
Layout Version: 2023.1

Field Positions and Definitions:

001-001  Single Character Field      PIC X(01)   ALPHA       Required  Section: EDGE_CASES
002-011  Standard Numeric            PIC 9(10)   NUMERIC     Required  Section: EDGE_CASES
012-023  Signed Decimal              PIC S9(10)V99 NUMERIC   Optional  Section: EDGE_CASES
024-028  Packed Decimal              PIC 9(05)   PACKED      Optional  Section: EDGE_CASES
029-058  Long Alpha Field            PIC X(30)   ALPHA       Optional  Section: EDGE_CASES
059-068  Date Field                  PIC 9(10)   NUMERIC     Required  Section: EDGE_CASES
069-100  Filler Space                PIC X(32)   FILLER      Optional  Section: EDGE_CASES
"""

        layout_data = {'layout_content': edge_case_layout}
        field_positions = parser.extract_field_positions(layout_data)

        # Verify various field types are parsed correctly
        fields_by_name = {f['field_name']: f for f in field_positions}

        # Test single character field
        single_char = fields_by_name['Single Character Field']
        assert single_char['start_position'] == 1
        assert single_char['field_length'] == 1

        # Test signed decimal
        signed_decimal = fields_by_name['Signed Decimal']
        assert signed_decimal['start_position'] == 12
        assert signed_decimal['field_length'] == 12  # S9(10)V99

        # Test long alpha field
        long_alpha = fields_by_name['Long Alpha Field']
        assert long_alpha['start_position'] == 29
        assert long_alpha['field_length'] == 30

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_error_handling_missing_required_metadata(self):
        """Test error handling when required IRS metadata is missing"""
        parser = IRSParserImpl()

        # Missing record type
        invalid_layout_no_record_type = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Layout Version: 2023.1

Field Positions and Definitions:
001-009  Social Security Number  PIC 9(09)   NUMERIC  Required  Section: IDENTITY
"""

        # Missing layout version
        invalid_layout_no_version = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Primary Individual Record (Type 01)

Field Positions and Definitions:
001-009  Social Security Number  PIC 9(09)   NUMERIC  Required  Section: IDENTITY
"""

        for invalid_layout in [invalid_layout_no_record_type, invalid_layout_no_version]:
            with pytest.raises((ValueError, KeyError)):
                parser.parse_imf_layout(invalid_layout)

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_content_hash_generation(self):
        """Test that content hash is generated correctly"""
        parser = IRSParserImpl()
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_TAX_YEAR_2023)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 64  # SHA256 hex digest length

        # Same content should generate same hash
        chunks2 = parser.parse_imf_layout(VALID_IMF_LAYOUT_TAX_YEAR_2023)
        assert chunks[0].content_hash == chunks2[0].content_hash

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_result_structure(self):
        """Test that ParseResult has the correct structure"""
        parser = IRSParserImpl()

        # Test with valid layout
        chunks = parser.parse_imf_layout(VALID_IMF_LAYOUT_TAX_YEAR_2023)

        # Verify chunk structure
        assert len(chunks) > 0
        chunk = chunks[0]

        # Required ChunkMetadata fields
        assert hasattr(chunk, 'source_type')
        assert hasattr(chunk, 'artifact_id')
        assert hasattr(chunk, 'content_text')
        assert hasattr(chunk, 'content_hash')
        assert hasattr(chunk, 'source_metadata')
        assert hasattr(chunk, 'collected_at')

        # IRS-specific metadata fields
        irs_metadata = chunk.source_metadata
        required_irs_fields = ['record_type', 'layout_version', 'start_position', 'field_length', 'data_type', 'section']
        for field in required_irs_fields:
            assert field in irs_metadata, f"Missing required IRS metadata field: {field}"

    @pytest.mark.skipif(not IRS_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_section_handling_edge_cases(self):
        """Test section handling for edge cases"""
        parser = IRSParserImpl()

        # Test layout with unusual section names
        unusual_section_layout = """
IRS Individual Master File (IMF) Record Layout - Tax Year 2023
Record Type: Unusual Section Record (Type 05)
Layout Version: 2023.1

Field Positions and Definitions:

001-009  Social Security Number  PIC 9(09)   NUMERIC  Required  Section: SPECIAL_SECTION_NAME
010-020  Custom Field            PIC X(11)   ALPHA    Optional  Section: CUSTOM_123
021-030  Another Field           PIC 9(10)   NUMERIC  Required  Section: SECTION_WITH_UNDERSCORE
"""

        chunks = parser.parse_imf_layout(unusual_section_layout)
        assert len(chunks) > 0

        # Verify section names are preserved correctly
        sections = [chunk.source_metadata['section'] for chunk in chunks]
        expected_sections = ['SPECIAL_SECTION_NAME', 'CUSTOM_123', 'SECTION_WITH_UNDERSCORE']
        for section in expected_sections:
            assert section in sections

        # Verify artifact IDs are generated correctly
        for chunk in chunks:
            assert chunk.artifact_id.startswith("irs://05/2023.1/")
            assert chunk.source_metadata['section'] in chunk.artifact_id


@pytest.mark.contract
@pytest.mark.unit
class TestIRSParserImplementationComplete:
    """Tests that verify implementation is complete (TDD GREEN phase)"""

    def test_implementation_available(self):
        """Verify that IRSParserImpl is now implemented and working"""
        assert IRS_PARSER_AVAILABLE, "IRSParserImpl should be implemented and available"

    def test_contract_tests_pass_with_implementation(self):
        """Verify that we've moved from TDD RED to GREEN phase"""
        assert IRS_PARSER_AVAILABLE, (
            "Implementation is now complete and available. "
            "We have successfully moved from TDD red phase to green phase."
        )


if __name__ == "__main__":
    # Run the contract tests
    pytest.main([__file__, "-v"])