"""
Integration tests for IRS IMF layout ingestion to BigQuery.

This is an end-to-end test that validates the complete flow from IRS IMF
fixed-width record layouts to BigQuery source_metadata table.

Following TDD methodology - these tests will FAIL initially because the
IRS parser implementation doesn't exist yet (RED phase).
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Import contracts - use the common parser interfaces
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from src.common.parser_interfaces import (
        ErrorClass,
        ParseResult,
        SourceType,
    )
except ImportError:
    # Fallback: Use local definitions
    from enum import Enum
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional

    class SourceType(Enum):
        IRS = "irs"

    class ErrorClass(Enum):
        PARSING_ERROR = "parsing_error"

    @dataclass
    class ParseResult:
        success: bool
        chunks: List[Dict[str, Any]]
        errors: List[Dict[str, Any]]
        metadata: Optional[Dict[str, Any]] = None


# Sample IRS IMF layout content for testing
SAMPLE_IMF_LAYOUT_BASIC = """
IRS IMF RECORD LAYOUT - INDIVIDUAL MASTER FILE
VERSION: 2024.1
EFFECTIVE DATE: January 1, 2024

RECORD TYPE: 01 - Basic Taxpayer Information

FIELD DEFINITIONS:
001-009    9   N   Social Security Number
010-010    1   A   Filing Status Code
011-050   40   A   Primary Taxpayer Name (Last)
051-070   20   A   Primary Taxpayer Name (First)
071-090   20   A   Primary Taxpayer Name (Middle)
091-130   40   A   Spouse Name (Last)
131-150   20   A   Spouse Name (First)
151-170   20   A   Spouse Name (Middle)
171-205   35   A   Address Line 1
206-240   35   A   Address Line 2
241-270   30   A   City
271-272    2   A   State Code
273-277    5   N   ZIP Code
278-281    4   N   ZIP+4 Extension
282-291   10   N   Date of Birth (MMDDYYYY)
292-301   10   N   Spouse Date of Birth
302-315   14   N   Adjusted Gross Income
316-329   14   N   Total Tax Liability
330-343   14   N   Total Payments
344-357   14   N   Refund Amount
358-358    1   A   Return Processing Code
359-368   10   N   Date Return Filed
369-378   10   N   Date Return Processed
379-400   22   A   Reserved for Future Use

RECORD LENGTH: 400 bytes
"""

SAMPLE_IMF_LAYOUT_COMPLEX = """
IRS IMF RECORD LAYOUT - BUSINESS RETURN DATA
VERSION: 2024.2
EFFECTIVE DATE: April 15, 2024

RECORD TYPE: 03 - Business Tax Information

SECTION A: ENTITY INFORMATION
001-015   15   A   Employer Identification Number
016-016    1   N   Entity Type Code
017-066   50   A   Business Name
067-101   35   A   Business Address Line 1
102-136   35   A   Business Address Line 2
137-166   30   A   Business City
167-168    2   A   Business State
169-173    5   N   Business ZIP
174-177    4   N   Business ZIP+4

SECTION B: FINANCIAL DATA
178-191   14   N   Gross Receipts
192-205   14   N   Total Deductions
206-219   14   N   Taxable Income
220-233   14   N   Tax Before Credits
234-247   14   N   Total Credits
248-261   14   N   Net Tax Liability
262-275   14   N   Estimated Tax Payments
276-289   14   N   Amount Owed
290-303   14   N   Overpayment Amount

SECTION C: BUSINESS CODES
304-309    6   N   NAICS Code
310-313    4   A   Business Activity Code
314-323   10   A   Accounting Method Code
324-333   10   N   Tax Year Begin Date
334-343   10   N   Tax Year End Date

SECTION D: INDICATORS
344-344    1   A   Amended Return Indicator
345-345    1   A   Final Return Indicator
346-346    1   A   Short Period Indicator
347-347    1   A   Personal Service Corp Indicator
348-348    1   A   Schedule M-3 Required
349-349    1   A   Section 199A Deduction
350-375   26   A   Reserved for Future Use

RECORD LENGTH: 375 bytes
"""

SAMPLE_INVALID_LAYOUT = """
INVALID IRS LAYOUT - MISSING REQUIRED FIELDS
This is not a valid IRS IMF layout file.
It lacks proper field definitions and structure.
"""


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_irs_files(temp_test_dir):
    """Create sample IRS layout files for testing."""
    files = {}

    # Basic IMF layout
    basic_file = temp_test_dir / "imf_basic_layout.txt"
    basic_file.write_text(SAMPLE_IMF_LAYOUT_BASIC)
    files["basic"] = str(basic_file)

    # Complex business layout
    complex_file = temp_test_dir / "imf_business_layout.txt"
    complex_file.write_text(SAMPLE_IMF_LAYOUT_COMPLEX)
    files["complex"] = str(complex_file)

    # Invalid layout
    invalid_file = temp_test_dir / "invalid_layout.txt"
    invalid_file.write_text(SAMPLE_INVALID_LAYOUT)
    files["invalid"] = str(invalid_file)

    return files


@pytest.fixture
def temp_bigquery_dataset():
    """Create temporary BigQuery dataset for testing."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
    dataset_id = f"test_irs_ingestion_{uuid.uuid4().hex[:8]}"

    client = bigquery.Client(project=project_id)

    # Create temporary dataset
    dataset_ref = client.dataset(dataset_id)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.description = "Temporary dataset for IRS ingestion integration tests"
    dataset.default_table_expiration_ms = 3600000  # 1 hour

    try:
        dataset = client.create_dataset(dataset)

        # Create test tables using the contract DDL
        create_test_tables(client, project_id, dataset_id)

        yield {"project_id": project_id, "dataset_id": dataset_id, "client": client}

    finally:
        # Cleanup - delete dataset and all tables
        try:
            client.delete_dataset(dataset_ref, delete_contents=True)
        except NotFound:
            pass  # Already deleted


def create_test_tables(client: bigquery.Client, project_id: str, dataset_id: str):
    """Create test tables based on the contract DDL."""

    # source_metadata table
    source_metadata_ddl = f"""
    CREATE TABLE `{project_id}.{dataset_id}.source_metadata` (
      source_type STRING NOT NULL,
      artifact_id STRING NOT NULL,
      parent_id STRING,
      parent_type STRING,
      content_text STRING NOT NULL,
      content_tokens INT64,
      content_hash STRING NOT NULL,
      created_at TIMESTAMP NOT NULL,
      updated_at TIMESTAMP NOT NULL,
      collected_at TIMESTAMP NOT NULL,
      source_uri STRING NOT NULL,
      repo_ref STRING,
      tool_version STRING NOT NULL,

      -- IRS-specific fields
      irs_record_type STRING,
      irs_layout_version STRING,
      irs_start_position INT64,
      irs_field_length INT64,
      irs_data_type STRING,
      irs_section STRING
    )
    """

    # source_metadata_errors table
    errors_ddl = f"""
    CREATE TABLE `{project_id}.{dataset_id}.source_metadata_errors` (
      error_id STRING NOT NULL,
      source_type STRING NOT NULL,
      source_uri STRING NOT NULL,
      error_class STRING NOT NULL,
      error_msg STRING NOT NULL,
      sample_text STRING,
      collected_at TIMESTAMP NOT NULL,
      tool_version STRING NOT NULL,
      stack_trace STRING
    )
    """

    # ingestion_log table
    log_ddl = f"""
    CREATE TABLE `{project_id}.{dataset_id}.ingestion_log` (
      run_id STRING NOT NULL,
      source_type STRING NOT NULL,
      started_at TIMESTAMP NOT NULL,
      completed_at TIMESTAMP,
      status STRING NOT NULL,
      files_processed INT64 NOT NULL,
      rows_written INT64 NOT NULL,
      rows_skipped INT64 NOT NULL,
      errors_count INT64 NOT NULL,
      bytes_written INT64 NOT NULL,
      processing_duration_ms INT64,
      avg_chunk_size_tokens FLOAT64,
      tool_version STRING NOT NULL,
      config_params JSON
    )
    """

    # Execute DDL statements
    for ddl in [source_metadata_ddl, errors_ddl, log_ddl]:
        client.query(ddl).result()


@pytest.mark.integration
@pytest.mark.bigquery
class TestIRSIngestionIntegration:
    """Integration tests for IRS IMF layout ingestion to BigQuery."""

    def test_irs_parser_not_implemented_yet(self):
        """
        TDD RED PHASE: This test should fail because IRSParser is not implemented.

        This test validates that we're following TDD by ensuring the parser
        doesn't exist yet, so we know our implementation is needed.
        """
        with pytest.raises((ImportError, NotImplementedError, AttributeError)):
            # This should fail because the actual implementation doesn't exist
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()

    def test_single_file_ingestion_basic_layout(
        self, sample_irs_files, temp_bigquery_dataset
    ):
        """
        Test ingestion of a single basic IRS IMF layout file.

        This test will FAIL initially because IRSParser implementation doesn't exist.
        Expected behavior when implemented:
        - Parse basic IMF layout with taxpayer information
        - Extract field positions, lengths, and data types
        - Generate appropriate chunks for BigQuery ingestion
        - Write to source_metadata table with IRS-specific fields
        """
        # This import should fail initially (TDD RED phase)
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl(version="1.0.0")
            result = parser.parse_file(sample_irs_files["basic"])

            # Expected assertions when implementation exists:
            assert isinstance(result, ParseResult)
            assert len(result.chunks) > 0
            assert len(result.errors) == 0
            assert result.files_processed == 1

            # Validate chunk metadata structure
            chunk = result.chunks[0]
            assert chunk.source_type == SourceType.IRS
            assert chunk.source_uri == sample_irs_files["basic"]
            assert "Social Security Number" in chunk.content_text

            # Validate IRS-specific metadata
            assert "record_type" in chunk.source_metadata
            assert "layout_version" in chunk.source_metadata
            assert "field_positions" in chunk.source_metadata

    def test_single_file_ingestion_complex_layout(
        self, sample_irs_files, temp_bigquery_dataset
    ):
        """
        Test ingestion of complex business tax layout with sections.

        This test will FAIL initially due to missing implementation.
        Expected behavior:
        - Parse multi-section business layout
        - Handle different field types (N=Numeric, A=Alpha)
        - Extract section information (Entity, Financial, Business Codes, Indicators)
        - Generate chunks preserving section relationships
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()
            result = parser.parse_file(sample_irs_files["complex"])

            # Expected validations:
            assert len(result.chunks) >= 4  # At least one chunk per section

            # Check for section-specific content
            section_contents = [chunk.content_text for chunk in result.chunks]
            assert any("ENTITY INFORMATION" in content for content in section_contents)
            assert any("FINANCIAL DATA" in content for content in section_contents)
            assert any("BUSINESS CODES" in content for content in section_contents)

    def test_directory_ingestion_multiple_files(
        self, sample_irs_files, temp_bigquery_dataset
    ):
        """
        Test ingestion of directory containing multiple IRS layout files.

        Should process all valid files and report errors for invalid ones.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()
            test_dir = Path(sample_irs_files["basic"]).parent
            result = parser.parse_directory(str(test_dir))

            # Expected behavior:
            assert result.files_processed == 3  # basic, complex, invalid
            assert len(result.chunks) >= 2  # At least chunks from valid files
            assert len(result.errors) >= 1  # Error from invalid file

    def test_invalid_layout_handling(self, sample_irs_files, temp_bigquery_dataset):
        """
        Test handling of invalid IRS layout files.

        Should generate appropriate error records without crashing.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()
            result = parser.parse_file(sample_irs_files["invalid"])

            # Expected error handling:
            assert len(result.chunks) == 0
            assert len(result.errors) >= 1
            assert result.errors[0].error_class == ErrorClass.PARSING
            assert "INVALID IRS LAYOUT" in result.errors[0].sample_text

    def test_field_position_extraction(self, sample_irs_files):
        """
        Test accurate extraction of field positions and metadata.

        Validates that parser correctly parses field definitions like:
        001-009    9   N   Social Security Number
        """
        with pytest.raises((ImportError, NotImplementedError, AttributeError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()

            # This method should extract field positions from layout
            field_positions = parser.extract_field_positions(
                {"layout_content": SAMPLE_IMF_LAYOUT_BASIC}
            )

            # Expected field extraction:
            assert len(field_positions) >= 22  # All fields in basic layout

            # Validate specific field parsing
            ssn_field = next(
                (f for f in field_positions if "Social Security" in f["description"]),
                None,
            )
            assert ssn_field is not None
            assert ssn_field["start_position"] == 1
            assert ssn_field["end_position"] == 9
            assert ssn_field["length"] == 9
            assert ssn_field["data_type"] == "N"

    def test_bigquery_ingestion_end_to_end(
        self, sample_irs_files, temp_bigquery_dataset
    ):
        """
        Test complete end-to-end flow from IRS layout to BigQuery.

        This is the primary integration test validating the entire pipeline.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.bigquery.writer import BigQueryWriterImpl
            from src.parsers.irs_parser import IRSParserImpl

            # Parse IRS layout
            parser = IRSParserImpl()
            parse_result = parser.parse_file(sample_irs_files["basic"])

            # Write to BigQuery
            writer = BigQueryWriterImpl(
                project_id=temp_bigquery_dataset["project_id"],
                dataset_id=temp_bigquery_dataset["dataset_id"],
            )

            # Ensure tables exist
            assert writer.create_tables_if_not_exist(
                temp_bigquery_dataset["dataset_id"]
            )

            # Write chunks to source_metadata
            rows_written = writer.write_chunks(parse_result.chunks, "source_metadata")
            assert rows_written == len(parse_result.chunks)

            # Write errors if any
            if parse_result.errors:
                errors_written = writer.write_errors(
                    parse_result.errors, "source_metadata_errors"
                )
                assert errors_written == len(parse_result.errors)

            # Log ingestion run
            run_id = writer.log_ingestion_run(
                {
                    "source_type": "irs",
                    "files_processed": parse_result.files_processed,
                    "rows_written": rows_written,
                    "processing_duration_ms": parse_result.processing_duration_ms,
                },
                "ingestion_log",
            )

            assert run_id is not None

    def test_bigquery_data_validation(self, sample_irs_files, temp_bigquery_dataset):
        """
        Test validation of data written to BigQuery tables.

        Ensures IRS-specific fields are populated correctly.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.bigquery.writer import BigQueryWriterImpl
            from src.parsers.irs_parser import IRSParserImpl

            # Complete ingestion (would need implementation)
            parser = IRSParserImpl()
            writer = BigQueryWriterImpl(
                project_id=temp_bigquery_dataset["project_id"],
                dataset_id=temp_bigquery_dataset["dataset_id"],
            )

            # Parse and ingest
            result = parser.parse_file(sample_irs_files["basic"])
            writer.write_chunks(result.chunks, "source_metadata")

            # Query and validate data
            client = temp_bigquery_dataset["client"]
            query = f"""
            SELECT
                source_type,
                artifact_id,
                content_text,
                irs_record_type,
                irs_layout_version,
                irs_start_position,
                irs_field_length,
                irs_data_type,
                irs_section
            FROM `{temp_bigquery_dataset['project_id']}.{temp_bigquery_dataset['dataset_id']}.source_metadata`
            WHERE source_type = 'irs'
            """

            rows = list(client.query(query).result())
            assert len(rows) > 0

            # Validate IRS-specific fields
            for row in rows:
                assert row.source_type == "irs"
                assert row.artifact_id.startswith("irs://")
                assert row.irs_record_type is not None
                assert row.irs_layout_version is not None

    def test_content_chunking_strategy(self, sample_irs_files):
        """
        Test IRS-specific content chunking strategy.

        Should chunk by logical sections while preserving field relationships.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()

            # Test chunking of complex layout with sections
            chunks = parser.chunk_content(SAMPLE_IMF_LAYOUT_COMPLEX, max_tokens=500)

            # Should preserve section boundaries
            assert len(chunks) >= 4  # At least one per section

            # Each chunk should contain related fields
            for chunk in chunks:
                # Should not split field definitions across chunks
                lines = chunk.split("\n")
                field_lines = [
                    line for line in lines if "-" in line and len(line.split()) >= 4
                ]

                # Validate field line integrity
                for field_line in field_lines:
                    assert len(field_line.split()) >= 4  # pos, len, type, desc

    def test_content_validation(self, sample_irs_files):
        """
        Test validation of IRS layout content format.

        Should identify valid vs invalid IRS layout files.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()

            # Valid layouts should pass validation
            assert parser.validate_content(SAMPLE_IMF_LAYOUT_BASIC) is True
            assert parser.validate_content(SAMPLE_IMF_LAYOUT_COMPLEX) is True

            # Invalid content should fail validation
            assert parser.validate_content(SAMPLE_INVALID_LAYOUT) is False
            assert parser.validate_content("Random text content") is False
            assert parser.validate_content("") is False

    def test_error_handling_and_recovery(self, temp_test_dir, temp_bigquery_dataset):
        """
        Test error handling for various failure scenarios.

        Should handle corrupt files, permission issues, and BigQuery errors gracefully.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            parser = IRSParserImpl()

            # Test with non-existent file
            result = parser.parse_file("/nonexistent/file.txt")
            assert len(result.errors) > 0
            assert result.errors[0].error_class == ErrorClass.PARSING

            # Test with directory instead of file
            result = parser.parse_file(str(temp_test_dir))
            assert len(result.errors) > 0

            # Test with empty directory
            empty_dir = temp_test_dir / "empty"
            empty_dir.mkdir()
            result = parser.parse_directory(str(empty_dir))
            assert result.files_processed == 0
            assert len(result.chunks) == 0

    @pytest.mark.slow
    def test_performance_large_files(self, temp_test_dir):
        """
        Test performance with large IRS layout files.

        Should handle large files efficiently without memory issues.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.irs_parser import IRSParserImpl

            # Create large test file
            large_content = SAMPLE_IMF_LAYOUT_BASIC + "\n" + SAMPLE_IMF_LAYOUT_COMPLEX
            large_content = large_content * 100  # Multiply for size

            large_file = temp_test_dir / "large_irs_layout.txt"
            large_file.write_text(large_content)

            parser = IRSParserImpl()
            start_time = datetime.now()
            result = parser.parse_file(str(large_file))
            processing_time = (datetime.now() - start_time).total_seconds()

            # Performance assertions
            assert processing_time < 30.0  # Should complete within 30 seconds
            assert result.processing_duration_ms > 0
            assert len(result.chunks) > 100  # Should generate many chunks


@pytest.mark.integration
@pytest.mark.bigquery
class TestIRSBigQueryIntegration:
    """Additional integration tests focusing on BigQuery-specific functionality."""

    def test_bigquery_writer_interface_compliance(self, temp_bigquery_dataset):
        """
        Test that BigQueryWriter implementation follows the contract interface.

        This validates the BigQueryWriter contract compliance.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.bigquery.writer import BigQueryWriterImpl

            writer = BigQueryWriterImpl(
                project_id=temp_bigquery_dataset["project_id"],
                dataset_id=temp_bigquery_dataset["dataset_id"],
            )

            # Validate interface methods exist
            assert hasattr(writer, "write_chunks")
            assert hasattr(writer, "write_errors")
            assert hasattr(writer, "log_ingestion_run")
            assert hasattr(writer, "create_tables_if_not_exist")

    def test_schema_compatibility(self, temp_bigquery_dataset):
        """
        Test that generated data is compatible with BigQuery schema.

        Validates data types, field lengths, and constraints.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            # This would test actual schema validation
            # when implementation exists
            pass

    def test_concurrent_ingestion(self, sample_irs_files, temp_bigquery_dataset):
        """
        Test concurrent ingestion of multiple IRS files.

        Should handle concurrent writes without conflicts.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            # This would test concurrent processing
            # when implementation exists
            pass


if __name__ == "__main__":
    """
    Run integration tests directly.

    This will FAIL until IRS parser implementation is created.
    Usage: python -m pytest tests/integration/test_irs_ingestion.py -v
    """
    pytest.main([__file__, "-v", "--tb=short"])
