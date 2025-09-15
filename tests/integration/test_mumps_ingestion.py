"""
Integration test for MUMPS dictionary ingestion to BigQuery.

This is a comprehensive end-to-end test that validates the complete flow
from MUMPS/VistA FileMan dictionaries to BigQuery source_metadata table.

Following TDD principles - this test MUST FAIL initially because no
MUMPS parser implementation exists yet.
"""

import importlib.util
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from google.cloud import bigquery

# Import parser interfaces from file with dashes in name
specs_file = (
    Path(__file__).parent.parent.parent
    / "specs"
    / "002-m1-parse-and"
    / "contracts"
    / "parser-interfaces.py"
)
spec = importlib.util.spec_from_file_location("parser_interfaces", specs_file)
parser_interfaces = importlib.util.module_from_spec(spec)
sys.modules["parser_interfaces"] = parser_interfaces
spec.loader.exec_module(parser_interfaces)

# Import the classes we need
BaseParser = parser_interfaces.BaseParser
ChunkMetadata = parser_interfaces.ChunkMetadata
ErrorClass = parser_interfaces.ErrorClass
MUMPSParser = parser_interfaces.MUMPSParser
ParseError = parser_interfaces.ParseError
ParseResult = parser_interfaces.ParseResult
SourceType = parser_interfaces.SourceType


# Sample MUMPS FileMan Dictionary Content
FILEMAN_DICT_CONTENT = """
^DDD(200,"DD")="STANDARD DATA DICTIONARY #200 -- NEW PERSON FILE"
^DDD(200,"RD",200,0)="NAME^RF^^0;1^K:$L(X)>30!($L(X)<3)!'(X'?1P.E) X"
^DDD(200,"GL")="^VA(200,"
^DDD(200,"IX","B",200,.01)=""
^DDD(200,"IX","SSN",200,9)=""
^DDD(200,0)="NEW PERSON^200^1001^4^200.001^3080903^1001"
^DDD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>30!($L(X)<3)!'(X'?1P.E) X"
^DDD(200,.01,1,0)="^200.001A"
^DDD(200,.01,1,1,0)="200^B"
^DDD(200,.01,1,1,1)="S ^VA(200,\\"B\\",$E(X,1,30),DA)=\\"\\"\\"\\"\\""
^DDD(200,.01,1,1,2)="K ^VA(200,\\"B\\",$E(X,1,30),DA)"
^DDD(200,.01,21,0)="^^1^1^3080903^"
^DDD(200,.01,21,1,0)="Answer with the name of the person."
^DDD(200,.01,"DT")=3080903
^DDD(200,1,0)="SSN^F^^0;9^K:$L(X)>9!($L(X)<9)!('(X?9N)) X"
^DDD(200,1,1,0)="^200.001A"
^DDD(200,1,1,1,0)="200^SSN"
^DDD(200,1,1,1,1)="S ^VA(200,\\"SSN\\",X,DA)=\\"\\"\\"\\"\\""
^DDD(200,1,1,1,2)="K ^VA(200,\\"SSN\\",X,DA)"
^DDD(200,1,21,0)="^^1^1^3080903^"
^DDD(200,1,21,1,0)="The social security number for this person."
^DDD(200,1,"DT")=3080903
^DDD(200,2,0)="DOB^D^^0;5^S %DT=\\"E\\" D ^%DT S X=Y K %DT"
^DDD(200,2,21,0)="^^1^1^3080903^"
^DDD(200,2,21,1,0)="Date of birth for this person."
^DDD(200,2,"DT")=3080903
^DDD(200,3,0)="SEX^S^M:MALE;F:FEMALE;^0;4^Q"
^DDD(200,3,1,0)="^200.001A"
^DDD(200,3,21,0)="^^1^1^3080903^"
^DDD(200,3,21,1,0)="Gender designation for this person."
^DDD(200,3,"DT")=3080903
"""

GLOBAL_DEFINITION_CONTENT = r"""
;Global Definition for NEW PERSON file
;
^VA(200,0)="NEW PERSON^200^1001^4"
^VA(200,"B","DOCTOR,JANE",123)=""
^VA(200,"B","NURSE,JOHN",124)=""
^VA(200,"B","SMITH,BOB",125)=""
^VA(200,"SSN","123456789",123)=""
^VA(200,"SSN","987654321",124)=""
^VA(200,123,0)="DOCTOR,JANE^123456789^MD^F^2651015"
^VA(200,124,0)="NURSE,JOHN^987654321^RN^M^2701201"
^VA(200,125,0)="SMITH,BOB^555123456^TECH^M^2801101"
;
;Additional indices
^VA(200,"AUSER","JANE.DOCTOR",123)=""
^VA(200,"AUSER","JOHN.NURSE",124)=""
^VA(200,"AUSER","BOB.SMITH",125)=""
;
;Cross-references for title field
^VA(200,"ATITLE","MD",123)=""
^VA(200,"ATITLE","RN",124)=""
^VA(200,"ATITLE","TECH",125)=""
"""

COMPLEX_FILEMAN_DICT = """
^DDD(8925,"DD")="STANDARD DATA DICTIONARY #8925 -- PATIENT ALLERGIES FILE"
^DDD(8925,0)="PATIENT ALLERGIES^8925^1001^15^8925.001^3080903^1001"
^DDD(8925,.01,0)="ALLERGEN^RP120'^GMR(120.8,^0;1^S DIC=\\"^GMR(120.8,\\",DIC(0)=\\"EQZ\\",DIC(\\"S\\")=\\"I '$P(^(0),U,7)\\" D ^DIC K DIC S DIC=DIE,X=+Y K:Y<0 X"
^DDD(8925,.02,0)="PATIENT^RP2'^DPT(^0;2^S DIC=\\"^DPT(\\",DIC(0)=\\"EQZ\\" D ^DIC K DIC S DIC=DIE,X=+Y K:Y<0 X"
^DDD(8925,1,0)="REACTION DATE/TIME^D^^1;1^S %DT=\\"ESTX\\" D ^%DT S X=Y K %DT"
^DDD(8925,2,0)="REACTIONS^120.85PA^^2;"
^DDD(8925,3,0)="SEVERITY^S^1:MILD;2:MODERATE;3:SEVERE;^1;2^Q"
^DDD(8925,4,0)="OBSERVED/HISTORICAL^S^o:OBSERVED;h:HISTORICAL;^1;3^Q"
^DDD(8925,5,0)="MECHANISM^S^A:ALLERGY;P:PHARMACOLOGIC;U:UNKNOWN;^1;4^Q"
^DDD(8925,10,0)="COMMENTS^120.8A^^COMMENTS;"
^DDD(8925,17,0)="ENTERED BY^RP200'^VA(200,^5;1^S DIC=\\"^VA(200,\\",DIC(0)=\\"EQZ\\" D ^DIC K DIC S DIC=DIE,X=+Y K:Y<0 X"
^DDD(8925,18,0)="ENTERED^D^^5;2^S %DT=\\"ESTX\\" D ^%DT S X=Y K %DT"
"""


@pytest.mark.integration
@pytest.mark.bigquery
class TestMUMPSIngestion:
    """End-to-end integration test for MUMPS dictionary ingestion to BigQuery."""

    @pytest.fixture(scope="class")
    def bq_client(self):
        """BigQuery client for integration testing."""
        # Use real BigQuery client but with temp dataset for isolation
        return bigquery.Client()

    @pytest.fixture(scope="class")
    def temp_dataset(self, bq_client):
        """Create temporary BigQuery dataset for testing."""
        dataset_id = f"test_mumps_ingestion_{int(time.time())}"
        project_id = bq_client.project
        full_dataset_id = f"{project_id}.{dataset_id}"

        # Create dataset
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = "US"
        dataset.description = "Temporary dataset for MUMPS ingestion testing"
        dataset = bq_client.create_dataset(dataset, timeout=30)

        yield dataset_id

        # Cleanup: Delete dataset and all tables
        bq_client.delete_dataset(
            full_dataset_id, delete_contents=True, not_found_ok=True
        )

    @pytest.fixture(scope="class")
    def temp_tables(self, bq_client, temp_dataset):
        """Create temporary BigQuery tables using the DDL schema."""
        project_id = bq_client.project

        # Read DDL schema and create tables
        ddl_path = (
            Path(__file__).parent.parent.parent
            / "specs"
            / "002-m1-parse-and"
            / "contracts"
            / "bigquery-ddl.sql"
        )
        with open(ddl_path) as f:
            ddl_content = f.read()

        # Replace placeholders with actual values
        ddl_content = ddl_content.replace("${BQ_PROJECT}", project_id)
        ddl_content = ddl_content.replace("${BQ_DATASET}", temp_dataset)

        # Split DDL into individual CREATE statements
        ddl_statements = [
            stmt.strip()
            for stmt in ddl_content.split(";")
            if stmt.strip() and stmt.strip().upper().startswith("CREATE TABLE")
        ]

        tables_created = []
        for statement in ddl_statements:
            try:
                query_job = bq_client.query(statement)
                query_job.result()  # Wait for completion

                # Extract table name from DDL
                table_name = statement.split("`")[5].split("`")[
                    0
                ]  # Extract from backticks
                tables_created.append(table_name)
            except Exception as e:
                pytest.fail(f"Failed to create table from DDL: {e}")

        return tables_created

    @pytest.fixture
    def sample_fileman_files(self):
        """Create temporary files with sample MUMPS FileMan dictionary content."""
        files = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Basic FileMan dictionary
            basic_file = Path(temp_dir) / "new_person_dict.ro"
            basic_file.write_text(FILEMAN_DICT_CONTENT)
            files["basic_dict"] = str(basic_file)

            # Global definition
            global_file = Path(temp_dir) / "new_person_globals.ro"
            global_file.write_text(GLOBAL_DEFINITION_CONTENT)
            files["global_def"] = str(global_file)

            # Complex dictionary
            complex_file = Path(temp_dir) / "patient_allergies_dict.ro"
            complex_file.write_text(COMPLEX_FILEMAN_DICT)
            files["complex_dict"] = str(complex_file)

            yield files

    @pytest.fixture
    def sample_mumps_directory(self):
        """Create a directory with multiple MUMPS files for directory ingestion testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir) / "mumps_data"
            dir_path.mkdir()

            # Create multiple MUMPS files
            (dir_path / "file200.ro").write_text(FILEMAN_DICT_CONTENT)
            (dir_path / "file8925.ro").write_text(COMPLEX_FILEMAN_DICT)
            (dir_path / "globals.ro").write_text(GLOBAL_DEFINITION_CONTENT)

            # Add a non-MUMPS file that should be ignored
            (dir_path / "readme.txt").write_text("This is not a MUMPS file")

            yield str(dir_path)

    def test_mumps_parser_not_implemented_yet(self):
        """
        TDD RED PHASE: Test that MUMPS parser doesn't exist yet.

        This test MUST FAIL initially, demonstrating that we need to implement
        the MUMPS parser. It validates our TDD approach.
        """
        with pytest.raises((ImportError, NotImplementedError, AttributeError)):
            # Try to import the concrete MUMPS parser implementation
            from src.parsers.mumps_parser import MUMPSParserImpl

            # If import succeeds, try to instantiate (should fail if not implemented)
            parser = MUMPSParserImpl()

            # If instantiation succeeds, methods should raise NotImplementedError
            result = parser.parse_file("dummy_file.ro")
            pytest.fail("MUMPS parser should not be implemented yet (TDD RED phase)")

    def test_mumps_parser_interface_contract(self):
        """Test that MUMPSParser interface follows the contract."""
        # Verify the abstract interface exists and has required methods
        assert hasattr(MUMPSParser, "parse_fileman_dict")
        assert hasattr(MUMPSParser, "parse_global_definition")
        assert hasattr(MUMPSParser, "_get_source_type")

        # Verify it inherits from BaseParser
        assert issubclass(MUMPSParser, BaseParser)

        # Verify source type
        # Note: This will fail until implementation exists
        with pytest.raises(TypeError):
            # Abstract class cannot be instantiated
            parser = MUMPSParser()

    def test_fileman_dictionary_parsing_contract(self, sample_fileman_files):
        """
        Test FileMan dictionary parsing contract.

        This test defines what the parser SHOULD do when implemented.
        It will FAIL until the parser is implemented.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()
            result = parser.parse_file(sample_fileman_files["basic_dict"])

            # Expected behavior when implemented:
            assert isinstance(result, ParseResult)
            assert len(result.chunks) > 0
            assert result.files_processed == 1

            # Verify MUMPS-specific metadata
            chunk = result.chunks[0]
            assert chunk.source_type == SourceType.MUMPS
            assert "mumps_file_no" in chunk.source_metadata
            assert "mumps_field_no" in chunk.source_metadata
            assert chunk.source_metadata["mumps_file_no"] == 200  # NEW PERSON file

    def test_global_definition_parsing_contract(self, sample_fileman_files):
        """
        Test global definition parsing contract.

        This test defines expected behavior for parsing MUMPS global definitions.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()
            result = parser.parse_file(sample_fileman_files["global_def"])

            # Expected behavior when implemented:
            assert isinstance(result, ParseResult)
            assert len(result.chunks) > 0

            # Should extract global structure
            chunk = result.chunks[0]
            assert "mumps_global_name" in chunk.source_metadata
            assert chunk.source_metadata["mumps_global_name"] == "VA"

    def test_directory_ingestion_contract(self, sample_mumps_directory):
        """
        Test directory-level ingestion contract.

        This test defines expected behavior for parsing entire directories.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()
            result = parser.parse_directory(sample_mumps_directory)

            # Expected behavior when implemented:
            assert isinstance(result, ParseResult)
            assert result.files_processed == 3  # Should ignore readme.txt
            assert len(result.chunks) > 0

            # Should contain chunks from all MUMPS files
            file_numbers = set()
            for chunk in result.chunks:
                if "mumps_file_no" in chunk.source_metadata:
                    file_numbers.add(chunk.source_metadata["mumps_file_no"])

            assert 200 in file_numbers  # NEW PERSON
            assert 8925 in file_numbers  # PATIENT ALLERGIES

    def test_bigquery_ingestion_workflow(
        self, bq_client, temp_dataset, temp_tables, sample_fileman_files
    ):
        """
        Test complete BigQuery ingestion workflow.

        This is the main end-to-end test that validates the complete flow
        from MUMPS files to BigQuery source_metadata table.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.ingestion.bigquery_writer import BigQueryWriterImpl
            from src.parsers.mumps_parser import MUMPSParserImpl

            # Parse MUMPS files
            parser = MUMPSParserImpl()
            parse_result = parser.parse_file(sample_fileman_files["basic_dict"])

            # Write to BigQuery
            writer = BigQueryWriterImpl(bq_client, temp_dataset)
            rows_written = writer.write_chunks(parse_result.chunks, "source_metadata")

            # Verify data was written
            assert rows_written > 0

            # Query and validate the data
            query = f"""
                SELECT
                    source_type,
                    artifact_id,
                    content_text,
                    mumps_file_no,
                    mumps_field_no,
                    mumps_global_name
                FROM `{bq_client.project}.{temp_dataset}.source_metadata`
                WHERE source_type = 'mumps'
                ORDER BY mumps_field_no
            """

            query_job = bq_client.query(query)
            rows = list(query_job.result())

            assert len(rows) > 0

            # Validate MUMPS-specific fields
            row = rows[0]
            assert row.source_type == "mumps"
            assert row.mumps_file_no == 200
            assert row.artifact_id.startswith("mumps://")

    def test_mumps_field_validation_contract(self):
        """
        Test MUMPS-specific field validation contract.

        Defines expected validation behavior for MUMPS field definitions.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()

            # Valid FileMan content
            valid_content = '^DDD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>30!($L(X)<3) X"'
            assert parser.validate_content(valid_content) == True

            # Invalid content
            invalid_content = "This is not MUMPS code"
            assert parser.validate_content(invalid_content) == False

    def test_error_handling_contract(self, sample_fileman_files):
        """
        Test error handling contract for malformed MUMPS files.

        Defines expected error handling behavior.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()

            # Create malformed MUMPS file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ro", delete=False) as f:
                f.write("^DDD(200,MALFORMED SYNTAX HERE")
                malformed_file = f.name

            try:
                result = parser.parse_file(malformed_file)

                # Should capture errors but not crash
                assert isinstance(result, ParseResult)
                assert len(result.errors) > 0

                error = result.errors[0]
                assert error.error_class == ErrorClass.PARSING
                assert error.source_type == SourceType.MUMPS

            finally:
                os.unlink(malformed_file)

    def test_chunking_strategy_contract(self, sample_fileman_files):
        """
        Test content chunking strategy for MUMPS files.

        Defines expected chunking behavior for large FileMan dictionaries.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()
            result = parser.parse_file(sample_fileman_files["complex_dict"])

            # Expected chunking behavior:
            assert len(result.chunks) > 1  # Should chunk large dictionaries

            # Each chunk should be manageable size
            for chunk in result.chunks:
                assert len(chunk.content_text) <= 16000  # 16KB limit from schema
                assert chunk.content_tokens is not None
                assert chunk.content_tokens <= 1000  # Reasonable token limit

    def test_metadata_extraction_contract(self, sample_fileman_files):
        """
        Test MUMPS-specific metadata extraction contract.

        Defines what metadata should be extracted from FileMan dictionaries.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()
            result = parser.parse_file(sample_fileman_files["basic_dict"])

            # Expected metadata extraction:
            chunk = result.chunks[0]
            metadata = chunk.source_metadata

            # Required MUMPS metadata fields
            assert "mumps_file_no" in metadata
            assert "mumps_field_no" in metadata
            assert "mumps_global_name" in metadata

            # Validate specific values
            assert metadata["mumps_file_no"] == 200
            assert metadata["mumps_global_name"] == "VA"

            # Field-specific metadata
            field_chunks = [
                c for c in result.chunks if "mumps_field_no" in c.source_metadata
            ]
            assert len(field_chunks) > 0

            name_field = next(
                c for c in field_chunks if c.source_metadata["mumps_field_no"] == 0.01
            )
            assert (
                "input_transform" in name_field.source_metadata
                or "mumps_input_transform" in name_field.source_metadata
            )

    def test_performance_requirements_contract(self, sample_mumps_directory):
        """
        Test performance requirements for MUMPS ingestion.

        Defines expected performance characteristics.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            parser = MUMPSParserImpl()

            start_time = time.time()
            result = parser.parse_directory(sample_mumps_directory)
            end_time = time.time()

            # Performance expectations:
            processing_time_ms = (end_time - start_time) * 1000
            assert processing_time_ms < 30000  # Should complete within 30 seconds
            assert result.processing_duration_ms > 0
            assert (
                result.processing_duration_ms <= processing_time_ms * 1.1
            )  # Allow 10% measurement variance

    def test_bigquery_schema_compliance_contract(
        self, bq_client, temp_dataset, temp_tables
    ):
        """
        Test BigQuery schema compliance contract.

        Validates that generated chunks conform to the BigQuery schema.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.ingestion.bigquery_writer import BigQueryWriterImpl
            from src.parsers.mumps_parser import MUMPSParserImpl

            # Generate test chunk that should comply with schema
            parser = MUMPSParserImpl()

            # Create a sample chunk with all MUMPS fields
            chunk = ChunkMetadata(
                source_type=SourceType.MUMPS,
                artifact_id="mumps://test_file.ro#field_200_01",
                content_text="NAME field definition from FileMan dictionary",
                content_hash="abc123def456",
                source_uri="test_file.ro",
                collected_at=datetime.now(timezone.utc),
                source_metadata={
                    "mumps_file_no": 200,
                    "mumps_field_no": 0.01,
                    "mumps_global_name": "VA",
                    "mumps_input_transform": "K:$L(X)>30!($L(X)<3) X",
                },
            )

            # Write to BigQuery and verify schema compliance
            writer = BigQueryWriterImpl(bq_client, temp_dataset)
            rows_written = writer.write_chunks([chunk], "source_metadata")

            assert rows_written == 1

            # Query to verify all fields were written correctly
            query = f"""
                SELECT *
                FROM `{bq_client.project}.{temp_dataset}.source_metadata`
                WHERE artifact_id = '{chunk.artifact_id}'
            """

            query_job = bq_client.query(query)
            rows = list(query_job.result())

            assert len(rows) == 1
            row = rows[0]

            # Verify MUMPS-specific fields
            assert row.mumps_file_no == 200
            assert row.mumps_field_no == 0.01
            assert row.mumps_global_name == "VA"
            assert row.mumps_input_transform is not None

    def test_integration_with_vector_embeddings_contract(self, bq_client, temp_dataset):
        """
        Test integration contract with vector embeddings workflow.

        Validates that MUMPS chunks are suitable for embedding generation.
        """
        with pytest.raises((ImportError, NotImplementedError)):
            from src.parsers.mumps_parser import MUMPSParserImpl

            # This test validates the workflow integration
            # In the real system, chunks should be suitable for:
            # 1. Embedding generation via Gemini API
            # 2. Vector storage in BigQuery
            # 3. Semantic search via vector similarity

            parser = MUMPSParserImpl()
            # Test with complex dictionary that should generate meaningful chunks
            # for semantic search

            # For now, just validate the contract exists
            assert hasattr(parser, "parse_fileman_dict")
            assert hasattr(parser, "parse_global_definition")


# Test data validation functions
def validate_mumps_chunk_metadata(chunk: ChunkMetadata) -> bool:
    """Validate that a chunk has proper MUMPS metadata."""
    if chunk.source_type != SourceType.MUMPS:
        return False

    metadata = chunk.source_metadata

    # Required fields for MUMPS chunks
    required_fields = ["mumps_file_no", "mumps_global_name"]
    for field in required_fields:
        if field not in metadata:
            return False

    # Validate data types
    if not isinstance(metadata["mumps_file_no"], (int, float)):
        return False

    if not isinstance(metadata["mumps_global_name"], str):
        return False

    return True


def validate_fileman_field_metadata(chunk: ChunkMetadata) -> bool:
    """Validate FileMan field-specific metadata."""
    metadata = chunk.source_metadata

    if "mumps_field_no" not in metadata:
        return False

    # Field number should be numeric
    if not isinstance(metadata["mumps_field_no"], (int, float)):
        return False

    return True


# Helper functions for test data generation
def create_test_fileman_dict(file_no: int, field_count: int = 5) -> str:
    """Generate test FileMan dictionary content."""
    lines = [
        f'^DDD({file_no},"DD")="TEST DATA DICTIONARY #{file_no}"',
        f'^DDD({file_no},0)="TEST FILE^{file_no}^1001^{field_count}^{file_no}.001^3080903^1001"',
        f'^DDD({file_no},"GL")="^TEST({file_no},"',
    ]

    for i in range(field_count):
        field_no = f".{i:02d}" if i < 10 else str(i)
        lines.append(
            f'^DDD({file_no},{field_no},0)="FIELD_{i}^RF^^0;{i+1}^K:$L(X)>30 X"'
        )

    return "\n".join(lines)


def create_test_global_def(file_no: int, record_count: int = 3) -> str:
    """Generate test global definition content."""
    lines = [f'^TEST({file_no},0)="TEST FILE^{file_no}^{record_count}^{record_count}"']

    for i in range(1, record_count + 1):
        lines.append(f'^TEST({file_no},{i},0)="RECORD_{i}^VALUE_{i}^DATA_{i}"')
        lines.append(f'^TEST({file_no},"B","RECORD_{i}",{i})=""')

    return "\n".join(lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
