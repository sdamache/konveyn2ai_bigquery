"""
Integration test: FastAPI project ingestion to BigQuery

This test validates the complete flow from FastAPI source code and OpenAPI specs
to BigQuery source_metadata table storage.

IMPORTANT: This test MUST FAIL initially (TDD Red phase) because:
1. FastAPIParser implementation doesn't exist yet
2. BigQuery writer implementation doesn't exist yet
3. AST and OpenAPI parsing logic not implemented

Test coverage:
- Python AST parsing for FastAPI routes and models
- OpenAPI specification parsing
- Route extraction with HTTP methods and paths
- Pydantic model parsing and schema extraction
- BigQuery ingestion with proper metadata fields
- Both individual file and project directory ingestion

Requirements:
- Real BigQuery testing (no mocks) with temp datasets for isolation
- FastAPI-specific metadata validation
- Contract compliance with parser-interfaces.py
"""

import json
import os
import tempfile
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch

import pytest
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Import contracts - these imports will FAIL until implementation exists
try:
    from specs.contracts.parser_interfaces import (
        FastAPIParser,
        BigQueryWriter,
        ChunkMetadata,
        ParseResult,
        SourceType,
        ErrorClass
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False

# Import implementation - these imports will FAIL until implementation exists
try:
    from src.parsers.fastapi_parser import FastAPIParser
    from src.storage.bigquery_writer import BigQueryWriter
    IMPLEMENTATION_AVAILABLE = True
except ImportError:
    IMPLEMENTATION_AVAILABLE = False


class TestFastAPIIngestion:
    """Integration tests for FastAPI project ingestion to BigQuery."""

    @pytest.fixture(scope="class")
    def temp_dataset_name(self):
        """Create temporary BigQuery dataset for testing isolation."""
        return f"test_fastapi_ingestion_{uuid.uuid4().hex[:8]}"

    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """Create BigQuery client for integration testing."""
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'konveyn2ai')
        return bigquery.Client(project=project_id)

    @pytest.fixture(scope="class")
    def bigquery_writer(self, bigquery_client, temp_dataset_name):
        """BigQuery writer instance - will fail until implementation exists."""
        if not IMPLEMENTATION_AVAILABLE:
            pytest.skip("BigQueryWriter implementation not available yet (TDD Red phase)")

        writer = BigQueryWriter(
            client=bigquery_client,
            dataset_name=temp_dataset_name
        )

        # Create temporary dataset and tables for testing
        writer.create_tables_if_not_exist(temp_dataset_name)

        yield writer

        # Cleanup: Delete temporary dataset after tests
        try:
            dataset_ref = bigquery_client.dataset(temp_dataset_name)
            bigquery_client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
        except Exception as e:
            print(f"Warning: Could not cleanup dataset {temp_dataset_name}: {e}")

    @pytest.fixture(scope="class")
    def fastapi_parser(self):
        """FastAPI parser instance - will fail until implementation exists."""
        if not IMPLEMENTATION_AVAILABLE:
            pytest.skip("FastAPIParser implementation not available yet (TDD Red phase)")
        return FastAPIParser()

    @pytest.fixture
    def sample_fastapi_route_file(self):
        """Create sample FastAPI route file for testing."""
        return textwrap.dedent("""
        from fastapi import FastAPI, HTTPException, Depends
        from pydantic import BaseModel
        from typing import List, Optional

        app = FastAPI(title="Sample API", version="1.0.0")

        class User(BaseModel):
            id: int
            username: str
            email: str
            is_active: bool = True
            profile: Optional[dict] = None

        class UserCreate(BaseModel):
            username: str
            email: str
            password: str

        class UserResponse(BaseModel):
            id: int
            username: str
            email: str
            is_active: bool

        @app.get("/users", response_model=List[UserResponse], tags=["users"])
        async def get_users(skip: int = 0, limit: int = 100):
            '''Get all users with pagination'''
            return []

        @app.get("/users/{user_id}", response_model=UserResponse, tags=["users"])
        async def get_user(user_id: int):
            '''Get user by ID'''
            if user_id < 1:
                raise HTTPException(status_code=404, detail="User not found")
            return {"id": user_id, "username": "test", "email": "test@example.com", "is_active": True}

        @app.post("/users", response_model=UserResponse, status_code=201, tags=["users"])
        async def create_user(user: UserCreate):
            '''Create a new user'''
            return {"id": 1, "username": user.username, "email": user.email, "is_active": True}

        @app.put("/users/{user_id}", response_model=UserResponse, tags=["users"])
        async def update_user(user_id: int, user: UserCreate):
            '''Update existing user'''
            return {"id": user_id, "username": user.username, "email": user.email, "is_active": True}

        @app.delete("/users/{user_id}", status_code=204, tags=["users"])
        async def delete_user(user_id: int):
            '''Delete user by ID'''
            pass
        """)

    @pytest.fixture
    def sample_fastapi_models_file(self):
        """Create sample Pydantic models file for testing."""
        return textwrap.dedent("""
        from pydantic import BaseModel, Field, validator
        from typing import List, Optional, Union
        from datetime import datetime
        from enum import Enum

        class UserRole(str, Enum):
            ADMIN = "admin"
            USER = "user"
            GUEST = "guest"

        class BaseUser(BaseModel):
            username: str = Field(..., min_length=3, max_length=50)
            email: str = Field(..., regex=r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$')

            @validator('email')
            def validate_email_domain(cls, v):
                if not v.endswith(('@company.com', '@partner.com')):
                    raise ValueError('Email must be from allowed domains')
                return v

        class User(BaseUser):
            id: int = Field(..., gt=0)
            role: UserRole = UserRole.USER
            is_active: bool = True
            created_at: datetime
            last_login: Optional[datetime] = None
            metadata: Optional[dict] = None

            class Config:
                schema_extra = {
                    "example": {
                        "id": 1,
                        "username": "johndoe",
                        "email": "john@company.com",
                        "role": "user",
                        "is_active": True,
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                }

        class UserStats(BaseModel):
            total_users: int
            active_users: int
            users_by_role: dict[UserRole, int]
            growth_rate: float = Field(..., ge=0, le=100)
        """)

    @pytest.fixture
    def sample_openapi_spec(self):
        """Create sample OpenAPI specification for testing."""
        return {
            "openapi": "3.0.2",
            "info": {
                "title": "Sample API",
                "version": "1.0.0",
                "description": "A sample FastAPI application for testing"
            },
            "servers": [
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {
                "/users": {
                    "get": {
                        "summary": "Get Users",
                        "operationId": "get_users_users_get",
                        "parameters": [
                            {
                                "name": "skip",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer", "default": 0}
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "required": False,
                                "schema": {"type": "integer", "default": 100}
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successful Response",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/UserResponse"}
                                        }
                                    }
                                }
                            }
                        },
                        "tags": ["users"]
                    },
                    "post": {
                        "summary": "Create User",
                        "operationId": "create_user_users_post",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserCreate"}
                                }
                            },
                            "required": True
                        },
                        "responses": {
                            "201": {
                                "description": "Successful Response",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/UserResponse"}
                                    }
                                }
                            }
                        },
                        "tags": ["users"]
                    }
                },
                "/users/{user_id}": {
                    "get": {
                        "summary": "Get User",
                        "operationId": "get_user_users__user_id__get",
                        "parameters": [
                            {
                                "name": "user_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "integer"}
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Successful Response",
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/UserResponse"}
                                    }
                                }
                            }
                        },
                        "tags": ["users"]
                    }
                }
            },
            "components": {
                "schemas": {
                    "UserCreate": {
                        "title": "UserCreate",
                        "required": ["username", "email", "password"],
                        "type": "object",
                        "properties": {
                            "username": {"type": "string"},
                            "email": {"type": "string"},
                            "password": {"type": "string"}
                        }
                    },
                    "UserResponse": {
                        "title": "UserResponse",
                        "required": ["id", "username", "email", "is_active"],
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "username": {"type": "string"},
                            "email": {"type": "string"},
                            "is_active": {"type": "boolean"}
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def temp_fastapi_project(self, sample_fastapi_route_file, sample_fastapi_models_file, sample_openapi_spec):
        """Create temporary FastAPI project directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create project structure
            (project_path / "app").mkdir()
            (project_path / "app" / "__init__.py").write_text("")

            # Write main route file
            (project_path / "app" / "main.py").write_text(sample_fastapi_route_file)

            # Write models file
            (project_path / "app" / "models.py").write_text(sample_fastapi_models_file)

            # Write OpenAPI spec
            with open(project_path / "openapi.json", "w") as f:
                json.dump(sample_openapi_spec, f, indent=2)

            # Create additional files
            (project_path / "requirements.txt").write_text("fastapi==0.104.1\nuvicorn==0.24.0\npydantic==2.5.0")

            yield project_path

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_contracts_available(self):
        """Test that parser interfaces contracts are available."""
        if not CONTRACTS_AVAILABLE:
            pytest.fail("Parser interface contracts not available - check specs/contracts/parser_interfaces.py")

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_implementation_not_available_yet(self):
        """Test that implementation is not available yet (TDD Red phase)."""
        assert not IMPLEMENTATION_AVAILABLE, (
            "Implementation already exists! This test should fail initially. "
            "Implementation should be created after this test is written (TDD approach)."
        )

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_parse_single_fastapi_file(self, fastapi_parser, temp_fastapi_project):
        """Test parsing a single FastAPI route file."""
        route_file = temp_fastapi_project / "app" / "main.py"

        # This will fail until parser implementation exists
        result = fastapi_parser.parse_file(str(route_file))

        # Validate parse result structure
        assert isinstance(result, ParseResult)
        assert isinstance(result.chunks, list)
        assert isinstance(result.errors, list)
        assert result.files_processed == 1
        assert result.processing_duration_ms > 0

        # Should find multiple chunks for routes and models
        assert len(result.chunks) >= 8, "Should find at least 8 chunks (5 routes + 3 models)"

        # Validate FastAPI-specific metadata
        route_chunks = [chunk for chunk in result.chunks
                       if chunk.source_metadata.get('fastapi_http_method')]
        assert len(route_chunks) == 5, "Should find 5 HTTP routes"

        # Check specific route metadata
        get_users_chunk = next((chunk for chunk in route_chunks
                               if chunk.source_metadata.get('fastapi_route_path') == '/users'
                               and chunk.source_metadata.get('fastapi_http_method') == 'GET'), None)
        assert get_users_chunk is not None, "Should find GET /users route"

        expected_metadata = {
            'fastapi_http_method': 'GET',
            'fastapi_route_path': '/users',
            'fastapi_operation_id': 'get_users',
            'fastapi_response_model': 'List[UserResponse]',
            'fastapi_status_codes': [200]
        }

        for key, expected_value in expected_metadata.items():
            assert get_users_chunk.source_metadata.get(key) == expected_value, \
                f"Expected {key}={expected_value}, got {get_users_chunk.source_metadata.get(key)}"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_parse_openapi_specification(self, fastapi_parser, temp_fastapi_project):
        """Test parsing OpenAPI specification file."""
        openapi_file = temp_fastapi_project / "openapi.json"

        # This will fail until parser implementation exists
        result = fastapi_parser.parse_file(str(openapi_file))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) >= 3, "Should find chunks for paths and schemas"

        # Check for path chunks
        path_chunks = [chunk for chunk in result.chunks
                      if 'openapi_path' in chunk.source_metadata]
        assert len(path_chunks) >= 2, "Should find at least 2 API paths"

        # Check for schema chunks
        schema_chunks = [chunk for chunk in result.chunks
                        if 'openapi_schema' in chunk.source_metadata]
        assert len(schema_chunks) >= 2, "Should find at least 2 schema definitions"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_parse_fastapi_project_directory(self, fastapi_parser, temp_fastapi_project):
        """Test parsing entire FastAPI project directory."""
        # This will fail until parser implementation exists
        result = fastapi_parser.parse_directory(str(temp_fastapi_project))

        assert isinstance(result, ParseResult)
        assert result.files_processed >= 3, "Should process at least 3 files (main.py, models.py, openapi.json)"
        assert len(result.chunks) >= 15, "Should find substantial number of chunks from all files"

        # Validate we have chunks from different source types
        source_files = set(chunk.source_uri for chunk in result.chunks)
        assert any('main.py' in uri for uri in source_files), "Should have chunks from main.py"
        assert any('models.py' in uri for uri in source_files), "Should have chunks from models.py"
        assert any('openapi.json' in uri for uri in source_files), "Should have chunks from openapi.json"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_pydantic_model_parsing(self, fastapi_parser, temp_fastapi_project):
        """Test parsing Pydantic models with proper metadata extraction."""
        models_file = temp_fastapi_project / "app" / "models.py"

        # This will fail until parser implementation exists
        result = fastapi_parser.parse_file(str(models_file))

        model_chunks = [chunk for chunk in result.chunks
                       if chunk.source_metadata.get('fastapi_model_type')]
        assert len(model_chunks) >= 4, "Should find at least 4 model definitions"

        # Check specific model metadata
        user_model = next((chunk for chunk in model_chunks
                          if 'User' in chunk.content_text and 'BaseUser' not in chunk.content_text), None)
        assert user_model is not None, "Should find User model"

        expected_fields = ['id', 'role', 'is_active', 'created_at', 'last_login', 'metadata']
        model_fields = user_model.source_metadata.get('fastapi_model_fields', [])
        for field in expected_fields:
            assert any(field in f for f in model_fields), f"Should find field {field} in User model"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_fastapi_content_validation(self, fastapi_parser):
        """Test content validation for FastAPI files."""
        # This will fail until parser implementation exists

        # Valid FastAPI content
        valid_content = """
        from fastapi import FastAPI
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}
        """
        assert fastapi_parser.validate_content(valid_content) is True

        # Invalid content (not FastAPI)
        invalid_content = """
        import random
        def some_function():
            return random.randint(1, 10)
        """
        assert fastapi_parser.validate_content(invalid_content) is False

        # OpenAPI spec validation
        openapi_content = '{"openapi": "3.0.2", "info": {"title": "Test", "version": "1.0.0"}}'
        assert fastapi_parser.validate_content(openapi_content) is True

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_end_to_end_bigquery_ingestion(self, fastapi_parser, bigquery_writer, temp_fastapi_project, temp_dataset_name):
        """Test complete end-to-end ingestion from FastAPI parsing to BigQuery storage."""
        # Parse the FastAPI project
        parse_result = fastapi_parser.parse_directory(str(temp_fastapi_project))

        # Write chunks to BigQuery
        chunks_written = bigquery_writer.write_chunks(parse_result.chunks, "source_metadata")
        assert chunks_written == len(parse_result.chunks), "All chunks should be written to BigQuery"

        # Write errors to BigQuery if any
        if parse_result.errors:
            errors_written = bigquery_writer.write_errors(parse_result.errors, "source_metadata_errors")
            assert errors_written == len(parse_result.errors)

        # Log ingestion run
        run_info = {
            "source_type": "fastapi",
            "source_path": str(temp_fastapi_project),
            "files_processed": parse_result.files_processed,
            "chunks_generated": len(parse_result.chunks),
            "errors_count": len(parse_result.errors),
            "processing_duration_ms": parse_result.processing_duration_ms,
            "parser_version": fastapi_parser.version
        }
        run_id = bigquery_writer.log_ingestion_run(run_info, "ingestion_log")
        assert run_id is not None, "Should return ingestion run ID"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_bigquery_data_validation(self, bigquery_client, temp_dataset_name, fastapi_parser, bigquery_writer, temp_fastapi_project):
        """Test that data written to BigQuery has correct structure and content."""
        # Parse and ingest data
        parse_result = fastapi_parser.parse_directory(str(temp_fastapi_project))
        bigquery_writer.write_chunks(parse_result.chunks, "source_metadata")

        # Query BigQuery to validate data
        query = f"""
        SELECT
            source_type,
            artifact_id,
            content_text,
            fastapi_http_method,
            fastapi_route_path,
            fastapi_operation_id,
            fastapi_response_model
        FROM `{bigquery_client.project}.{temp_dataset_name}.source_metadata`
        WHERE source_type = 'fastapi'
        AND fastapi_http_method IS NOT NULL
        ORDER BY fastapi_route_path, fastapi_http_method
        """

        rows = list(bigquery_client.query(query))
        assert len(rows) >= 5, "Should have at least 5 route records in BigQuery"

        # Validate specific route data
        get_users_row = next((row for row in rows
                             if row.fastapi_route_path == '/users'
                             and row.fastapi_http_method == 'GET'), None)
        assert get_users_row is not None, "Should find GET /users route in BigQuery"
        assert get_users_row.fastapi_operation_id == 'get_users'
        assert 'List[UserResponse]' in get_users_row.fastapi_response_model
        assert 'async def get_users' in get_users_row.content_text

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_error_handling_and_logging(self, fastapi_parser):
        """Test error handling for invalid FastAPI files."""
        # Create invalid Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax $$$ @#!")
            invalid_file = f.name

        try:
            # This will fail until parser implementation exists
            result = fastapi_parser.parse_file(invalid_file)

            # Should have errors but not crash
            assert isinstance(result, ParseResult)
            assert len(result.errors) > 0, "Should capture parsing errors"
            assert result.files_processed == 1, "Should still count as processed"

            # Check error details
            error = result.errors[0]
            assert error.source_type == SourceType.FASTAPI
            assert error.error_class == ErrorClass.PARSING
            assert 'syntax' in error.error_msg.lower()

        finally:
            os.unlink(invalid_file)

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_chunking_strategy(self, fastapi_parser, sample_fastapi_route_file):
        """Test that FastAPI content is chunked appropriately."""
        # Create a large FastAPI file to test chunking
        large_content = sample_fastapi_route_file + "\n" * 50 + sample_fastapi_route_file

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_content)
            large_file = f.name

        try:
            # This will fail until parser implementation exists
            result = fastapi_parser.parse_file(large_file)

            # Should create multiple chunks for large files
            assert len(result.chunks) > 1, "Should create multiple chunks for large files"

            # Chunks should have proper overlap and structure
            for chunk in result.chunks:
                assert len(chunk.content_text) > 0, "Chunks should have content"
                assert chunk.content_hash is not None, "Chunks should have content hash"
                assert chunk.content_tokens is not None, "Chunks should have token count"

        finally:
            os.unlink(large_file)

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_artifact_id_generation(self, fastapi_parser, temp_fastapi_project):
        """Test that artifact IDs are generated consistently and uniquely."""
        route_file = temp_fastapi_project / "app" / "main.py"

        # Parse same file twice
        result1 = fastapi_parser.parse_file(str(route_file))
        result2 = fastapi_parser.parse_file(str(route_file))

        # Artifact IDs should be consistent between runs
        artifact_ids_1 = [chunk.artifact_id for chunk in result1.chunks]
        artifact_ids_2 = [chunk.artifact_id for chunk in result2.chunks]
        assert artifact_ids_1 == artifact_ids_2, "Artifact IDs should be deterministic"

        # All artifact IDs should be unique within a parse result
        assert len(set(artifact_ids_1)) == len(artifact_ids_1), "All artifact IDs should be unique"

        # Artifact IDs should follow expected pattern
        for artifact_id in artifact_ids_1:
            assert artifact_id.startswith('fastapi://'), "Artifact IDs should start with fastapi://"
            assert str(route_file) in artifact_id, "Artifact IDs should contain file path"

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_source_metadata_completeness(self, fastapi_parser, temp_fastapi_project):
        """Test that all required source metadata fields are populated."""
        result = fastapi_parser.parse_directory(str(temp_fastapi_project))

        required_fields = [
            'source_type', 'artifact_id', 'content_text', 'content_hash',
            'source_uri', 'collected_at'
        ]

        for chunk in result.chunks:
            # Check required common fields
            for field in required_fields:
                value = getattr(chunk, field, None)
                assert value is not None, f"Required field {field} is missing in chunk {chunk.artifact_id}"
                assert value != "", f"Required field {field} is empty in chunk {chunk.artifact_id}"

            # Check source type
            assert chunk.source_type == SourceType.FASTAPI

            # Check that FastAPI-specific metadata is present where appropriate
            if any(keyword in chunk.content_text.lower() for keyword in ['@app.', 'async def', 'def ']):
                # Should have FastAPI-specific metadata for route chunks
                if '@app.' in chunk.content_text:
                    assert chunk.source_metadata.get('fastapi_http_method') is not None
                    assert chunk.source_metadata.get('fastapi_route_path') is not None

    @pytest.mark.integration
    @pytest.mark.bigquery
    def test_performance_benchmarks(self, fastapi_parser, temp_fastapi_project):
        """Test that parsing performance meets acceptable benchmarks."""
        start_time = datetime.now()
        result = fastapi_parser.parse_directory(str(temp_fastapi_project))
        end_time = datetime.now()

        total_duration_ms = (end_time - start_time).total_seconds() * 1000

        # Performance assertions
        assert result.processing_duration_ms <= total_duration_ms, "Reported duration should be <= actual duration"
        assert result.processing_duration_ms < 5000, "Parsing should complete within 5 seconds for small project"

        # Throughput assertions
        if result.files_processed > 0:
            ms_per_file = result.processing_duration_ms / result.files_processed
            assert ms_per_file < 2000, "Should process each file in under 2 seconds"

        if len(result.chunks) > 0:
            ms_per_chunk = result.processing_duration_ms / len(result.chunks)
            assert ms_per_chunk < 500, "Should generate each chunk in under 500ms"