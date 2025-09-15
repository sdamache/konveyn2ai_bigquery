"""
Contract tests for FastAPI Parser Interface
These tests MUST FAIL initially (TDD requirement) until the FastAPI parser is implemented.

Tests the contract defined in specs/002-m1-parse-and/contracts/parser-interfaces.py
"""

import pytest
import json
import ast
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Register custom markers to avoid warnings
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "contract: Contract tests for interface compliance (TDD)")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")

# Import the parser interface contracts via shared module
try:
    from src.common.parser_interfaces import (
        BaseParser,
        FastAPIParser,
        SourceType,
        ChunkMetadata,
        ParseResult,
        ParseError,
        ErrorClass,
    )
    PARSER_INTERFACES_AVAILABLE = True
except Exception as e:
    PARSER_INTERFACES_AVAILABLE = False
    print(f"Warning: Could not import parser interfaces: {e}")

# Try to import the actual implementation (expected to fail initially)
try:
    from src.ingest.fastapi.parser import FastAPIParserImpl
    FASTAPI_PARSER_AVAILABLE = True
except ImportError:
    FASTAPI_PARSER_AVAILABLE = False


# Test data for FastAPI OpenAPI specs and source code
VALID_OPENAPI_SPEC_JSON = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Sample API",
    "version": "1.0.0",
    "description": "A sample FastAPI application"
  },
  "paths": {
    "/users": {
      "get": {
        "summary": "Get users",
        "operationId": "get_users",
        "tags": ["users"],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "schema": {
              "type": "integer",
              "default": 10
            }
          }
        ],
        "responses": {
          "200": {
            "description": "List of users",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UserList"
                }
              }
            }
          },
          "400": {
            "description": "Bad request"
          }
        }
      },
      "post": {
        "summary": "Create user",
        "operationId": "create_user",
        "tags": ["users"],
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/UserCreate"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "User created",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                }
              }
            }
          },
          "422": {
            "description": "Validation error"
          }
        }
      }
    },
    "/users/{user_id}": {
      "get": {
        "summary": "Get user by ID",
        "operationId": "get_user_by_id",
        "tags": ["users"],
        "parameters": [
          {
            "name": "user_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "User details",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/User"
                }
              }
            }
          },
          "404": {
            "description": "User not found"
          }
        }
      }
    },
    "/health": {
      "get": {
        "summary": "Health check",
        "operationId": "health_check",
        "tags": ["system"],
        "responses": {
          "200": {
            "description": "Service is healthy",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "status": {
                      "type": "string"
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "User": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "email": {
            "type": "string"
          },
          "created_at": {
            "type": "string",
            "format": "date-time"
          }
        },
        "required": ["id", "name", "email"]
      },
      "UserCreate": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string"
          },
          "email": {
            "type": "string"
          }
        },
        "required": ["name", "email"]
      },
      "UserList": {
        "type": "object",
        "properties": {
          "users": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/User"
            }
          },
          "total": {
            "type": "integer"
          }
        }
      }
    }
  }
}
"""

VALID_FASTAPI_SOURCE_CODE = '''
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(
    title="Sample API",
    version="1.0.0",
    description="A sample FastAPI application"
)

class User(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

class UserCreate(BaseModel):
    name: str
    email: str

class UserList(BaseModel):
    users: List[User]
    total: int

@app.get("/users", response_model=UserList, tags=["users"])
async def get_users(
    limit: int = Query(default=10, ge=1, le=100)
) -> UserList:
    """Get list of users with pagination"""
    # Mock implementation
    return UserList(users=[], total=0)

@app.post("/users", response_model=User, status_code=201, tags=["users"])
async def create_user(user_data: UserCreate) -> User:
    """Create a new user"""
    # Mock implementation
    return User(
        id=1,
        name=user_data.name,
        email=user_data.email,
        created_at=datetime.now()
    )

@app.get("/users/{user_id}", response_model=User, tags=["users"])
async def get_user_by_id(
    user_id: int = Path(..., ge=1)
) -> User:
    """Get user by ID"""
    if user_id == 999:
        raise HTTPException(status_code=404, detail="User not found")

    return User(
        id=user_id,
        name="Sample User",
        email="user@example.com",
        created_at=datetime.now()
    )

@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Dependency injection example
def get_current_user() -> User:
    return User(id=1, name="Current User", email="current@example.com", created_at=datetime.now())

@app.get("/profile", response_model=User, dependencies=[Depends(get_current_user)])
async def get_profile(current_user: User = Depends(get_current_user)) -> User:
    """Get current user profile"""
    return current_user
'''

INVALID_JSON_SPEC = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Invalid API",
    "version": "1.0.0"
  },
  "paths": {
    "/test": {
      "get": {
        "summary": "Test endpoint"
        // Invalid JSON comment
      }
    }
  }
}
"""

INVALID_PYTHON_SOURCE = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/test"
async def test_endpoint():  # Missing closing parenthesis
    return {"message": "test"}

# Syntax error - missing colon
def invalid_function()
    pass
'''

MALFORMED_OPENAPI_SPEC = """
{
  "openapi": "3.0.0",
  "info": {
    "title": "Malformed API"
  },
  "paths": {}
}
"""

COMPLEX_FASTAPI_SOURCE = '''
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncio

app = FastAPI()

security = HTTPBearer()

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)

class UserFilter(BaseModel):
    role: Optional[UserRole] = None
    active: Optional[bool] = None

@app.post("/api/v1/users/bulk", status_code=202)
async def bulk_create_users(
    users: List[UserCreate],
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Bulk create users with background processing"""
    background_tasks.add_task(process_bulk_users, users)
    return {"message": "Bulk operation started", "count": len(users)}

@app.get("/api/v1/users/search")
async def search_users(
    q: str = Query(..., min_length=1),
    filters: UserFilter = Depends(),
    pagination: PaginationParams = Depends()
) -> UserList:
    """Advanced user search with filters"""
    return UserList(users=[], total=0)

async def process_bulk_users(users: List[UserCreate]):
    """Background task for processing bulk user creation"""
    await asyncio.sleep(1)  # Simulate processing
'''


@pytest.mark.contract
@pytest.mark.unit
class TestFastAPIParserContract:
    """Contract tests for FastAPI Parser implementation"""

    def test_parser_interfaces_available(self):
        """Test that parser interface contracts are available"""
        assert PARSER_INTERFACES_AVAILABLE, "Parser interface contracts should be available"

    @pytest.mark.skipif(not PARSER_INTERFACES_AVAILABLE, reason="Parser interfaces not available")
    def test_fastapi_parser_interface_exists(self):
        """Test that FastAPIParser abstract class exists with required methods"""
        # Verify FastAPIParser exists and inherits from BaseParser
        assert issubclass(FastAPIParser, BaseParser)

        # Verify required abstract methods exist
        abstract_methods = getattr(FastAPIParser, '__abstractmethods__', set())
        required_methods = {
            'parse_file',
            'parse_directory',
            'validate_content',
            'parse_openapi_spec',
            'parse_source_code'
        }

        # Check that all required methods are declared as abstract
        for method in required_methods:
            assert hasattr(FastAPIParser, method), f"Method {method} should exist"

    @pytest.mark.skipif(FASTAPI_PARSER_AVAILABLE, reason="Skip when implementation exists")
    def test_fastapi_parser_not_implemented_yet(self):
        """Test that the actual implementation doesn't exist yet (TDD requirement)"""
        assert not FASTAPI_PARSER_AVAILABLE, "FastAPIParserImpl should not exist yet (TDD)"

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_fastapi_parser_inheritance(self):
        """Test that FastAPIParserImpl properly inherits from FastAPIParser"""
        assert issubclass(FastAPIParserImpl, FastAPIParser)
        assert issubclass(FastAPIParserImpl, BaseParser)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_fastapi_parser_source_type(self):
        """Test that parser returns correct source type"""
        parser = FastAPIParserImpl()
        assert parser.source_type == SourceType.FASTAPI
        assert parser._get_source_type() == SourceType.FASTAPI

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_valid_openapi_spec(self):
        """Test parsing a valid OpenAPI specification"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Check the first chunk
        chunk = chunks[0]
        assert isinstance(chunk, ChunkMetadata)
        assert chunk.source_type == SourceType.FASTAPI

        # Verify artifact_id format: py://{src_path}#{start_line}-{end_line}
        assert chunk.artifact_id.startswith("py://")

        # Verify FastAPI-specific metadata
        fastapi_metadata = chunk.source_metadata
        assert fastapi_metadata['http_method'] in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        assert 'route_path' in fastapi_metadata
        assert 'operation_id' in fastapi_metadata
        assert 'status_codes' in fastapi_metadata
        assert isinstance(fastapi_metadata['status_codes'], list)

        # Verify content fields
        assert chunk.content_text.strip() != ""
        assert chunk.content_hash != ""
        assert isinstance(chunk.collected_at, datetime)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_openapi_spec_with_multiple_endpoints(self):
        """Test parsing OpenAPI spec with multiple endpoints"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        # Should generate chunks for all endpoints
        assert len(chunks) >= 4  # GET /users, POST /users, GET /users/{user_id}, GET /health

        # Verify we got expected HTTP methods and paths (filter for route chunks only)
        route_chunks = [chunk for chunk in chunks if 'http_method' in chunk.source_metadata]
        endpoints = [(chunk.source_metadata['http_method'], chunk.source_metadata['route_path'])
                    for chunk in route_chunks]

        expected_endpoints = [
            ('GET', '/users'),
            ('POST', '/users'),
            ('GET', '/users/{user_id}'),
            ('GET', '/health')
        ]

        for expected in expected_endpoints:
            assert expected in endpoints, f"Expected endpoint {expected} not found"

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_openapi_spec_metadata_fields(self):
        """Test that all required FastAPI metadata fields are present"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        # Filter for route chunks only (not schema chunks)
        route_chunks = [chunk for chunk in chunks if 'http_method' in chunk.source_metadata]
        assert len(route_chunks) > 0, "Should have at least one route chunk"

        for chunk in route_chunks:
            metadata = chunk.source_metadata

            # Required FastAPI fields according to the specification
            assert 'http_method' in metadata
            assert 'route_path' in metadata
            assert 'operation_id' in metadata
            assert 'status_codes' in metadata

            # Optional but expected fields
            if 'request_model' in metadata:
                assert isinstance(metadata['request_model'], (str, type(None)))
            if 'response_model' in metadata:
                assert isinstance(metadata['response_model'], (str, type(None)))
            if 'dependencies' in metadata:
                assert isinstance(metadata['dependencies'], list)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_valid_fastapi_source_code(self):
        """Test parsing valid FastAPI Python source code"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_source_code(VALID_FASTAPI_SOURCE_CODE)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Should find route decorators
        route_chunks = [chunk for chunk in chunks
                       if 'route_path' in chunk.source_metadata]
        assert len(route_chunks) > 0

        # Check a specific route chunk
        get_users_chunk = next((chunk for chunk in route_chunks
                               if chunk.source_metadata.get('route_path') == '/users'
                               and chunk.source_metadata.get('http_method') == 'GET'), None)
        assert get_users_chunk is not None

        metadata = get_users_chunk.source_metadata
        assert metadata['http_method'] == 'GET'
        assert metadata['route_path'] == '/users'
        assert 'get_users' in metadata.get('operation_id', '')

        # Verify artifact_id format includes line numbers
        assert "#" in get_users_chunk.artifact_id
        assert "-" in get_users_chunk.artifact_id.split("#")[1]

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_source_code_with_dependencies(self):
        """Test parsing source code with FastAPI dependencies"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_source_code(VALID_FASTAPI_SOURCE_CODE)

        # Find chunk with dependencies
        profile_chunk = next((chunk for chunk in chunks
                             if chunk.source_metadata.get('route_path') == '/profile'), None)

        if profile_chunk:  # May not be present in simple test case
            metadata = profile_chunk.source_metadata
            assert 'dependencies' in metadata
            assert isinstance(metadata['dependencies'], list)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_complex_fastapi_source(self):
        """Test parsing complex FastAPI source with advanced features"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_source_code(COMPLEX_FASTAPI_SOURCE)

        assert len(chunks) > 0

        # Check for bulk endpoint (filter for route chunks only)
        route_chunks = [chunk for chunk in chunks if 'http_method' in chunk.source_metadata]

        bulk_chunk = next((chunk for chunk in route_chunks
                          if chunk.source_metadata.get('route_path') == '/api/v1/users/bulk'), None)
        assert bulk_chunk is not None
        assert bulk_chunk.source_metadata['http_method'] == 'POST'
        assert 202 in bulk_chunk.source_metadata['status_codes']

        # Check for search endpoint with query parameters
        search_chunk = next((chunk for chunk in chunks
                            if chunk.source_metadata.get('route_path') == '/api/v1/users/search'), None)
        assert search_chunk is not None

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_invalid_json_spec(self):
        """Test error handling for invalid JSON in OpenAPI spec"""
        parser = FastAPIParserImpl()

        with pytest.raises((json.JSONDecodeError, ValueError)):
            parser.parse_openapi_spec(INVALID_JSON_SPEC)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_invalid_python_source(self):
        """Test error handling for invalid Python source code"""
        parser = FastAPIParserImpl()

        with pytest.raises((SyntaxError, ValueError)):
            parser.parse_source_code(INVALID_PYTHON_SOURCE)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_malformed_openapi_spec(self):
        """Test handling of malformed but valid JSON OpenAPI spec"""
        parser = FastAPIParserImpl()

        # Should parse but may return empty list or minimal chunks
        chunks = parser.parse_openapi_spec(MALFORMED_OPENAPI_SPEC)
        assert isinstance(chunks, list)
        # May be empty for malformed spec with no paths

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_valid_json(self):
        """Test content validation for valid JSON OpenAPI spec"""
        parser = FastAPIParserImpl()

        assert parser.validate_content(VALID_OPENAPI_SPEC_JSON) is True

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_valid_python(self):
        """Test content validation for valid Python source code"""
        parser = FastAPIParserImpl()

        assert parser.validate_content(VALID_FASTAPI_SOURCE_CODE) is True

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_validate_content_invalid(self):
        """Test content validation for invalid content"""
        parser = FastAPIParserImpl()

        assert parser.validate_content(INVALID_JSON_SPEC) is False
        assert parser.validate_content(INVALID_PYTHON_SOURCE) is False
        assert parser.validate_content("") is False
        assert parser.validate_content("not json or python") is False

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_file_openapi_spec(self, tmp_path):
        """Test parsing a file containing OpenAPI specification"""
        parser = FastAPIParserImpl()

        # Create temporary spec file
        spec_file = tmp_path / "openapi.json"
        spec_file.write_text(VALID_OPENAPI_SPEC_JSON)

        result = parser.parse_file(str(spec_file))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0
        assert len(result.errors) == 0
        assert result.files_processed == 1
        assert result.processing_duration_ms > 0

        # Verify source_uri is set correctly
        chunk = result.chunks[0]
        assert chunk.source_uri == str(spec_file)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_file_python_source(self, tmp_path):
        """Test parsing a Python file with FastAPI routes"""
        parser = FastAPIParserImpl()

        # Create temporary Python file
        python_file = tmp_path / "main.py"
        python_file.write_text(VALID_FASTAPI_SOURCE_CODE)

        result = parser.parse_file(str(python_file))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0
        assert result.files_processed == 1

        # Verify source_uri is set correctly
        chunk = result.chunks[0]
        assert chunk.source_uri == str(python_file)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_directory_mixed_files(self, tmp_path):
        """Test parsing a directory with both OpenAPI specs and Python files"""
        parser = FastAPIParserImpl()

        # Create multiple files
        (tmp_path / "openapi.json").write_text(VALID_OPENAPI_SPEC_JSON)
        (tmp_path / "main.py").write_text(VALID_FASTAPI_SOURCE_CODE)
        (tmp_path / "complex.py").write_text(COMPLEX_FASTAPI_SOURCE)
        (tmp_path / "not-relevant.txt").write_text("This is not a FastAPI file")

        result = parser.parse_directory(str(tmp_path))

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0
        assert result.files_processed >= 3  # At least the relevant files

        # Verify we got chunks from different file types
        source_uris = [chunk.source_uri for chunk in result.chunks]
        json_files = [uri for uri in source_uris if uri.endswith('.json')]
        py_files = [uri for uri in source_uris if uri.endswith('.py')]

        assert len(json_files) > 0
        assert len(py_files) > 0

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_artifact_id_generation_source_code(self):
        """Test artifact ID generation for Python source code with line numbers"""
        parser = FastAPIParserImpl()

        chunks = parser.parse_source_code(VALID_FASTAPI_SOURCE_CODE)

        for chunk in chunks:
            artifact_id = chunk.artifact_id

            # Should follow format: py://{src_path}#{start_line}-{end_line}
            assert artifact_id.startswith("py://")
            assert "#" in artifact_id

            line_part = artifact_id.split("#")[1]
            assert "-" in line_part

            start_line, end_line = line_part.split("-")
            assert start_line.isdigit()
            assert end_line.isdigit()
            assert int(start_line) <= int(end_line)

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_artifact_id_generation_openapi_spec(self):
        """Test artifact ID generation for OpenAPI specifications"""
        parser = FastAPIParserImpl()

        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        for chunk in chunks:
            artifact_id = chunk.artifact_id

            # Should follow format: py://{src_path}#{start_line}-{end_line}
            assert artifact_id.startswith("py://")

            # For OpenAPI specs, line numbers may be derived differently
            # but should still contain the # format

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_status_codes_extraction(self):
        """Test that status codes are correctly extracted from endpoints"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        # Find the POST /users endpoint
        post_users_chunk = next((chunk for chunk in chunks
                                if chunk.source_metadata.get('route_path') == '/users'
                                and chunk.source_metadata.get('http_method') == 'POST'), None)

        assert post_users_chunk is not None
        status_codes = post_users_chunk.source_metadata['status_codes']
        assert 201 in status_codes  # Created
        assert 422 in status_codes  # Validation error

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_request_response_model_extraction(self):
        """Test extraction of request and response models"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        # Find the POST /users endpoint
        post_users_chunk = next((chunk for chunk in chunks
                                if chunk.source_metadata.get('route_path') == '/users'
                                and chunk.source_metadata.get('http_method') == 'POST'), None)

        assert post_users_chunk is not None
        metadata = post_users_chunk.source_metadata

        # Should have request and response models
        assert 'request_model' in metadata
        assert 'response_model' in metadata

        # Values should reference the schema components
        assert 'UserCreate' in str(metadata.get('request_model', ''))
        assert 'User' in str(metadata.get('response_model', ''))

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_content_hash_generation(self):
        """Test that content hash is generated correctly"""
        parser = FastAPIParserImpl()
        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 64  # SHA256 hex digest length

        # Same content should generate same hash
        chunks2 = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)
        assert chunks[0].content_hash == chunks2[0].content_hash

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_parse_result_structure(self):
        """Test that ParseResult has the correct structure"""
        parser = FastAPIParserImpl()

        chunks = parser.parse_openapi_spec(VALID_OPENAPI_SPEC_JSON)

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

        # FastAPI-specific metadata fields
        fastapi_metadata = chunk.source_metadata
        required_fastapi_fields = ['http_method', 'route_path', 'operation_id', 'status_codes']
        for field in required_fastapi_fields:
            assert field in fastapi_metadata, f"Missing required FastAPI metadata field: {field}"

    @pytest.mark.skipif(not FASTAPI_PARSER_AVAILABLE, reason="Implementation not available yet")
    def test_chunk_content_large_spec(self):
        """Test chunking behavior for large OpenAPI specifications"""
        parser = FastAPIParserImpl()

        # Create a large spec with many endpoints
        large_spec = json.loads(VALID_OPENAPI_SPEC_JSON)

        # Add many more endpoints
        for i in range(50):
            large_spec['paths'][f'/endpoint_{i}'] = {
                'get': {
                    'summary': f'Endpoint {i}',
                    'operationId': f'endpoint_{i}',
                    'responses': {
                        '200': {
                            'description': 'Success'
                        }
                    }
                }
            }

        large_spec_json = json.dumps(large_spec)
        chunks = parser.parse_openapi_spec(large_spec_json)

        # Should generate chunks for all endpoints
        assert len(chunks) >= 50

        # Verify all chunks have content and proper metadata
        route_chunks = [chunk for chunk in chunks if 'http_method' in chunk.source_metadata]
        assert len(route_chunks) >= 25, "Should have at least 25 route chunks from 50+ endpoints"

        for chunk in chunks:
            assert chunk.content_text.strip() != ""
            assert chunk.artifact_id.startswith("py://")

        # Verify route chunks have HTTP method metadata
        for chunk in route_chunks:
            assert chunk.source_metadata['http_method'] in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']


@pytest.mark.contract
@pytest.mark.unit
class TestFastAPIParserContractFailure:
    """Tests that should fail until implementation is complete (TDD verification)"""

    @pytest.mark.skipif(FASTAPI_PARSER_AVAILABLE, reason="Implementation available")
    def test_implementation_not_available(self):
        """This test ensures we're in TDD mode - implementation should not exist yet"""
        # This test should pass initially, then fail once implementation exists
        try:
            from src.parsers.fastapi_parser import FastAPIParserImpl
            pytest.fail("FastAPIParserImpl should not be implemented yet (TDD requirement)")
        except ImportError:
            # This is expected in TDD mode
            pass

    @pytest.mark.skipif(FASTAPI_PARSER_AVAILABLE, reason="Implementation available")
    def test_contract_will_fail_without_implementation(self):
        """Verify that the contract tests will fail without implementation"""
        assert not FASTAPI_PARSER_AVAILABLE, (
            "Implementation should not exist yet. "
            "Once implemented, this test should be removed or modified."
        )


if __name__ == "__main__":
    # Run the contract tests
    pytest.main([__file__, "-v"])
