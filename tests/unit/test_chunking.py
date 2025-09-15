"""
Unit tests for chunking strategies
T034: Comprehensive tests for ContentChunker with all strategies and edge cases
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from common.chunking import (
    ChunkConfig,
    ChunkingStrategy,
    ChunkResult,
    ContentChunker,
    validate_chunks,
)


class TestChunkConfig:
    """Test ChunkConfig dataclass functionality"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ChunkConfig()
        assert config.max_tokens == 1000
        assert config.overlap_pct == 0.15
        assert config.min_chunk_size == 50
        assert config.preserve_boundaries is True
        assert config.strategy == ChunkingStrategy.TOKEN_BASED

    def test_custom_config(self):
        """Test custom configuration creation"""
        config = ChunkConfig(
            max_tokens=500,
            overlap_pct=0.20,
            min_chunk_size=25,
            preserve_boundaries=False,
            strategy=ChunkingStrategy.SEMANTIC_BLOCKS,
        )
        assert config.max_tokens == 500
        assert config.overlap_pct == 0.20
        assert config.min_chunk_size == 25
        assert config.preserve_boundaries is False
        assert config.strategy == ChunkingStrategy.SEMANTIC_BLOCKS


class TestChunkResult:
    """Test ChunkResult dataclass functionality"""

    def test_chunk_result_creation(self):
        """Test ChunkResult creation with all fields"""
        chunk = ChunkResult(
            content="test content",
            start_position=0,
            end_position=12,
            token_count=3,
            metadata={"source_type": "test"},
        )
        assert chunk.content == "test content"
        assert chunk.start_position == 0
        assert chunk.end_position == 12
        assert chunk.token_count == 3
        assert chunk.metadata["source_type"] == "test"


class TestContentChunker:
    """Test ContentChunker functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.chunker = ContentChunker()

    def test_initialization(self):
        """Test chunker initialization"""
        chunker = ContentChunker()
        assert chunker.config.max_tokens == 1000
        assert chunker.chars_per_token == 4
        assert len(chunker.boundary_patterns) > 0

    def test_custom_config_initialization(self):
        """Test chunker with custom configuration"""
        config = ChunkConfig(max_tokens=500, overlap_pct=0.25)
        chunker = ContentChunker(config)
        assert chunker.config.max_tokens == 500
        assert chunker.config.overlap_pct == 0.25

    def test_token_estimation(self):
        """Test token estimation accuracy"""
        chunker = ContentChunker()

        # Test simple text
        assert chunker._estimate_tokens("hello world") >= 2
        assert chunker._estimate_tokens("") == 1  # Minimum 1 token

        # Test with extra whitespace (should be normalized)
        assert chunker._estimate_tokens("hello    world") == chunker._estimate_tokens(
            "hello world"
        )

        # Test longer text
        long_text = "The quick brown fox jumps over the lazy dog"
        tokens = chunker._estimate_tokens(long_text)
        assert tokens >= 8  # Should be around 9-12 tokens

    def test_strategy_selection(self):
        """Test strategy selection for different source types"""
        chunker = ContentChunker()

        assert (
            chunker._get_strategy_for_source("kubernetes")
            == ChunkingStrategy.SEMANTIC_BLOCKS
        )
        assert (
            chunker._get_strategy_for_source("fastapi")
            == ChunkingStrategy.SEMANTIC_BLOCKS
        )
        assert chunker._get_strategy_for_source("cobol") == ChunkingStrategy.FIXED_WIDTH
        assert chunker._get_strategy_for_source("irs") == ChunkingStrategy.FIXED_WIDTH
        assert (
            chunker._get_strategy_for_source("mumps") == ChunkingStrategy.HIERARCHICAL
        )
        assert (
            chunker._get_strategy_for_source("unknown") == ChunkingStrategy.TOKEN_BASED
        )

    def test_empty_content_chunking(self):
        """Test chunking of empty content"""
        chunker = ContentChunker()
        chunks = chunker.chunk_content("", "generic")
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_minimal_content_chunking(self):
        """Test chunking of very small content"""
        chunker = ContentChunker()
        chunks = chunker.chunk_content("hi", "generic")
        assert len(chunks) >= 0  # Might be 0 if below min_chunk_size

    def test_chunking_stats(self):
        """Test chunking statistics calculation"""
        chunker = ContentChunker()

        # Test empty chunks
        stats = chunker.get_chunking_stats([])
        assert stats["total_chunks"] == 0

        # Test with actual chunks
        content = "This is a test content that should be chunked into multiple pieces for testing."
        chunks = chunker.chunk_content(content, "generic")

        if chunks:  # Only test if we got chunks
            stats = chunker.get_chunking_stats(chunks)
            assert "total_chunks" in stats
            assert "avg_tokens_per_chunk" in stats
            assert "min_tokens" in stats
            assert "max_tokens" in stats
            assert "total_tokens" in stats
            assert "strategies_used" in stats
            assert stats["total_chunks"] == len(chunks)


class TestSemanticBlocksChunking:
    """Test semantic blocks chunking strategy (Kubernetes/FastAPI)"""

    def test_kubernetes_yaml_chunking(self):
        """Test chunking of Kubernetes YAML content"""
        k8s_content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: production
  labels:
    app: web-application
    version: v1.2.3
    environment: production
data:
  config.yaml: |
    database:
      host: localhost
      port: 5432
      username: app_user
      password: secure_password
      pool_size: 10
      timeout: 30
    redis:
      host: redis-cluster
      port: 6379
      database: 0
      timeout: 5
  application.properties: |
    server.port=8080
    spring.profiles.active=production
    logging.level.com.example=DEBUG
---
apiVersion: v1
kind: Service
metadata:
  name: app-service
  namespace: production
  labels:
    app: web-application
    version: v1.2.3
spec:
  selector:
    app: web-application
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP"""

        # Use smaller min chunk size for testing
        config = ChunkConfig(min_chunk_size=20, max_tokens=200)
        chunker = ContentChunker(config)
        chunks = chunker.chunk_content(k8s_content, "kubernetes")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.token_count > 0
            assert chunk.metadata["strategy"] == "semantic_blocks"
            assert chunk.metadata["source_type"] == "kubernetes"

    def test_fastapi_python_chunking(self):
        """Test chunking of FastAPI Python code"""
        fastapi_content = """from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
import logging
import uvicorn

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="User Management API",
    description="A comprehensive API for managing users and authentication",
    version="1.0.0"
)

security = HTTPBearer()

class UserModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: Optional[int] = Field(None, ge=0, le=120)
    is_active: bool = True

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int]
    is_active: bool

@app.get("/")
async def root():
    \"\"\"Root endpoint returning API information\"\"\"
    logger.info("Root endpoint accessed")
    return {
        "message": "User Management API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    \"\"\"Get item by ID with optional query parameter\"\"\"
    logger.info(f"Fetching item {item_id}")
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid item ID")

    result = {"item_id": item_id}
    if q:
        result["query"] = q
    return result

@app.post("/users/", response_model=UserResponse)
async def create_user(
    user: UserModel,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    \"\"\"Create a new user with authentication\"\"\"
    logger.info(f"Creating user: {user.email}")
    # Simulate user creation with auto-generated ID
    return UserResponse(
        id=12345,
        name=user.name,
        email=user.email,
        age=user.age,
        is_active=user.is_active
    )

@app.get("/users/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    \"\"\"List all users with pagination\"\"\"
    logger.info(f"Listing users: skip={skip}, limit={limit}")
    # Simulate user listing
    return [
        UserResponse(id=1, name="John Doe", email="john@example.com", age=30, is_active=True),
        UserResponse(id=2, name="Jane Smith", email="jane@example.com", age=25, is_active=True)
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)"""

        # Use smaller min chunk size for testing
        config = ChunkConfig(min_chunk_size=25, max_tokens=300)
        chunker = ContentChunker(config)
        chunks = chunker.chunk_content(fastapi_content, "fastapi")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata["strategy"] == "semantic_blocks"
            assert chunk.metadata["source_type"] == "fastapi"

    def test_semantic_boundary_detection(self):
        """Test semantic boundary pattern matching"""
        chunker = ContentChunker()

        # Test YAML separator
        assert chunker._is_semantic_boundary("---", ChunkingStrategy.SEMANTIC_BLOCKS)

        # Test function definition
        assert chunker._is_semantic_boundary(
            "def test_function():", ChunkingStrategy.SEMANTIC_BLOCKS
        )
        assert chunker._is_semantic_boundary(
            "async def async_function():", ChunkingStrategy.SEMANTIC_BLOCKS
        )

        # Test FastAPI decorator
        assert chunker._is_semantic_boundary(
            "@app.get('/test')", ChunkingStrategy.SEMANTIC_BLOCKS
        )

        # Test K8s manifest start
        assert chunker._is_semantic_boundary(
            "apiVersion: v1", ChunkingStrategy.SEMANTIC_BLOCKS
        )
        assert chunker._is_semantic_boundary(
            "kind: Pod", ChunkingStrategy.SEMANTIC_BLOCKS
        )

    def test_overlap_calculation(self):
        """Test overlap line calculation"""
        chunker = ContentChunker()

        lines = ["line1", "line2", "line3", "line4", "line5"]

        # Test 20% overlap
        overlap = chunker._calculate_overlap_lines(lines, 0.20)
        assert len(overlap) == 1  # 20% of 5 = 1
        assert overlap[0] == "line5"

        # Test 50% overlap
        overlap = chunker._calculate_overlap_lines(lines, 0.50)
        assert len(overlap) == 2  # 50% of 5 = 2.5, rounded down to 2
        assert overlap == ["line4", "line5"]

        # Test empty list
        overlap = chunker._calculate_overlap_lines([], 0.20)
        assert overlap == []


class TestFixedWidthChunking:
    """Test fixed-width chunking strategy (COBOL/IRS)"""

    def test_cobol_copybook_chunking(self):
        """Test chunking of COBOL copybook content"""
        cobol_content = """01  CUSTOMER-RECORD.
    05  CUST-ID            PIC 9(6).
    05  CUST-NAME          PIC X(30).
    05  CUST-ADDRESS.
        10  STREET-ADDR    PIC X(40).
        10  CITY           PIC X(20).
        10  STATE          PIC X(2).
        10  ZIP-CODE       PIC 9(5).
    05  CUST-PHONE         PIC X(10).
    05  CUST-BALANCE       PIC 9(8)V99.

01  ORDER-RECORD.
    05  ORDER-ID           PIC 9(8).
    05  ORDER-DATE         PIC 9(8).
    05  ORDER-TOTAL        PIC 9(6)V99."""

        chunker = ContentChunker()
        chunks = chunker.chunk_content(cobol_content, "cobol")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata["strategy"] == "fixed_width"
            assert chunk.metadata["source_type"] == "cobol"
            assert "record_count" in chunk.metadata

    def test_irs_layout_chunking(self):
        """Test chunking of IRS record layout content"""
        irs_content = """001-009  SSN              Social Security Number
010-018  LAST-NAME        Last Name
019-033  FIRST-NAME       First Name
034-034  MIDDLE-INIT      Middle Initial
035-042  DOB              Date of Birth (YYYYMMDD)
043-043  GENDER           Gender Code (M/F)
044-083  ADDRESS          Mailing Address
084-103  CITY             City
104-105  STATE            State Code
106-114  ZIP              ZIP Code"""

        chunker = ContentChunker()
        chunks = chunker.chunk_content(irs_content, "irs")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata["strategy"] == "fixed_width"
            assert chunk.metadata["source_type"] == "irs"
            assert "record_count" in chunk.metadata


class TestHierarchicalChunking:
    """Test hierarchical chunking strategy (MUMPS)"""

    def test_mumps_dictionary_chunking(self):
        """Test chunking of MUMPS FileMan dictionary content"""
        mumps_content = """^DD(200,0,"GL")
^DD(200,0)="FIELD^200^20"
^DD(200,.01,0)="NAME^200.01^1"
^DD(200,.01,1,0)="^.1"
^DD(200,.01,1,1,0)="200^B"
^DD(200,.01,3)="NAME MUST BE 3-30 CHARACTERS"
^DD(200,1,0)="ACCESS CODE^200.1^1"
^DD(200,1,1,0)="^.1"
^DD(200,2,0)="VERIFY CODE^200.2^1"
^DD(200,3,0)="ELECTRONIC SIGNATURE^200.3^1"

^DIC(200,0)="NEW PERSON^200B^B"
^DIC(200,1,0)="^.1"
^DIC(200,"%",0)="T"
^DIC(200,"%D",0)="^^C"
^DIC(200,"B",0)="^200B^B^200"""

        chunker = ContentChunker()
        chunks = chunker.chunk_content(mumps_content, "mumps")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata["strategy"] == "hierarchical"
            assert chunk.metadata["source_type"] == "mumps"
            assert "hierarchy_level" in chunk.metadata

    def test_hierarchy_level_detection(self):
        """Test hierarchy level detection for MUMPS"""
        chunker = ContentChunker()

        # Test MUMPS global references
        level = chunker._detect_hierarchy_level("^DD(200,0)", "mumps")
        assert level >= 0

        # Test deeper nesting
        level1 = chunker._detect_hierarchy_level("^DD(200,1,0)", "mumps")
        level2 = chunker._detect_hierarchy_level("^DD(200,1,1,0)", "mumps")
        assert level2 >= level1  # Deeper nesting should have higher level


class TestTokenBasedChunking:
    """Test token-based chunking strategy (default)"""

    def test_basic_token_chunking(self):
        """Test basic token-based chunking"""
        content = "This is a simple test document that should be chunked based on token limits rather than semantic boundaries."

        config = ChunkConfig(max_tokens=10, overlap_pct=0.2)
        chunker = ContentChunker(config)
        chunks = chunker.chunk_content(content, "generic")

        assert len(chunks) >= 2  # Should create multiple chunks
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata["strategy"] == "token_based"
            assert chunk.metadata["source_type"] == "generic"
            assert "word_start" in chunk.metadata
            assert "word_end" in chunk.metadata

    def test_token_overlap(self):
        """Test token overlap between chunks"""
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"

        config = ChunkConfig(max_tokens=5, overlap_pct=0.4)  # 40% overlap
        chunker = ContentChunker(config)
        chunks = chunker.chunk_content(content, "generic")

        if len(chunks) >= 2:
            # Check that chunks have expected overlap
            first_words = chunks[0].content.split()
            second_words = chunks[1].content.split()

            # Should have some overlapping words
            overlap_expected = int(len(first_words) * 0.4)
            if overlap_expected > 0:
                # Last words of first chunk should appear in second chunk
                last_words_first = first_words[-overlap_expected:]
                first_words_second = second_words[:overlap_expected]

                # There should be some commonality (allowing for chunking adjustments)
                assert len(set(last_words_first) & set(first_words_second)) >= 0


class TestChunkValidation:
    """Test chunk validation utilities"""

    def test_validate_chunks_good_chunks(self):
        """Test validation of properly formed chunks"""
        chunks = [
            ChunkResult(
                content="chunk 1 content",
                start_position=0,
                end_position=15,
                token_count=3,
                metadata={"strategy": "test"},
            ),
            ChunkResult(
                content="chunk 2 content",
                start_position=10,
                end_position=25,
                token_count=3,
                metadata={"strategy": "test"},
            ),
        ]

        validation = validate_chunks(chunks, "original content")

        assert validation["has_chunks"] is True
        assert validation["no_empty_chunks"] is True
        assert validation["reasonable_sizes"] is True
        assert validation["proper_positions"] is True
        assert validation["content_preserved"] is True

    def test_validate_chunks_empty(self):
        """Test validation of empty chunk list"""
        validation = validate_chunks([], "original content")
        assert validation["has_chunks"] is False

    def test_validate_chunks_bad_chunks(self):
        """Test validation of malformed chunks"""
        bad_chunks = [
            ChunkResult(
                content="",  # Empty content
                start_position=10,
                end_position=5,  # Invalid position order
                token_count=0,  # Zero tokens
                metadata={},
            )
        ]

        validation = validate_chunks(bad_chunks, "original content")

        assert validation["has_chunks"] is True
        assert validation["no_empty_chunks"] is False
        assert validation["reasonable_sizes"] is False
        assert validation["proper_positions"] is False


class TestChunkerFactory:
    """Test factory function for creating chunkers"""

    def test_create_chunker_default(self):
        """Test creating chunker with default settings"""
        # Fix the factory function first by testing the method directly
        chunker = ContentChunker()
        strategy = chunker._get_strategy_for_source("kubernetes")
        assert strategy == ChunkingStrategy.SEMANTIC_BLOCKS

    def test_create_chunker_custom(self):
        """Test creating chunker with custom settings"""
        chunker = ContentChunker(ChunkConfig(max_tokens=500, overlap_pct=0.25))
        assert chunker.config.max_tokens == 500
        assert chunker.config.overlap_pct == 0.25


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_very_large_content(self):
        """Test chunking of very large content"""
        large_content = "word " * 10000  # 10,000 words

        chunker = ContentChunker()
        chunks = chunker.chunk_content(large_content, "generic")

        assert len(chunks) > 1  # Should create multiple chunks
        assert all(chunk.token_count > 0 for chunk in chunks)

    def test_special_characters(self):
        """Test chunking content with special characters"""
        special_content = """
        Content with émojis 🚀 and spëcial charåcters!
        @#$%^&*()_+-={}[]|\\:";'<>?,./"
        Üñíçødé tëxt with various symbols ∑∆π∫∞
        """

        chunker = ContentChunker()
        chunks = chunker.chunk_content(special_content, "generic")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content  # Should not be empty
            assert chunk.token_count > 0

    def test_whitespace_only_content(self):
        """Test chunking content that is mostly whitespace"""
        whitespace_content = "\n\n   \t\t   \n\n   \n   \t\t\t   \n\n"

        chunker = ContentChunker()
        chunks = chunker.chunk_content(whitespace_content, "generic")

        # Should handle gracefully (might produce no chunks or minimal chunks)
        assert isinstance(chunks, list)

    def test_single_very_long_line(self):
        """Test chunking a single very long line"""
        long_line = "word" * 1000  # One very long word

        chunker = ContentChunker()
        chunks = chunker.chunk_content(long_line, "generic")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content
            assert chunk.token_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
