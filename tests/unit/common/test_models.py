"""
Unit tests for common models and data structures.
"""

import pytest
from pydantic import ValidationError

# Clean import pattern using centralized utilities - no sys.path needed with PYTHONPATH=src
from common.models import QueryRequest, Snippet, SearchRequest, AdviceRequest


class TestQueryRequest:
    """Test QueryRequest model validation and serialization."""

    def test_valid_query_request(self):
        """Test valid QueryRequest creation."""

        query = QueryRequest(
            question="How do I implement authentication?", role="backend engineer"
        )

        assert query.question == "How do I implement authentication?"
        assert query.role == "backend engineer"

    def test_query_request_validation(self):
        """Test QueryRequest validation rules."""

        # Test empty question - this is actually valid in the current model
        query = QueryRequest(question="", role="developer")
        assert query.question == ""
        assert query.role == "developer"

        # Test empty role - this is also valid in the current model
        query = QueryRequest(question="test question", role="")
        assert query.question == "test question"
        assert query.role == ""

    def test_query_request_missing_fields(self):
        """Test QueryRequest with missing required fields."""

        # Missing question
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(role="developer")

        assert "Field required" in str(exc_info.value)

        # Missing role - should NOT raise error because role has default value
        query = QueryRequest(question="test question")
        assert query.question == "test question"
        assert query.role == "developer"  # default value

    def test_query_request_serialization(self):
        """Test QueryRequest serialization to dict."""

        query = QueryRequest(
            question="How do I implement authentication?", role="backend engineer"
        )

        data = query.model_dump()

        assert data == {
            "question": "How do I implement authentication?",
            "role": "backend engineer",
        }

    def test_query_request_deserialization(self):
        """Test QueryRequest deserialization from dict."""

        data = {
            "question": "How do I implement authentication?",
            "role": "backend engineer",
        }

        query = QueryRequest(**data)

        assert query.question == "How do I implement authentication?"
        assert query.role == "backend engineer"

    def test_query_request_with_special_characters(self):
        """Test QueryRequest with special characters."""

        query = QueryRequest(
            question="How do I handle unicode: 你好? And symbols: !@#$%",
            role="backend engineer",
        )

        assert "你好" in query.question
        assert "!@#$%" in query.question

    def test_query_request_long_text(self):
        """Test QueryRequest with very long text."""

        long_question = "How do I implement authentication? " * 1000
        long_role = "very detailed backend engineer with lots of experience"

        query = QueryRequest(question=long_question, role=long_role)

        assert len(query.question) > 1000
        assert query.role == long_role


class TestSnippet:
    """Test Snippet model validation and serialization."""

    def test_valid_snippet(self):
        """Test valid Snippet creation."""

        snippet = Snippet(
            file_path="src/auth/jwt.py",
            content="def create_jwt_token(data: dict) -> str:\n    return jwt.encode(data, SECRET_KEY)",
        )

        assert snippet.file_path == "src/auth/jwt.py"
        assert "def create_jwt_token" in snippet.content

    def test_snippet_validation(self):
        """Test Snippet validation rules."""

        # Test empty file_path - should be allowed (no min_length constraint)
        snippet = Snippet(file_path="", content="some content")
        assert snippet.file_path == ""
        assert snippet.content == "some content"

        # Test empty content - should be allowed (no min_length constraint)
        snippet = Snippet(file_path="test.py", content="")
        assert snippet.file_path == "test.py"
        assert snippet.content == ""

    def test_snippet_missing_fields(self):
        """Test Snippet with missing required fields."""

        # Missing file_path
        with pytest.raises(ValidationError) as exc_info:
            Snippet(content="some content")

        assert "Field required" in str(exc_info.value)

        # Missing content
        with pytest.raises(ValidationError) as exc_info:
            Snippet(file_path="test.py")

        assert "Field required" in str(exc_info.value)

    def test_snippet_serialization(self):
        """Test Snippet serialization to dict."""

        snippet = Snippet(
            file_path="src/auth/jwt.py",
            content="def create_jwt_token(data: dict) -> str:\n    return jwt.encode(data, SECRET_KEY)",
        )

        data = snippet.model_dump()

        assert data["file_path"] == "src/auth/jwt.py"
        assert "def create_jwt_token" in data["content"]

    def test_snippet_with_various_file_types(self):
        """Test Snippet with different file types and extensions."""

        file_types = [
            ("script.py", "python code"),
            ("style.css", "css content"),
            ("page.html", "html content"),
            ("config.json", "json content"),
            ("README.md", "markdown content"),
            ("Dockerfile", "docker content"),
            ("src/deep/nested/file.ts", "typescript content"),
            ("../relative/path.js", "javascript content"),
        ]

        for file_path, content in file_types:
            snippet = Snippet(file_path=file_path, content=content)
            assert snippet.file_path == file_path
            assert snippet.content == content

    def test_snippet_with_large_content(self):
        """Test Snippet with large content."""

        large_content = "def function():\n    pass\n" * 10000

        snippet = Snippet(file_path="large_file.py", content=large_content)

        assert len(snippet.content) > 100000
        assert snippet.file_path == "large_file.py"

    def test_snippet_with_binary_like_content(self):
        """Test Snippet with binary-like or special content."""

        special_contents = [
            "Binary content: \x00\x01\x02\x03",
            "Unicode: 你好世界 🌍",
            "Quotes: 'single' \"double\" `backtick`",
            "Newlines:\n\r\n\t\v\f",
            "SQL: SELECT * FROM table WHERE id = 'value'",
            "Regex: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
        ]

        for content in special_contents:
            snippet = Snippet(file_path="test.py", content=content)
            assert snippet.content == content


class TestSearchRequest:
    """Test SearchRequest model validation and serialization."""

    def test_valid_search_request(self):
        """Test valid SearchRequest creation."""

        search = SearchRequest(query="authentication implementation", k=5)

        assert search.query == "authentication implementation"
        assert search.k == 5

    def test_search_request_default_k(self):
        """Test SearchRequest with default k value."""

        search = SearchRequest(query="test query")

        assert search.k == 5  # Default value

    def test_search_request_validation(self):
        """Test SearchRequest validation rules."""

        # Test empty query - should be allowed (no min_length constraint)
        search = SearchRequest(query="", k=5)
        assert search.query == ""
        assert search.k == 5

        # Test invalid k values
        invalid_k_values = [0, -1, 101, "five"]

        for invalid_k in invalid_k_values:
            with pytest.raises(ValidationError):
                SearchRequest(query="test", k=invalid_k)

    def test_search_request_k_boundaries(self):
        """Test SearchRequest k value boundaries."""

        # Minimum valid k
        search = SearchRequest(query="test", k=1)
        assert search.k == 1

        # Maximum valid k
        search = SearchRequest(query="test", k=20)
        assert search.k == 20

    def test_search_request_serialization(self):
        """Test SearchRequest serialization."""

        search = SearchRequest(query="authentication", k=7)
        data = search.model_dump()

        assert data == {"query": "authentication", "k": 7}


class TestAdviceRequest:
    """Test AdviceRequest model validation and serialization."""

    def test_valid_advice_request(self):
        """Test valid AdviceRequest creation."""

        chunks = [
            Snippet(file_path="test.py", content="def test(): pass"),
            Snippet(file_path="main.py", content="if __name__ == '__main__': pass"),
        ]

        advice = AdviceRequest(role="backend engineer", chunks=chunks)

        assert advice.role == "backend engineer"
        assert len(advice.chunks) == 2

    def test_advice_request_validation(self):
        """Test AdviceRequest validation rules."""

        # Test missing role
        with pytest.raises(ValidationError) as exc_info:
            AdviceRequest(chunks=[])

        assert "Field required" in str(exc_info.value)

    def test_advice_request_with_snippet_objects(self):
        """Test AdviceRequest with Snippet objects."""

        snippets = [
            Snippet(file_path="test.py", content="def test(): pass"),
            Snippet(file_path="main.py", content="if __name__ == '__main__': pass"),
        ]

        advice = AdviceRequest(role="backend engineer", chunks=snippets)

        assert len(advice.chunks) == 2
        assert advice.chunks[0].file_path == "test.py"

    def test_advice_request_empty_chunks(self):
        """Test AdviceRequest with empty chunks."""

        advice = AdviceRequest(role="developer", chunks=[])

        assert advice.role == "developer"
        assert len(advice.chunks) == 0

    def test_advice_request_serialization(self):
        """Test AdviceRequest serialization."""

        chunks = [Snippet(file_path="test.py", content="code")]

        advice = AdviceRequest(role="developer", chunks=chunks)
        data = advice.model_dump()

        assert data["role"] == "developer"
        assert len(data["chunks"]) == 1


class TestModelInteroperability:
    """Test interoperability between different models."""

    def test_query_to_search_request(self):
        """Test converting QueryRequest to SearchRequest."""

        query = QueryRequest(
            question="How do I implement authentication?", role="backend engineer"
        )

        # Simulate conversion logic
        search = SearchRequest(query=query.question, k=5)

        assert search.query == query.question
        assert search.k == 5

    def test_snippets_to_advice_request(self):
        """Test converting Snippets to AdviceRequest."""

        snippets = [
            Snippet(file_path="auth.py", content="def authenticate(): pass"),
            Snippet(file_path="models.py", content="class User: pass"),
        ]

        advice = AdviceRequest(role="backend engineer", chunks=snippets)

        assert len(advice.chunks) == 2
        assert advice.chunks[0].file_path == "auth.py"

    def test_model_json_compatibility(self):
        """Test JSON serialization/deserialization compatibility."""

        import json

        # Create instances of all models
        query = QueryRequest(question="test", role="developer")
        snippet = Snippet(file_path="test.py", content="code")
        search = SearchRequest(query="test", k=5)
        advice = AdviceRequest(role="dev", chunks=[snippet])

        models = [query, snippet, search, advice]

        for model in models:
            # Serialize to JSON
            json_str = model.model_dump_json()

            # Deserialize back
            data = json.loads(json_str)

            # Create new instance
            new_model = type(model)(**data)

            # Should be equivalent
            assert new_model.model_dump() == model.model_dump()


class TestModelValidationEdgeCases:
    """Test edge cases for model validation."""

    def test_whitespace_handling(self):
        """Test handling of whitespace in string fields."""

        # Leading/trailing whitespace should be preserved
        query = QueryRequest(question="  spaced question  ", role="  spaced role  ")

        assert query.question == "  spaced question  "
        assert query.role == "  spaced role  "

    def test_newlines_and_tabs(self):
        """Test handling of newlines and tabs in content."""

        content_with_formatting = """def function():
\t# Indented comment
\tif True:
\t\tprint("Hello\\nWorld")
\treturn True"""

        snippet = Snippet(file_path="formatted.py", content=content_with_formatting)

        assert "\t" in snippet.content
        assert "\n" in snippet.content

    def test_unicode_normalization(self):
        """Test Unicode character handling."""

        unicode_texts = [
            "café",  # Composed
            "café",  # Decomposed
            "🚀 rocket emoji",
            "中文字符",
            "العربية",
            "русский",
        ]

        for text in unicode_texts:
            query = QueryRequest(question=text, role="developer")
            assert query.question == text

    def test_very_long_strings(self):
        """Test handling of very long strings."""

        # Very long question (10MB)
        long_question = "a" * (10 * 1024 * 1024)

        # Should handle without issues (unless there are specific limits)
        query = QueryRequest(question=long_question, role="developer")
        assert len(query.question) == 10 * 1024 * 1024

    def test_numeric_string_coercion(self):
        """Test that numeric values are properly handled."""

        # These should fail as they're not strings
        with pytest.raises(ValidationError):
            QueryRequest(question=123, role="developer")

        with pytest.raises(ValidationError):
            QueryRequest(question="test", role=456)

        with pytest.raises(ValidationError):
            Snippet(file_path=789, content="content")

    def test_none_values(self):
        """Test handling of None values."""

        with pytest.raises(ValidationError):
            QueryRequest(question=None, role="developer")

        with pytest.raises(ValidationError):
            Snippet(file_path="test.py", content=None)
