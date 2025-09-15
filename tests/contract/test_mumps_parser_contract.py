"""
Contract tests for MUMPS Parser Interface
These tests MUST FAIL initially (TDD requirement) until the MUMPS parser is implemented.

Tests the contract defined in specs/002-m1-parse-and/contracts/parser-interfaces.py
"""

import pytest
import json
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

# Register custom markers to avoid warnings
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "contract: Contract tests for interface compliance (TDD)")
    config.addinivalue_line("markers", "unit: Unit tests for individual components")

# Import the parser interface contracts
try:
    import sys
    import os
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)

    # Import using the module loading approach for hyphenated filenames
    import importlib.util
    parser_interfaces_path = os.path.join(project_root, "specs", "002-m1-parse-and", "contracts", "parser-interfaces.py")
    spec = importlib.util.spec_from_file_location("parser_interfaces", parser_interfaces_path)
    parser_interfaces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_interfaces)

    BaseParser = parser_interfaces.BaseParser
    MUMPSParser = parser_interfaces.MUMPSParser
    SourceType = parser_interfaces.SourceType
    ChunkMetadata = parser_interfaces.ChunkMetadata
    ParseResult = parser_interfaces.ParseResult
    ParseError = parser_interfaces.ParseError
    ErrorClass = parser_interfaces.ErrorClass

    PARSER_INTERFACES_AVAILABLE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    PARSER_INTERFACES_AVAILABLE = False
    print(f"Warning: Could not import parser interfaces: {e}")


# Skip all tests if parser interfaces are not available
@pytest.mark.skipif(not PARSER_INTERFACES_AVAILABLE, reason="Parser interfaces not available")
class TestMUMPSParserContract:
    """Contract tests for MUMPS Parser implementation"""

    @pytest.fixture
    def sample_fileman_dict(self):
        """Sample FileMan data dictionary content"""
        return '''
^DD(200,0)="NAME^RF^^0;1^K:$L(X)>200!($L(X)<3)!'(X'?1P.E) X"
^DD(200,0,"NM","NEW PERSON")=
^DD(200,0,"UP")=
^DD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>200!($L(X)<3)!'(X'?1P.E) X"
^DD(200,.01,1,0)="^.1"
^DD(200,.01,1,1,0)="200^B"
^DD(200,.01,1,1,1)="S ^VA(200,\\"B\\",$E(X,1,30),DA)="
^DD(200,.01,1,1,2)="K ^VA(200,\\"B\\",$E(X,1,30),DA)"
^DD(200,.01,3)="Type a Name, between 3 and 200 characters in length."
^DD(200,.01,"DT")=2921204
        '''

    @pytest.fixture
    def sample_global_definition(self):
        """Sample MUMPS global variable definition"""
        return '''
^VA(200,0)="NEW PERSON^200P^23^23"
^VA(200,1,0)="PROGRAMMER,ONE^201^1^2901101.12^2901101^PROGRAMMER^1^0^0"
^VA(200,1,1)="123 MAIN ST"
^VA(200,1,2)="ANYTOWN^ST^12345"
^VA(200,1,200)="P"
^VA(200,2,0)="USER,ANOTHER^202^2^2901101.13^2901101^USER^1^0^0"
        '''

    @pytest.fixture
    def expected_mumps_fields(self):
        """Expected MUMPS-specific fields in source_metadata"""
        return {
            'global_name': str,
            'node_path': str,
            'file_no': str,
            'field_no': str,
            'xrefs': dict,
            'input_transform': str
        }

    @pytest.mark.contract
    def test_mumps_parser_class_exists(self):
        """MUST FAIL: Test that MUMPSParser class exists and inherits from BaseParser"""
        # This test will fail until MUMPSParser is implemented
        try:
            # Try to import an actual implementation
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()
            assert isinstance(parser, BaseParser)
            assert parser.source_type == SourceType.MUMPS
        except ImportError:
            pytest.fail("MUMPSParser implementation not found in src.parsers.mumps_parser")

    @pytest.mark.contract
    def test_mumps_parser_abstract_methods_implemented(self):
        """MUST FAIL: Test that all abstract methods are implemented"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Test that required abstract methods exist and are callable
            assert hasattr(parser, 'parse_fileman_dict')
            assert callable(getattr(parser, 'parse_fileman_dict'))

            assert hasattr(parser, 'parse_global_definition')
            assert callable(getattr(parser, 'parse_global_definition'))

            assert hasattr(parser, 'parse_file')
            assert callable(getattr(parser, 'parse_file'))

            assert hasattr(parser, 'parse_directory')
            assert callable(getattr(parser, 'parse_directory'))

            assert hasattr(parser, 'validate_content')
            assert callable(getattr(parser, 'validate_content'))

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")

    @pytest.mark.contract
    def test_parse_fileman_dict_method_signature(self, sample_fileman_dict):
        """MUST FAIL: Test parse_fileman_dict method signature and return type"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Test method exists and returns correct type
            result = parser.parse_fileman_dict(sample_fileman_dict)
            assert isinstance(result, list)
            assert all(isinstance(chunk, ChunkMetadata) for chunk in result)

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"parse_fileman_dict method not properly implemented: {e}")

    @pytest.mark.contract
    def test_parse_global_definition_method_signature(self, sample_global_definition):
        """MUST FAIL: Test parse_global_definition method signature and return type"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Test method exists and returns correct type
            result = parser.parse_global_definition(sample_global_definition)
            assert isinstance(result, list)
            assert all(isinstance(chunk, ChunkMetadata) for chunk in result)

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"parse_global_definition method not properly implemented: {e}")

    @pytest.mark.contract
    def test_artifact_id_format_mumps_protocol(self, sample_fileman_dict):
        """MUST FAIL: Test that artifact IDs follow mumps://{global_name}/{node_path} format"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            chunks = parser.parse_fileman_dict(sample_fileman_dict)

            for chunk in chunks:
                assert chunk.artifact_id.startswith("mumps://")
                # Should follow pattern: mumps://{global_name}/{node_path}
                parts = chunk.artifact_id.replace("mumps://", "").split("/")
                assert len(parts) >= 2, f"Invalid artifact_id format: {chunk.artifact_id}"

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"Artifact ID format validation failed: {e}")

    @pytest.mark.contract
    def test_mumps_specific_metadata_fields(self, sample_fileman_dict, expected_mumps_fields):
        """MUST FAIL: Test that MUMPS-specific fields are present in source_metadata"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            chunks = parser.parse_fileman_dict(sample_fileman_dict)

            for chunk in chunks:
                assert chunk.source_type == SourceType.MUMPS
                assert isinstance(chunk.source_metadata, dict)

                # Verify required MUMPS fields are present
                for field_name, field_type in expected_mumps_fields.items():
                    assert field_name in chunk.source_metadata, f"Missing field: {field_name}"
                    if chunk.source_metadata[field_name] is not None:
                        assert isinstance(chunk.source_metadata[field_name], field_type), \
                            f"Field {field_name} should be {field_type}, got {type(chunk.source_metadata[field_name])}"

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"MUMPS metadata fields validation failed: {e}")

    @pytest.mark.contract
    def test_fileman_dict_parsing_produces_chunks(self, sample_fileman_dict):
        """MUST FAIL: Test that FileMan dictionary parsing produces meaningful chunks"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            chunks = parser.parse_fileman_dict(sample_fileman_dict)

            assert len(chunks) > 0, "Should produce at least one chunk from FileMan dictionary"

            for chunk in chunks:
                assert len(chunk.content_text) > 0, "Chunk content should not be empty"
                assert chunk.content_hash, "Content hash should be generated"
                assert chunk.source_metadata.get('global_name'), "global_name should be extracted"
                assert chunk.source_metadata.get('file_no'), "file_no should be extracted"

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"FileMan dictionary parsing failed: {e}")

    @pytest.mark.contract
    def test_global_definition_parsing_produces_chunks(self, sample_global_definition):
        """MUST FAIL: Test that global definition parsing produces meaningful chunks"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            chunks = parser.parse_global_definition(sample_global_definition)

            assert len(chunks) > 0, "Should produce at least one chunk from global definition"

            for chunk in chunks:
                assert len(chunk.content_text) > 0, "Chunk content should not be empty"
                assert chunk.content_hash, "Content hash should be generated"
                assert chunk.source_metadata.get('global_name'), "global_name should be extracted"
                assert chunk.source_metadata.get('node_path'), "node_path should be extracted"

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"Global definition parsing failed: {e}")

    @pytest.mark.contract
    def test_validate_content_method(self, sample_fileman_dict, sample_global_definition):
        """MUST FAIL: Test that validate_content correctly identifies MUMPS content"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Should validate MUMPS content as true
            assert parser.validate_content(sample_fileman_dict) == True
            assert parser.validate_content(sample_global_definition) == True

            # Should reject non-MUMPS content
            assert parser.validate_content("This is not MUMPS content") == False
            assert parser.validate_content("") == False
            assert parser.validate_content("SELECT * FROM table") == False

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"validate_content method failed: {e}")

    @pytest.mark.contract
    def test_parse_file_method_integration(self, tmp_path):
        """MUST FAIL: Test that parse_file method works with actual files"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Create a test MUMPS file
            test_file = tmp_path / "test_mumps.m"
            test_file.write_text('''
^DD(200,0)="NAME^RF^^0;1^K:$L(X)>200!($L(X)<3)!'(X'?1P.E) X"
^DD(200,.01,0)="NAME^RF^^0;1^K:$L(X)>200!($L(X)<3)!'(X'?1P.E) X"
''')

            result = parser.parse_file(str(test_file))

            assert isinstance(result, ParseResult)
            assert result.files_processed == 1
            assert len(result.chunks) > 0
            assert len(result.errors) == 0

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"parse_file method failed: {e}")

    @pytest.mark.contract
    def test_parse_directory_method_integration(self, tmp_path):
        """MUST FAIL: Test that parse_directory method works with directories"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Create test directory with MUMPS files
            (tmp_path / "file1.m").write_text('^DD(200,0)="NAME^RF^^0;1"')
            (tmp_path / "file2.m").write_text('^VA(200,1,0)="PROGRAMMER,ONE^201"')
            (tmp_path / "readme.txt").write_text("Not a MUMPS file")

            result = parser.parse_directory(str(tmp_path))

            assert isinstance(result, ParseResult)
            assert result.files_processed >= 2  # Should process MUMPS files
            assert len(result.chunks) > 0

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"parse_directory method failed: {e}")

    @pytest.mark.contract
    def test_error_handling_for_invalid_mumps_content(self):
        """MUST FAIL: Test that parser handles invalid MUMPS content gracefully"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            invalid_content = "This is definitely not valid MUMPS content!"

            # Should either return empty chunks or produce parse errors
            chunks = parser.parse_fileman_dict(invalid_content)
            assert isinstance(chunks, list)
            # Either no chunks produced or errors are tracked elsewhere

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            # Should handle errors gracefully, not crash
            pytest.fail(f"Parser should handle invalid content gracefully: {e}")

    @pytest.mark.contract
    def test_chunk_metadata_completeness(self, sample_fileman_dict):
        """MUST FAIL: Test that all required ChunkMetadata fields are populated"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            chunks = parser.parse_fileman_dict(sample_fileman_dict)

            for chunk in chunks:
                # Test required fields are populated
                assert chunk.source_type == SourceType.MUMPS
                assert chunk.artifact_id and len(chunk.artifact_id) > 0
                assert chunk.content_text and len(chunk.content_text) > 0
                assert chunk.content_hash and len(chunk.content_hash) > 0
                assert isinstance(chunk.collected_at, datetime)
                assert isinstance(chunk.source_metadata, dict)

                # Content tokens should be estimated if provided
                if chunk.content_tokens is not None:
                    assert chunk.content_tokens > 0

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")
        except Exception as e:
            pytest.fail(f"ChunkMetadata completeness validation failed: {e}")


# Additional tests for edge cases and comprehensive coverage
@pytest.mark.skipif(not PARSER_INTERFACES_AVAILABLE, reason="Parser interfaces not available")
class TestMUMPSParserEdgeCases:
    """Edge case tests for MUMPS Parser contract compliance"""

    @pytest.mark.contract
    def test_empty_content_handling(self):
        """MUST FAIL: Test handling of empty or whitespace-only content"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Test empty string
            chunks = parser.parse_fileman_dict("")
            assert isinstance(chunks, list)

            # Test whitespace only
            chunks = parser.parse_fileman_dict("   \n\t  ")
            assert isinstance(chunks, list)

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")

    @pytest.mark.contract
    def test_large_content_chunking(self):
        """MUST FAIL: Test that large MUMPS content is properly chunked"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            # Create large content that should be chunked
            large_content = "\n".join([
                f'^DD(200,{i},0)="FIELD{i}^RF^^0;{i}^K:$L(X)>200!($L(X)<3)!\'(X\'?1P.E) X"'
                for i in range(100)
            ])

            chunks = parser.parse_fileman_dict(large_content)

            # Should produce multiple chunks for large content
            if len(large_content) > 2000:  # Assuming reasonable chunk size
                assert len(chunks) > 1, "Large content should be split into multiple chunks"

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")

    @pytest.mark.contract
    def test_unicode_content_handling(self):
        """MUST FAIL: Test handling of Unicode characters in MUMPS content"""
        try:
            from src.parsers.mumps_parser import MUMPSParser as ActualMUMPSParser
            parser = ActualMUMPSParser()

            unicode_content = '''
^DD(200,0)="NATÉ^RF^^0;1^K:$L(X)>200!($L(X)<3)!'(X'?1P.E) X"
^DD(200,.01,0)="ÑÁmÉ^RF^^0;1^Unicode test content"
'''

            chunks = parser.parse_fileman_dict(unicode_content)
            assert isinstance(chunks, list)

            # Should handle Unicode content without errors
            for chunk in chunks:
                assert isinstance(chunk.content_text, str)

        except ImportError:
            pytest.fail("MUMPSParser implementation not found")