"""
FastAPI Parser Implementation with AST and OpenAPI introspection
T024: Implements comprehensive FastAPI source code and OpenAPI specification parsing

This parser extends the FastAPIParser contract to provide:
- Python AST parsing for FastAPI decorators and route functions
- OpenAPI specification parsing (JSON/YAML)
- Pydantic model schema extraction and validation
- Route path and HTTP method detection
- Request/response model analysis
- Dependency injection pattern recognition
- Content chunking optimized for Python functions and classes
- Metadata extraction for API documentation and validation
"""

import ast
import json
import yaml
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Iterator, Union
from datetime import datetime, timezone
from pathlib import Path
import logging

# Import contract interfaces
import sys
import importlib.util

# Load parser interfaces dynamically
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
parser_interfaces_path = os.path.join(project_root, "specs", "002-m1-parse-and", "contracts", "parser-interfaces.py")

# Load the contract interfaces (required for proper contract compliance)
spec = importlib.util.spec_from_file_location("parser_interfaces", parser_interfaces_path)
parser_interfaces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_interfaces)

BaseParser = parser_interfaces.BaseParser
FastAPIParser = parser_interfaces.FastAPIParser
SourceType = parser_interfaces.SourceType
ChunkMetadata = parser_interfaces.ChunkMetadata
ParseResult = parser_interfaces.ParseResult
ParseError = parser_interfaces.ParseError
ErrorClass = parser_interfaces.ErrorClass

# Import common utilities
sys.path.append(os.path.join(project_root, "src"))
from common.chunking import ContentChunker, ChunkConfig, ChunkingStrategy
from common.ids import ArtifactIDGenerator, ContentHashGenerator
from common.normalize import ContentNormalizer


class FastAPIParserImpl(FastAPIParser):
    """
    FastAPI Parser Implementation using AST and OpenAPI introspection

    Provides comprehensive parsing of FastAPI applications including:
    - Route decorators and handler functions
    - Pydantic model definitions
    - OpenAPI/Swagger specifications
    - Dependency injection patterns
    - Request/response model schemas
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__(version)
        self.logger = logging.getLogger(__name__)

        # Initialize utilities
        self.chunker = ContentChunker(ChunkConfig(
            max_tokens=1000,
            overlap_pct=0.15,
            strategy=ChunkingStrategy.SEMANTIC_BLOCKS
        ))
        self.id_generator = ArtifactIDGenerator("fastapi")
        self.hash_generator = ContentHashGenerator()
        self.normalizer = ContentNormalizer()

        # FastAPI-specific patterns
        self.fastapi_decorators = [
            r'@app\.(get|post|put|delete|patch|head|options|trace)',
            r'@router\.(get|post|put|delete|patch|head|options|trace)',
            r'@.*\.(get|post|put|delete|patch|head|options|trace)'
        ]

        self.pydantic_model_patterns = [
            r'class\s+\w+\s*\(\s*BaseModel\s*\)',
            r'class\s+\w+\s*\(\s*.*BaseModel.*\)',
            r'from\s+pydantic\s+import.*BaseModel'
        ]

    def _get_source_type(self) -> SourceType:
        """Return FastAPI source type"""
        return SourceType.FASTAPI

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a single file (Python or JSON/YAML OpenAPI spec)"""
        start_time = time.time()
        chunks = []
        errors = []

        try:
            if not os.path.exists(file_path):
                errors.append(ParseError(
                    source_type=self.source_type,
                    source_uri=file_path,
                    error_class=ErrorClass.PARSING,
                    error_msg=f"File not found: {file_path}",
                    collected_at=datetime.now(timezone.utc)
                ))
                return ParseResult(chunks, errors, 0, int((time.time() - start_time) * 1000))

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Determine file type and parse accordingly
            if file_path.endswith(('.json', '.yaml', '.yml')):
                # OpenAPI specification
                file_chunks = self.parse_openapi_spec(content)
                for chunk in file_chunks:
                    chunk.source_uri = file_path
                chunks.extend(file_chunks)
            elif file_path.endswith('.py'):
                # Python source code
                file_chunks = self.parse_source_code(content)
                for chunk in file_chunks:
                    chunk.source_uri = file_path
                chunks.extend(file_chunks)
            else:
                # Try to validate as either
                if self.validate_content(content):
                    if content.strip().startswith(('{', '---', 'openapi:')):
                        file_chunks = self.parse_openapi_spec(content)
                    else:
                        file_chunks = self.parse_source_code(content)

                    for chunk in file_chunks:
                        chunk.source_uri = file_path
                    chunks.extend(file_chunks)

        except Exception as e:
            errors.append(ParseError(
                source_type=self.source_type,
                source_uri=file_path,
                error_class=ErrorClass.PARSING,
                error_msg=str(e),
                stack_trace=str(e),
                collected_at=datetime.now(datetime.timezone.utc)
            ))

        processing_duration = int((time.time() - start_time) * 1000)
        return ParseResult(chunks, errors, 1, processing_duration)

    def parse_directory(self, directory_path: str) -> ParseResult:
        """Parse all FastAPI-related files in a directory"""
        start_time = time.time()
        all_chunks = []
        all_errors = []
        files_processed = 0

        try:
            path = Path(directory_path)
            if not path.exists():
                all_errors.append(ParseError(
                    source_type=self.source_type,
                    source_uri=directory_path,
                    error_class=ErrorClass.PARSING,
                    error_msg=f"Directory not found: {directory_path}",
                    collected_at=datetime.now(timezone.utc)
                ))
                return ParseResult(all_chunks, all_errors, 0, int((time.time() - start_time) * 1000))

            # Find relevant files
            relevant_files = []
            for pattern in ['**/*.py', '**/*.json', '**/*.yaml', '**/*.yml']:
                relevant_files.extend(path.glob(pattern))

            for file_path in relevant_files:
                # Skip common non-relevant files
                if any(skip in str(file_path) for skip in [
                    '__pycache__', '.pyc', '.git', 'node_modules',
                    '.env', 'requirements.txt', 'setup.py'
                ]):
                    continue

                result = self.parse_file(str(file_path))
                all_chunks.extend(result.chunks)
                all_errors.extend(result.errors)
                files_processed += result.files_processed

        except Exception as e:
            all_errors.append(ParseError(
                source_type=self.source_type,
                source_uri=directory_path,
                error_class=ErrorClass.PARSING,
                error_msg=f"Error processing directory: {str(e)}",
                collected_at=datetime.now(datetime.timezone.utc)
            ))

        processing_duration = int((time.time() - start_time) * 1000)
        return ParseResult(all_chunks, all_errors, files_processed, processing_duration)

    def validate_content(self, content: str) -> bool:
        """Validate if content is FastAPI-related"""
        if not content or not content.strip():
            return False

        # Check for FastAPI Python code
        if self._is_fastapi_python_content(content):
            return True

        # Check for OpenAPI specification
        if self._is_openapi_spec(content):
            return True

        return False

    def parse_openapi_spec(self, spec_content: str) -> List[ChunkMetadata]:
        """Parse OpenAPI specification and extract endpoint metadata"""
        chunks = []

        try:
            # Try JSON first, then YAML
            try:
                spec = json.loads(spec_content)
            except json.JSONDecodeError:
                spec = yaml.safe_load(spec_content)

            if not isinstance(spec, dict) or 'openapi' not in spec:
                raise ValueError("Invalid OpenAPI specification format")

            # Parse paths and operations
            paths = spec.get('paths', {})
            for path, path_item in paths.items():
                for method, operation in path_item.items():
                    if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'TRACE']:
                        chunk = self._create_openapi_endpoint_chunk(
                            path, method.upper(), operation, spec_content
                        )
                        chunks.append(chunk)

            # Parse component schemas
            components = spec.get('components', {})
            schemas = components.get('schemas', {})
            for schema_name, schema_def in schemas.items():
                chunk = self._create_openapi_schema_chunk(
                    schema_name, schema_def, spec_content
                )
                chunks.append(chunk)

        except Exception as e:
            self.logger.error(f"Error parsing OpenAPI spec: {e}")
            raise ValueError(f"Failed to parse OpenAPI specification: {e}")

        return chunks

    def parse_source_code(self, python_code: str) -> List[ChunkMetadata]:
        """Parse Python source code for FastAPI routes and models"""
        chunks = []

        try:
            # Parse the AST
            tree = ast.parse(python_code)

            # Extract route functions and Pydantic models
            chunks.extend(self._extract_route_functions(tree, python_code))
            chunks.extend(self._extract_pydantic_models(tree, python_code))
            chunks.extend(self._extract_dependency_functions(tree, python_code))

        except SyntaxError as e:
            self.logger.error(f"Syntax error in Python code: {e}")
            raise SyntaxError(f"Invalid Python syntax: {e}")

        return chunks

    def _is_fastapi_python_content(self, content: str) -> bool:
        """Check if content is FastAPI Python code"""
        fastapi_indicators = [
            'from fastapi import',
            'import fastapi',
            '@app.',
            '@router.',
            'FastAPI(',
            'APIRouter(',
            'BaseModel'
        ]

        return any(indicator in content for indicator in fastapi_indicators)

    def _is_openapi_spec(self, content: str) -> bool:
        """Check if content is an OpenAPI specification"""
        content = content.strip()

        # JSON format
        if content.startswith('{'):
            try:
                spec = json.loads(content)
                return 'openapi' in spec or 'swagger' in spec
            except:
                return False

        # YAML format
        if content.startswith(('openapi:', '---', 'swagger:')):
            try:
                spec = yaml.safe_load(content)
                return isinstance(spec, dict) and ('openapi' in spec or 'swagger' in spec)
            except:
                return False

        return False

    def _create_openapi_endpoint_chunk(self, path: str, method: str,
                                     operation: Dict, spec_content: str) -> ChunkMetadata:
        """Create chunk for OpenAPI endpoint"""
        operation_id = operation.get('operationId', f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}")

        # Extract status codes
        status_codes = []
        responses = operation.get('responses', {})
        for status_code in responses.keys():
            try:
                status_codes.append(int(status_code))
            except ValueError:
                pass  # Skip non-numeric status codes like 'default'

        # Extract request/response models
        request_model = None
        response_model = None

        request_body = operation.get('requestBody', {})
        if request_body:
            content = request_body.get('content', {})
            json_content = content.get('application/json', {})
            schema = json_content.get('schema', {})
            request_model = schema.get('$ref', str(schema))

        # Get primary response model (usually 200 or 201)
        for status in ['200', '201', '202']:
            if status in responses:
                response_content = responses[status].get('content', {})
                json_content = response_content.get('application/json', {})
                schema = json_content.get('schema', {})
                response_model = schema.get('$ref', str(schema))
                break

        # Create content chunk
        chunk_content = json.dumps({
            'path': path,
            'method': method,
            'operation': operation
        }, indent=2)

        # Generate artifact ID with line numbers (estimated from content)
        lines = spec_content.split('\n')
        start_line = 1
        end_line = len(lines)

        # Try to find approximate line numbers
        path_search = f'"{path}"'
        for i, line in enumerate(lines):
            if path_search in line:
                start_line = i + 1
                break

        artifact_id = self.id_generator.generate_artifact_id(
            f"openapi_spec_{operation_id}",
            {"start_line": start_line, "end_line": end_line}
        )

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=chunk_content,
            content_hash=self.hash_generator.generate_content_hash(chunk_content, "fastapi"),
            collected_at=datetime.now(timezone.utc),
            source_metadata={
                'http_method': method,
                'route_path': path,
                'operation_id': operation_id,
                'status_codes': status_codes,
                'request_model': request_model,
                'response_model': response_model,
                'tags': operation.get('tags', []),
                'summary': operation.get('summary', ''),
                'description': operation.get('description', ''),
                'openapi_path': path,
                'openapi_operation': operation_id
            }
        )

    def _create_openapi_schema_chunk(self, schema_name: str, schema_def: Dict,
                                   spec_content: str) -> ChunkMetadata:
        """Create chunk for OpenAPI schema definition"""
        chunk_content = json.dumps({
            'schema_name': schema_name,
            'schema_definition': schema_def
        }, indent=2)

        # Generate artifact ID
        artifact_id = self.id_generator.generate_artifact_id(
            f"openapi_schema_{schema_name}",
            {"start_line": 1, "end_line": 10}  # Estimated
        )

        # Extract field information
        properties = schema_def.get('properties', {})
        required_fields = schema_def.get('required', [])

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=chunk_content,
            content_hash=self.hash_generator.generate_content_hash(chunk_content, "fastapi"),
            collected_at=datetime.now(timezone.utc),
            source_metadata={
                'openapi_schema': schema_name,
                'schema_type': schema_def.get('type', 'object'),
                'schema_properties': list(properties.keys()),
                'required_fields': required_fields,
                'description': schema_def.get('description', ''),
                'example': schema_def.get('example')
            }
        )

    def _extract_route_functions(self, tree: ast.AST, source_code: str) -> List[ChunkMetadata]:
        """Extract FastAPI route functions from AST"""
        chunks = []
        lines = source_code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has FastAPI route decorators
                route_info = self._analyze_route_decorators(node)
                if route_info:
                    chunk = self._create_route_function_chunk(
                        node, route_info, source_code, lines
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_pydantic_models(self, tree: ast.AST, source_code: str) -> List[ChunkMetadata]:
        """Extract Pydantic model classes from AST"""
        chunks = []
        lines = source_code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from BaseModel
                if self._is_pydantic_model(node):
                    chunk = self._create_pydantic_model_chunk(
                        node, source_code, lines
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_dependency_functions(self, tree: ast.AST, source_code: str) -> List[ChunkMetadata]:
        """Extract dependency injection functions"""
        chunks = []
        lines = source_code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is used as a dependency
                if self._is_dependency_function(node, tree):
                    chunk = self._create_dependency_function_chunk(
                        node, source_code, lines
                    )
                    chunks.append(chunk)

        return chunks

    def _analyze_route_decorators(self, func_node: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Analyze function decorators for FastAPI route information"""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                # Handle @app.get("/path") style decorators
                if isinstance(decorator.func, ast.Attribute):
                    method = decorator.func.attr.upper()
                    if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'TRACE']:
                        path = None
                        status_code = None

                        # Extract path from first argument
                        if decorator.args:
                            if isinstance(decorator.args[0], ast.Str):
                                path = decorator.args[0].s
                            elif isinstance(decorator.args[0], ast.Constant):
                                path = decorator.args[0].value

                        # Extract status_code from keywords
                        for keyword in decorator.keywords:
                            if keyword.arg == 'status_code':
                                if isinstance(keyword.value, ast.Num):
                                    status_code = keyword.value.n
                                elif isinstance(keyword.value, ast.Constant):
                                    status_code = keyword.value.value

                        return {
                            'http_method': method,
                            'route_path': path or f"/{func_node.name}",
                            'status_code': status_code or (200 if method == 'GET' else 201 if method == 'POST' else 200),
                            'operation_id': func_node.name
                        }

        return None

    def _is_pydantic_model(self, class_node: ast.ClassDef) -> bool:
        """Check if class inherits from BaseModel"""
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == 'BaseModel':
                return True
            elif isinstance(base, ast.Attribute) and base.attr == 'BaseModel':
                return True
        return False

    def _is_dependency_function(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is used as FastAPI dependency"""
        func_name = func_node.name

        # Look for Depends(function_name) patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'Depends':
                    if node.args and isinstance(node.args[0], ast.Name):
                        if node.args[0].id == func_name:
                            return True
        return False

    def _create_route_function_chunk(self, func_node: ast.FunctionDef,
                                   route_info: Dict[str, Any],
                                   source_code: str, lines: List[str]) -> ChunkMetadata:
        """Create chunk for FastAPI route function"""
        start_line = func_node.lineno
        end_line = func_node.end_lineno or start_line

        # Extract function source
        function_lines = lines[start_line - 1:end_line]

        # Include decorators
        if func_node.decorator_list:
            decorator_start = func_node.decorator_list[0].lineno
            function_lines = lines[decorator_start - 1:end_line]
            start_line = decorator_start

        chunk_content = '\n'.join(function_lines)

        # Analyze function signature for dependencies and parameters
        dependencies = []
        parameters = []
        response_model = None

        for arg in func_node.args.args:
            if arg.annotation:
                param_info = {
                    'name': arg.arg,
                    'type': ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                }
                parameters.append(param_info)

        # Look for response_model in decorators
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if keyword.arg == 'response_model':
                        if hasattr(ast, 'unparse'):
                            response_model = ast.unparse(keyword.value)
                        else:
                            response_model = str(keyword.value)

        artifact_id = self.id_generator.generate_artifact_id(
            f"route_{func_node.name}",
            {"start_line": start_line, "end_line": end_line}
        )

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=chunk_content,
            content_hash=self.hash_generator.generate_content_hash(chunk_content, "fastapi"),
            collected_at=datetime.now(timezone.utc),
            source_metadata={
                'http_method': route_info['http_method'],
                'route_path': route_info['route_path'],
                'operation_id': route_info['operation_id'],
                'status_codes': [route_info['status_code']],
                'response_model': response_model,
                'function_name': func_node.name,
                'parameters': parameters,
                'dependencies': dependencies,
                'start_line': start_line,
                'end_line': end_line,
                'is_async': isinstance(func_node, ast.AsyncFunctionDef)
            }
        )

    def _create_pydantic_model_chunk(self, class_node: ast.ClassDef,
                                   source_code: str, lines: List[str]) -> ChunkMetadata:
        """Create chunk for Pydantic model class"""
        start_line = class_node.lineno
        end_line = class_node.end_lineno or start_line

        # Extract class source
        class_lines = lines[start_line - 1:end_line]
        chunk_content = '\n'.join(class_lines)

        # Analyze class fields
        fields = []
        validators = []

        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                # Field with type annotation
                field_info = {
                    'name': node.target.id,
                    'type': ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation),
                    'has_default': node.value is not None
                }
                fields.append(field_info)
            elif isinstance(node, ast.FunctionDef) and node.name.startswith('validate_'):
                # Validator function
                validators.append(node.name)

        artifact_id = self.id_generator.generate_artifact_id(
            f"model_{class_node.name}",
            {"start_line": start_line, "end_line": end_line}
        )

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=chunk_content,
            content_hash=self.hash_generator.generate_content_hash(chunk_content, "fastapi"),
            collected_at=datetime.now(timezone.utc),
            source_metadata={
                'fastapi_model_type': 'pydantic',
                'fastapi_model_name': class_node.name,
                'fastapi_model_fields': [f"{field['name']}: {field['type']}" for field in fields],
                'model_validators': validators,
                'field_count': len(fields),
                'start_line': start_line,
                'end_line': end_line,
                'base_classes': [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) for base in class_node.bases]
            }
        )

    def _create_dependency_function_chunk(self, func_node: ast.FunctionDef,
                                        source_code: str, lines: List[str]) -> ChunkMetadata:
        """Create chunk for dependency function"""
        start_line = func_node.lineno
        end_line = func_node.end_lineno or start_line

        # Extract function source
        function_lines = lines[start_line - 1:end_line]
        chunk_content = '\n'.join(function_lines)

        artifact_id = self.id_generator.generate_artifact_id(
            f"dependency_{func_node.name}",
            {"start_line": start_line, "end_line": end_line}
        )

        return ChunkMetadata(
            source_type=self.source_type,
            artifact_id=artifact_id,
            content_text=chunk_content,
            content_hash=self.hash_generator.generate_content_hash(chunk_content, "fastapi"),
            collected_at=datetime.now(timezone.utc),
            source_metadata={
                'fastapi_dependency': True,
                'function_name': func_node.name,
                'dependency_type': 'function',
                'start_line': start_line,
                'end_line': end_line,
                'is_async': isinstance(func_node, ast.AsyncFunctionDef)
            }
        )