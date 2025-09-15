"""
Shared import module for parser interfaces
This ensures all modules use the same instance of the parser interface classes
"""

import os
import sys
import importlib.util

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Load parser interfaces module once
parser_interfaces_path = os.path.join(project_root, "specs", "002-m1-parse-and", "contracts", "parser-interfaces.py")
spec = importlib.util.spec_from_file_location("parser_interfaces", parser_interfaces_path)
parser_interfaces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser_interfaces)

# Export all the classes
IRSParser = parser_interfaces.IRSParser
BaseParser = parser_interfaces.BaseParser
SourceType = parser_interfaces.SourceType
ChunkMetadata = parser_interfaces.ChunkMetadata
ParseResult = parser_interfaces.ParseResult
ParseError = parser_interfaces.ParseError
ErrorClass = parser_interfaces.ErrorClass
KubernetesParser = parser_interfaces.KubernetesParser
FastAPIParser = parser_interfaces.FastAPIParser
COBOLParser = parser_interfaces.COBOLParser
MUMPSParser = parser_interfaces.MUMPSParser

__all__ = [
    'IRSParser', 'BaseParser', 'SourceType', 'ChunkMetadata',
    'ParseResult', 'ParseError', 'ErrorClass', 'KubernetesParser',
    'FastAPIParser', 'COBOLParser', 'MUMPSParser'
]