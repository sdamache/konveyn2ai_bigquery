"""
MUMPS/VistA FileMan Dictionary Parser Module

This module provides parsing capabilities for MUMPS/VistA FileMan data dictionaries
and global variable definitions, implementing the contract defined in parser-interfaces.py.

Key Components:
- MUMPSParserImpl: Main parser implementation
- MUMPS regex patterns for FileMan dictionary parsing
- Global variable structure analysis
- Hierarchical content chunking
- VistA-specific metadata extraction

Usage:
    from src.ingest.mumps import MUMPSParserImpl, create_mumps_parser

    parser = create_mumps_parser()
    result = parser.parse_file("sample.ro")
"""

from .parser import MUMPSParserImpl, create_mumps_parser, validate_mumps_parser

__all__ = [
    'MUMPSParserImpl',
    'create_mumps_parser',
    'validate_mumps_parser'
]

__version__ = "1.0.0"