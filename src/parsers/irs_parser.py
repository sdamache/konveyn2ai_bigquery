"""
IRS Parser Compatibility Module
Bridges the import path expected by tests to the actual implementation
"""

# Import the actual implementation
from src.ingest.irs.parser import IRSParserImpl

# Re-export for compatibility
__all__ = ['IRSParserImpl']