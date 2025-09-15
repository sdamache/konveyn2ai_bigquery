# Import the actual implementation from the ingest module
from src.ingest.mumps.parser import MUMPSParserImpl as MUMPSParser

# Re-export for backward compatibility with tests
__all__ = ["MUMPSParser"]
