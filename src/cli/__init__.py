"""
CLI entrypoints for M1 multi-source ingestion
Provides command-line interface for all parser types with standardized options
"""

from .main import main, create_parser
from .commands import (
    ingest_kubernetes,
    ingest_fastapi,
    ingest_cobol,
    ingest_irs,
    ingest_mumps,
    setup_environment
)

__version__ = "1.0.0"

__all__ = [
    "main",
    "create_parser",
    "ingest_kubernetes",
    "ingest_fastapi",
    "ingest_cobol",
    "ingest_irs",
    "ingest_mumps",
    "setup_environment"
]