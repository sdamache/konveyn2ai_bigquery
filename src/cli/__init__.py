"""
CLI entrypoints for M1 multi-source ingestion
Provides command-line interface for all parser types with standardized options
"""

from .commands import (
    ingest_cobol,
    ingest_fastapi,
    ingest_irs,
    ingest_kubernetes,
    ingest_mumps,
    setup_environment,
)
from .main import create_parser, main

__version__ = "1.0.0"

__all__ = [
    "main",
    "create_parser",
    "ingest_kubernetes",
    "ingest_fastapi",
    "ingest_cobol",
    "ingest_irs",
    "ingest_mumps",
    "setup_environment",
]
