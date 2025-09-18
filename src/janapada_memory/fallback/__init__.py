"""
Fallback implementations for vector search operations.

This package contains fallback components that provide degraded functionality
when primary BigQuery services are unavailable.
"""

from .local_vector_index import LocalVectorIndex

__all__ = ["LocalVectorIndex"]