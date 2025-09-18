"""
BigQuery adapters for vector search operations.

This package contains adapters that handle BigQuery-specific operations
including SQL query construction, parameter encoding, and result parsing.
"""

from .bigquery_adapter import BigQueryAdapter

__all__ = ["BigQueryAdapter"]
