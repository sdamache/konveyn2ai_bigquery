"""
Janapada Memory - BigQuery Vector Store Implementation

This module implements the BigQuery-based vector storage system for KonveyN2AI,
replacing the previous Vertex AI implementation with a more scalable solution.
"""

__version__ = "1.0.0"
__author__ = "KonveyN2AI Team"

try:
    from .connections.bigquery_connection import BigQueryConnectionManager
    from .bigquery_vector_store import BigQueryVectorStore
    from .bigquery_vector_index import BigQueryVectorIndex
    from .memory_service import JanapadaMemoryService, create_memory_service
    from .dimension_reducer import DimensionReducer
    from .migration_manager import MigrationManager
    from .schema_manager import SchemaManager

    # Legacy alias for backward compatibility
    BigQueryConnection = BigQueryConnectionManager
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import all components: {e}. Please install required dependencies."
    )

    # Provide stub classes for missing dependencies
    class BigQueryConnectionManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("google-cloud-bigquery not installed")

    # Legacy alias for backward compatibility
    BigQueryConnection = BigQueryConnectionManager

    class BigQueryVectorIndex:
        def __init__(self, *args, **kwargs):
            raise ImportError("google-cloud-bigquery not installed")

    class JanapadaMemoryService:
        def __init__(self, *args, **kwargs):
            raise ImportError("google-cloud-bigquery not installed")

    def create_memory_service(*args, **kwargs):
        raise ImportError("google-cloud-bigquery not installed")

    class BigQueryVectorStore:
        def __init__(self, *args, **kwargs):
            raise ImportError("google-cloud-bigquery not installed")

    class SchemaManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("google-cloud-bigquery not installed")

    class DimensionReducer:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-learn not installed")

    class MigrationManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("Required dependencies not installed")


__all__ = [
    "BigQueryConnection",  # Legacy - deprecated
    "BigQueryConnectionManager",  # New preferred connection manager
    "BigQueryVectorStore",
    "BigQueryVectorIndex",  # Main vector index implementation
    "JanapadaMemoryService",  # Primary service interface
    "create_memory_service",  # Factory function
    "SchemaManager",
    "DimensionReducer",
    "MigrationManager",
]
