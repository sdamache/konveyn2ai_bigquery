"""
Janapada Memory - BigQuery Vector Store Implementation

This module implements the BigQuery-based vector storage system for KonveyN2AI,
replacing the previous Vertex AI implementation with a more scalable solution.
"""

__version__ = "1.0.0"
__author__ = "KonveyN2AI Team"

try:
    from .bigquery_connection import BigQueryConnection
    from .bigquery_vector_store import BigQueryVectorStore
    from .schema_manager import SchemaManager
    from .dimension_reducer import DimensionReducer
    from .migration_manager import MigrationManager
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all components: {e}. Please install required dependencies.")
    
    # Provide stub classes for missing dependencies
    class BigQueryConnection:
        def __init__(self, *args, **kwargs):
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
    "BigQueryConnection",
    "BigQueryVectorStore", 
    "SchemaManager",
    "DimensionReducer",
    "MigrationManager"
]