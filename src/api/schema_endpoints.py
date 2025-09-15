"""
FastAPI endpoints for schema management.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..janapada_memory import SchemaManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schema", tags=["Schema Management"])


# Request Models
class CreateTablesRequest(BaseModel):
    """Request model for table creation."""

    force_recreate: bool = Field(
        False, description="Whether to recreate existing tables"
    )
    tables: list[str] = Field(["all"], description="List of tables to create or 'all'")
    partition_expiration_days: int = Field(
        365, ge=1, le=7300, description="Partition expiration in days"
    )


class CreateIndexRequest(BaseModel):
    """Request model for single index creation."""

    name: str = Field(..., description="Index name")
    table: str = Field(..., description="Table name")
    column: str = Field(..., description="Column name")
    index_type: str = Field("IVF", description="Index type (IVF, TREE_AH)")
    distance_type: str = Field(
        "COSINE", description="Distance type (COSINE, EUCLIDEAN, DOT_PRODUCT)"
    )
    options: Optional[dict[str, Any]] = Field(None, description="Index options")


class CreateIndexesRequest(BaseModel):
    """Request model for index creation."""

    indexes: list[CreateIndexRequest] = Field(
        ..., description="List of indexes to create"
    )


class ValidateSchemaRequest(BaseModel):
    """Request model for schema validation."""

    validate_data: bool = Field(False, description="Whether to validate data quality")
    check_indexes: bool = Field(True, description="Whether to check index status")
    sample_size: int = Field(
        1000, ge=100, le=10000, description="Sample size for data validation"
    )


# Response Models
class TableInfo(BaseModel):
    """Table information model."""

    name: str
    full_name: str
    schema: list[dict[str, str]]
    partitioning: dict[str, Any]
    clustering: list[str]
    created_at: Optional[str] = None


class CreateTablesResponse(BaseModel):
    """Response model for table creation."""

    tables_created: list[TableInfo]
    creation_time_ms: int
    dataset_id: str
    project_id: str


class IndexInfo(BaseModel):
    """Index information model."""

    name: str
    table: str
    column: str
    index_type: str
    distance_type: Optional[str] = None
    status: str
    coverage_percentage: float
    options: Optional[dict[str, Any]] = None


class CreateIndexesResponse(BaseModel):
    """Response model for index creation."""

    indexes_created: list[IndexInfo]
    creation_time_ms: int


class ValidationResult(BaseModel):
    """Validation result model."""

    overall_status: str
    tables: dict[str, Any]
    indexes: dict[str, Any]
    recommendations: list[str]
    timestamp: str


class DatasetInfo(BaseModel):
    """Dataset information model."""

    dataset_id: str
    project_id: str
    location: str
    created: Optional[str] = None
    modified: Optional[str] = None
    description: Optional[str] = None
    labels: dict[str, str] = {}


class TablesListResponse(BaseModel):
    """Response model for listing tables."""

    tables: list[dict[str, Any]]
    dataset_info: DatasetInfo
    count: int


class IndexesListResponse(BaseModel):
    """Response model for listing indexes."""

    indexes: list[IndexInfo]
    count: int


# Dependency to get schema manager instance
def get_schema_manager() -> SchemaManager:
    """Get schema manager instance."""
    return SchemaManager()


@router.post(
    "/tables",
    response_model=CreateTablesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create BigQuery tables",
    description="Create BigQuery tables with partitioning and clustering.",
)
async def create_tables(
    request: CreateTablesRequest = CreateTablesRequest(),
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> CreateTablesResponse:
    """Create BigQuery tables."""
    try:
        result = schema_manager.create_tables(
            tables=request.tables,
            partition_expiration_days=request.partition_expiration_days,
            force_recreate=request.force_recreate,
        )

        # Convert to response format
        tables_created = []
        for table_info in result["tables_created"]:
            tables_created.append(TableInfo(**table_info))

        return CreateTablesResponse(
            tables_created=tables_created,
            creation_time_ms=result["creation_time_ms"],
            dataset_id=result["dataset_id"],
            project_id=result["project_id"],
        )

    except ValueError as e:
        if "already exists" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": "TableAlreadyExists", "message": str(e)},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "InvalidRequest", "message": str(e)},
            )
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during table creation",
        )


@router.get(
    "/tables",
    response_model=TablesListResponse,
    summary="List BigQuery tables",
    description="List all tables in the dataset with schema information.",
)
async def list_tables(
    include_schema: bool = Query(
        True, description="Whether to include schema information"
    ),
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> TablesListResponse:
    """List BigQuery tables."""
    try:
        result = schema_manager.list_tables(include_schema=include_schema)

        dataset_info = DatasetInfo(**result["dataset_info"])

        return TablesListResponse(
            tables=result["tables"], dataset_info=dataset_info, count=result["count"]
        )

    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing tables",
        )


@router.delete(
    "/tables",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete all tables",
    description="Delete all tables in the dataset (requires confirmation).",
)
async def delete_tables(
    confirm: bool = Query(
        False, description="Confirmation required to delete all tables"
    ),
    schema_manager: SchemaManager = Depends(get_schema_manager),
):
    """Delete all tables."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ConfirmationRequired",
                "message": "Must set confirm=true to delete all tables",
            },
        )

    try:
        schema_manager.delete_all_tables(confirm=True)

    except Exception as e:
        logger.error(f"Failed to delete tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during table deletion",
        )


@router.post(
    "/indexes",
    response_model=CreateIndexesResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create vector indexes",
    description="Create vector indexes for similarity search.",
)
async def create_indexes(
    request: CreateIndexesRequest,
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> CreateIndexesResponse:
    """Create vector indexes."""
    try:
        # Convert request to format expected by schema manager
        indexes_spec = []
        for index_req in request.indexes:
            index_spec = {
                "name": index_req.name,
                "table": index_req.table,
                "column": index_req.column,
                "index_type": index_req.index_type,
                "distance_type": index_req.distance_type,
                "options": index_req.options or {},
            }
            indexes_spec.append(index_spec)

        result = schema_manager.create_indexes(indexes_spec)

        # Convert to response format
        indexes_created = []
        for index_info in result["indexes_created"]:
            indexes_created.append(IndexInfo(**index_info))

        return CreateIndexesResponse(
            indexes_created=indexes_created, creation_time_ms=result["creation_time_ms"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidIndexSpec", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during index creation",
        )


@router.get(
    "/indexes",
    response_model=IndexesListResponse,
    summary="List vector indexes",
    description="List all vector indexes in the dataset.",
)
async def list_indexes(
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> IndexesListResponse:
    """List vector indexes."""
    try:
        result = schema_manager.list_indexes()

        indexes = []
        for index_info in result["indexes"]:
            indexes.append(IndexInfo(**index_info))

        return IndexesListResponse(indexes=indexes, count=result["count"])

    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing indexes",
        )


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Validate schema",
    description="Validate BigQuery schema and data quality.",
)
async def validate_schema(
    request: ValidateSchemaRequest = ValidateSchemaRequest(),
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> ValidationResult:
    """Validate schema and data quality."""
    try:
        result = schema_manager.validate_schema(
            validate_data=request.validate_data,
            check_indexes=request.check_indexes,
            sample_size=request.sample_size,
        )

        return ValidationResult(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "InvalidValidationRequest", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during schema validation",
        )


@router.get(
    "/info",
    response_model=dict[str, Any],
    summary="Get dataset info",
    description="Get information about the BigQuery dataset.",
)
async def get_dataset_info(
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> dict[str, Any]:
    """Get dataset information."""
    try:
        dataset_info = schema_manager.get_dataset_info()

        # Add table and index counts
        tables_info = schema_manager.list_tables(include_schema=False)
        indexes_info = schema_manager.list_indexes()

        dataset_info.update(
            {
                "table_count": tables_info["count"],
                "index_count": indexes_info["count"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        return dataset_info

    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting dataset information",
        )


@router.get(
    "/tables/{table_name}/schema",
    response_model=list[dict[str, Any]],
    summary="Get table schema",
    description="Get detailed schema information for a specific table.",
)
async def get_table_schema(
    table_name: str, schema_manager: SchemaManager = Depends(get_schema_manager)
) -> list[dict[str, Any]]:
    """Get table schema."""
    try:
        schema_info = schema_manager.get_table_schema(table_name)
        return schema_info

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table '{table_name}' not found",
            )

        logger.error(f"Failed to get table schema: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting table schema",
        )


@router.get(
    "/indexes/{index_name}/status",
    response_model=dict[str, Any],
    summary="Get index status",
    description="Get detailed status information for a specific vector index.",
)
async def get_index_status(
    index_name: str, schema_manager: SchemaManager = Depends(get_schema_manager)
) -> dict[str, Any]:
    """Get index status."""
    try:
        status_info = schema_manager.get_index_status(index_name)
        return status_info

    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index '{index_name}' not found",
            )

        logger.error(f"Failed to get index status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while getting index status",
        )


@router.post(
    "/setup",
    response_model=dict[str, Any],
    summary="Setup complete schema",
    description="Setup complete BigQuery schema with tables and indexes.",
)
async def setup_schema(
    schema_manager: SchemaManager = Depends(get_schema_manager),
) -> dict[str, Any]:
    """Setup complete schema."""
    try:
        # Create dataset
        dataset_result = schema_manager.create_dataset()

        # Create tables
        tables_result = schema_manager.create_tables()

        # Create default vector index
        default_index = [
            {
                "name": "embedding_similarity_index",
                "table": "source_embeddings",
                "column": "embedding",
                "index_type": "IVF",
                "distance_type": "COSINE",
                "options": {"num_lists": 1000, "fraction_lists_to_search": 0.01},
            }
        ]

        indexes_result = schema_manager.create_indexes(default_index)

        return {
            "status": "completed",
            "dataset": dataset_result,
            "tables": {
                "created": len(tables_result["tables_created"]),
                "names": [t["name"] for t in tables_result["tables_created"]],
            },
            "indexes": {
                "created": len(indexes_result["indexes_created"]),
                "names": [i["name"] for i in indexes_result["indexes_created"]],
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Schema setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during schema setup",
        )
