"""
BigQuery Schema Manager

Handles BigQuery schema operations including table creation, index management,
partitioning, clustering, and schema validation for the vector store.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from google.cloud import bigquery
from google.cloud.exceptions import Conflict, NotFound

from .bigquery_connection import BigQueryConnection

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages BigQuery schema for vector storage operations."""

    # Table schemas definition
    SCHEMAS = {
        "source_metadata": [
            bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("artifact_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("text_content", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("kind", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("api_path", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("record_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("partition_date", "DATE", mode="REQUIRED"),
        ],
        "source_embeddings": [
            bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("embedding_model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("partition_date", "DATE", mode="REQUIRED"),
        ],
        "gap_metrics": [
            bigquery.SchemaField("analysis_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("metric_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("metric_value", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("partition_date", "DATE", mode="REQUIRED"),
        ],
    }

    # Clustering configuration
    CLUSTERING_CONFIG = {
        "source_metadata": ["artifact_type", "source", "chunk_id"],
        "source_embeddings": ["chunk_id"],
        "gap_metrics": ["analysis_id", "metric_type", "chunk_id"],
    }

    def __init__(
        self,
        connection: Optional[BigQueryConnection] = None,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize schema manager.

        Args:
            connection: BigQuery connection instance
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        if connection:
            self.connection = connection
        else:
            self.connection = BigQueryConnection(
                project_id=project_id, dataset_id=dataset_id
            )

        self.project_id = self.connection.project_id
        self.dataset_id = self.connection.dataset_id

        logger.info(
            f"Schema manager initialized for {self.project_id}.{self.dataset_id}"
        )

    def create_dataset(self, location: str = "us-central1") -> dict[str, Any]:
        """
        Create BigQuery dataset if it doesn't exist.

        Args:
            location: Dataset location

        Returns:
            Dataset creation result
        """
        try:
            dataset = self.connection.create_dataset(
                dataset_id=self.dataset_id, location=location, exists_ok=True
            )

            return {
                "dataset_id": dataset.dataset_id,
                "project_id": dataset.project,
                "location": dataset.location,
                "created": dataset.created is not None,
                "full_dataset_id": dataset.full_dataset_id,
            }

        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    def create_tables(
        self,
        tables: list[str] = None,
        partition_expiration_days: int = 365,
        force_recreate: bool = False,
    ) -> dict[str, Any]:
        """
        Create all required tables with partitioning and clustering.

        Args:
            tables: List of table names to create (default: all tables)
            partition_expiration_days: Partition expiration in days
            force_recreate: Whether to drop and recreate existing tables

        Returns:
            Table creation results
        """
        if tables is None or tables == ["all"]:
            tables = list(self.SCHEMAS.keys())

        created_tables = []
        errors = []

        for table_name in tables:
            try:
                table_result = self._create_single_table(
                    table_name=table_name,
                    partition_expiration_days=partition_expiration_days,
                    force_recreate=force_recreate,
                )
                created_tables.append(table_result)
                logger.info(f"Table {table_name} created successfully")

            except Exception as e:
                error_msg = f"Failed to create table {table_name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        result = {
            "tables_created": created_tables,
            "creation_time_ms": int(datetime.now().timestamp() * 1000),
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
        }

        if errors:
            result["errors"] = errors

        return result

    def _create_single_table(
        self,
        table_name: str,
        partition_expiration_days: int = 365,
        force_recreate: bool = False,
    ) -> dict[str, Any]:
        """Create a single table with proper configuration."""
        if table_name not in self.SCHEMAS:
            raise ValueError(f"Unknown table: {table_name}")

        table_ref = self.connection.client.dataset(self.dataset_id).table(table_name)

        # Check if table exists
        try:
            existing_table = self.connection.client.get_table(table_ref)
            if not force_recreate:
                raise Conflict(f"Table {table_name} already exists")
            else:
                # Delete existing table
                self.connection.client.delete_table(table_ref)
                logger.info(f"Deleted existing table {table_name}")
        except NotFound:
            pass  # Table doesn't exist, which is fine

        # Create table
        table = bigquery.Table(table_ref, schema=self.SCHEMAS[table_name])

        # Configure partitioning
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="partition_date",
            expiration_ms=partition_expiration_days * 24 * 60 * 60 * 1000,
        )

        # Configure clustering
        if table_name in self.CLUSTERING_CONFIG:
            table.clustering_fields = self.CLUSTERING_CONFIG[table_name]

        # Create the table
        created_table = self.connection.client.create_table(table)

        return {
            "name": table_name,
            "full_name": created_table.full_table_id,
            "schema": [
                {"name": field.name, "type": field.field_type}
                for field in created_table.schema
            ],
            "partitioning": {
                "type": "TIME",
                "field": "partition_date",
                "expiration_days": partition_expiration_days,
            },
            "clustering": self.CLUSTERING_CONFIG.get(table_name, []),
            "created_at": (
                created_table.created.isoformat() if created_table.created else None
            ),
        }

    def create_indexes(self, indexes: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Create vector indexes for similarity search.

        Args:
            indexes: List of index specifications

        Returns:
            Index creation results
        """
        created_indexes = []
        errors = []
        start_time = datetime.now()

        for index_spec in indexes:
            try:
                index_result = self._create_vector_index(index_spec)
                created_indexes.append(index_result)
                logger.info(f"Index {index_spec['name']} created successfully")

            except Exception as e:
                error_msg = (
                    f"Failed to create index {index_spec.get('name', 'unknown')}: {e}"
                )
                logger.error(error_msg)
                errors.append(error_msg)

        creation_time = int((datetime.now() - start_time).total_seconds() * 1000)

        result = {"indexes_created": created_indexes, "creation_time_ms": creation_time}

        if errors:
            result["errors"] = errors

        return result

    def _create_vector_index(self, index_spec: dict[str, Any]) -> dict[str, Any]:
        """Create a single vector index."""
        name = index_spec["name"]
        table_name = index_spec["table"]
        column = index_spec["column"]
        index_type = index_spec.get("index_type", "IVF")
        distance_type = index_spec.get("distance_type", "COSINE")
        options = index_spec.get("options", {})

        # Validate parameters
        if index_type not in ["IVF", "TREE_AH"]:
            raise ValueError(f"Invalid index type: {index_type}")

        if distance_type not in ["COSINE", "EUCLIDEAN", "DOT_PRODUCT"]:
            raise ValueError(f"Invalid distance type: {distance_type}")

        # Build CREATE VECTOR INDEX statement
        full_table_name = f"`{self.project_id}.{self.dataset_id}.{table_name}`"

        sql_parts = [
            f"CREATE VECTOR INDEX IF NOT EXISTS `{name}`",
            f"ON {full_table_name}({column})",
            "OPTIONS(",
        ]

        option_parts = [
            f"index_type='{index_type}'",
            f"distance_type='{distance_type}'",
        ]

        # Add IVF-specific options
        if index_type == "IVF":
            if "num_lists" in options:
                num_lists = options["num_lists"]
                if not (1 <= num_lists <= 5000):
                    raise ValueError(
                        f"num_lists must be between 1 and 5000, got {num_lists}"
                    )
                option_parts.append(f'ivf_options=JSON"{{"num_lists": {num_lists}}}""')

            if "fraction_lists_to_search" in options:
                fraction = options["fraction_lists_to_search"]
                if not (0.0 < fraction <= 1.0):
                    raise ValueError(
                        f"fraction_lists_to_search must be between 0 and 1, got {fraction}"
                    )
                option_parts.append(f"fraction_lists_to_search={fraction}")

        sql_parts.append(", ".join(option_parts) + ")")
        create_sql = " ".join(sql_parts)

        # Execute index creation
        query_job = self.connection.client.query(create_sql)
        query_job.result()  # Wait for completion

        return {
            "name": name,
            "table": table_name,
            "column": column,
            "index_type": index_type,
            "distance_type": distance_type,
            "status": "CREATING",  # Index creation is async
            "coverage_percentage": 0.0,  # Will be updated as index builds
            "options": options,
        }

    def list_tables(self, include_schema: bool = True) -> dict[str, Any]:
        """
        List all tables in the dataset.

        Args:
            include_schema: Whether to include schema information

        Returns:
            Table listing with metadata
        """
        try:
            tables = self.connection.list_tables()
            table_info = []

            for table in tables:
                table_data = {
                    "name": table.table_id,
                    "full_name": table.full_table_id,
                    "table_type": table.table_type,
                    "num_rows": table.num_rows,
                    "num_bytes": table.num_bytes,
                    "created": table.created.isoformat() if table.created else None,
                    "modified": table.modified.isoformat() if table.modified else None,
                }

                if include_schema:
                    full_table = self.connection.get_table(table.table_id)
                    table_data["schema"] = [
                        {
                            "name": field.name,
                            "type": field.field_type,
                            "mode": field.mode,
                        }
                        for field in full_table.schema
                    ]

                    if full_table.time_partitioning:
                        table_data["partitioning"] = {
                            "type": full_table.time_partitioning.type_.name,
                            "field": full_table.time_partitioning.field,
                            "expiration_ms": full_table.time_partitioning.expiration_ms,
                        }

                    if full_table.clustering_fields:
                        table_data["clustering"] = full_table.clustering_fields

                table_info.append(table_data)

            return {
                "tables": table_info,
                "dataset_info": self.get_dataset_info(),
                "count": len(table_info),
            }

        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise

    def list_indexes(self) -> dict[str, Any]:
        """
        List all vector indexes in the dataset.

        Returns:
            Index listing with status
        """
        try:
            # Query information schema for vector indexes
            query = f"""
            SELECT
                index_name,
                table_name,
                indexed_columns,
                index_type,
                status,
                coverage_percentage
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
            ORDER BY index_name
            """

            results = self.connection.execute_query(query)
            indexes = []

            for row in results:
                indexes.append(
                    {
                        "name": row.index_name,
                        "table": row.table_name,
                        "column": (
                            row.indexed_columns[0] if row.indexed_columns else None
                        ),
                        "index_type": row.index_type,
                        "status": row.status,
                        "coverage_percentage": (
                            float(row.coverage_percentage)
                            if row.coverage_percentage
                            else 0.0
                        ),
                    }
                )

            return {"indexes": indexes, "count": len(indexes)}

        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            # Return empty list if information schema is not available
            return {"indexes": [], "count": 0}

    def get_dataset_info(self) -> dict[str, Any]:
        """Get dataset information."""
        try:
            dataset = self.connection.get_dataset()
            return {
                "dataset_id": dataset.dataset_id,
                "project_id": dataset.project,
                "location": dataset.location,
                "created": dataset.created.isoformat() if dataset.created else None,
                "modified": dataset.modified.isoformat() if dataset.modified else None,
                "description": dataset.description,
                "labels": dict(dataset.labels) if dataset.labels else {},
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            raise

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get table schema information."""
        try:
            table = self.connection.get_table(table_name)
            return [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
                for field in table.schema
            ]
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            raise

    def get_table_partition_info(self, table_name: str) -> dict[str, Any]:
        """Get table partitioning information."""
        try:
            table = self.connection.get_table(table_name)
            if table.time_partitioning:
                return {
                    "type": table.time_partitioning.type_.name,
                    "field": table.time_partitioning.field,
                    "expiration_days": (
                        table.time_partitioning.expiration_ms // (24 * 60 * 60 * 1000)
                        if table.time_partitioning.expiration_ms
                        else None
                    ),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get partition info for table {table_name}: {e}")
            raise

    def get_table_clustering_info(self, table_name: str) -> dict[str, Any]:
        """Get table clustering information."""
        try:
            table = self.connection.get_table(table_name)
            if table.clustering_fields:
                return {"clustering_fields": list(table.clustering_fields)}
            return None
        except Exception as e:
            logger.error(f"Failed to get clustering info for table {table_name}: {e}")
            raise

    def get_index_status(self, index_name: str) -> dict[str, Any]:
        """Get vector index status."""
        try:
            query = f"""
            SELECT
                index_name,
                table_name,
                status,
                coverage_percentage,
                creation_time
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
            WHERE index_name = '{index_name}'
            """

            results = list(self.connection.execute_query(query))
            if not results:
                raise NotFound(f"Index {index_name} not found")

            row = results[0]
            return {
                "name": row.index_name,
                "table": row.table_name,
                "status": row.status,
                "coverage_percentage": (
                    float(row.coverage_percentage) if row.coverage_percentage else 0.0
                ),
                "creation_time": (
                    row.creation_time.isoformat() if row.creation_time else None
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get status for index {index_name}: {e}")
            raise

    def validate_schema(
        self,
        validate_data: bool = False,
        check_indexes: bool = True,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        """
        Validate BigQuery schema and data quality.

        Args:
            validate_data: Whether to perform data quality checks
            check_indexes: Whether to check index status
            sample_size: Sample size for data validation

        Returns:
            Validation results
        """
        validation_result = {
            "overall_status": "VALID",
            "tables": {},
            "indexes": {},
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Validate tables
        try:
            tables_info = self.list_tables(include_schema=True)
            expected_tables = set(self.SCHEMAS.keys())
            existing_tables = {table["name"] for table in tables_info["tables"]}

            for table_name in expected_tables:
                table_validation = self._validate_single_table(
                    table_name,
                    table_name in existing_tables,
                    validate_data,
                    sample_size,
                )
                validation_result["tables"][table_name] = table_validation

                if not table_validation.get("schema_valid", True):
                    validation_result["overall_status"] = "INVALID"

        except Exception as e:
            logger.error(f"Table validation failed: {e}")
            validation_result["overall_status"] = "INVALID"
            validation_result["tables"]["validation_error"] = str(e)

        # Validate indexes
        if check_indexes:
            try:
                indexes_info = self.list_indexes()
                for index in indexes_info["indexes"]:
                    index_validation = self._validate_single_index(index)
                    validation_result["indexes"][index["name"]] = index_validation

                    if index_validation.get("status") not in ["ACTIVE", "CREATING"]:
                        validation_result["overall_status"] = "INVALID"

            except Exception as e:
                logger.error(f"Index validation failed: {e}")
                validation_result["indexes"]["validation_error"] = str(e)

        # Generate recommendations
        validation_result["recommendations"] = self._generate_recommendations(
            validation_result
        )

        return validation_result

    def _validate_single_table(
        self,
        table_name: str,
        exists: bool,
        validate_data: bool = False,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        """Validate a single table."""
        validation = {
            "exists": exists,
            "schema_valid": False,
            "row_count": 0,
            "data_quality_score": 1.0,
        }

        if not exists:
            return validation

        try:
            # Schema validation
            table_schema = self.get_table_schema(table_name)
            expected_schema = self.SCHEMAS[table_name]

            schema_fields = {field["name"]: field["type"] for field in table_schema}
            expected_fields = {
                field.name: field.field_type for field in expected_schema
            }

            validation["schema_valid"] = schema_fields == expected_fields

            # Row count
            query = f"SELECT COUNT(*) as row_count FROM `{self.project_id}.{self.dataset_id}.{table_name}`"
            result = list(self.connection.execute_query(query))[0]
            validation["row_count"] = result.row_count

            # Data validation if requested
            if validate_data and validation["row_count"] > 0:
                data_quality = self._validate_table_data(table_name, sample_size)
                validation["data_quality_score"] = data_quality

        except Exception as e:
            logger.error(f"Table validation failed for {table_name}: {e}")
            validation["error"] = str(e)

        return validation

    def _validate_single_index(self, index: dict[str, Any]) -> dict[str, Any]:
        """Validate a single index."""
        return {
            "exists": True,
            "status": index.get("status", "UNKNOWN"),
            "performance_score": min(1.0, index.get("coverage_percentage", 0) / 100.0),
        }

    def _validate_table_data(self, table_name: str, sample_size: int) -> float:
        """Validate table data quality."""
        try:
            query = f"""
            SELECT
                COUNT(*) as total_rows,
                COUNT(DISTINCT chunk_id) as unique_chunk_ids,
                COUNTIF(chunk_id IS NULL) as null_chunk_ids,
                COUNTIF(LENGTH(text_content) < 10) as short_content
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            LIMIT {sample_size}
            """

            result = list(self.connection.execute_query(query))[0]

            # Calculate quality score (0-1)
            total_rows = result.total_rows
            if total_rows == 0:
                return 1.0

            quality_issues = (
                result.null_chunk_ids  # Null chunk IDs
                + (total_rows - result.unique_chunk_ids)  # Duplicate chunk IDs
                + result.short_content  # Very short content
            )

            quality_score = max(0.0, 1.0 - (quality_issues / total_rows))
            return quality_score

        except Exception as e:
            logger.error(f"Data validation failed for {table_name}: {e}")
            return 0.0

    def _generate_recommendations(self, validation_result: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check for missing tables
        for table_name, table_info in validation_result["tables"].items():
            if not table_info.get("exists", False):
                recommendations.append(f"Create missing table: {table_name}")
            elif not table_info.get("schema_valid", False):
                recommendations.append(f"Fix schema issues in table: {table_name}")
            elif table_info.get("data_quality_score", 1.0) < 0.9:
                recommendations.append(f"Improve data quality in table: {table_name}")

        # Check for missing or problematic indexes
        for index_name, index_info in validation_result["indexes"].items():
            if index_info.get("status") == "ERROR":
                recommendations.append(f"Fix failed index: {index_name}")
            elif index_info.get("performance_score", 1.0) < 0.8:
                recommendations.append(
                    f"Rebuild index for better coverage: {index_name}"
                )

        return recommendations

    def delete_table(self, table_name: str, not_found_ok: bool = True) -> None:
        """Delete a table."""
        try:
            table_ref = self.connection.client.dataset(self.dataset_id).table(
                table_name
            )
            self.connection.client.delete_table(table_ref, not_found_ok=not_found_ok)
            logger.info(f"Table {table_name} deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete table {table_name}: {e}")
            raise

    def delete_all_tables(self, confirm: bool = False) -> None:
        """Delete all tables in the dataset."""
        if not confirm:
            raise ValueError("Must confirm deletion of all tables")

        try:
            tables = self.connection.list_tables()
            for table in tables:
                self.delete_table(table.table_id, not_found_ok=True)

            logger.info(f"All tables deleted from dataset {self.dataset_id}")
        except Exception as e:
            logger.error(f"Failed to delete all tables: {e}")
            raise
