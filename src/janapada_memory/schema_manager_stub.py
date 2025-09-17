"""In-memory stub implementation of SchemaManager for tests and local dev."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.cloud.exceptions import Conflict, NotFound

from .schema_manager import SchemaManager


class InMemorySchemaManager:
    """Lightweight drop-in replacement for SchemaManager backed by dictionaries."""

    SCHEMAS = SchemaManager.SCHEMAS
    CLUSTERING_CONFIG = SchemaManager.CLUSTERING_CONFIG

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> None:
        self.project_id = project_id or "stub-project"
        self.dataset_id = dataset_id or "stub_dataset"

        self._tables: Dict[str, Dict[str, Any]] = {}
        self._indexes: Dict[str, Dict[str, Any]] = {}
        self._dataset_info = {
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "location": "us-central1",
            "created": datetime.utcnow().isoformat() + "Z",
            "modified": datetime.utcnow().isoformat() + "Z",
            "description": "In-memory BigQuery dataset stub",
            "labels": {"environment": "stub"},
        }

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def create_dataset(self, location: str = "us-central1") -> dict[str, Any]:
        self._dataset_info["location"] = location
        self._dataset_info["modified"] = datetime.utcnow().isoformat() + "Z"
        return {
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "location": location,
            "created": True,
            "full_dataset_id": f"{self.project_id}:{self.dataset_id}",
        }

    def get_dataset_info(self) -> dict[str, Any]:
        return deepcopy(self._dataset_info)

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def create_tables(
        self,
        tables: Optional[List[str]] = None,
        partition_expiration_days: int = 365,
        force_recreate: bool = False,
    ) -> dict[str, Any]:
        if tables is None or tables == ["all"]:
            tables = list(self.SCHEMAS.keys())

        created_tables: List[dict[str, Any]] = []
        errors: List[str] = []

        for table_name in tables:
            if table_name not in self.SCHEMAS:
                errors.append(f"Unknown table: {table_name}")
                continue

            if table_name in self._tables and not force_recreate:
                errors.append(f"Table {table_name} already exists")
                continue

            schema_fields = self.SCHEMAS[table_name]
            table_info = {
                "name": table_name,
                "full_name": f"{self.project_id}.{self.dataset_id}.{table_name}",
                "schema": [
                    {"name": field.name, "type": field.field_type}
                    for field in schema_fields
                ],
                "partitioning": {
                    "type": "TIME",
                    "field": "partition_date",
                    "expiration_days": partition_expiration_days,
                },
                "clustering": list(self.CLUSTERING_CONFIG.get(table_name, [])),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }

            self._tables[table_name] = {
                "info": table_info,
                "schema": deepcopy(schema_fields),
                "rows": [],
            }
            created_tables.append(table_info)

        result = {
            "tables_created": created_tables,
            "creation_time_ms": int(datetime.utcnow().timestamp() * 1000),
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
        }

        if errors:
            result["errors"] = errors

        return result

    def delete_all_tables(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError("Confirmation required to delete all tables")
        self._tables.clear()
        self._indexes.clear()

    def list_tables(self, include_schema: bool = True) -> dict[str, Any]:
        tables = []
        for table_name, table_data in self._tables.items():
            info = deepcopy(table_data["info"])
            info.update(
                {
                    "table_type": "TABLE",
                    "num_rows": len(table_data["rows"]),
                    "num_bytes": len(table_data["rows"]) * 1024,
                    "modified": datetime.utcnow().isoformat() + "Z",
                }
            )
            if include_schema:
                info["schema"] = [
                    {
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                    }
                    for field in table_data["schema"]
                ]
            tables.append(info)

        return {
            "tables": tables,
            "dataset_info": self.get_dataset_info(),
            "count": len(tables),
        }

    def get_table_schema(self, table_name: str) -> List[dict[str, Any]]:
        if table_name not in self._tables:
            raise NotFound(f"Table {table_name} not found")
        return [
            {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description,
            }
            for field in self._tables[table_name]["schema"]
        ]

    def get_table_partition_info(self, table_name: str) -> Optional[dict[str, Any]]:
        if table_name not in self._tables:
            return None
        partitioning = self._tables[table_name]["info"].get("partitioning")
        return deepcopy(partitioning) if partitioning else None

    def get_table_clustering_info(self, table_name: str) -> Optional[dict[str, Any]]:
        if table_name not in self._tables:
            return None
        clustering = self._tables[table_name]["info"].get("clustering")
        return {"clustering_fields": list(clustering)} if clustering else None

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def create_indexes(self, indexes: List[dict[str, Any]]) -> dict[str, Any]:
        created_indexes: List[dict[str, Any]] = []
        errors: List[str] = []

        for index_spec in indexes:
            name = index_spec.get("name")
            table = index_spec.get("table")
            column = index_spec.get("column")
            index_type = index_spec.get("index_type", "IVF")
            distance_type = index_spec.get("distance_type", "COSINE")
            options = index_spec.get("options", {}) or {}

            if not name or not table or not column:
                errors.append("Missing required index parameters")
                continue

            if table not in self._tables:
                if table in self.SCHEMAS:
                    self.create_tables(tables=[table], force_recreate=False)
                else:
                    errors.append(f"Table {table} does not exist")
                    continue

            schema_fields = {field.name for field in self._tables[table]["schema"]}
            if column not in schema_fields:
                errors.append(f"Column {column} does not exist in {table}")
                continue

            if index_type not in ["IVF", "TREE_AH", "BTREE"]:
                errors.append(f"Invalid index type: {index_type}")
                continue

            if index_type in {"IVF", "TREE_AH"} and distance_type not in [
                "COSINE",
                "EUCLIDEAN",
                "DOT_PRODUCT",
            ]:
                errors.append(f"Invalid distance type: {distance_type}")
                continue

            num_lists = options.get("num_lists")
            if num_lists is not None and not (1 <= num_lists <= 5000):
                errors.append(f"num_lists must be between 1 and 5000, got {num_lists}")
                continue

            fraction = options.get("fraction_lists_to_search")
            if fraction is not None and not (0.0 < fraction <= 1.0):
                errors.append(
                    f"fraction_lists_to_search must be between 0 and 1, got {fraction}"
                )
                continue

            index_info = {
                "name": name,
                "table": table,
                "column": column,
                "index_type": index_type,
                "distance_type": distance_type,
                "status": "ACTIVE",
                "coverage_percentage": 100.0,
                "options": options,
            }
            self._indexes[name] = index_info
            created_indexes.append(index_info)

        result = {
            "indexes_created": created_indexes,
            "creation_time_ms": int(datetime.utcnow().timestamp() * 1000),
        }
        if errors:
            result["errors"] = errors
        return result

    def list_indexes(self) -> dict[str, Any]:
        return {"indexes": list(self._indexes.values()), "count": len(self._indexes)}

    def get_index_status(self, index_name: str) -> dict[str, Any]:
        if index_name not in self._indexes:
            raise NotFound(f"Index {index_name} not found")
        return deepcopy(self._indexes[index_name])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_schema(
        self,
        validate_data: bool = False,
        check_indexes: bool = True,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        result = {
            "overall_status": "VALID",
            "tables": {},
            "indexes": {},
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        for table_name, schema in self.SCHEMAS.items():
            exists = table_name in self._tables
            schema_valid = exists
            if exists:
                stored = {
                    field.name: field.field_type
                    for field in self._tables[table_name]["schema"]
                }
                expected = {field.name: field.field_type for field in schema}
                schema_valid = stored == expected

            table_info = {
                "exists": exists,
                "schema_valid": schema_valid,
                "row_count": len(self._tables.get(table_name, {}).get("rows", [])),
                "data_quality_score": 1.0,
            }

            if not exists or not schema_valid:
                result["overall_status"] = "INVALID"

            result["tables"][table_name] = table_info

        if check_indexes:
            for name, info in self._indexes.items():
                result["indexes"][name] = {
                    "exists": True,
                    "status": info.get("status", "ACTIVE"),
                    "performance_score": 1.0,
                }
        else:
            result["indexes"] = {}

        if validate_data:
            result["data_quality"] = {
                "embedding_quality": {
                    "dimension_consistency": True,
                    "nan_count": 0,
                    "infinity_count": 0,
                    "zero_vector_count": 0,
                },
                "metadata_quality": {
                    "missing_fields": 0,
                    "duplicate_chunk_ids": 0,
                    "invalid_artifact_types": 0,
                },
                "referential_integrity": {
                    "orphaned_embeddings": 0,
                    "missing_embeddings": 0,
                },
            }

        result["recommendations"] = [
            "Schedule nightly schema validation",
            "Review index coverage metrics weekly",
        ]

        return result
