"""
Vector search configuration data model for BigQuery Memory Adapter.

This module provides the VectorSearchConfig dataclass which encapsulates
all configuration parameters needed for BigQuery vector similarity search
operations, including validation and type safety.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class DistanceType(Enum):
    """Supported distance metrics for vector similarity search."""

    COSINE = "COSINE"
    DOT_PRODUCT = "DOT_PRODUCT"
    EUCLIDEAN = "EUCLIDEAN"  # Added for completeness, though BigQuery primarily uses COSINE/DOT_PRODUCT


class VectorSearchConfigError(ValueError):
    """Custom exception for vector search configuration errors."""

    pass


@dataclass(frozen=True)
class VectorSearchConfig:
    """
    Immutable configuration for BigQuery vector search operations.

    This dataclass validates all configuration parameters and ensures
    type safety for vector search operations against BigQuery.

    Attributes:
        project_id: Google Cloud project ID containing the dataset
        dataset_id: BigQuery dataset ID containing vector embeddings
        table_name: Name of the table containing vectors (default: source_embeddings)
        top_k: Number of nearest neighbors to return (1-1000)
        distance_type: Distance metric for similarity calculation
        distance_threshold: Optional threshold for filtering results by distance
        timeout_ms: Query timeout in milliseconds (default: 500ms per performance requirement)
    """

    project_id: str
    dataset_id: str
    table_name: str = "source_embeddings"
    top_k: int = 10
    distance_type: DistanceType = DistanceType.COSINE
    distance_threshold: Optional[float] = None
    timeout_ms: int = 500  # Performance requirement: <500ms

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_project_id()
        self._validate_dataset_id()
        self._validate_table_name()
        self._validate_top_k()
        self._validate_distance_threshold()
        self._validate_timeout()

    def _validate_project_id(self) -> None:
        """
        Validate BigQuery project ID format.

        Project IDs must:
        - Be 6-30 characters long
        - Start with a lowercase letter
        - Contain only lowercase letters, numbers, and hyphens
        - Not end with a hyphen
        """
        if not self.project_id:
            raise VectorSearchConfigError("project_id cannot be empty")

        # BigQuery project ID validation regex
        project_id_pattern = r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$"

        if not re.match(project_id_pattern, self.project_id):
            raise VectorSearchConfigError(
                f"Invalid project_id format: {self.project_id}. "
                "Must be 6-30 characters, start with a letter, "
                "and contain only lowercase letters, numbers, and hyphens."
            )

    def _validate_dataset_id(self) -> None:
        """
        Validate BigQuery dataset ID accessibility.

        Dataset IDs must:
        - Be 1-1024 characters long
        - Contain only letters, numbers, and underscores
        """
        if not self.dataset_id:
            raise VectorSearchConfigError("dataset_id cannot be empty")

        # BigQuery dataset ID validation regex
        dataset_id_pattern = r"^[a-zA-Z0-9_]{1,1024}$"

        if not re.match(dataset_id_pattern, self.dataset_id):
            raise VectorSearchConfigError(
                f"Invalid dataset_id format: {self.dataset_id}. "
                "Must contain only letters, numbers, and underscores."
            )

    def _validate_table_name(self) -> None:
        """
        Validate BigQuery table name format.

        Table names must:
        - Be 1-1024 characters long
        - Contain only letters, numbers, and underscores
        """
        if not self.table_name:
            raise VectorSearchConfigError("table_name cannot be empty")

        table_name_pattern = r"^[a-zA-Z0-9_]{1,1024}$"

        if not re.match(table_name_pattern, self.table_name):
            raise VectorSearchConfigError(
                f"Invalid table_name format: {self.table_name}. "
                "Must contain only letters, numbers, and underscores."
            )

    def _validate_top_k(self) -> None:
        """
        Validate top_k bounds (1-1000).

        BigQuery VECTOR_SEARCH supports top_k values from 1 to 1000.
        """
        if not isinstance(self.top_k, int):
            raise VectorSearchConfigError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )

        if self.top_k < 1 or self.top_k > 1000:
            raise VectorSearchConfigError(
                f"top_k must be between 1 and 1000, got {self.top_k}"
            )

    def _validate_distance_threshold(self) -> None:
        """Validate distance threshold if provided."""
        if self.distance_threshold is not None:
            if not isinstance(self.distance_threshold, (int, float)):
                raise VectorSearchConfigError(
                    f"distance_threshold must be a number, got {type(self.distance_threshold).__name__}"
                )

            # Distance thresholds depend on the metric type
            if self.distance_type == DistanceType.COSINE:
                # Cosine distance ranges from 0 to 2
                if self.distance_threshold < 0 or self.distance_threshold > 2:
                    raise VectorSearchConfigError(
                        f"COSINE distance_threshold must be between 0 and 2, got {self.distance_threshold}"
                    )
            elif self.distance_type == DistanceType.DOT_PRODUCT:
                # DOT_PRODUCT can be negative or positive
                pass  # No specific bounds for dot product

    def _validate_timeout(self) -> None:
        """Validate timeout milliseconds."""
        if not isinstance(self.timeout_ms, int):
            raise VectorSearchConfigError(
                f"timeout_ms must be an integer, got {type(self.timeout_ms).__name__}"
            )

        if self.timeout_ms < 1:
            raise VectorSearchConfigError(
                f"timeout_ms must be positive, got {self.timeout_ms}"
            )

        # Warn if timeout exceeds performance requirement
        if self.timeout_ms > 500:
            import logging

            logging.warning(
                "timeout_ms=%d exceeds 500ms performance requirement", self.timeout_ms
            )

    @property
    def full_table_reference(self) -> str:
        """Get the fully qualified BigQuery table reference."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_name}"

    @property
    def is_within_performance_target(self) -> bool:
        """Check if configuration meets performance requirements."""
        return self.timeout_ms <= 500

    def to_bigquery_params(self) -> dict:
        """
        Convert configuration to BigQuery query parameters.

        Returns:
            Dictionary of parameters suitable for BigQuery query execution
        """
        params = {
            "table_reference": self.full_table_reference,
            "top_k": self.top_k,
            "distance_type": self.distance_type.value,
            "timeout_ms": self.timeout_ms,
        }

        if self.distance_threshold is not None:
            params["distance_threshold"] = self.distance_threshold

        return params

    @classmethod
    def from_env(cls, top_k: int = 10) -> "VectorSearchConfig":
        """
        Create configuration from environment variables.

        Args:
            top_k: Number of results to return (default: 10)

        Returns:
            VectorSearchConfig instance

        Raises:
            VectorSearchConfigError: If environment variables are invalid
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        dataset_id = os.getenv("BIGQUERY_DATASET_ID", "semantic_gap_detector")
        table_name = os.getenv("BIGQUERY_TABLE_NAME", "source_embeddings")
        distance_type_str = os.getenv("BIGQUERY_DISTANCE_TYPE", "COSINE")
        timeout_ms = int(os.getenv("BIGQUERY_TIMEOUT_MS", "500"))

        try:
            distance_type = DistanceType[distance_type_str.upper()]
        except KeyError:
            raise VectorSearchConfigError(
                f"Invalid BIGQUERY_DISTANCE_TYPE: {distance_type_str}. "
                f"Must be one of {[e.value for e in DistanceType]}"
            )

        return cls(
            project_id=project_id,
            dataset_id=dataset_id,
            table_name=table_name,
            top_k=top_k,
            distance_type=distance_type,
            timeout_ms=timeout_ms,
        )