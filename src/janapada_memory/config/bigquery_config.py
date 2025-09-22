"""
BigQuery configuration module for Janapada Memory.

This module handles BigQuery-specific configuration including project settings,
dataset configuration, authentication, and graceful error handling for missing
credentials or invalid settings.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BigQueryConfig:
    """Immutable BigQuery configuration with validation."""

    project_id: str
    dataset_id: str
    table_prefix: str = ""
    location: str = "us-central1"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_project_id()
        self._validate_dataset_id()

    def _validate_project_id(self) -> None:
        """Validate BigQuery project ID format."""
        if not self.project_id:
            raise ValueError("project_id cannot be empty")

        if not self.project_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Invalid project_id format: {self.project_id}. "
                "Must contain only alphanumeric characters, hyphens, and underscores."
            )

    def _validate_dataset_id(self) -> None:
        """Validate BigQuery dataset ID format."""
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")

        # Allow hyphens for test datasets (e.g., test-dataset)
        if not self.dataset_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid dataset_id format: {self.dataset_id}. "
                "Must contain only alphanumeric characters, underscores, and hyphens."
            )

    @property
    def table_name(self) -> str:
        """Get the full table name with prefix."""
        if self.table_prefix:
            return f"{self.table_prefix}_source_embeddings"
        return "source_embeddings"

    @property
    def full_table_ref(self) -> str:
        """Get the full table reference in project.dataset.table format."""
        return f"{self.project_id}.{self.dataset_id}.{self.table_name}"


class BigQueryConfigError(Exception):
    """Custom exception for BigQuery configuration errors."""

    pass


class BigQueryConfigManager:
    """Manages BigQuery configuration loading and validation."""

    @staticmethod
    def load_from_environment() -> BigQueryConfig:
        """
        Load BigQuery configuration from environment variables.

        Environment Variables:
            GOOGLE_CLOUD_PROJECT: BigQuery project ID
            BIGQUERY_DATASET_ID: BigQuery dataset ID
            BIGQUERY_TABLE_PREFIX: Optional table prefix
            BIGQUERY_LOCATION: BigQuery location (default: us-central1)

        Returns:
            BigQueryConfig: Validated configuration object

        Raises:
            BigQueryConfigError: If required environment variables are missing
                                or configuration is invalid
        """
        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            # Try new standardized variable first, then fall back to legacy
            dataset_id = os.getenv("BIGQUERY_EMBEDDINGS_DATASET_ID") or os.getenv(
                "BIGQUERY_DATASET_ID"
            )  # Legacy fallback
            table_prefix = os.getenv("BIGQUERY_TABLE_PREFIX", "")
            location = os.getenv("BIGQUERY_LOCATION", "us-central1")

            # Apply defaults for missing required values
            if not project_id:
                project_id = "konveyn2ai"
                logger.warning(
                    "GOOGLE_CLOUD_PROJECT not set, using default: %s", project_id
                )

            if not dataset_id:
                dataset_id = "semantic_gap_detector"
                logger.warning(
                    "BIGQUERY_EMBEDDINGS_DATASET_ID not set, using default: %s",
                    dataset_id,
                )

            config = BigQueryConfig(
                project_id=project_id,
                dataset_id=dataset_id,
                table_prefix=table_prefix,
                location=location,
            )

            logger.info(
                "BigQuery configuration loaded successfully: %s.%s",
                config.project_id,
                config.dataset_id,
            )

            return config

        except ValueError as e:
            raise BigQueryConfigError(f"Invalid BigQuery configuration: {e}") from e
        except Exception as e:
            raise BigQueryConfigError(
                f"Failed to load BigQuery configuration: {e}"
            ) from e

    @staticmethod
    def handle_missing_credentials() -> None:
        """
        Handle missing Google Cloud credentials gracefully.

        This method provides helpful error messages and guidance when
        authentication fails due to missing or invalid credentials.
        """
        logger.error(
            "Google Cloud credentials not found. Please ensure one of the following:\n"
            "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
            "2. Run 'gcloud auth application-default login'\n"
            "3. Use a service account key file\n"
            "4. Run on Google Cloud with default service account"
        )

    @staticmethod
    def validate_bigquery_access(config: BigQueryConfig) -> bool:
        """
        Validate that BigQuery dataset is accessible with current credentials.

        Args:
            config: BigQuery configuration to validate

        Returns:
            bool: True if dataset is accessible, False otherwise
        """
        try:
            from google.cloud import bigquery
            from google.cloud.exceptions import NotFound, Forbidden

            client = bigquery.Client(
                project=config.project_id, location=config.location
            )
            dataset_ref = f"{config.project_id}.{config.dataset_id}"

            # Try to access the dataset
            dataset = client.get_dataset(dataset_ref)
            logger.info("Successfully validated access to dataset: %s", dataset_ref)
            return True

        except NotFound:
            logger.error(
                "Dataset not found: %s. Please run 'make setup' to create tables.",
                dataset_ref,
            )
            return False
        except Forbidden:
            logger.error(
                "Access denied to dataset: %s. Please check BigQuery permissions.",
                dataset_ref,
            )
            return False
        except Exception as e:
            logger.error("Failed to validate BigQuery access: %s", e)
            BigQueryConfigManager.handle_missing_credentials()
            return False


# Global configuration instance
_bigquery_config: Optional[BigQueryConfig] = None


def get_bigquery_config() -> BigQueryConfig:
    """
    Get the global BigQuery configuration instance.

    This function implements lazy loading of the configuration and caches
    the result for subsequent calls.

    Returns:
        BigQueryConfig: The global configuration instance

    Raises:
        BigQueryConfigError: If configuration cannot be loaded or is invalid
    """
    global _bigquery_config

    if _bigquery_config is None:
        _bigquery_config = BigQueryConfigManager.load_from_environment()

    return _bigquery_config


def reset_bigquery_config() -> None:
    """Reset the global configuration instance. Useful for testing."""
    global _bigquery_config
    _bigquery_config = None
