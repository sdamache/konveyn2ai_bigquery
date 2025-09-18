"""
BigQuery connection manager for Memory Adapter.

This module manages the BigQuery client lifecycle, handles authentication
using Application Default Credentials (ADC), performs connection health checks,
and provides structured logging for debugging and monitoring.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass
import uuid

from google.cloud import bigquery
from google.cloud.exceptions import (
    GoogleCloudError,
    NotFound,
    Forbidden,
    BadRequest,
    Conflict,
)
from google.api_core import exceptions as api_exceptions

from ..config import BigQueryConfig, get_bigquery_config

# Configure structured logging
logger = logging.getLogger(__name__)


@dataclass
class ConnectionHealth:
    """Represents the health status of a BigQuery connection."""

    is_healthy: bool
    last_check: float  # Unix timestamp
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    connection_id: Optional[str] = None


class BigQueryConnectionError(Exception):
    """Custom exception for BigQuery connection issues."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class BigQueryConnectionManager:
    """
    Manages BigQuery client lifecycle and connection health.

    This class handles:
    - BigQuery client creation and lifecycle management
    - Application Default Credentials (ADC) authentication
    - Connection health monitoring and diagnostics
    - Structured logging with correlation IDs
    - Graceful error handling and recovery
    """

    def __init__(self, config: Optional[BigQueryConfig] = None):
        """
        Initialize the BigQuery connection manager.

        Args:
            config: BigQuery configuration. If None, loads from environment.
        """
        self.config = config or get_bigquery_config()
        self.connection_id = str(uuid.uuid4())
        self._client: Optional[bigquery.Client] = None
        self._health_status: Optional[ConnectionHealth] = None

        # Configure structured logging
        self.logger = logging.getLogger(f"{__name__}.{self.connection_id[:8]}")

        self.logger.info(
            "BigQuery connection manager initialized",
            extra={
                "connection_id": self.connection_id,
                "project_id": self.config.project_id,
                "dataset_id": self.config.dataset_id,
            },
        )

    @property
    def client(self) -> bigquery.Client:
        """
        Get the BigQuery client, creating it if necessary.

        Returns:
            Configured BigQuery client

        Raises:
            BigQueryConnectionError: If client creation fails
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> bigquery.Client:
        """
        Create a new BigQuery client with proper configuration.

        Returns:
            Configured BigQuery client

        Raises:
            BigQueryConnectionError: If client creation fails
        """
        try:
            start_time = time.time()

            # Create client with configuration
            client = bigquery.Client(
                project=self.config.project_id, location=self.config.location
            )

            creation_time_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "BigQuery client created successfully",
                extra={
                    "connection_id": self.connection_id,
                    "project_id": self.config.project_id,
                    "location": self.config.location,
                    "creation_time_ms": creation_time_ms,
                },
            )

            return client

        except api_exceptions.GoogleAPICallError as e:
            error_msg = f"Failed to create BigQuery client: {e}"
            self.logger.error(
                "BigQuery client creation failed",
                extra={
                    "connection_id": self.connection_id,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                    "remediation": "Check authentication and project permissions",
                },
            )
            raise BigQueryConnectionError(error_msg, e)

        except Exception as e:
            error_msg = f"Unexpected error creating BigQuery client: {e}"
            self.logger.error(
                "Unexpected BigQuery client error",
                extra={"connection_id": self.connection_id, "error": str(e)},
            )
            raise BigQueryConnectionError(error_msg, e)

    def check_health(self, force_refresh: bool = False) -> ConnectionHealth:
        """
        Check the health of the BigQuery connection.

        Args:
            force_refresh: Force a new health check even if cached result exists

        Returns:
            ConnectionHealth status
        """
        # Return cached result if available and not forcing refresh
        if (
            not force_refresh
            and self._health_status is not None
            and (time.time() - self._health_status.last_check) < 30
        ):  # 30 second cache
            return self._health_status

        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Test basic connectivity with a simple query
            query = "SELECT 1 as health_check"
            query_job = self.client.query(query)
            list(query_job)  # Execute the query

            response_time_ms = (time.time() - start_time) * 1000

            self._health_status = ConnectionHealth(
                is_healthy=True,
                last_check=time.time(),
                response_time_ms=response_time_ms,
                connection_id=self.connection_id,
            )

            self.logger.info(
                "BigQuery health check passed",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "response_time_ms": response_time_ms,
                },
            )

        except (NotFound, Forbidden, BadRequest) as e:
            error_msg = f"BigQuery health check failed: {e}"
            self._health_status = ConnectionHealth(
                is_healthy=False,
                last_check=time.time(),
                error_message=error_msg,
                connection_id=self.connection_id,
            )

            self.logger.error(
                "BigQuery health check failed",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_code": getattr(e, "code", None),
                },
            )

        except Exception as e:
            error_msg = f"Unexpected error in health check: {e}"
            self._health_status = ConnectionHealth(
                is_healthy=False,
                last_check=time.time(),
                error_message=error_msg,
                connection_id=self.connection_id,
            )

            self.logger.error(
                "Unexpected health check error",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

        return self._health_status

    def check_dataset_access(self) -> bool:
        """
        Check if the configured dataset is accessible.

        Returns:
            True if dataset is accessible, False otherwise
        """
        correlation_id = str(uuid.uuid4())

        try:
            dataset_ref = f"{self.config.project_id}.{self.config.dataset_id}"
            dataset = self.client.get_dataset(dataset_ref)

            self.logger.info(
                "Dataset access verified",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "dataset_ref": dataset_ref,
                    "dataset_created": (
                        dataset.created.isoformat() if dataset.created else None
                    ),
                },
            )
            return True

        except NotFound:
            self.logger.error(
                "Dataset not found",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "dataset_ref": f"{self.config.project_id}.{self.config.dataset_id}",
                    "remediation": "Run 'make setup' to create required tables",
                },
            )
            return False

        except Forbidden:
            self.logger.error(
                "Dataset access forbidden",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "dataset_ref": f"{self.config.project_id}.{self.config.dataset_id}",
                    "remediation": "Check BigQuery IAM permissions",
                },
            )
            return False

        except Exception as e:
            self.logger.error(
                "Unexpected error checking dataset access",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    def check_table_access(self, table_name: str) -> bool:
        """
        Check if a specific table is accessible.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table is accessible, False otherwise
        """
        correlation_id = str(uuid.uuid4())

        try:
            table_ref = (
                f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"
            )
            table = self.client.get_table(table_ref)

            self.logger.info(
                "Table access verified",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "table_ref": table_ref,
                    "num_rows": table.num_rows,
                    "schema_length": len(table.schema),
                },
            )
            return True

        except NotFound:
            self.logger.warning(
                "Table not found",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "table_ref": f"{self.config.project_id}.{self.config.dataset_id}.{table_name}",
                    "remediation": f"Create table {table_name} or run 'make setup'",
                },
            )
            return False

        except Forbidden:
            self.logger.error(
                "Table access forbidden",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "table_ref": f"{self.config.project_id}.{self.config.dataset_id}.{table_name}",
                    "remediation": "Check BigQuery table-level permissions",
                },
            )
            return False

        except Exception as e:
            self.logger.error(
                "Unexpected error checking table access",
                extra={
                    "connection_id": self.connection_id,
                    "correlation_id": correlation_id,
                    "table_name": table_name,
                    "error": str(e),
                },
            )
            return False

    @contextmanager
    def query_context(self, operation: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for BigQuery operations with structured logging.

        Args:
            operation: Description of the operation being performed

        Yields:
            Dictionary containing correlation_id and other context data
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        context = {
            "correlation_id": correlation_id,
            "connection_id": self.connection_id,
            "operation": operation,
            "start_time": start_time,
        }

        self.logger.info(
            f"BigQuery operation started: {operation}",
            extra=context,
        )

        try:
            yield context

            elapsed_ms = (time.time() - start_time) * 1000
            context["elapsed_ms"] = elapsed_ms

            self.logger.info(
                f"BigQuery operation completed: {operation}",
                extra=context,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            context.update(
                {
                    "elapsed_ms": elapsed_ms,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

            self.logger.error(
                f"BigQuery operation failed: {operation}",
                extra=context,
            )
            raise

    def reset_connection(self) -> None:
        """Reset the BigQuery client connection."""
        correlation_id = str(uuid.uuid4())

        self.logger.info(
            "Resetting BigQuery connection",
            extra={
                "connection_id": self.connection_id,
                "correlation_id": correlation_id,
            },
        )

        self._client = None
        self._health_status = None

        # Generate new connection ID for the reset connection
        self.connection_id = str(uuid.uuid4())

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the current connection.

        Returns:
            Dictionary with connection information
        """
        info = {
            "connection_id": self.connection_id,
            "project_id": self.config.project_id,
            "dataset_id": self.config.dataset_id,
            "location": self.config.location,
            "client_created": self._client is not None,
        }

        if self._health_status:
            info.update(
                {
                    "last_health_check": self._health_status.last_check,
                    "is_healthy": self._health_status.is_healthy,
                    "last_response_time_ms": self._health_status.response_time_ms,
                }
            )

        return info

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup if needed."""
        if exc_type is not None:
            self.logger.error(
                "BigQuery connection manager exiting due to error",
                extra={
                    "connection_id": self.connection_id,
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error": str(exc_val) if exc_val else None,
                },
            )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"BigQueryConnectionManager("
            f"connection_id='{self.connection_id[:8]}...', "
            f"project='{self.config.project_id}', "
            f"dataset='{self.config.dataset_id}', "
            f"healthy={self._health_status.is_healthy if self._health_status else 'unknown'}"
            f")"
        )
