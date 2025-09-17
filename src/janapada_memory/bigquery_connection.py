"""
BigQuery Connection and Authentication Manager

Handles secure connection to BigQuery with proper authentication,
credential management, and connection pooling.
"""

import logging
import os
import time
from typing import Any, Optional

from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

logger = logging.getLogger(__name__)


class BigQueryConnection:
    """Manages BigQuery connections with authentication and error handling."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        location: str = "us-central1",
        credentials_path: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize BigQuery connection.

        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            location: BigQuery dataset location
            credentials_path: Path to service account credentials
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        # NOTE: default to hackathon project/dataset only when env vars are unset to
        # keep local tooling runnable; production deploys must supply explicit values.
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.dataset_id = dataset_id or os.getenv(
            "BIGQUERY_DATASET_ID", "semantic_gap_detector"
        )
        self.location = location
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = None
        self._dataset_ref = None
        self._connection_validated = False

        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID must be provided via parameter or GOOGLE_CLOUD_PROJECT env var"
            )

        logger.info(
            f"Initializing BigQuery connection to project: {self.project_id}, dataset: {self.dataset_id}"
        )

    @property
    def client(self) -> bigquery.Client:
        """Get BigQuery client, creating if necessary."""
        if self._client is None:
            self._create_client()
        return self._client

    @property
    def dataset_ref(self) -> bigquery.DatasetReference:
        """Get dataset reference."""
        if self._dataset_ref is None:
            self._dataset_ref = bigquery.DatasetReference(
                self.project_id, self.dataset_id
            )
        return self._dataset_ref

    def _create_client(self) -> None:
        """Create and authenticate BigQuery client."""
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                logger.info(
                    f"Using service account credentials from: {self.credentials_path}"
                )
                self._client = bigquery.Client.from_service_account_json(
                    self.credentials_path, project=self.project_id
                )
            else:
                logger.info("Using Application Default Credentials")
                credentials, project = default()
                self._client = bigquery.Client(
                    credentials=credentials, project=self.project_id
                )

            # Test connection
            self._validate_connection()
            logger.info("BigQuery client created successfully")

        except DefaultCredentialsError as e:
            logger.error("Failed to authenticate with Google Cloud")
            raise ValueError(
                "Failed to authenticate with Google Cloud. "
                "Please set GOOGLE_APPLICATION_CREDENTIALS or configure Application Default Credentials"
            ) from e
        except Exception as e:
            logger.error(f"Failed to create BigQuery client: {e}")
            raise

    def _validate_connection(self) -> None:
        """Validate BigQuery connection and permissions."""
        try:
            # Test basic connectivity
            query = "SELECT 1 as test_connection"
            query_job = self._client.query(query)
            results = list(query_job.result())

            if not results or results[0].test_connection != 1:
                raise RuntimeError("Connection test failed")

            # Test dataset access
            try:
                dataset = self._client.get_dataset(self.dataset_ref)
                logger.info(f"Successfully connected to dataset: {dataset.dataset_id}")
            except NotFound:
                logger.warning(
                    f"Dataset {self.dataset_id} does not exist, will be created during setup"
                )

            self._connection_validated = True
            logger.info("BigQuery connection validation successful")

        except Exception as e:
            logger.error(f"BigQuery connection validation failed: {e}")
            raise

    def execute_query(
        self,
        query: str,
        job_config: Optional[bigquery.QueryJobConfig] = None,
        timeout: float = 30.0,
    ) -> bigquery.table.RowIterator:
        """
        Execute a BigQuery query with retry logic.

        Args:
            query: SQL query to execute
            job_config: Query job configuration
            timeout: Query timeout in seconds

        Returns:
            Query results iterator
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Executing query (attempt {attempt + 1}/{self.max_retries}): {query[:100]}..."
                )

                query_job = self.client.query(query, job_config=job_config)
                results = query_job.result(timeout=timeout)

                logger.debug(
                    f"Query completed successfully. Rows processed: {query_job.total_bytes_processed}"
                )
                return results

            except Exception as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Query failed after {self.max_retries} attempts: {query[:100]}..."
                    )
                    raise

                time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

    def insert_rows(
        self,
        table_ref: bigquery.TableReference,
        rows: list,
        ignore_unknown_values: bool = False,
        skip_invalid_rows: bool = False,
    ) -> list:
        """
        Insert rows into BigQuery table with error handling.

        Args:
            table_ref: Table reference
            rows: List of row dictionaries to insert
            ignore_unknown_values: Whether to ignore unknown fields
            skip_invalid_rows: Whether to skip invalid rows

        Returns:
            List of insertion errors (empty if successful)
        """
        for attempt in range(self.max_retries):
            try:
                table = self.client.get_table(table_ref)
                errors = self.client.insert_rows_json(
                    table,
                    rows,
                    ignore_unknown_values=ignore_unknown_values,
                    skip_invalid_rows=skip_invalid_rows,
                )

                if errors:
                    logger.warning(f"Insert errors: {errors}")
                else:
                    logger.debug(
                        f"Successfully inserted {len(rows)} rows into {table_ref.table_id}"
                    )

                return errors

            except Exception as e:
                logger.warning(f"Insert attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    logger.error(f"Insert failed after {self.max_retries} attempts")
                    raise

                time.sleep(self.retry_delay * (2**attempt))

    def create_dataset(
        self,
        dataset_id: Optional[str] = None,
        location: Optional[str] = None,
        exists_ok: bool = True,
    ) -> bigquery.Dataset:
        """
        Create BigQuery dataset.

        Args:
            dataset_id: Dataset ID (defaults to self.dataset_id)
            location: Dataset location (defaults to self.location)
            exists_ok: Don't raise error if dataset exists

        Returns:
            Created or existing dataset
        """
        dataset_id = dataset_id or self.dataset_id
        location = location or self.location

        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location

        try:
            created_dataset = self.client.create_dataset(dataset, exists_ok=exists_ok)
            logger.info(f"Dataset {dataset_id} created successfully in {location}")
            return created_dataset

        except Exception as e:
            logger.error(f"Failed to create dataset {dataset_id}: {e}")
            raise

    def get_dataset(self, dataset_id: Optional[str] = None) -> bigquery.Dataset:
        """
        Get BigQuery dataset.

        Args:
            dataset_id: Dataset ID (defaults to self.dataset_id)

        Returns:
            Dataset object
        """
        dataset_id = dataset_id or self.dataset_id
        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)

        try:
            dataset = self.client.get_dataset(dataset_ref)
            return dataset

        except NotFound:
            logger.error(f"Dataset {dataset_id} not found")
            raise
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            raise

    def delete_dataset(
        self,
        dataset_id: Optional[str] = None,
        delete_contents: bool = False,
        not_found_ok: bool = True,
    ) -> None:
        """
        Delete BigQuery dataset.

        Args:
            dataset_id: Dataset ID (defaults to self.dataset_id)
            delete_contents: Whether to delete all tables in dataset
            not_found_ok: Don't raise error if dataset doesn't exist
        """
        dataset_id = dataset_id or self.dataset_id
        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)

        try:
            self.client.delete_dataset(
                dataset_ref, delete_contents=delete_contents, not_found_ok=not_found_ok
            )
            logger.info(f"Dataset {dataset_id} deleted successfully")

        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            raise

    def list_tables(self, dataset_id: Optional[str] = None) -> list:
        """
        List tables in dataset.

        Args:
            dataset_id: Dataset ID (defaults to self.dataset_id)

        Returns:
            List of table references
        """
        dataset_id = dataset_id or self.dataset_id
        dataset_ref = bigquery.DatasetReference(self.project_id, dataset_id)

        try:
            tables = list(self.client.list_tables(dataset_ref))
            logger.debug(f"Found {len(tables)} tables in dataset {dataset_id}")
            return tables

        except Exception as e:
            logger.error(f"Failed to list tables in dataset {dataset_id}: {e}")
            raise

    def get_table(
        self, table_id: str, dataset_id: Optional[str] = None
    ) -> bigquery.Table:
        """
        Get BigQuery table.

        Args:
            table_id: Table ID
            dataset_id: Dataset ID (defaults to self.dataset_id)

        Returns:
            Table object
        """
        dataset_id = dataset_id or self.dataset_id
        table_ref = self.client.dataset(dataset_id).table(table_id)

        try:
            table = self.client.get_table(table_ref)
            return table

        except NotFound:
            logger.error(f"Table {table_id} not found in dataset {dataset_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to get table {table_id}: {e}")
            raise

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on BigQuery connection.

        Returns:
            Health status dictionary
        """
        health_status = {
            "status": "unknown",
            "bigquery_connection": False,
            "dataset_accessible": False,
            "project_id": self.project_id,
            "dataset_id": self.dataset_id,
            "timestamp": time.time(),
        }

        try:
            # Test basic connection
            query = "SELECT 1 as health_check"
            results = list(self.execute_query(query, timeout=10.0))

            if results and results[0].health_check == 1:
                health_status["bigquery_connection"] = True

            # Test dataset access
            try:
                self.get_dataset()
                health_status["dataset_accessible"] = True
            except NotFound:
                logger.warning("Dataset not found during health check")
                health_status["dataset_accessible"] = False

            # Determine overall status
            if (
                health_status["bigquery_connection"]
                and health_status["dataset_accessible"]
            ):
                health_status["status"] = "healthy"
            elif health_status["bigquery_connection"]:
                health_status["status"] = "partial"  # Connected but dataset issues
            else:
                health_status["status"] = "unhealthy"

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def close(self) -> None:
        """Close BigQuery connection and cleanup resources."""
        if self._client:
            self._client.close()
            self._client = None
            self._connection_validated = False
            logger.info("BigQuery connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
