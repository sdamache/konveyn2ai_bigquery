"""
BigQuery Storage Write API client with batch processing
T022: High-performance BigQuery writer with retry logic and batch optimization
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from google.cloud import bigquery
from google.cloud.bigquery import WriteDisposition, CreateDisposition
from google.cloud.exceptions import GoogleCloudError, NotFound, Conflict
from google.api_core import retry, exceptions


# Import our contract types (update path based on project structure)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from specs.contracts.parser_interfaces import ChunkMetadata, ParseError, SourceType, ErrorClass
except ImportError:
    # Fallback: define minimal types for now
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any, Optional

    class SourceType(Enum):
        KUBERNETES = "kubernetes"
        FASTAPI = "fastapi"
        COBOL = "cobol"
        IRS = "irs"
        MUMPS = "mumps"

    class ErrorClass(Enum):
        PARSING = "parsing"
        VALIDATION = "validation"
        INGESTION = "ingestion"

    @dataclass
    class ChunkMetadata:
        source_type: SourceType
        artifact_id: str
        parent_id: Optional[str] = None
        parent_type: Optional[str] = None
        content_text: str = ""
        content_tokens: Optional[int] = None
        content_hash: str = ""
        source_uri: str = ""
        repo_ref: Optional[str] = None
        collected_at: Optional[datetime] = None
        source_metadata: Dict[str, Any] = None

    @dataclass
    class ParseError:
        source_type: SourceType
        source_uri: str
        error_class: ErrorClass
        error_msg: str
        sample_text: Optional[str] = None
        stack_trace: Optional[str] = None
        collected_at: Optional[datetime] = None


@dataclass
class WriteResult:
    """Result of a BigQuery write operation"""
    rows_written: int
    errors_count: int
    processing_time_ms: int
    run_id: str
    table_name: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchConfig:
    """Configuration for batch writing"""
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 300
    use_storage_write_api: bool = True


class BigQueryWriter:
    """High-performance BigQuery writer with batch processing and retry logic"""

    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None,
                 location: str = "US", batch_config: Optional[BatchConfig] = None):
        self.project_id = project_id or os.getenv("BQ_PROJECT", "konveyn2ai")
        self.dataset_id = dataset_id or os.getenv("BQ_DATASET", "source_ingestion")
        self.location = location
        self.batch_config = batch_config or BatchConfig()

        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Table schemas from contract
        self._setup_table_schemas()

    def _setup_table_schemas(self):
        """Define table schemas based on contract"""
        self.schemas = {
            "source_metadata": [
                bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("artifact_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("parent_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("parent_type", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("content_text", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("content_tokens", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("content_hash", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("source_uri", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("repo_ref", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("collected_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("source_metadata", "STRING", mode="NULLABLE"),
            ],
            "source_metadata_errors": [
                bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("source_uri", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("error_class", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("error_msg", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("sample_text", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("stack_trace", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("collected_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            "ingestion_log": [
                bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("source_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("started_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("completed_at", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("files_processed", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("chunks_created", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("errors_encountered", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("processing_duration_ms", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("config_used", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("error_summary", "STRING", mode="NULLABLE"),
            ]
        }

    def create_dataset_if_not_exists(self, location: str = "US") -> bool:
        """Create BigQuery dataset if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id, project=self.project_id)

        try:
            # Try to get the dataset
            dataset = self.client.get_dataset(dataset_ref)
            self.logger.info(f"Dataset {self.project_id}.{self.dataset_id} already exists")
            return False  # Already exists

        except NotFound:
            # Create the dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            dataset.description = f"M1 ingestion dataset for storing source metadata, embeddings, and analysis results"

            try:
                dataset = self.client.create_dataset(dataset, exists_ok=True)
                self.logger.info(f"Created dataset {self.project_id}.{self.dataset_id}")
                return True  # Created

            except Conflict:
                # Dataset was created by another process
                self.logger.info(f"Dataset {self.project_id}.{self.dataset_id} created by another process")
                return False

        except Exception as e:
            self.logger.error(f"Error checking/creating dataset {self.project_id}.{self.dataset_id}: {e}")
            raise

    def create_tables_if_not_exist(self) -> Dict[str, bool]:
        """Create BigQuery tables if they don't exist"""
        results = {}

        for table_name, schema in self.schemas.items():
            table_ref = self.client.dataset(self.dataset_id).table(table_name)

            try:
                # Try to get the table
                self.client.get_table(table_ref)
                results[table_name] = False  # Already exists
                self.logger.info(f"Table {table_name} already exists")

            except NotFound:
                # Create the table
                table = bigquery.Table(table_ref, schema=schema)

                # Set additional properties
                table.clustering_fields = self._get_clustering_fields(table_name)
                table.time_partitioning = self._get_partitioning(table_name)

                try:
                    table = self.client.create_table(table)
                    results[table_name] = True  # Created
                    self.logger.info(f"Created table {table_name}")

                except Conflict:
                    # Table was created by another process
                    results[table_name] = False
                    self.logger.info(f"Table {table_name} created by another process")

            except Exception as e:
                self.logger.error(f"Error checking/creating table {table_name}: {e}")
                raise

        return results

    def _get_clustering_fields(self, table_name: str) -> Optional[List[str]]:
        """Get clustering fields for table optimization"""
        clustering_config = {
            "source_metadata": ["source_type", "content_hash"],
            "source_metadata_errors": ["source_type", "error_class"],
            "ingestion_log": ["source_type", "status"]
        }
        return clustering_config.get(table_name)

    def _get_partitioning(self, table_name: str) -> Optional[bigquery.TimePartitioning]:
        """Get time partitioning for table optimization"""
        partitioning_config = {
            "source_metadata": "collected_at",
            "source_metadata_errors": "collected_at",
            "ingestion_log": "started_at"
        }

        field = partitioning_config.get(table_name)
        if field:
            return bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=field
            )
        return None

    def write_chunks(self, chunks: List[ChunkMetadata], run_id: str,
                    enable_upsert: bool = True) -> WriteResult:
        """Write chunks to source_metadata table with idempotent upsert logic (T032)"""
        if not chunks:
            return WriteResult(
                rows_written=0,
                errors_count=0,
                processing_time_ms=0,
                run_id=run_id,
                table_name="source_metadata",
                success=True
            )

        start_time = time.time()

        try:
            # Convert chunks to BigQuery rows
            rows = [self._chunk_to_row(chunk) for chunk in chunks]

            if enable_upsert:
                # Implement idempotent upsert: check for existing content hashes
                existing_hashes = self._get_existing_content_hashes([row["content_hash"] for row in rows])
                new_rows = [row for row in rows if row["content_hash"] not in existing_hashes]

                self.logger.info(f"Upsert logic: {len(chunks)} total chunks, {len(new_rows)} new chunks, {len(existing_hashes)} duplicates skipped")

                if not new_rows:
                    processing_time = int((time.time() - start_time) * 1000)
                    return WriteResult(
                        rows_written=0,
                        errors_count=0,
                        processing_time_ms=processing_time,
                        run_id=run_id,
                        table_name="source_metadata",
                        success=True,
                        error_message="All chunks already exist (duplicates skipped)"
                    )

                rows = new_rows

            # Write in batches
            total_written = 0
            total_errors = 0

            for batch in self._batch_iterator(rows, self.batch_config.batch_size):
                result = self._write_batch("source_metadata", batch, run_id)
                total_written += result.rows_written
                total_errors += result.errors_count

                if not result.success:
                    self.logger.warning(f"Batch write had errors: {result.error_message}")

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                rows_written=total_written,
                errors_count=total_errors,
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata",
                success=total_errors == 0
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error writing chunks: {e}")

            return WriteResult(
                rows_written=0,
                errors_count=len(chunks),
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata",
                success=False,
                error_message=str(e)
            )

    def write_errors(self, errors: List[ParseError], run_id: str) -> WriteResult:
        """Write errors to source_metadata_errors table"""
        if not errors:
            return WriteResult(
                rows_written=0,
                errors_count=0,
                processing_time_ms=0,
                run_id=run_id,
                table_name="source_metadata_errors",
                success=True
            )

        start_time = time.time()

        try:
            # Convert errors to BigQuery rows
            rows = [self._error_to_row(error) for error in errors]

            # Write in batches
            total_written = 0
            total_errors = 0

            for batch in self._batch_iterator(rows, self.batch_config.batch_size):
                result = self._write_batch("source_metadata_errors", batch, run_id)
                total_written += result.rows_written
                total_errors += result.errors_count

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                rows_written=total_written,
                errors_count=total_errors,
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata_errors",
                success=total_errors == 0
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error writing errors: {e}")

            return WriteResult(
                rows_written=0,
                errors_count=len(errors),
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata_errors",
                success=False,
                error_message=str(e)
            )

    def log_ingestion_run(self, run_info: Dict[str, Any]) -> WriteResult:
        """Log ingestion run to ingestion_log table"""
        start_time = time.time()

        try:
            # Convert run info to BigQuery row
            row = self._run_info_to_row(run_info)

            # Write single row
            result = self._write_batch("ingestion_log", [row], run_info.get("run_id", "unknown"))

            processing_time = int((time.time() - start_time) * 1000)

            return WriteResult(
                rows_written=result.rows_written,
                errors_count=result.errors_count,
                processing_time_ms=processing_time,
                run_id=run_info.get("run_id", "unknown"),
                table_name="ingestion_log",
                success=result.success,
                error_message=result.error_message
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error logging ingestion run: {e}")

            return WriteResult(
                rows_written=0,
                errors_count=1,
                processing_time_ms=processing_time,
                run_id=run_info.get("run_id", "unknown"),
                table_name="ingestion_log",
                success=False,
                error_message=str(e)
            )

    def _write_batch(self, table_name: str, rows: List[Dict[str, Any]], run_id: str) -> WriteResult:
        """Write a batch of rows to BigQuery with retry logic"""
        table_ref = self.client.dataset(self.dataset_id).table(table_name)

        @retry.Retry(
            predicate=retry.if_transient_error,
            maximum=self.batch_config.max_retries,
            initial=self.batch_config.retry_delay_seconds
        )
        def _do_write():
            if self.batch_config.use_storage_write_api:
                return self._write_with_storage_api(table_ref, rows)
            else:
                return self._write_with_load_job(table_ref, rows)

        start_time = time.time()

        try:
            errors = _do_write()
            processing_time = int((time.time() - start_time) * 1000)

            if errors:
                self.logger.warning(f"Batch write to {table_name} had {len(errors)} errors")
                return WriteResult(
                    rows_written=len(rows) - len(errors),
                    errors_count=len(errors),
                    processing_time_ms=processing_time,
                    run_id=run_id,
                    table_name=table_name,
                    success=False,
                    error_message=f"Batch had {len(errors)} errors"
                )
            else:
                return WriteResult(
                    rows_written=len(rows),
                    errors_count=0,
                    processing_time_ms=processing_time,
                    run_id=run_id,
                    table_name=table_name,
                    success=True
                )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Failed to write batch to {table_name}: {e}")

            return WriteResult(
                rows_written=0,
                errors_count=len(rows),
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name=table_name,
                success=False,
                error_message=str(e)
            )

    def _write_with_storage_api(self, table_ref: bigquery.TableReference,
                               rows: List[Dict[str, Any]]) -> List[Dict]:
        """Write using BigQuery Storage Write API (preferred)"""
        return self._write_with_insert_rows(table_ref, rows)

    def _write_with_insert_rows(self, table_ref: bigquery.TableReference,
                               rows: List[Dict[str, Any]]) -> List[Dict]:
        """Write using insert_rows (streaming inserts)"""
        table = self.client.get_table(table_ref)
        errors = self.client.insert_rows_json(table, rows)
        return errors

    def _write_with_load_job(self, table_ref: bigquery.TableReference,
                            rows: List[Dict[str, Any]]) -> List[Dict]:
        """Write using load job (batch loading)"""
        job_config = bigquery.LoadJobConfig(
            write_disposition=WriteDisposition.WRITE_APPEND,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )

        # Convert rows to NDJSON
        ndjson_data = '\n'.join(json.dumps(row) for row in rows)

        job = self.client.load_table_from_json(
            json_rows=rows,
            destination=table_ref,
            job_config=job_config
        )

        job.result(timeout=self.batch_config.timeout_seconds)

        if job.errors:
            return job.errors
        else:
            return []

    def _batch_iterator(self, items: List[Any], batch_size: int) -> Iterator[List[Any]]:
        """Iterate over items in batches"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _chunk_to_row(self, chunk: ChunkMetadata) -> Dict[str, Any]:
        """Convert ChunkMetadata to BigQuery row"""
        # Ensure source_type is a string value, not the enum representation
        if hasattr(chunk.source_type, 'value'):
            source_type_str = chunk.source_type.value
        else:
            source_type_str = str(chunk.source_type).lower()

        # Convert source_metadata to JSON string for BigQuery STRING field
        source_metadata_dict = chunk.source_metadata or {}
        source_metadata_json = json.dumps(source_metadata_dict) if source_metadata_dict else None

        return {
            "source_type": source_type_str,
            "artifact_id": chunk.artifact_id,
            "parent_id": chunk.parent_id,
            "parent_type": chunk.parent_type,
            "content_text": chunk.content_text,
            "content_tokens": chunk.content_tokens,
            "content_hash": chunk.content_hash,
            "source_uri": chunk.source_uri,
            "repo_ref": chunk.repo_ref,
            "collected_at": chunk.collected_at.isoformat() if chunk.collected_at else datetime.now(timezone.utc).isoformat(),
            "source_metadata": source_metadata_json
        }

    def _error_to_row(self, error: ParseError) -> Dict[str, Any]:
        """Convert ParseError to BigQuery row"""
        return {
            "source_type": error.source_type.value if isinstance(error.source_type, SourceType) else str(error.source_type),
            "source_uri": error.source_uri,
            "error_class": error.error_class.value if isinstance(error.error_class, ErrorClass) else str(error.error_class),
            "error_msg": error.error_msg,
            "sample_text": error.sample_text,
            "stack_trace": error.stack_trace,
            "collected_at": error.collected_at.isoformat() if error.collected_at else datetime.now(timezone.utc).isoformat()
        }

    def _run_info_to_row(self, run_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert run info to BigQuery row"""
        # Convert config_used to JSON string for BigQuery STRING field
        config_used_dict = run_info.get("config_used", {})
        config_used_json = json.dumps(config_used_dict) if config_used_dict else None

        return {
            "run_id": run_info.get("run_id"),
            "source_type": run_info.get("source_type"),
            "started_at": run_info.get("started_at", datetime.now(timezone.utc)).isoformat(),
            "completed_at": run_info.get("completed_at").isoformat() if run_info.get("completed_at") else None,
            "status": run_info.get("status", "unknown"),
            "files_processed": run_info.get("files_processed"),
            "chunks_created": run_info.get("chunks_created"),
            "errors_encountered": run_info.get("errors_encountered"),
            "processing_duration_ms": run_info.get("processing_duration_ms"),
            "config_used": config_used_json,
            "error_summary": run_info.get("error_summary")
        }

    @contextmanager
    def ingestion_run_context(self, run_id: str, source_type: str, source_uri: str,
                             config_used: Optional[Dict[str, Any]] = None):
        """Context manager for comprehensive ingestion run tracking (T033)"""
        start_time = datetime.now(timezone.utc)

        # Log start of run
        self.log_ingestion_run({
            "run_id": run_id,
            "source_type": source_type,
            "started_at": start_time,
            "status": "started",
            "config_used": config_used or {}
        })

        run_stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "errors_encountered": 0
        }

        try:
            yield run_stats

            # Log successful completion
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - start_time).total_seconds() * 1000)

            self.log_ingestion_run({
                "run_id": run_id,
                "source_type": source_type,
                "started_at": start_time,
                "completed_at": completed_at,
                "status": "completed",
                "files_processed": run_stats["files_processed"],
                "chunks_created": run_stats["chunks_created"],
                "errors_encountered": run_stats["errors_encountered"],
                "processing_duration_ms": duration_ms,
                "config_used": config_used or {}
            })

        except Exception as e:
            # Log failed run
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - start_time).total_seconds() * 1000)

            self.log_ingestion_run({
                "run_id": run_id,
                "source_type": source_type,
                "started_at": start_time,
                "completed_at": completed_at,
                "status": "failed",
                "files_processed": run_stats["files_processed"],
                "chunks_created": run_stats["chunks_created"],
                "errors_encountered": run_stats["errors_encountered"],
                "processing_duration_ms": duration_ms,
                "config_used": config_used or {},
                "error_summary": str(e)
            })
            raise

    @contextmanager
    def transaction_context(self, run_id: str):
        """Context manager for transaction-like behavior (legacy)"""
        start_time = datetime.now(timezone.utc)

        try:
            yield self
        except Exception as e:
            # Log failed run
            self.log_ingestion_run({
                "run_id": run_id,
                "source_type": "unknown",
                "started_at": start_time,
                "completed_at": datetime.now(timezone.utc),
                "status": "failed",
                "error_summary": str(e)
            })
            raise
        else:
            # Log successful run
            self.log_ingestion_run({
                "run_id": run_id,
                "source_type": "unknown",
                "started_at": start_time,
                "completed_at": datetime.now(timezone.utc),
                "status": "completed"
            })

    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics about a table"""
        table_ref = self.client.dataset(self.dataset_id).table(table_name)

        try:
            table = self.client.get_table(table_ref)

            return {
                "table_name": table_name,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "schema_fields": len(table.schema),
                "clustering_fields": table.clustering_fields,
                "partitioning": table.time_partitioning.field if table.time_partitioning else None
            }

        except NotFound:
            return {
                "table_name": table_name,
                "exists": False
            }

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from tables"""
        cutoff_date = datetime.now(timezone.utc).date()
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)

        results = {}

        for table_name in self.schemas.keys():
            if table_name == "source_metadata":
                field = "collected_at"
            elif table_name == "source_metadata_errors":
                field = "collected_at"
            elif table_name == "ingestion_log":
                field = "started_at"
            else:
                continue

            query = f"""
            DELETE FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            WHERE DATE({field}) < @cutoff_date
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "DATE", cutoff_date)
                ]
            )

            try:
                job = self.client.query(query, job_config=job_config)
                job.result()
                results[table_name] = job.num_dml_affected_rows or 0

            except Exception as e:
                self.logger.error(f"Error cleaning up {table_name}: {e}")
                results[table_name] = -1

        return results

    def _get_existing_content_hashes(self, content_hashes: List[str]) -> set:
        """Check which content hashes already exist in source_metadata table (T032)"""
        if not content_hashes:
            return set()

        try:
            # Create parameterized query to check for existing hashes
            hash_params = [bigquery.ScalarQueryParameter(f"hash_{i}", "STRING", hash_val)
                          for i, hash_val in enumerate(content_hashes)]

            hash_conditions = " OR ".join([f"content_hash = @hash_{i}" for i in range(len(content_hashes))])

            query = f"""
            SELECT DISTINCT content_hash
            FROM `{self.project_id}.{self.dataset_id}.source_metadata`
            WHERE {hash_conditions}
            """

            job_config = bigquery.QueryJobConfig(query_parameters=hash_params)
            job = self.client.query(query, job_config=job_config)
            results = job.result()

            existing_hashes = {row.content_hash for row in results}
            self.logger.debug(f"Found {len(existing_hashes)} existing content hashes out of {len(content_hashes)} checked")

            return existing_hashes

        except Exception as e:
            self.logger.warning(f"Error checking existing content hashes, proceeding without deduplication: {e}")
            # Return empty set to continue with all writes if hash check fails
            return set()

    def upsert_chunks_by_artifact_id(self, chunks: List[ChunkMetadata], run_id: str) -> WriteResult:
        """Upsert chunks using MERGE statement for more sophisticated deduplication (T032)"""
        if not chunks:
            return WriteResult(
                rows_written=0, errors_count=0, processing_time_ms=0,
                run_id=run_id, table_name="source_metadata", success=True
            )

        start_time = time.time()

        try:
            # Create temporary table with new data
            temp_table_id = f"temp_upsert_{run_id.replace('-', '_')}"
            temp_table_ref = self.client.dataset(self.dataset_id).table(temp_table_id)

            # Write chunks to temporary table
            rows = [self._chunk_to_row(chunk) for chunk in chunks]

            # Create temporary table
            temp_table = bigquery.Table(temp_table_ref, schema=self.schemas["source_metadata"])
            temp_table = self.client.create_table(temp_table)

            # Insert data into temp table
            errors = self.client.insert_rows_json(temp_table, rows)
            if errors:
                raise Exception(f"Failed to insert into temporary table: {errors}")

            # Execute MERGE statement
            merge_query = f"""
            MERGE `{self.project_id}.{self.dataset_id}.source_metadata` AS target
            USING `{self.project_id}.{self.dataset_id}.{temp_table_id}` AS source
            ON target.content_hash = source.content_hash
            WHEN NOT MATCHED THEN
                INSERT (
                    source_type, artifact_id, parent_id, parent_type,
                    content_text, content_tokens, content_hash, source_uri,
                    repo_ref, collected_at, source_metadata
                )
                VALUES (
                    source.source_type, source.artifact_id, source.parent_id, source.parent_type,
                    source.content_text, source.content_tokens, source.content_hash, source.source_uri,
                    source.repo_ref, source.collected_at, source.source_metadata
                )
            """

            job = self.client.query(merge_query)
            job.result()

            # Clean up temporary table
            self.client.delete_table(temp_table_ref)

            processing_time = int((time.time() - start_time) * 1000)
            rows_affected = job.num_dml_affected_rows or 0

            self.logger.info(f"MERGE operation completed: {rows_affected} rows affected")

            return WriteResult(
                rows_written=rows_affected,
                errors_count=0,
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata",
                success=True
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Error in upsert operation: {e}")

            # Clean up temporary table if it exists
            try:
                self.client.delete_table(temp_table_ref)
            except:
                pass

            return WriteResult(
                rows_written=0,
                errors_count=len(chunks),
                processing_time_ms=processing_time,
                run_id=run_id,
                table_name="source_metadata",
                success=False,
                error_message=str(e)
            )


# Factory function
def create_bigquery_writer(project_id: Optional[str] = None,
                          dataset_id: Optional[str] = None,
                          batch_size: int = 1000) -> BigQueryWriter:
    """Create BigQuery writer with specified configuration"""
    batch_config = BatchConfig(batch_size=batch_size)
    return BigQueryWriter(project_id, dataset_id, batch_config=batch_config)


# Utility functions
def test_connection(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Test BigQuery connection and permissions"""
    try:
        client = bigquery.Client(project=project_id)

        # Test basic query
        query = "SELECT 1 as test_column"
        job = client.query(query)
        result = list(job.result())

        return {
            "connection_successful": True,
            "project_id": client.project,
            "location": client.location,
            "test_query_result": result[0].test_column if result else None
        }

    except Exception as e:
        return {
            "connection_successful": False,
            "error": str(e)
        }