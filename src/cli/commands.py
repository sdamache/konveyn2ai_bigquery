"""
CLI command implementations for M1 multi-source ingestion
Implements the actual command logic called by main.py
"""

import argparse
from pathlib import Path
from typing import Any

from common import (
    create_bigquery_writer,
)
from common.logging import LogCategory, create_error_handler, get_logger
from ingest.cobol.parser import COBOLParserImpl
from ingest.fastapi.parser import FastAPIParserImpl
from ingest.irs.parser import IRSParserImpl

# Import parsers
from ingest.k8s.parser import KubernetesParserImpl
from ingest.mumps.parser import MUMPSParserImpl


def setup_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Setup environment and create BigQuery tables"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    logger.info("Setting up M1 ingestion environment", category=LogCategory.SYSTEM)

    try:
        # Create BigQuery writer and setup tables
        writer = create_bigquery_writer(
            project_id=args.project, dataset_id=args.dataset
        )

        # Create dataset if it doesn't exist
        dataset_created = writer.create_dataset_if_not_exists(location=args.location)

        # Create tables if they don't exist
        tables_created = writer.create_tables_if_not_exist()

        # Test connection
        stats = {}
        for table_name in [
            "source_metadata",
            "source_metadata_errors",
            "ingestion_log",
        ]:
            table_stats = writer.get_table_stats(table_name)
            stats[table_name] = table_stats

        logger.info(
            "Environment setup completed successfully",
            category=LogCategory.SYSTEM,
            metadata={"tables_created": tables_created, "table_stats": stats},
        )

        return {
            "success": True,
            "dataset_created": dataset_created,
            "tables_created": tables_created,
            "table_stats": stats,
            "project": args.project,
            "dataset": args.dataset,
        }

    except Exception as e:
        error_handler.handle_system_error(
            "unknown", "setup", e, {"project": args.project, "dataset": args.dataset}
        )

        return {
            "success": False,
            "error": str(e),
            "project": args.project,
            "dataset": args.dataset,
        }


def ingest_kubernetes(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Ingest Kubernetes manifests"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    source_path = Path(args.source)
    logger.log_ingestion_start(run_id, "kubernetes", str(source_path))

    try:
        # Initialize parser
        parser = KubernetesParserImpl()

        # Check if dry run
        if args.dry_run:
            return _dry_run_analysis(parser, source_path, "kubernetes", run_id)

        # Parse content
        if source_path.is_file():
            result = parser.parse_file(str(source_path))
        else:
            result = parser.parse_directory(str(source_path))

        # Apply max rows limit
        if args.max_rows and len(result.chunks) > args.max_rows:
            result.chunks = result.chunks[: args.max_rows]

        # Output handling
        if args.output == "bigquery":
            # Write to BigQuery with enhanced run tracking (T031-T033)
            writer = create_bigquery_writer(
                project_id=args.project, dataset_id=args.dataset
            )

            config_used = {
                "source_path": str(source_path),
                "dry_run": args.dry_run,
                "max_rows": args.max_rows,
                "namespace": getattr(args, "namespace", None),
                "live_cluster": getattr(args, "live_cluster", False),
            }

            # Use comprehensive run tracking context (T033)
            with writer.ingestion_run_context(
                run_id, "kubernetes", str(source_path), config_used
            ) as run_stats:
                # Write chunks with idempotent upsert logic (T032)
                chunk_result = writer.write_chunks(
                    result.chunks, run_id, enable_upsert=True
                )
                error_result = writer.write_errors(result.errors, run_id)

                # Update run statistics for tracking
                run_stats["files_processed"] = result.files_processed
                run_stats["chunks_created"] = len(result.chunks)
                run_stats["errors_encountered"] = len(result.errors)

            logger.log_ingestion_complete(
                run_id,
                "kubernetes",
                str(source_path),
                len(result.chunks),
                result.processing_duration_ms,
            )

            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "bigquery_rows": chunk_result.rows_written,
                "processing_duration_ms": result.processing_duration_ms,
                "duplicates_skipped": len(result.chunks) - chunk_result.rows_written,
            }

        else:
            # Console or JSON output
            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "content_length": len(c.content_text),
                    }
                    for c in result.chunks
                ],
                "processing_duration_ms": result.processing_duration_ms,
            }

    except Exception as e:
        error_handler.handle_ingestion_error(run_id, "kubernetes", str(source_path), e)

        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "source": str(source_path),
        }


def ingest_fastapi(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Ingest FastAPI source code and specs"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    source_path = Path(args.source)
    logger.log_ingestion_start(run_id, "fastapi", str(source_path))

    try:
        # Initialize parser
        parser = FastAPIParserImpl()

        # Check if dry run
        if args.dry_run:
            return _dry_run_analysis(parser, source_path, "fastapi", run_id)

        # Parse content
        if source_path.is_file():
            result = parser.parse_file(str(source_path))
        else:
            result = parser.parse_directory(str(source_path))

        # Apply max rows limit
        if args.max_rows and len(result.chunks) > args.max_rows:
            result.chunks = result.chunks[: args.max_rows]

        # Output handling
        if args.output == "bigquery":
            # Write to BigQuery with enhanced run tracking (T031-T033)
            writer = create_bigquery_writer(
                project_id=args.project, dataset_id=args.dataset
            )

            config_used = {
                "source_path": str(source_path),
                "dry_run": args.dry_run,
                "max_rows": args.max_rows,
                "include_tests": getattr(args, "include_tests", False),
            }

            # Use comprehensive run tracking context (T033)
            with writer.ingestion_run_context(
                run_id, "fastapi", str(source_path), config_used
            ) as run_stats:
                # Write chunks with idempotent upsert logic (T032)
                chunk_result = writer.write_chunks(
                    result.chunks, run_id, enable_upsert=True
                )
                error_result = writer.write_errors(result.errors, run_id)

                # Update run statistics for tracking
                run_stats["files_processed"] = result.files_processed
                run_stats["chunks_created"] = len(result.chunks)
                run_stats["errors_encountered"] = len(result.errors)

            logger.log_ingestion_complete(
                run_id,
                "fastapi",
                str(source_path),
                len(result.chunks),
                result.processing_duration_ms,
            )

            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "bigquery_rows": chunk_result.rows_written,
                "processing_duration_ms": result.processing_duration_ms,
                "duplicates_skipped": len(result.chunks) - chunk_result.rows_written,
            }

        else:
            # Console or JSON output
            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "content_length": len(c.content_text),
                    }
                    for c in result.chunks
                ],
                "processing_duration_ms": result.processing_duration_ms,
            }

    except Exception as e:
        error_handler.handle_ingestion_error(run_id, "fastapi", str(source_path), e)

        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "source": str(source_path),
        }


def ingest_cobol(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Ingest COBOL copybooks"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    source_path = Path(args.source)
    logger.log_ingestion_start(run_id, "cobol", str(source_path))

    try:
        # Initialize parser
        parser = COBOLParserImpl()

        # Check if dry run
        if args.dry_run:
            return _dry_run_analysis(parser, source_path, "cobol", run_id)

        # Parse content
        if source_path.is_file():
            result = parser.parse_file(str(source_path))
        else:
            result = parser.parse_directory(str(source_path))

        # Apply max rows limit
        if args.max_rows and len(result.chunks) > args.max_rows:
            result.chunks = result.chunks[: args.max_rows]

        # Output handling
        if args.output == "bigquery":
            # Write to BigQuery with enhanced run tracking (T031-T033)
            writer = create_bigquery_writer(
                project_id=args.project, dataset_id=args.dataset
            )

            config_used = {
                "source_path": str(source_path),
                "dry_run": args.dry_run,
                "max_rows": args.max_rows,
                "encoding": getattr(args, "encoding", "utf-8"),
            }

            # Use comprehensive run tracking context (T033)
            with writer.ingestion_run_context(
                run_id, "cobol", str(source_path), config_used
            ) as run_stats:
                # Write chunks with idempotent upsert logic (T032)
                chunk_result = writer.write_chunks(
                    result.chunks, run_id, enable_upsert=True
                )
                error_result = writer.write_errors(result.errors, run_id)

                # Update run statistics for tracking
                run_stats["files_processed"] = result.files_processed
                run_stats["chunks_created"] = len(result.chunks)
                run_stats["errors_encountered"] = len(result.errors)

            logger.log_ingestion_complete(
                run_id,
                "cobol",
                str(source_path),
                len(result.chunks),
                result.processing_duration_ms,
            )

            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "bigquery_rows": chunk_result.rows_written,
                "processing_duration_ms": result.processing_duration_ms,
                "duplicates_skipped": len(result.chunks) - chunk_result.rows_written,
            }

        else:
            # Console or JSON output
            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "content_length": len(c.content_text),
                    }
                    for c in result.chunks
                ],
                "processing_duration_ms": result.processing_duration_ms,
            }

    except Exception as e:
        error_handler.handle_ingestion_error(run_id, "cobol", str(source_path), e)

        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "source": str(source_path),
        }


def ingest_irs(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Ingest IRS record layouts"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    source_path = Path(args.source)
    logger.log_ingestion_start(run_id, "irs", str(source_path))

    try:
        # Initialize parser
        parser = IRSParserImpl()

        # Check if dry run
        if args.dry_run:
            return _dry_run_analysis(parser, source_path, "irs", run_id)

        # Parse content
        if source_path.is_file():
            result = parser.parse_file(str(source_path))
        else:
            result = parser.parse_directory(str(source_path))

        # Apply max rows limit
        if args.max_rows and len(result.chunks) > args.max_rows:
            result.chunks = result.chunks[: args.max_rows]

        # Output handling
        if args.output == "bigquery":
            # Write to BigQuery with enhanced run tracking (T031-T033)
            writer = create_bigquery_writer(
                project_id=args.project, dataset_id=args.dataset
            )

            config_used = {
                "source_path": str(source_path),
                "dry_run": args.dry_run,
                "max_rows": args.max_rows,
                "layout_version": getattr(args, "layout_version", None),
            }

            # Use comprehensive run tracking context (T033)
            with writer.ingestion_run_context(
                run_id, "irs", str(source_path), config_used
            ) as run_stats:
                # Write chunks with idempotent upsert logic (T032)
                chunk_result = writer.write_chunks(
                    result.chunks, run_id, enable_upsert=True
                )
                error_result = writer.write_errors(result.errors, run_id)

                # Update run statistics for tracking
                run_stats["files_processed"] = result.files_processed
                run_stats["chunks_created"] = len(result.chunks)
                run_stats["errors_encountered"] = len(result.errors)

            logger.log_ingestion_complete(
                run_id,
                "irs",
                str(source_path),
                len(result.chunks),
                result.processing_duration_ms,
            )

            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "bigquery_rows": chunk_result.rows_written,
                "processing_duration_ms": result.processing_duration_ms,
                "duplicates_skipped": len(result.chunks) - chunk_result.rows_written,
            }

        else:
            # Console or JSON output
            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "content_length": len(c.content_text),
                    }
                    for c in result.chunks
                ],
                "processing_duration_ms": result.processing_duration_ms,
            }

    except Exception as e:
        error_handler.handle_ingestion_error(run_id, "irs", str(source_path), e)

        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "source": str(source_path),
        }


def ingest_mumps(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    """Ingest MUMPS/VistA dictionaries"""
    logger = get_logger()
    error_handler = create_error_handler(logger)

    source_path = Path(args.source)
    logger.log_ingestion_start(run_id, "mumps", str(source_path))

    try:
        # Initialize parser
        parser = MUMPSParserImpl()

        # Check if dry run
        if args.dry_run:
            return _dry_run_analysis(parser, source_path, "mumps", run_id)

        # Parse content
        if source_path.is_file():
            result = parser.parse_file(str(source_path))
        else:
            result = parser.parse_directory(str(source_path))

        # Apply max rows limit
        if args.max_rows and len(result.chunks) > args.max_rows:
            result.chunks = result.chunks[: args.max_rows]

        # Output handling
        if args.output == "bigquery":
            # Write to BigQuery with enhanced run tracking (T031-T033)
            writer = create_bigquery_writer(
                project_id=args.project, dataset_id=args.dataset
            )

            config_used = {
                "source_path": str(source_path),
                "dry_run": args.dry_run,
                "max_rows": args.max_rows,
                "fileman_only": getattr(args, "fileman_only", False),
            }

            # Use comprehensive run tracking context (T033)
            with writer.ingestion_run_context(
                run_id, "mumps", str(source_path), config_used
            ) as run_stats:
                # Write chunks with idempotent upsert logic (T032)
                chunk_result = writer.write_chunks(
                    result.chunks, run_id, enable_upsert=True
                )
                error_result = writer.write_errors(result.errors, run_id)

                # Update run statistics for tracking
                run_stats["files_processed"] = result.files_processed
                run_stats["chunks_created"] = len(result.chunks)
                run_stats["errors_encountered"] = len(result.errors)

            logger.log_ingestion_complete(
                run_id,
                "mumps",
                str(source_path),
                len(result.chunks),
                result.processing_duration_ms,
            )

            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "bigquery_rows": chunk_result.rows_written,
                "processing_duration_ms": result.processing_duration_ms,
                "duplicates_skipped": len(result.chunks) - chunk_result.rows_written,
            }

        else:
            # Console or JSON output
            return {
                "success": True,
                "run_id": run_id,
                "files_processed": result.files_processed,
                "chunks_created": len(result.chunks),
                "errors_encountered": len(result.errors),
                "chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "content_length": len(c.content_text),
                    }
                    for c in result.chunks
                ],
                "processing_duration_ms": result.processing_duration_ms,
            }

    except Exception as e:
        error_handler.handle_ingestion_error(run_id, "mumps", str(source_path), e)

        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "source": str(source_path),
        }


def _dry_run_analysis(
    parser, source_path: Path, source_type: str, run_id: str
) -> dict[str, Any]:
    """Perform dry run analysis without actual ingestion"""
    logger = get_logger()

    logger.info(
        f"Performing dry run analysis for {source_type}",
        category=LogCategory.CLI,
        run_id=run_id,
    )

    try:
        files_to_process = []
        total_size = 0

        if source_path.is_file():
            if parser.validate_content(source_path.read_text()):
                files_to_process.append(str(source_path))
                total_size = source_path.stat().st_size
        else:
            # Find all files that would be processed
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    try:
                        if parser.validate_content(file_path.read_text()):
                            files_to_process.append(str(file_path))
                            total_size += file_path.stat().st_size
                    except Exception:
                        # Skip files that can't be read
                        continue

        return {
            "success": True,
            "run_id": run_id,
            "dry_run": True,
            "files_to_process": len(files_to_process),
            "estimated_chunks": len(files_to_process) * 2,  # Rough estimate
            "total_size_bytes": total_size,
            "files": files_to_process[:10],  # Show first 10 files
            "source_type": source_type,
        }

    except Exception as e:
        logger.error(
            f"Dry run analysis failed for {source_type}",
            category=LogCategory.CLI,
            run_id=run_id,
            error=e,
        )

        return {
            "success": False,
            "run_id": run_id,
            "dry_run": True,
            "error": str(e),
            "source_type": source_type,
        }
