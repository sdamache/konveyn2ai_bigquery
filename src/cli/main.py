"""
Main CLI entrypoint for M1 multi-source ingestion
T029: Provides standardized command-line interface with argument parsing, dry-run support, and output formatting
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from common.logging import setup_logging, get_logger, log_cli_command, LogCategory
from common import generate_run_id
from .commands import (
    ingest_kubernetes,
    ingest_fastapi,
    ingest_cobol,
    ingest_irs,
    ingest_mumps,
    setup_environment
)


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser with all subcommands"""

    parser = argparse.ArgumentParser(
        prog="m1-ingest",
        description="M1 Multi-Source Ingestion for BigQuery AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup environment and BigQuery tables
  m1-ingest setup

  # Ingest Kubernetes manifests
  m1-ingest k8s --source ./k8s-manifests/ --output bigquery

  # Dry run FastAPI analysis
  m1-ingest fastapi --source ./api/ --dry-run --output console

  # Ingest with custom BigQuery settings
  m1-ingest cobol --source ./copybooks/ --project my-project --dataset my_data

  # JSON output for automation
  m1-ingest irs --source ./layouts/ --output json --max-rows 1000

Environment Variables:
  BQ_PROJECT          BigQuery project ID (default: konveyn2ai)
  BQ_DATASET          BigQuery dataset name (default: source_ingestion)
  LOG_LEVEL           Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
  LOG_FILE            Log file path (default: console only)
  LOG_CONSOLE         Enable console logging: true/false (default: true)
        """
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Set logging level"
    )

    parser.add_argument(
        "--log-file",
        default=os.getenv("LOG_FILE"),
        help="Log to file instead of console"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (except errors)"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Available ingestion commands",
        help="Use <command> --help for command-specific options"
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Setup environment and create BigQuery tables"
    )
    add_bigquery_args(setup_parser)

    # Common parser for ingestion commands
    def add_ingestion_args(cmd_parser):
        """Add common arguments for ingestion commands"""
        cmd_parser.add_argument(
            "--source", "-s",
            required=True,
            help="Source file or directory to ingest"
        )

        cmd_parser.add_argument(
            "--output", "-o",
            choices=["console", "json", "bigquery"],
            default="bigquery",
            help="Output format (default: bigquery)"
        )

        cmd_parser.add_argument(
            "--dry-run", "-n",
            action="store_true",
            help="Show what would be processed without executing"
        )

        cmd_parser.add_argument(
            "--max-rows",
            type=int,
            help="Limit number of rows to process"
        )

        cmd_parser.add_argument(
            "--run-id",
            help="Custom run ID for tracking (auto-generated if not provided)"
        )

        add_bigquery_args(cmd_parser)

    # Kubernetes ingestion
    k8s_parser = subparsers.add_parser(
        "k8s",
        aliases=["kubernetes"],
        help="Ingest Kubernetes YAML/JSON manifests"
    )
    add_ingestion_args(k8s_parser)
    k8s_parser.add_argument(
        "--namespace",
        help="Filter by Kubernetes namespace"
    )
    k8s_parser.add_argument(
        "--live-cluster",
        action="store_true",
        help="Extract from live cluster (requires kubectl access)"
    )

    # FastAPI ingestion
    fastapi_parser = subparsers.add_parser(
        "fastapi",
        aliases=["api"],
        help="Ingest FastAPI source code and OpenAPI specs"
    )
    add_ingestion_args(fastapi_parser)
    fastapi_parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in analysis"
    )

    # COBOL ingestion
    cobol_parser = subparsers.add_parser(
        "cobol",
        help="Ingest COBOL copybooks and data definitions"
    )
    add_ingestion_args(cobol_parser)
    cobol_parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for COBOL files (default: utf-8)"
    )

    # IRS ingestion
    irs_parser = subparsers.add_parser(
        "irs",
        help="Ingest IRS IMF record layouts"
    )
    add_ingestion_args(irs_parser)
    irs_parser.add_argument(
        "--layout-version",
        help="IRS layout version identifier"
    )

    # MUMPS ingestion
    mumps_parser = subparsers.add_parser(
        "mumps",
        aliases=["vista"],
        help="Ingest MUMPS/VistA FileMan dictionaries"
    )
    add_ingestion_args(mumps_parser)
    mumps_parser.add_argument(
        "--fileman-only",
        action="store_true",
        help="Process only FileMan dictionaries (skip globals)"
    )

    return parser


def add_bigquery_args(parser: argparse.ArgumentParser):
    """Add BigQuery-specific arguments"""
    parser.add_argument(
        "--project",
        default=os.getenv("BQ_PROJECT", "konveyn2ai"),
        help="BigQuery project ID"
    )

    parser.add_argument(
        "--dataset",
        default=os.getenv("BQ_DATASET", "source_ingestion"),
        help="BigQuery dataset name"
    )

    parser.add_argument(
        "--location",
        default="US",
        help="BigQuery location (default: US)"
    )


def format_output(result: Dict[str, Any], output_format: str) -> str:
    """Format command result for output"""
    if output_format == "json":
        return json.dumps(result, indent=2, default=str)

    elif output_format == "console":
        # Human-readable console output
        lines = []
        lines.append(f"✅ Command: {result.get('command', 'unknown')}")
        lines.append(f"📊 Status: {result.get('status', 'unknown')}")

        if result.get('run_id'):
            lines.append(f"🔄 Run ID: {result['run_id']}")

        if result.get('duration_ms'):
            lines.append(f"⏱️  Duration: {result['duration_ms']}ms")

        if result.get('files_processed'):
            lines.append(f"📁 Files processed: {result['files_processed']}")

        if result.get('chunks_created'):
            lines.append(f"🧩 Chunks created: {result['chunks_created']}")

        if result.get('errors_encountered'):
            lines.append(f"❌ Errors: {result['errors_encountered']}")

        if result.get('bigquery_rows'):
            lines.append(f"💾 BigQuery rows: {result['bigquery_rows']}")

        if result.get('error'):
            lines.append(f"💥 Error: {result['error']}")

        return "\n".join(lines)

    else:  # bigquery
        # For BigQuery output, just show summary
        return f"✅ Ingestion completed. Run ID: {result.get('run_id')}"


def main():
    """Main CLI entrypoint"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        console_output=not args.quiet
    )

    # Log the command execution
    log_cli_command(args.command or "help", vars(args))

    # Handle no command
    if not args.command:
        parser.print_help()
        return 1

    # Generate run ID if not provided
    run_id = getattr(args, 'run_id', None) or generate_run_id()

    try:
        start_time = time.time()

        # Execute command
        if args.command == "setup":
            result = setup_environment(args)

        elif args.command in ["k8s", "kubernetes"]:
            result = ingest_kubernetes(args, run_id)

        elif args.command in ["fastapi", "api"]:
            result = ingest_fastapi(args, run_id)

        elif args.command == "cobol":
            result = ingest_cobol(args, run_id)

        elif args.command == "irs":
            result = ingest_irs(args, run_id)

        elif args.command in ["mumps", "vista"]:
            result = ingest_mumps(args, run_id)

        else:
            logger.error(f"Unknown command: {args.command}", category=LogCategory.CLI)
            return 1

        # Add timing and status info
        duration_ms = int((time.time() - start_time) * 1000)
        result.update({
            "command": args.command,
            "duration_ms": duration_ms,
            "status": "success" if result.get("success", True) else "failed"
        })

        # Output result
        output = format_output(result, getattr(args, 'output', 'console'))
        if not args.quiet:
            print(output)

        # Log completion
        logger.info(
            f"CLI command completed: {args.command}",
            category=LogCategory.CLI,
            run_id=run_id,
            duration_ms=duration_ms,
            metadata=result
        )

        return 0 if result.get("success", True) else 1

    except KeyboardInterrupt:
        logger.warning("Command interrupted by user", category=LogCategory.CLI, run_id=run_id)
        return 130  # Standard exit code for Ctrl+C

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        logger.error(
            f"CLI command failed: {args.command}",
            category=LogCategory.CLI,
            run_id=run_id,
            duration_ms=duration_ms,
            error=e
        )

        if not args.quiet:
            print(f"❌ Error: {str(e)}")

        return 1


if __name__ == "__main__":
    sys.exit(main())