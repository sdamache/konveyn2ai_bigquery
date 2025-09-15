"""
Error handling and logging infrastructure with structured JSON logs
T030: Provides comprehensive logging system for M1 ingestion pipeline
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import ulid
from contextlib import contextmanager


class LogLevel(Enum):
    """Standard log levels with numeric values"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for structured logging"""
    INGESTION = "ingestion"
    PARSING = "parsing"
    BIGQUERY = "bigquery"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    CLI = "cli"


@dataclass
class LogEntry:
    """Structured log entry for JSON logging"""
    timestamp: str
    level: str
    category: str
    message: str
    run_id: Optional[str] = None
    source_type: Optional[str] = None
    source_uri: Optional[str] = None
    duration_ms: Optional[int] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        # Remove None values for cleaner JSON
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(data, separators=(',', ':'))


class StructuredLogger:
    """Structured logger with JSON output and correlation tracking"""

    def __init__(self, name: str = "m1_ingestion",
                 level: LogLevel = LogLevel.INFO,
                 output_file: Optional[str] = None,
                 console_output: bool = True,
                 correlation_id: Optional[str] = None):
        self.name = name
        self.level = level
        self.correlation_id = correlation_id or str(ulid.new())

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)

        # Clear any existing handlers
        self.logger.handlers = []

        # Console handler for structured output
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level.value)
            console_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(console_handler)

        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setLevel(level.value)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)

    def _create_log_entry(self, level: str, category: LogCategory, message: str,
                         **kwargs) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            category=category.value,
            message=message,
            correlation_id=self.correlation_id,
            **kwargs
        )

    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message"""
        if self.level.value <= LogLevel.DEBUG.value:
            entry = self._create_log_entry("DEBUG", category, message, **kwargs)
            self.logger.debug(entry.to_json())

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message"""
        if self.level.value <= LogLevel.INFO.value:
            entry = self._create_log_entry("INFO", category, message, **kwargs)
            self.logger.info(entry.to_json())

    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message"""
        if self.level.value <= LogLevel.WARNING.value:
            entry = self._create_log_entry("WARNING", category, message, **kwargs)
            self.logger.warning(entry.to_json())

    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM,
             error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        error_details = None
        error_type = None

        if error:
            error_type = type(error).__name__
            error_details = {
                "exception_message": str(error),
                "exception_type": error_type,
                "traceback": traceback.format_exc() if error else None
            }

        entry = self._create_log_entry("ERROR", category, message,
                                     error_type=error_type,
                                     error_details=error_details,
                                     **kwargs)
        self.logger.error(entry.to_json())

    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                error: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        error_details = None
        error_type = None

        if error:
            error_type = type(error).__name__
            error_details = {
                "exception_message": str(error),
                "exception_type": error_type,
                "traceback": traceback.format_exc()
            }

        entry = self._create_log_entry("CRITICAL", category, message,
                                     error_type=error_type,
                                     error_details=error_details,
                                     **kwargs)
        self.logger.critical(entry.to_json())

    def log_ingestion_start(self, run_id: str, source_type: str, source_uri: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log ingestion start event"""
        self.info(
            f"Starting ingestion for {source_type}",
            category=LogCategory.INGESTION,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            metadata=metadata or {}
        )

    def log_ingestion_complete(self, run_id: str, source_type: str, source_uri: str,
                              chunks_created: int, duration_ms: int,
                              metadata: Optional[Dict[str, Any]] = None):
        """Log successful ingestion completion"""
        self.info(
            f"Ingestion completed for {source_type}: {chunks_created} chunks in {duration_ms}ms",
            category=LogCategory.INGESTION,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            duration_ms=duration_ms,
            metadata={
                **(metadata or {}),
                "chunks_created": chunks_created,
                "success": True
            }
        )

    def log_ingestion_error(self, run_id: str, source_type: str, source_uri: str,
                           error: Exception, duration_ms: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log ingestion error"""
        self.error(
            f"Ingestion failed for {source_type}: {str(error)}",
            category=LogCategory.INGESTION,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            duration_ms=duration_ms,
            error=error,
            metadata={
                **(metadata or {}),
                "success": False
            }
        )

    def log_parsing_progress(self, run_id: str, source_type: str, source_uri: str,
                            files_processed: int, chunks_generated: int,
                            errors_encountered: int = 0):
        """Log parsing progress"""
        self.info(
            f"Parsing progress for {source_type}: {files_processed} files, {chunks_generated} chunks, {errors_encountered} errors",
            category=LogCategory.PARSING,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            metadata={
                "files_processed": files_processed,
                "chunks_generated": chunks_generated,
                "errors_encountered": errors_encountered
            }
        )

    def log_bigquery_operation(self, run_id: str, operation: str, table_name: str,
                              rows_affected: int, duration_ms: int,
                              success: bool = True, error: Optional[Exception] = None):
        """Log BigQuery operations"""
        level_method = self.info if success else self.error
        message = f"BigQuery {operation} on {table_name}: {rows_affected} rows in {duration_ms}ms"

        level_method(
            message,
            category=LogCategory.BIGQUERY,
            run_id=run_id,
            duration_ms=duration_ms,
            error=error,
            metadata={
                "operation": operation,
                "table_name": table_name,
                "rows_affected": rows_affected,
                "success": success
            }
        )

    def log_performance_metric(self, run_id: str, metric_name: str, value: Union[int, float],
                              unit: str, source_type: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        self.info(
            f"Performance metric {metric_name}: {value} {unit}",
            category=LogCategory.PERFORMANCE,
            run_id=run_id,
            source_type=source_type,
            metadata={
                **(metadata or {}),
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit
            }
        )

    @contextmanager
    def operation_context(self, operation_name: str, run_id: Optional[str] = None,
                         source_type: Optional[str] = None, source_uri: Optional[str] = None):
        """Context manager for tracking operation duration and outcomes"""
        start_time = datetime.now(timezone.utc)
        op_run_id = run_id or str(ulid.new())

        self.info(
            f"Starting operation: {operation_name}",
            category=LogCategory.SYSTEM,
            run_id=op_run_id,
            source_type=source_type,
            source_uri=source_uri,
            metadata={"operation": operation_name, "status": "started"}
        )

        try:
            yield op_run_id
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            self.info(
                f"Operation completed: {operation_name}",
                category=LogCategory.SYSTEM,
                run_id=op_run_id,
                source_type=source_type,
                source_uri=source_uri,
                duration_ms=duration_ms,
                metadata={"operation": operation_name, "status": "completed"}
            )

        except Exception as e:
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

            self.error(
                f"Operation failed: {operation_name}",
                category=LogCategory.SYSTEM,
                run_id=op_run_id,
                source_type=source_type,
                source_uri=source_uri,
                duration_ms=duration_ms,
                error=e,
                metadata={"operation": operation_name, "status": "failed"}
            )
            raise


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs"""

    def format(self, record):
        # The message should already be JSON from StructuredLogger
        return record.getMessage()


class ErrorHandler:
    """Centralized error handling with categorization and reporting"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def handle_parsing_error(self, run_id: str, source_type: str, source_uri: str,
                           error: Exception, sample_content: Optional[str] = None):
        """Handle parsing errors with context"""
        self.logger.error(
            f"Parsing error in {source_type}",
            category=LogCategory.PARSING,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            error=error,
            metadata={
                "sample_content": sample_content[:200] if sample_content else None,
                "error_location": "parser"
            }
        )

    def handle_validation_error(self, run_id: str, source_type: str, source_uri: str,
                              error: Exception, validation_rule: str):
        """Handle validation errors"""
        self.logger.error(
            f"Validation error in {source_type}: {validation_rule}",
            category=LogCategory.VALIDATION,
            run_id=run_id,
            source_type=source_type,
            source_uri=source_uri,
            error=error,
            metadata={
                "validation_rule": validation_rule,
                "error_location": "validation"
            }
        )

    def handle_bigquery_error(self, run_id: str, operation: str, table_name: str,
                             error: Exception, rows_attempted: Optional[int] = None):
        """Handle BigQuery operation errors"""
        self.logger.error(
            f"BigQuery error during {operation} on {table_name}",
            category=LogCategory.BIGQUERY,
            run_id=run_id,
            error=error,
            metadata={
                "operation": operation,
                "table_name": table_name,
                "rows_attempted": rows_attempted,
                "error_location": "bigquery"
            }
        )

    def handle_system_error(self, run_id: str, component: str, error: Exception,
                           context: Optional[Dict[str, Any]] = None):
        """Handle system/infrastructure errors"""
        self.logger.error(
            f"System error in {component}",
            category=LogCategory.SYSTEM,
            run_id=run_id,
            error=error,
            metadata={
                "component": component,
                "context": context or {},
                "error_location": "system"
            }
        )


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "m1_ingestion") -> StructuredLogger:
    """Get or create global logger instance"""
    global _global_logger

    if _global_logger is None:
        # Configure from environment
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_file = os.getenv("LOG_FILE")
        console_output = os.getenv("LOG_CONSOLE", "true").lower() == "true"

        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL
        }

        _global_logger = StructuredLogger(
            name=name,
            level=level_map.get(log_level, LogLevel.INFO),
            output_file=log_file,
            console_output=console_output
        )

    return _global_logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None,
                 console_output: bool = True) -> StructuredLogger:
    """Setup structured logging for the application"""
    global _global_logger

    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL
    }

    _global_logger = StructuredLogger(
        name="m1_ingestion",
        level=level_map.get(log_level.upper(), LogLevel.INFO),
        output_file=log_file,
        console_output=console_output
    )

    return _global_logger


def create_error_handler(logger: Optional[StructuredLogger] = None) -> ErrorHandler:
    """Create error handler with logger"""
    if logger is None:
        logger = get_logger()
    return ErrorHandler(logger)


# Utility functions for common logging patterns
def log_function_call(func):
    """Decorator to log function calls with timing"""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"

        with logger.operation_context(f"function_call:{func_name}"):
            return func(*args, **kwargs)

    return wrapper


def log_cli_command(command: str, args: Dict[str, Any]):
    """Log CLI command execution"""
    logger = get_logger()
    logger.info(
        f"CLI command executed: {command}",
        category=LogCategory.CLI,
        metadata={
            "command": command,
            "arguments": args
        }
    )


def log_configuration(config: Dict[str, Any]):
    """Log application configuration at startup"""
    logger = get_logger()
    logger.info(
        "Application configuration loaded",
        category=LogCategory.SYSTEM,
        metadata={"configuration": config}
    )