"""
Datetime serialization utilities for KonveyN2AI BigQuery project.

This module provides standardized datetime handling across all components,
ensuring consistent JSON serialization and BigQuery compatibility.
"""

import json
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def datetime_to_iso(dt: datetime) -> str:
    """Convert datetime object to ISO string."""
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def safe_datetime_serializer(obj: Any) -> str:
    """JSON serializer that handles datetime objects."""
    if isinstance(obj, datetime):
        return datetime_to_iso(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_dumps_safe(data: Any, **kwargs) -> str:
    """JSON dumps with datetime serialization support."""
    return json.dumps(data, default=safe_datetime_serializer, **kwargs)


def prepare_metadata_for_json(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare metadata dictionary for JSON serialization by converting datetime objects.

    This function recursively traverses the metadata and converts any datetime
    objects to ISO format strings, making it safe for JSON serialization.
    """
    if not isinstance(metadata, dict):
        return metadata

    cleaned_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, datetime):
            cleaned_metadata[key] = datetime_to_iso(value)
        elif isinstance(value, dict):
            cleaned_metadata[key] = prepare_metadata_for_json(value)
        elif isinstance(value, list):
            cleaned_metadata[key] = [
                prepare_metadata_for_json(item) if isinstance(item, dict)
                else datetime_to_iso(item) if isinstance(item, datetime)
                else item
                for item in value
            ]
        else:
            cleaned_metadata[key] = value

    return cleaned_metadata


# Make commonly used functions available at module level
__all__ = [
    'now_iso',
    'datetime_to_iso',
    'safe_datetime_serializer',
    'json_dumps_safe',
    'prepare_metadata_for_json'
]