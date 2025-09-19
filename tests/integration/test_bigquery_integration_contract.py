"""Wrapper to execute the BigQuery integration contract defined in specs."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_spec_module(module_name: str, relative_path: str):
    """Dynamically load a spec module by relative path."""
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec module {module_name} from {full_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_spec_module = _load_spec_module(
    "specs_004_bigquery_memory_adapter_contracts_bigquery_integration_contract",
    "specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py",
)


class TestBigQueryIntegrationContract(
    _spec_module.TestBigQueryIntegrationContract
):
    """Execute BigQuery integration contract from specs."""


class TestBigQueryVectorIndexIntegration(
    _spec_module.TestBigQueryVectorIndexIntegration
):
    """Execute BigQueryVectorIndex integration contract."""
