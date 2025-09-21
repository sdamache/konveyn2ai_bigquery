"""Gap Analysis Package

Provides deterministic gap analysis for documentation quality assessment across
multiple artifact types including Kubernetes, FastAPI, COBOL, IRS layouts, and
MUMPS/FileMan.  Updated for Issue #5 to expose validation helpers to pipeline
callers.
"""

from .rule_loader import RuleLoader, load_rules, RuleValidationError
from .rule_validator import RuleConfigValidator
from .semantic_candidates import (
    SemanticCandidateBuilder,
    build_semantic_candidates,
)
from .semantic_support import (
    SemanticSupportFetcher,
    fetch_semantic_support,
)
from .metrics_runner import GapMetricRecord, GapMetricsRunner

__all__ = [
    "RuleConfigValidator",
    "RuleValidationError",
    "RuleLoader",
    "load_rules",
    "SemanticCandidateBuilder",
    "build_semantic_candidates",
    "SemanticSupportFetcher",
    "fetch_semantic_support",
    "GapMetricsRunner",
    "GapMetricRecord",
]
__version__ = "1.0.0"
