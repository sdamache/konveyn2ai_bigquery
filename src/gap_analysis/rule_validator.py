"""Rule configuration validation helpers for Issue #5 (T001/T005/T010).

The JSON Schema enforces structural constraints, but we still need deterministic
runtime checks for values that the schema cannot express (for example, ensuring
confidence weights sum to 1.0 within a tolerance).  These helpers provide a
lightweight validation layer that the pipeline can reuse before executing rule
SQL against BigQuery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import math


class RuleValidationError(ValueError):
    """Raised when a rule configuration violates deterministic guardrails."""


@dataclass
class RuleConfigValidator:
    """Validate rule dictionaries emitted by the configuration loader.

    Notes
    -----
    * Addresses Issue #5 Task T001 by ensuring confidence weight maps sum to 1.
    * Complements Task T005 by checking that required fields exist before rules
      are evaluated.
    * Provides the backbone for Task T010 (rule linting) once integrated with
      command-line tooling.
    """

    weight_tolerance: float = 1e-6

    def validate(self, rule_config: Dict[str, Any]) -> None:
        """Run all registered validators against a single rule definition."""

        self._validate_confidence_weights(rule_config)
        self._validate_severity(rule_config)
        self._validate_required_strings(rule_config)
        self._validate_semantic_probe(rule_config)

    # ------------------------------------------------------------------
    # Individual validation helpers
    # ------------------------------------------------------------------
    def _validate_confidence_weights(self, rule_config: Dict[str, Any]) -> None:
        """Ensure confidence weights exist, are bounded, and sum to 1.0."""

        weights = rule_config.get("confidence_weights")
        if not isinstance(weights, dict) or not weights:
            raise RuleValidationError(
                "confidence_weights must be a non-empty mapping of component -> weight"
            )

        total = 0.0
        for component, value in weights.items():
            if not isinstance(value, (int, float)):
                raise RuleValidationError(
                    f"Weight for component '{component}' must be numeric"
                )
            if not 0.0 <= value <= 1.0:
                raise RuleValidationError(
                    f"Weight for component '{component}' must be between 0 and 1"
                )
            total += float(value)

        if not math.isclose(
            total, 1.0, rel_tol=self.weight_tolerance, abs_tol=self.weight_tolerance
        ):
            raise RuleValidationError(
                f"confidence_weights must sum to 1.0 (got {total:.6f})"
            )

    def _validate_severity(self, rule_config: Dict[str, Any]) -> None:
        """Check severity bounds at runtime to defend against schema drift."""

        severity = rule_config.get("severity")
        if severity is None:
            raise RuleValidationError("severity field is required")

        if not isinstance(severity, int) or not 1 <= severity <= 5:
            raise RuleValidationError("severity must be an integer between 1 and 5")

    def _validate_required_strings(self, rule_config: Dict[str, Any]) -> None:
        """Ensure critical string fields are populated for deterministic evaluation."""

        for field_name in (
            "rule_name",
            "artifact_type",
            "evaluation_sql",
            "suggested_fix_template",
        ):
            value = rule_config.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise RuleValidationError(f"{field_name} must be a non-empty string")

    def _validate_semantic_probe(self, rule_config: Dict[str, Any]) -> None:
        """Validate optional semantic probe configuration."""

        probe = rule_config.get("semantic_probe")
        if probe is None:
            return

        if not isinstance(probe, dict):
            raise RuleValidationError("semantic_probe must be an object if provided")

        query_text = probe.get("query_text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise RuleValidationError("semantic_probe.query_text must be a non-empty string")
        if len(query_text.strip()) < 10:
            raise RuleValidationError(
                "semantic_probe.query_text must be at least 10 characters to yield meaningful search context"
            )

        top_k = probe.get("top_k", 5)
        if not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise RuleValidationError("semantic_probe.top_k must be an integer between 1 and 50")

        threshold = probe.get("similarity_threshold")
        if threshold is not None:
            if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
                raise RuleValidationError(
                    "semantic_probe.similarity_threshold must be a number between 0 and 1"
                )

        weight = probe.get("weight")
        if weight is not None:
            if not isinstance(weight, (int, float)) or not 0.0 <= float(weight) <= 1.0:
                raise RuleValidationError(
                    "semantic_probe.weight must be a number between 0 and 1 when provided"
                )

        vector_table = probe.get("vector_table")
        if vector_table is not None and (not isinstance(vector_table, str) or not vector_table.strip()):
            raise RuleValidationError(
                "semantic_probe.vector_table, when provided, must be a non-empty string"
            )


__all__ = ["RuleConfigValidator", "RuleValidationError"]
