"""Rule configuration loader that wires runtime validation (Issue #5).

This module provides a deterministic way to load rule YAML files from
``configs/rules`` while enforcing the guardrails captured in
``RuleConfigValidator``.  It covers Task T001 (schema-level checks) and extends
into Task T010 by being the single ingestion path the pipeline can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict, Any

import yaml

from .rule_validator import RuleConfigValidator, RuleValidationError


DEFAULT_RULES_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"


@dataclass
class RuleLoader:
    """Load and validate Issue #5 rule definitions from disk."""

    rules_dir: Path = DEFAULT_RULES_DIR
    validator: RuleConfigValidator = field(default_factory=RuleConfigValidator)

    def load(self) -> List[Dict[str, Any]]:
        """Load every rule file under ``rules_dir`` and validate each document."""

        if not self.rules_dir.exists():
            raise FileNotFoundError(f"Rules directory not found: {self.rules_dir}")

        rules: List[Dict[str, Any]] = []
        for yaml_path in sorted(self.rules_dir.rglob("*.yaml")):
            rules.extend(self._load_rule_file(yaml_path))
        return rules

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_rule_file(self, yaml_path: Path) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        with yaml_path.open("r", encoding="utf-8") as handle:
            loaded_docs = list(yaml.safe_load_all(handle))

        for doc_index, doc in enumerate(loaded_docs, start=1):
            if doc is None:
                continue
            if not isinstance(doc, dict):
                raise RuleValidationError(
                    f"Rule document #{doc_index} in {yaml_path} must be a mapping"
                )

            self.validator.validate(doc)
            documents.append(doc)

        return documents


def load_rules(rules_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Functional helper so callers do not need to instantiate ``RuleLoader``."""

    loader = RuleLoader(rules_dir=rules_dir or DEFAULT_RULES_DIR)
    return loader.load()


__all__ = ["RuleLoader", "load_rules", "RuleValidationError"]
