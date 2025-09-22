"""
Confidence Calculation Engine for Gap Analysis Rules

This module implements the deterministic confidence scoring algorithm that quantifies
how certain we are about rule evaluation results based on completeness, quality,
and penalty factors.
"""

import re
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceComponent(Enum):
    """Components that contribute to confidence scoring."""

    FIELD_COMPLETENESS = "field_completeness"
    CONTENT_QUALITY = "content_quality"
    STRUCTURE_QUALITY = "structure_quality"
    VOCABULARY_QUALITY = "vocabulary_quality"
    SEMANTIC_SUPPORT = "semantic_support"
    CRITICAL_PENALTIES = "critical_penalties"
    SECURITY_PENALTIES = "security_penalties"
    COMPLIANCE_PENALTIES = "compliance_penalties"


@dataclass
class ConfidenceWeights:
    """Weights for confidence calculation components (Issue #5 T006)."""

    field_completeness: float = 0.6
    content_quality: float = 0.4
    semantic_support_weight: float = 0.0
    structure_weight: float = 0.3
    vocabulary_weight: float = 0.3
    length_weight: float = 0.4
    critical_penalty: float = 0.20
    security_penalty: float = 0.15
    compliance_penalty: float = 0.10

    def validate(self) -> None:
        """Validate that weights are in valid ranges and properly normalized."""

        if not (0 <= self.field_completeness <= 1):
            raise ValueError(
                f"field_completeness must be [0,1], got {self.field_completeness}"
            )
        if not (0 <= self.content_quality <= 1):
            raise ValueError(
                f"content_quality must be [0,1], got {self.content_quality}"
            )
        if not math.isclose(
            self.field_completeness + self.content_quality, 1.0, rel_tol=1e-9
        ):
            raise ValueError("field_completeness + content_quality must sum to 1.0")

        if not (0 <= self.semantic_support_weight <= 1):
            raise ValueError(
                f"semantic_support_weight must be [0,1], got {self.semantic_support_weight}"
            )

        if not (0 <= self.length_weight <= 1):
            raise ValueError(f"length_weight must be [0,1], got {self.length_weight}")
        if not (0 <= self.structure_weight <= 1):
            raise ValueError(
                f"structure_weight must be [0,1], got {self.structure_weight}"
            )
        if not (0 <= self.vocabulary_weight <= 1):
            raise ValueError(
                f"vocabulary_weight must be [0,1], got {self.vocabulary_weight}"
            )
        if not math.isclose(
            self.length_weight + self.structure_weight + self.vocabulary_weight,
            1.0,
            rel_tol=1e-9,
        ):
            raise ValueError(
                "length_weight + structure_weight + vocabulary_weight must sum to 1.0"
            )

        for label, value in (
            ("critical_penalty", self.critical_penalty),
            ("security_penalty", self.security_penalty),
            ("compliance_penalty", self.compliance_penalty),
        ):
            if not (0 <= value <= 1):
                raise ValueError(f"{label} must be [0,1], got {value}")


@dataclass
class FieldAnalysis:
    """Analysis results for field presence and quality."""

    required_fields_present: int
    total_required_fields: int
    optional_fields_present: int
    total_optional_fields: int
    critical_missing_fields: int
    field_values: Dict[str, Any]

    @property
    def completeness_ratio(self) -> float:
        """Calculate field completeness ratio."""
        if self.total_required_fields == 0:
            return 1.0
        return self.required_fields_present / self.total_required_fields


@dataclass
class ContentQuality:
    """Content quality assessment results."""

    total_content_length: int
    meaningful_content_length: int
    min_required_length: int
    has_proper_formatting: bool
    has_consistent_style: bool
    has_domain_vocabulary: bool
    has_placeholders: bool
    structure_score: float
    vocabulary_score: float

    @property
    def length_score(self) -> float:
        """Calculate length-based quality score."""
        if self.min_required_length == 0:
            return 1.0
        return min(1.0, self.meaningful_content_length / self.min_required_length)


@dataclass
class PenaltyAssessment:
    """Assessment of penalty factors that reduce confidence."""

    critical_violations: List[str]
    security_violations: List[str]
    compliance_violations: List[str]
    format_violations: List[str]

    @property
    def total_critical_penalties(self) -> int:
        """Count of critical violations."""
        return len(self.critical_violations)

    @property
    def total_security_penalties(self) -> int:
        """Count of security violations."""
        return len(self.security_violations)

    @property
    def total_compliance_penalties(self) -> int:
        """Count of compliance violations."""
        return len(self.compliance_violations)


@dataclass
class ConfidenceResult:
    """Complete confidence calculation result."""

    final_confidence: float
    base_completeness: float
    quality_multiplier: float
    total_penalties: float
    component_scores: Dict[ConfidenceComponent, float]
    breakdown: Dict[str, float]

    def round_precision(self, decimals: int = 3) -> "ConfidenceResult":
        """Round confidence to specified decimal places for consistency."""
        return ConfidenceResult(
            final_confidence=round(self.final_confidence, decimals),
            base_completeness=round(self.base_completeness, decimals),
            quality_multiplier=round(self.quality_multiplier, decimals),
            total_penalties=round(self.total_penalties, decimals),
            component_scores={
                k: round(v, decimals) for k, v in self.component_scores.items()
            },
            breakdown={k: round(v, decimals) for k, v in self.breakdown.items()},
        )


class ConfidenceCalculator:
    """
    Deterministic confidence calculation engine.

    Implements the confidence scoring algorithm:
    confidence = (base_completeness * base_weight)
                  + (quality_multiplier * quality_weight)
                  - penalty_deductions

    All calculations are deterministic and reproducible.
    """

    def __init__(self, weights: Optional[ConfidenceWeights] = None):
        """Initialize calculator with optional custom weights."""
        self.weights = weights or ConfidenceWeights()
        self.weights.validate()

        # Precompiled regex patterns for performance
        self._domain_patterns = {
            "kubernetes": re.compile(
                r"\b(pod|service|deployment|ingress|configmap|secret|namespace)\b",
                re.IGNORECASE,
            ),
            "fastapi": re.compile(
                r"\b(endpoint|route|schema|model|dependency|middleware|response)\b",
                re.IGNORECASE,
            ),
            "cobol": re.compile(
                r"\b(pic|copy|section|division|procedure|working|storage)\b",
                re.IGNORECASE,
            ),
            "irs": re.compile(
                r"\b(field|record|position|length|format|required|valid)\b",
                re.IGNORECASE,
            ),
            "mumps": re.compile(
                r"\b(routine|fileman|input|output|transform|help|prompt)\b",
                re.IGNORECASE,
            ),
        }

        self._placeholder_patterns = re.compile(
            r"\b(todo|fixme|tbd|placeholder|replace|change|update)\b", re.IGNORECASE
        )

    def calculate_confidence(
        self,
        field_analysis: FieldAnalysis,
        content_quality: ContentQuality,
        penalty_assessment: PenaltyAssessment,
        artifact_type: str = "unknown",
        semantic_similarity: float | None = None,
    ) -> ConfidenceResult:
        """
        Calculate final confidence score using the weighted algorithm (Issue #5 T006).

        Args:
            field_analysis: Field presence and completeness analysis
            content_quality: Content quality metrics
            penalty_assessment: Penalty factors
            artifact_type: Type of artifact for domain-specific scoring

        Returns:
            Complete confidence calculation result
        """
        # Step 1: Calculate base completeness
        base_completeness = self._calculate_base_completeness(field_analysis)

        # Step 2: Calculate quality multiplier
        quality_multiplier = self._calculate_quality_multiplier(
            content_quality, artifact_type
        )

        # Step 3: Calculate penalty deductions
        total_penalties = self._calculate_penalties(penalty_assessment)

        # Step 4: Apply final formula with bounds checking. Weighting makes the
        # configurable split between raw completeness and quality explicit for
        # downstream consumers.
        weighted_base = base_completeness * self.weights.field_completeness
        weighted_quality = quality_multiplier * self.weights.content_quality
        semantic_component = 0.0
        if semantic_similarity is not None:
            semantic_component = (
                max(0.0, min(1.0, semantic_similarity))
                * self.weights.semantic_support_weight
            )

        raw_confidence = (
            weighted_base + weighted_quality + semantic_component - total_penalties
        )
        final_confidence = max(0.0, min(1.0, raw_confidence))

        # Step 5: Build component breakdown
        component_scores = {
            ConfidenceComponent.FIELD_COMPLETENESS: weighted_base,
            ConfidenceComponent.CONTENT_QUALITY: weighted_quality,
            ConfidenceComponent.STRUCTURE_QUALITY: content_quality.structure_score,
            ConfidenceComponent.VOCABULARY_QUALITY: content_quality.vocabulary_score,
            ConfidenceComponent.SEMANTIC_SUPPORT: semantic_component,
            ConfidenceComponent.CRITICAL_PENALTIES: penalty_assessment.total_critical_penalties
            * self.weights.critical_penalty,
            ConfidenceComponent.SECURITY_PENALTIES: penalty_assessment.total_security_penalties
            * self.weights.security_penalty,
            ConfidenceComponent.COMPLIANCE_PENALTIES: penalty_assessment.total_compliance_penalties
            * self.weights.compliance_penalty,
        }

        breakdown = {
            "base_completeness": base_completeness,
            "quality_multiplier": quality_multiplier,
            "weighted_base": weighted_base,
            "weighted_quality": weighted_quality,
            "length_score": content_quality.length_score,
            "structure_score": content_quality.structure_score,
            "vocabulary_score": content_quality.vocabulary_score,
            "semantic_component": semantic_component,
            "critical_penalties": penalty_assessment.total_critical_penalties,
            "security_penalties": penalty_assessment.total_security_penalties,
            "compliance_penalties": penalty_assessment.total_compliance_penalties,
            "total_penalties": total_penalties,
            "raw_confidence": raw_confidence,
            "final_confidence": final_confidence,
        }

        result = ConfidenceResult(
            final_confidence=final_confidence,
            base_completeness=base_completeness,
            quality_multiplier=quality_multiplier,
            total_penalties=total_penalties,
            component_scores=component_scores,
            breakdown=breakdown,
        )

        return result.round_precision(3)

    def _calculate_base_completeness(self, field_analysis: FieldAnalysis) -> float:
        """Calculate base completeness score from field analysis."""
        if field_analysis.total_required_fields == 0:
            return 1.0

        return (
            field_analysis.required_fields_present
            / field_analysis.total_required_fields
        )

    def _calculate_quality_multiplier(
        self, content_quality: ContentQuality, artifact_type: str
    ) -> float:
        """Calculate quality multiplier from content analysis."""
        # Length component
        length_score = content_quality.length_score

        # Structure component
        structure_score = self._calculate_structure_score(content_quality)

        # Vocabulary component (domain-specific)
        vocabulary_score = self._calculate_vocabulary_score(
            content_quality, artifact_type
        )

        # Weighted combination
        quality_multiplier = (
            length_score * self.weights.length_weight
            + structure_score * self.weights.structure_weight
            + vocabulary_score * self.weights.vocabulary_weight
        )

        return min(1.0, quality_multiplier)

    def _calculate_structure_score(self, content_quality: ContentQuality) -> float:
        """Calculate structure quality score."""
        structure_factors = []

        if content_quality.has_proper_formatting:
            structure_factors.append(1.0)
        else:
            structure_factors.append(0.0)

        if content_quality.has_consistent_style:
            structure_factors.append(1.0)
        else:
            structure_factors.append(0.0)

        # Average of structure factors
        if structure_factors:
            return sum(structure_factors) / len(structure_factors)
        return 0.0

    def _calculate_vocabulary_score(
        self, content_quality: ContentQuality, artifact_type: str
    ) -> float:
        """Calculate vocabulary quality score with domain-specific patterns."""
        vocab_score = 0.0

        # Domain vocabulary presence (50% weight)
        if content_quality.has_domain_vocabulary:
            vocab_score += 0.5

        # No placeholder text (30% weight)
        if not content_quality.has_placeholders:
            vocab_score += 0.3

        # Meaningful content vs total length ratio (20% weight)
        if content_quality.total_content_length > 0:
            meaning_ratio = (
                content_quality.meaningful_content_length
                / content_quality.total_content_length
            )
            vocab_score += 0.2 * meaning_ratio

        return min(1.0, vocab_score)

    def _calculate_penalties(self, penalty_assessment: PenaltyAssessment) -> float:
        """Calculate total penalty deductions."""
        critical_penalties = (
            penalty_assessment.total_critical_penalties * self.weights.critical_penalty
        )
        security_penalties = (
            penalty_assessment.total_security_penalties * self.weights.security_penalty
        )
        compliance_penalties = (
            penalty_assessment.total_compliance_penalties
            * self.weights.compliance_penalty
        )

        total_penalties = critical_penalties + security_penalties + compliance_penalties

        # Cap total penalties at 1.0 to prevent negative confidence
        return min(1.0, total_penalties)

    def analyze_content_quality(
        self, text_content: str, artifact_type: str, min_length: int = 10
    ) -> ContentQuality:
        """
        Analyze content quality for confidence calculation.

        Args:
            text_content: Raw content to analyze
            artifact_type: Type of artifact for domain-specific analysis
            min_length: Minimum required content length

        Returns:
            ContentQuality analysis results
        """
        # Length analysis
        total_length = len(text_content)
        meaningful_length = len(text_content.strip())

        # Formatting analysis
        has_proper_formatting = self._check_formatting(text_content, artifact_type)
        has_consistent_style = self._check_style_consistency(
            text_content, artifact_type
        )

        # Vocabulary analysis
        has_domain_vocab = self._check_domain_vocabulary(text_content, artifact_type)
        has_placeholders = self._placeholder_patterns.search(text_content) is not None

        # Structure and vocabulary scores
        structure_score = self._calculate_structure_score_from_content(
            text_content, artifact_type
        )
        vocabulary_score = 1.0 if has_domain_vocab and not has_placeholders else 0.5

        return ContentQuality(
            total_content_length=total_length,
            meaningful_content_length=meaningful_length,
            min_required_length=min_length,
            has_proper_formatting=has_proper_formatting,
            has_consistent_style=has_consistent_style,
            has_domain_vocabulary=has_domain_vocab,
            has_placeholders=has_placeholders,
            structure_score=structure_score,
            vocabulary_score=vocabulary_score,
        )

    def _check_formatting(self, content: str, artifact_type: str) -> bool:
        """Check if content has proper formatting for its type."""
        content = content.strip()

        if artifact_type == "kubernetes":
            # Check for YAML structure
            return ":" in content and ("\n" in content or len(content.split(":")) > 1)
        elif artifact_type == "fastapi":
            # Check for Python structure
            return "def " in content or "class " in content or "@" in content
        elif artifact_type == "cobol":
            # Check for COBOL structure
            return "PIC " in content.upper() or "COPY " in content.upper()
        elif artifact_type == "irs":
            # Check for structured field definitions
            return "Field" in content or "Position" in content or "Length" in content
        elif artifact_type == "mumps":
            # Check for MUMPS routine structure
            return ";" in content or "^" in content

        return len(content) > 0

    def _check_style_consistency(self, content: str, artifact_type: str) -> bool:
        """Check for consistent style within content."""
        lines = content.split("\n")
        if len(lines) < 2:
            return True

        # Check indentation consistency
        indented_lines = [
            line for line in lines if line.strip() and line.startswith(" ")
        ]
        if len(indented_lines) > 1:
            indent_levels = [len(line) - len(line.lstrip()) for line in indented_lines]
            # Check if indentation follows a pattern (multiples of 2 or 4)
            consistent_indent = all(
                level % 2 == 0 or level % 4 == 0 for level in indent_levels
            )
            return consistent_indent

        return True

    def _check_domain_vocabulary(self, content: str, artifact_type: str) -> bool:
        """Check if content uses appropriate domain vocabulary."""
        if artifact_type in self._domain_patterns:
            pattern = self._domain_patterns[artifact_type]
            return pattern.search(content) is not None
        return True  # Unknown types pass vocabulary check

    def _calculate_structure_score_from_content(
        self, content: str, artifact_type: str
    ) -> float:
        """Calculate structure score directly from content."""
        score = 0.0

        # Basic structure presence (0.4 weight)
        if self._check_formatting(content, artifact_type):
            score += 0.4

        # Style consistency (0.3 weight)
        if self._check_style_consistency(content, artifact_type):
            score += 0.3

        # Content organization (0.3 weight)
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) > 1:
            score += 0.3
        elif len(non_empty_lines) == 1:
            score += 0.15

        return min(1.0, score)


# Example usage and testing
def example_confidence_calculations():
    """Demonstrate confidence calculation with examples."""
    calculator = ConfidenceCalculator()

    # Example 1: High-quality Kubernetes deployment
    field_analysis = FieldAnalysis(
        required_fields_present=3,
        total_required_fields=3,
        optional_fields_present=2,
        total_optional_fields=3,
        critical_missing_fields=0,
        field_values={
            "description": "Production nginx server",
            "app_label": "nginx",
            "resource_limits": "512Mi",
        },
    )

    content_quality = ContentQuality(
        total_content_length=200,
        meaningful_content_length=190,
        min_required_length=50,
        has_proper_formatting=True,
        has_consistent_style=True,
        has_domain_vocabulary=True,
        has_placeholders=False,
        structure_score=0.9,
        vocabulary_score=0.8,
    )

    penalty_assessment = PenaltyAssessment(
        critical_violations=[],
        security_violations=[],
        compliance_violations=[],
        format_violations=[],
    )

    result = calculator.calculate_confidence(
        field_analysis, content_quality, penalty_assessment, "kubernetes"
    )

    print("Example 1 - High Quality Kubernetes:")
    print(f"Final Confidence: {result.final_confidence}")
    print(f"Base Completeness: {result.base_completeness}")
    print(f"Quality Multiplier: {result.quality_multiplier}")
    print(f"Total Penalties: {result.total_penalties}")
    print()

    # Example 2: Poor-quality FastAPI endpoint
    field_analysis2 = FieldAnalysis(
        required_fields_present=1,
        total_required_fields=3,
        optional_fields_present=0,
        total_optional_fields=2,
        critical_missing_fields=2,
        field_values={"function_name": "get_user"},
    )

    content_quality2 = ContentQuality(
        total_content_length=30,
        meaningful_content_length=25,
        min_required_length=50,
        has_proper_formatting=True,
        has_consistent_style=True,
        has_domain_vocabulary=False,
        has_placeholders=True,
        structure_score=0.4,
        vocabulary_score=0.2,
    )

    penalty_assessment2 = PenaltyAssessment(
        critical_violations=["missing_docstring", "missing_response_model"],
        security_violations=[],
        compliance_violations=[],
        format_violations=[],
    )

    result2 = calculator.calculate_confidence(
        field_analysis2, content_quality2, penalty_assessment2, "fastapi"
    )

    print("Example 2 - Poor Quality FastAPI:")
    print(f"Final Confidence: {result2.final_confidence}")
    print(f"Base Completeness: {result2.base_completeness}")
    print(f"Quality Multiplier: {result2.quality_multiplier}")
    print(f"Total Penalties: {result2.total_penalties}")


if __name__ == "__main__":
    example_confidence_calculations()
