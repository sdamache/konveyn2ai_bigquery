"""
Content normalization utility for consistent hashing
T021: Implements content preprocessing to ensure deterministic hashing across environments
"""

import json
import re
import unicodedata
from typing import Any, Optional

import yaml


class ContentNormalizer:
    """Normalizes content for consistent hashing and processing"""

    def __init__(
        self, preserve_semantics: bool = True, aggressive_whitespace: bool = False
    ):
        self.preserve_semantics = preserve_semantics
        self.aggressive_whitespace = aggressive_whitespace

        # Comment patterns for different languages
        self.comment_patterns = {
            "python": [
                r"#.*$",  # Python comments
                r'""".*?"""',  # Python docstrings (multiline)
                r"'''.*?'''",  # Python docstrings (single quotes)
            ],
            "yaml": [
                r"#.*$",  # YAML comments
            ],
            "javascript": [
                r"//.*$",  # Single line comments
                r"/\*.*?\*/",  # Multi-line comments
            ],
            "cobol": [
                r"^\s*\*.*$",  # COBOL comment lines
            ],
            "mumps": [
                r";.*$",  # MUMPS comments
            ],
        }

    def normalize_content(
        self, content: str, source_type: str, options: Optional[dict[str, bool]] = None
    ) -> str:
        """
        Normalize content for consistent processing

        Args:
            content: Raw content to normalize
            source_type: Source type for specific normalization rules
            options: Override normalization options

        Returns:
            Normalized content string
        """
        if not content:
            return ""

        # Apply options overrides
        if options is None:
            options = {}
        opts = {
            "remove_comments": options.get("remove_comments", False),
            "normalize_encoding": options.get("normalize_encoding", True),
            "normalize_line_endings": options.get("normalize_line_endings", True),
            "trim_whitespace": options.get("trim_whitespace", True),
            "sort_keys": options.get("sort_keys", False),
            "remove_metadata": options.get("remove_metadata", False),
        }

        normalized = content

        # Step 1: Unicode normalization
        if opts["normalize_encoding"]:
            normalized = self._normalize_encoding(normalized)

        # Step 2: Line ending normalization
        if opts["normalize_line_endings"]:
            normalized = self._normalize_line_endings(normalized)

        # Step 3: Source-specific normalization
        normalized = self._normalize_by_source_type(normalized, source_type, opts)

        # Step 4: General whitespace cleanup
        if opts["trim_whitespace"]:
            normalized = self._normalize_whitespace(normalized, source_type)

        # Step 5: Remove comments if requested
        if opts["remove_comments"]:
            normalized = self._remove_comments(normalized, source_type)

        return normalized

    def _normalize_encoding(self, content: str) -> str:
        """Normalize Unicode encoding for consistent processing"""
        # Normalize Unicode to NFC form
        normalized = unicodedata.normalize("NFC", content)

        # Replace common problematic characters
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u00a0": " ",  # Non-breaking space
        }

        for unicode_char, replacement in replacements.items():
            normalized = normalized.replace(unicode_char, replacement)

        return normalized

    def _normalize_line_endings(self, content: str) -> str:
        """Normalize line endings to Unix-style (LF)"""
        # Convert Windows (CRLF) and Mac (CR) line endings to Unix (LF)
        content = content.replace("\r\n", "\n")
        content = content.replace("\r", "\n")
        return content

    def _normalize_by_source_type(
        self, content: str, source_type: str, options: dict[str, bool]
    ) -> str:
        """Apply source-specific normalization rules"""
        source_type = source_type.lower()

        if source_type in ["kubernetes", "yaml"]:
            return self._normalize_yaml_content(content, options)
        elif source_type in ["fastapi", "python"]:
            return self._normalize_python_content(content, options)
        elif source_type == "cobol":
            return self._normalize_cobol_content(content, options)
        elif source_type == "irs":
            return self._normalize_irs_content(content, options)
        elif source_type == "mumps":
            return self._normalize_mumps_content(content, options)
        elif source_type in ["json"]:
            return self._normalize_json_content(content, options)
        else:
            return self._normalize_generic_content(content, options)

    def _normalize_yaml_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize YAML content while preserving structure"""
        try:
            if options.get("sort_keys", False):
                # Parse and re-serialize to sort keys
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    return yaml.dump(data, default_flow_style=False, sort_keys=True)
        except yaml.YAMLError:
            # If parsing fails, proceed with text-based normalization
            pass

        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace but preserve indentation
            line = line.rstrip()

            # Skip empty lines unless they're significant for YAML structure
            if line.strip() or self.preserve_semantics:
                normalized_lines.append(line)

        result = "\n".join(normalized_lines)

        # Remove metadata if requested
        if options.get("remove_metadata", False):
            result = self._remove_yaml_metadata(result)

        return result

    def _normalize_python_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize Python content while preserving syntax"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace but preserve indentation
            original_line = line
            line = line.rstrip()

            # Preserve empty lines for Python syntax
            if line or self.preserve_semantics:
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_cobol_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize COBOL content (position-sensitive)"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # COBOL is position-sensitive, so minimal normalization
            # Only remove trailing whitespace
            line = line.rstrip()
            if line or self.preserve_semantics:
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_irs_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize IRS content (fixed-width format)"""
        # IRS layouts are position-sensitive, minimal normalization
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Only remove trailing whitespace
            line = line.rstrip()
            if line.strip():  # Remove empty lines
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_mumps_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize MUMPS content"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()

            # MUMPS can be sensitive to spacing in some contexts
            if line.strip() or self.preserve_semantics:
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_json_content(self, content: str, options: dict[str, bool]) -> str:
        """Normalize JSON content"""
        try:
            # Parse and re-serialize for consistent formatting
            data = json.loads(content)
            return json.dumps(
                data,
                sort_keys=options.get("sort_keys", False),
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except json.JSONDecodeError:
            # Fall back to text normalization
            return self._normalize_generic_content(content, options)

    def _normalize_generic_content(self, content: str, options: dict[str, bool]) -> str:
        """Generic content normalization"""
        lines = content.split("\n")
        normalized_lines = []

        for line in lines:
            line = line.rstrip()
            if line.strip() or self.preserve_semantics:
                normalized_lines.append(line)

        return "\n".join(normalized_lines)

    def _normalize_whitespace(self, content: str, source_type: str) -> str:
        """Normalize whitespace based on source type requirements"""
        if self.aggressive_whitespace:
            # Aggressive normalization - collapse all whitespace
            lines = content.split("\n")
            normalized_lines = []

            for line in lines:
                # Collapse multiple spaces to single space (except for indentation)
                if line.strip():
                    leading_spaces = len(line) - len(line.lstrip())
                    content_part = re.sub(r"\s+", " ", line.strip())
                    normalized_lines.append(" " * leading_spaces + content_part)
                else:
                    normalized_lines.append("")

            return "\n".join(normalized_lines)
        else:
            # Conservative whitespace normalization
            lines = content.split("\n")
            normalized_lines = []

            prev_empty = False
            for line in lines:
                is_empty = not line.strip()

                # Remove consecutive empty lines (but keep one)
                if not (is_empty and prev_empty):
                    normalized_lines.append(line.rstrip())

                prev_empty = is_empty

            return "\n".join(normalized_lines)

    def _remove_comments(self, content: str, source_type: str) -> str:
        """Remove comments based on source type"""
        source_type = source_type.lower()

        if source_type in self.comment_patterns:
            patterns = self.comment_patterns[source_type]

            for pattern in patterns:
                if ".*?" in pattern:  # Multiline pattern
                    content = re.sub(pattern, "", content, flags=re.DOTALL)
                else:  # Single line pattern
                    content = re.sub(pattern, "", content, flags=re.MULTILINE)

        return content

    def _remove_yaml_metadata(self, content: str) -> str:
        """Remove YAML metadata fields that might vary"""
        metadata_fields = [
            r"^\s*creationTimestamp:.*$",
            r"^\s*resourceVersion:.*$",
            r"^\s*uid:.*$",
            r"^\s*generation:.*$",
            r"^\s*timestamp:.*$",
        ]

        lines = content.split("\n")
        filtered_lines = []

        for line in lines:
            is_metadata = any(
                re.match(pattern, line, re.IGNORECASE) for pattern in metadata_fields
            )
            if not is_metadata:
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def extract_significant_tokens(self, content: str, source_type: str) -> list[str]:
        """Extract significant tokens for fuzzy matching"""
        # Normalize content first
        normalized = self.normalize_content(
            content, source_type, {"remove_comments": True, "trim_whitespace": True}
        )

        # Extract tokens based on source type
        if source_type.lower() in ["python", "fastapi"]:
            return self._extract_python_tokens(normalized)
        elif source_type.lower() in ["yaml", "kubernetes"]:
            return self._extract_yaml_tokens(normalized)
        elif source_type.lower() == "cobol":
            return self._extract_cobol_tokens(normalized)
        elif source_type.lower() == "irs":
            return self._extract_irs_tokens(normalized)
        elif source_type.lower() == "mumps":
            return self._extract_mumps_tokens(normalized)
        else:
            return self._extract_generic_tokens(normalized)

    def _extract_python_tokens(self, content: str) -> list[str]:
        """Extract significant Python tokens"""
        tokens = []

        # Function definitions
        tokens.extend(re.findall(r"def\s+(\w+)", content))
        tokens.extend(re.findall(r"class\s+(\w+)", content))

        # FastAPI decorators
        tokens.extend(re.findall(r"@app\.(\w+)", content))

        # Import statements
        tokens.extend(re.findall(r"from\s+(\w+)", content))
        tokens.extend(re.findall(r"import\s+(\w+)", content))

        return tokens

    def _extract_yaml_tokens(self, content: str) -> list[str]:
        """Extract significant YAML tokens"""
        tokens = []

        # YAML keys
        tokens.extend(re.findall(r"^(\w+):", content, re.MULTILINE))

        # Kubernetes-specific
        tokens.extend(re.findall(r"kind:\s*(\w+)", content))
        tokens.extend(re.findall(r"apiVersion:\s*([^\s]+)", content))

        return tokens

    def _extract_cobol_tokens(self, content: str) -> list[str]:
        """Extract significant COBOL tokens"""
        tokens = []

        # Level numbers and names
        tokens.extend(re.findall(r"^\s*(\d{2})\s+([A-Z0-9_-]+)", content, re.MULTILINE))

        # PIC clauses
        tokens.extend(re.findall(r"PIC\s+([A-Z0-9\(\)]+)", content))

        return [str(token) for token in tokens if token]

    def _extract_irs_tokens(self, content: str) -> list[str]:
        """Extract significant IRS tokens"""
        tokens = []

        # Field positions
        tokens.extend(re.findall(r"(\d{3}-\d{3})", content))

        # Field names
        tokens.extend(re.findall(r"\d{3}-\d{3}\s+\d+\s+[A-Z]\s+([A-Z\s]+)", content))

        return tokens

    def _extract_mumps_tokens(self, content: str) -> list[str]:
        """Extract significant MUMPS tokens"""
        tokens = []

        # Global references
        tokens.extend(re.findall(r"\^([A-Z0-9_]+)", content))

        # Field numbers
        tokens.extend(re.findall(r"^\s*(\d+)\s+", content, re.MULTILINE))

        return tokens

    def _extract_generic_tokens(self, content: str) -> list[str]:
        """Extract generic significant tokens"""
        # Simple word tokenization
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", content)
        return [token for token in tokens if len(token) > 2]

    def compare_content_similarity(
        self, content1: str, content2: str, source_type: str
    ) -> dict[str, float]:
        """Compare content similarity for deduplication"""
        # Normalize both contents
        norm1 = self.normalize_content(content1, source_type)
        norm2 = self.normalize_content(content2, source_type)

        # Exact match after normalization
        exact_match = norm1 == norm2

        # Token-based similarity
        tokens1 = set(self.extract_significant_tokens(content1, source_type))
        tokens2 = set(self.extract_significant_tokens(content2, source_type))

        if tokens1 or tokens2:
            token_similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            token_similarity = 1.0 if exact_match else 0.0

        # Line-based similarity
        lines1 = set(line.strip() for line in norm1.split("\n") if line.strip())
        lines2 = set(line.strip() for line in norm2.split("\n") if line.strip())

        if lines1 or lines2:
            line_similarity = len(lines1 & lines2) / len(lines1 | lines2)
        else:
            line_similarity = 1.0 if exact_match else 0.0

        return {
            "exact_match": 1.0 if exact_match else 0.0,
            "token_similarity": token_similarity,
            "line_similarity": line_similarity,
            "overall_similarity": (token_similarity + line_similarity) / 2,
        }


# Factory function
def create_normalizer(
    preserve_semantics: bool = True, aggressive_whitespace: bool = False
) -> ContentNormalizer:
    """Create content normalizer with specified options"""
    return ContentNormalizer(preserve_semantics, aggressive_whitespace)


# Utility functions
def normalize_for_hashing(content: str, source_type: str) -> str:
    """Quick normalization for consistent hashing"""
    normalizer = create_normalizer(preserve_semantics=False, aggressive_whitespace=True)
    return normalizer.normalize_content(
        content,
        source_type,
        {
            "remove_comments": True,
            "normalize_encoding": True,
            "normalize_line_endings": True,
            "trim_whitespace": True,
        },
    )


def normalize_for_display(content: str, source_type: str) -> str:
    """Normalization for human-readable display"""
    normalizer = create_normalizer(preserve_semantics=True, aggressive_whitespace=False)
    return normalizer.normalize_content(
        content,
        source_type,
        {
            "remove_comments": False,
            "normalize_encoding": True,
            "normalize_line_endings": True,
            "trim_whitespace": True,
        },
    )


def detect_content_changes(
    original: str, modified: str, source_type: str
) -> dict[str, Any]:
    """Detect what changed between two versions of content"""
    normalizer = create_normalizer()

    # Normalize both versions
    norm_original = normalizer.normalize_content(original, source_type)
    norm_modified = normalizer.normalize_content(modified, source_type)

    # Get similarity metrics
    similarity = normalizer.compare_content_similarity(original, modified, source_type)

    # Analyze changes
    orig_lines = set(norm_original.split("\n"))
    mod_lines = set(norm_modified.split("\n"))

    added_lines = mod_lines - orig_lines
    removed_lines = orig_lines - mod_lines

    return {
        "has_changes": norm_original != norm_modified,
        "similarity_score": similarity["overall_similarity"],
        "lines_added": len(added_lines),
        "lines_removed": len(removed_lines),
        "lines_changed": len(added_lines) + len(removed_lines),
        "change_summary": {
            "added": list(added_lines)[:5],  # First 5 additions
            "removed": list(removed_lines)[:5],  # First 5 removals
        },
    }
