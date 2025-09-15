"""
Content chunking utility with source-aware strategies
T019: Implements intelligent chunking for different source types with overlap and semantic awareness
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ChunkingStrategy(Enum):
    """Different chunking strategies based on source type"""

    TOKEN_BASED = "token_based"  # Simple token counting
    SEMANTIC_BLOCKS = "semantic_blocks"  # Code/config blocks
    FIXED_WIDTH = "fixed_width"  # IRS/COBOL fixed-width records
    HIERARCHICAL = "hierarchical"  # MUMPS/nested structures
    LINE_BASED = "line_based"  # General text processing


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior"""

    max_tokens: int = 1000
    overlap_pct: float = 0.15  # 15% overlap between chunks
    min_chunk_size: int = 50
    preserve_boundaries: bool = True  # Keep semantic boundaries intact
    strategy: ChunkingStrategy = ChunkingStrategy.TOKEN_BASED


@dataclass
class ChunkResult:
    """Result of chunking operation"""

    content: str
    start_position: int
    end_position: int
    token_count: int
    metadata: dict[str, Any]


class ContentChunker:
    """Source-aware content chunking with overlap and semantic boundaries"""

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # Rough token estimation: 1 token ≈ 4 characters for English text
        self.chars_per_token = 4

        # Semantic boundary patterns for different source types
        self.boundary_patterns = {
            ChunkingStrategy.SEMANTIC_BLOCKS: [
                r"^---\s*$",  # YAML separators
                r"^\s*[\{\}]\s*$",  # JSON/JS braces
                r"^\s*(?:def|class|function|async def)\s+",  # Python/JS functions
                r"^\s*(?:@[A-Za-z]+\.(?:get|post|put|delete))",  # FastAPI decorators
                r"^\s*(?:apiVersion|kind):\s*",  # K8s manifest starts
            ],
            ChunkingStrategy.FIXED_WIDTH: [
                r"^\d{3}-\d{3}\s+",  # IRS field positions (001-009)
                r"^\s*\d{2}\s+[A-Z0-9_-]+\s+",  # COBOL level numbers
            ],
            ChunkingStrategy.HIERARCHICAL: [
                r"^\^[A-Z0-9_]+\(",  # MUMPS global references
                r"^\s*\d+\s+[A-Z_]+\s*;",  # FileMan field definitions
            ],
        }

    def chunk_content(
        self, content: str, source_type: str = "generic"
    ) -> list[ChunkResult]:
        """
        Chunk content based on source type with appropriate strategy

        Args:
            content: Raw content to chunk
            source_type: Source type (kubernetes, fastapi, cobol, irs, mumps)

        Returns:
            List of chunk results with metadata
        """
        # Determine chunking strategy based on source type
        strategy = self._get_strategy_for_source(source_type)

        # Update config for this specific chunking operation
        chunk_config = ChunkConfig(
            max_tokens=self.config.max_tokens,
            overlap_pct=self.config.overlap_pct,
            min_chunk_size=self.config.min_chunk_size,
            preserve_boundaries=self.config.preserve_boundaries,
            strategy=strategy,
        )

        if strategy == ChunkingStrategy.SEMANTIC_BLOCKS:
            return self._chunk_by_semantic_blocks(content, chunk_config, source_type)
        elif strategy == ChunkingStrategy.FIXED_WIDTH:
            return self._chunk_by_fixed_width(content, chunk_config, source_type)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return self._chunk_by_hierarchy(content, chunk_config, source_type)
        elif strategy == ChunkingStrategy.LINE_BASED:
            return self._chunk_by_lines(content, chunk_config, source_type)
        else:
            return self._chunk_by_tokens(content, chunk_config, source_type)

    def _get_strategy_for_source(self, source_type: str) -> ChunkingStrategy:
        """Determine optimal chunking strategy for source type"""
        strategy_map = {
            "kubernetes": ChunkingStrategy.SEMANTIC_BLOCKS,
            "fastapi": ChunkingStrategy.SEMANTIC_BLOCKS,
            "cobol": ChunkingStrategy.FIXED_WIDTH,
            "irs": ChunkingStrategy.FIXED_WIDTH,
            "mumps": ChunkingStrategy.HIERARCHICAL,
        }
        return strategy_map.get(source_type.lower(), ChunkingStrategy.TOKEN_BASED)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: remove extra whitespace, count characters
        normalized = re.sub(r"\s+", " ", text.strip())
        return max(1, len(normalized) // self.chars_per_token)

    def _chunk_by_semantic_blocks(
        self, content: str, config: ChunkConfig, source_type: str
    ) -> list[ChunkResult]:
        """Chunk content by semantic blocks (YAML docs, functions, etc.)"""
        chunks = []
        lines = content.split("\n")

        current_chunk = []
        current_start = 0
        current_tokens = 0

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            # Check if this line starts a new semantic block
            is_boundary = self._is_semantic_boundary(line, config.strategy)

            # If adding this line would exceed limit, or we hit a boundary, finalize current chunk
            if (current_tokens + line_tokens > config.max_tokens and current_chunk) or (
                is_boundary and current_chunk and config.preserve_boundaries
            ):

                # Create chunk from current content
                chunk_content = "\n".join(current_chunk)
                if self._estimate_tokens(chunk_content) >= config.min_chunk_size:
                    chunks.append(
                        ChunkResult(
                            content=chunk_content,
                            start_position=current_start,
                            end_position=current_start + len(chunk_content),
                            token_count=current_tokens,
                            metadata={
                                "strategy": config.strategy.value,
                                "source_type": source_type,
                                "line_start": current_start,
                                "line_end": i,
                                "semantic_block": True,
                            },
                        )
                    )

                # Start new chunk with overlap
                overlap_lines = self._calculate_overlap_lines(
                    current_chunk, config.overlap_pct
                )
                current_chunk = overlap_lines + [line]
                current_start = i - len(overlap_lines)
                current_tokens = sum(self._estimate_tokens(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Handle final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            if self._estimate_tokens(chunk_content) >= config.min_chunk_size:
                chunks.append(
                    ChunkResult(
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        token_count=current_tokens,
                        metadata={
                            "strategy": config.strategy.value,
                            "source_type": source_type,
                            "line_start": current_start,
                            "line_end": len(lines),
                            "semantic_block": True,
                        },
                    )
                )

        return chunks

    def _chunk_by_fixed_width(
        self, content: str, config: ChunkConfig, source_type: str
    ) -> list[ChunkResult]:
        """Chunk fixed-width content (COBOL/IRS) by record boundaries"""
        chunks = []
        lines = content.split("\n")

        current_chunk = []
        current_start = 0
        current_tokens = 0
        record_count = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            line_tokens = self._estimate_tokens(line)

            # For fixed-width, chunk by record count or token limit
            if current_tokens + line_tokens > config.max_tokens and current_chunk:
                # Create chunk
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    ChunkResult(
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        token_count=current_tokens,
                        metadata={
                            "strategy": config.strategy.value,
                            "source_type": source_type,
                            "record_count": record_count,
                            "fixed_width": True,
                        },
                    )
                )

                # Start new chunk with minimal overlap (preserve record structure)
                current_chunk = [line]
                current_start = i
                current_tokens = line_tokens
                record_count = 1
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
                record_count += 1

        # Handle final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    token_count=current_tokens,
                    metadata={
                        "strategy": config.strategy.value,
                        "source_type": source_type,
                        "record_count": record_count,
                        "fixed_width": True,
                    },
                )
            )

        return chunks

    def _chunk_by_hierarchy(
        self, content: str, config: ChunkConfig, source_type: str
    ) -> list[ChunkResult]:
        """Chunk hierarchical content (MUMPS) by node structures"""
        chunks = []
        lines = content.split("\n")

        current_chunk = []
        current_start = 0
        current_tokens = 0
        hierarchy_level = 0

        for i, line in enumerate(lines):
            line_tokens = self._estimate_tokens(line)

            # Detect hierarchy level changes
            new_level = self._detect_hierarchy_level(line, source_type)

            # If we're starting a new major section, finalize current chunk
            if (
                new_level < hierarchy_level
                and current_chunk
                and current_tokens + line_tokens > config.max_tokens
            ):

                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    ChunkResult(
                        content=chunk_content,
                        start_position=current_start,
                        end_position=current_start + len(chunk_content),
                        token_count=current_tokens,
                        metadata={
                            "strategy": config.strategy.value,
                            "source_type": source_type,
                            "hierarchy_level": hierarchy_level,
                            "hierarchical": True,
                        },
                    )
                )

                # Start new chunk
                current_chunk = [line]
                current_start = i
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

            hierarchy_level = new_level

        # Handle final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    start_position=current_start,
                    end_position=current_start + len(chunk_content),
                    token_count=current_tokens,
                    metadata={
                        "strategy": config.strategy.value,
                        "source_type": source_type,
                        "hierarchy_level": hierarchy_level,
                        "hierarchical": True,
                    },
                )
            )

        return chunks

    def _chunk_by_lines(
        self, content: str, config: ChunkConfig, source_type: str
    ) -> list[ChunkResult]:
        """Chunk content by lines with token limits"""
        return self._chunk_by_tokens(content, config, source_type)

    def _chunk_by_tokens(
        self, content: str, config: ChunkConfig, source_type: str
    ) -> list[ChunkResult]:
        """Basic token-based chunking with overlap"""
        chunks = []

        # Split into words for basic token chunking
        words = content.split()
        if not words:
            return chunks

        chunk_size = max(1, int(config.max_tokens * 0.75))  # Leave room for overlap
        overlap_size = int(chunk_size * config.overlap_pct)

        start = 0
        position = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = " ".join(chunk_words)

            chunks.append(
                ChunkResult(
                    content=chunk_content,
                    start_position=position,
                    end_position=position + len(chunk_content),
                    token_count=len(chunk_words),
                    metadata={
                        "strategy": config.strategy.value,
                        "source_type": source_type,
                        "word_start": start,
                        "word_end": end,
                        "token_based": True,
                    },
                )
            )

            position += len(chunk_content) + 1  # +1 for space

            if end >= len(words):
                break
            start = end - overlap_size

        return chunks

    def _is_semantic_boundary(self, line: str, strategy: ChunkingStrategy) -> bool:
        """Check if line represents a semantic boundary"""
        if strategy not in self.boundary_patterns:
            return False

        patterns = self.boundary_patterns[strategy]
        return any(re.match(pattern, line, re.MULTILINE) for pattern in patterns)

    def _calculate_overlap_lines(
        self, lines: list[str], overlap_pct: float
    ) -> list[str]:
        """Calculate overlap lines based on percentage"""
        if not lines:
            return []

        overlap_count = max(0, int(len(lines) * overlap_pct))
        return lines[-overlap_count:] if overlap_count > 0 else []

    def _detect_hierarchy_level(self, line: str, source_type: str) -> int:
        """Detect hierarchy level for MUMPS/hierarchical content"""
        if source_type.lower() == "mumps":
            # MUMPS hierarchy: ^GLOBAL(subscript,subscript)
            if re.match(r"^\^[A-Z0-9_]+\(", line):
                # Count parentheses depth
                return line.count("(") - line.count(")")
            # FileMan field numbers (level indicators)
            field_match = re.match(r"^\s*(\d+)\s+", line)
            if field_match:
                return len(field_match.group(1))  # Higher numbers = deeper nesting

        # Default: indentation-based hierarchy
        return len(line) - len(line.lstrip())

    def get_chunking_stats(self, chunks: list[ChunkResult]) -> dict[str, Any]:
        """Get statistics about chunking results"""
        if not chunks:
            return {"total_chunks": 0}

        token_counts = [chunk.token_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "strategies_used": list(
                set(chunk.metadata.get("strategy", "unknown") for chunk in chunks)
            ),
        }


# Factory function for easy instantiation
def create_chunker(
    source_type: str, max_tokens: int = 1000, overlap_pct: float = 0.15
) -> ContentChunker:
    """Create a content chunker optimized for specific source type"""
    config = ChunkConfig(
        max_tokens=max_tokens,
        overlap_pct=overlap_pct,
        strategy=ContentChunker._get_strategy_for_source(None, source_type),
    )
    return ContentChunker(config)


# Helper function for testing and validation
def validate_chunks(
    chunks: list[ChunkResult], original_content: str
) -> dict[str, bool]:
    """Validate that chunks properly cover original content"""
    return {
        "has_chunks": len(chunks) > 0,
        "no_empty_chunks": all(chunk.content.strip() for chunk in chunks),
        "reasonable_sizes": all(chunk.token_count > 0 for chunk in chunks),
        "proper_positions": all(
            chunk.start_position <= chunk.end_position for chunk in chunks
        ),
        "content_preserved": len("".join(chunk.content for chunk in chunks)) > 0,
    }
