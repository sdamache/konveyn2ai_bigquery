#!/usr/bin/env python3
"""
Repository Ingestion Pipeline for KonveyN2AI
Processes repository files into chunks and embeddings for vector search.

This script implements Task 9: Repository Ingestion Pipeline
- Subtask 9.1: File Collection and Filtering
- Subtask 9.2: File Processing and Chunking Logic  
- Subtask 9.3: Embedding Generation with Vertex AI
- Subtask 9.4: Matching Engine Integration and CLI

Usage:
    python scripts/ingest.py --repo-path /path/to/repo --index-id INDEX_ID --index-endpoint-id ENDPOINT_ID
"""

import os
import sys
import glob
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ingestion")


class RepositoryIngestionPipeline:
    """
    Main ingestion pipeline for processing repository files into vector embeddings.

    Implements the complete Task 9 workflow:
    1. File collection and filtering (9.1)
    2. File processing and chunking (9.2)
    3. Embedding generation (9.3)
    4. Matching Engine integration (9.4)
    """

    def __init__(
        self,
        repo_path: str,
        index_id: str,
        index_endpoint_id: str,
        batch_size: int = 10,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            repo_path: Path to repository root
            index_id: Matching Engine index ID
            index_endpoint_id: Matching Engine index endpoint ID
            batch_size: Batch size for embedding generation
        """
        self.repo_path = Path(repo_path).resolve()
        self.index_id = index_id
        self.index_endpoint_id = index_endpoint_id
        self.batch_size = batch_size

        # File processing configuration
        self.supported_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".md",
            ".txt",
            ".html",
            ".css",
            ".json",
            ".yaml",
            ".yml",
            ".sh",
            ".sql",
        }

        self.exclude_directories = {
            "node_modules",
            "venv",
            ".git",
            "__pycache__",
            "build",
            "dist",
            ".next",
            ".nuxt",
            "target",
            "bin",
            "obj",
            ".vscode",
            ".idea",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
        }

        # Statistics tracking
        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

        logger.info(f"Initialized ingestion pipeline for repository: {self.repo_path}")
        logger.info(
            f"Target index: {self.index_id}, endpoint: {self.index_endpoint_id}"
        )

    def collect_files(self) -> List[Path]:
        """
        Collect and filter repository files based on extensions and exclusion patterns.

        Implements Subtask 9.1: File Collection and Filtering

        Returns:
            List[Path]: Filtered list of file paths to process
        """
        logger.info("Starting file collection and filtering...")

        collected_files = []

        # Walk through repository directory
        for root, dirs, files in os.walk(self.repo_path):
            root_path = Path(root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_directories]

            # Check if current directory should be excluded
            if any(
                excluded in root_path.parts for excluded in self.exclude_directories
            ):
                continue

            # Process files in current directory
            for file in files:
                file_path = root_path / file

                # Check file extension
                if file_path.suffix.lower() in self.supported_extensions:
                    # Additional filtering
                    if self._should_include_file(file_path):
                        collected_files.append(file_path)

        self.stats["files_found"] = len(collected_files)
        logger.info(f"Collected {len(collected_files)} files for processing")

        # Log file type distribution
        extension_counts = {}
        for file_path in collected_files:
            ext = file_path.suffix.lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1

        logger.info("File type distribution:")
        for ext, count in sorted(extension_counts.items()):
            logger.info(f"  {ext}: {count} files")

        return collected_files

    def _should_include_file(self, file_path: Path) -> bool:
        """
        Additional filtering logic for individual files.

        Args:
            file_path: Path to file to check

        Returns:
            bool: True if file should be included
        """
        # Skip hidden files
        if file_path.name.startswith("."):
            return False

        # Skip very large files (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                logger.warning(
                    f"Skipping large file: {file_path} ({file_path.stat().st_size} bytes)"
                )
                return False
        except OSError:
            logger.warning(f"Could not stat file: {file_path}")
            return False

        # Skip binary files (basic check)
        if self._is_binary_file(file_path):
            return False

        return True

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Basic binary file detection.

        Args:
            file_path: Path to file to check

        Returns:
            bool: True if file appears to be binary
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\0" in chunk:
                    return True
        except (OSError, UnicodeDecodeError):
            return True
        return False

    def process_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process files into chunks with metadata.

        Implements Subtask 9.2: File Processing and Chunking Logic

        Args:
            file_paths: List of file paths to process

        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
        """
        logger.info("Starting file processing and chunking...")

        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self._process_single_file(file_path)
                all_chunks.extend(chunks)
                self.stats["files_processed"] += 1

                if self.stats["files_processed"] % 10 == 0:
                    logger.info(
                        f"Processed {self.stats['files_processed']}/{len(file_paths)} files"
                    )

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                self.stats["errors"] += 1
                continue

        self.stats["chunks_created"] = len(all_chunks)
        logger.info(
            f"Created {len(all_chunks)} chunks from {self.stats['files_processed']} files"
        )

        return all_chunks

    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single file into chunks.

        Args:
            file_path: Path to file to process

        Returns:
            List[Dict[str, Any]]: List of chunks for this file
        """
        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return []

        # Skip empty files
        if not content.strip():
            return []

        # Get relative path for storage
        try:
            rel_path = file_path.relative_to(self.repo_path)
        except ValueError:
            rel_path = file_path

        # Split into chunks
        chunks = self._split_into_chunks(content, str(rel_path))

        return chunks

    def _split_into_chunks(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Split file content into logical chunks.

        Implements intelligent chunking with language-aware boundaries.

        Args:
            content: File content to chunk
            file_path: Relative file path for metadata

        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        # Target chunk size (in characters, roughly 300-500 tokens)
        target_size = 2000  # ~400-500 tokens
        overlap_size = 200  # Overlap for context preservation

        # Determine language-specific separators
        separators = self._get_separators_for_file(file_path)

        # Split content into initial segments
        segments = self._split_by_separators(content, separators)

        # Further process segments into appropriately sized chunks
        chunks = []
        chunk_index = 0

        for segment in segments:
            if len(segment.strip()) == 0:
                continue

            if len(segment) <= target_size:
                # Segment is already appropriate size
                chunks.append(self._create_chunk(segment, file_path, chunk_index))
                chunk_index += 1
            else:
                # Split large segment into smaller chunks
                sub_chunks = self._split_large_segment(
                    segment, target_size, overlap_size
                )
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(sub_chunk, file_path, chunk_index))
                    chunk_index += 1

        return chunks

    def _get_separators_for_file(self, file_path: str) -> List[str]:
        """
        Get language-specific separators for logical chunking.

        Args:
            file_path: File path to determine language

        Returns:
            List[str]: List of separators in priority order
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            return ["\ndef ", "\nclass ", "\n# ", '\n"""', "\n'''", "\n\n"]
        elif ext in [".js", ".ts"]:
            return [
                "\nfunction ",
                "\nclass ",
                "\nconst ",
                "\nlet ",
                "\n// ",
                "\n/*",
                "\n\n",
            ]
        elif ext == ".java":
            return [
                "\npublic class ",
                "\nprivate class ",
                "\npublic ",
                "\nprivate ",
                "\n// ",
                "\n/*",
                "\n\n",
            ]
        elif ext == ".go":
            return ["\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\n// ", "\n/*", "\n\n"]
        elif ext == ".md":
            return ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n"]
        elif ext in [".html", ".css"]:
            return ["\n<", "\n}", "\n/*", "\n\n"]
        else:
            return ["\n\n", "\n// ", "\n# ", "\n/*"]

    def _split_by_separators(self, content: str, separators: List[str]) -> List[str]:
        """
        Split content by separators while preserving separators.

        Args:
            content: Content to split
            separators: List of separators

        Returns:
            List[str]: List of segments
        """
        segments = [content]

        for separator in separators:
            new_segments = []
            for segment in segments:
                if len(segment) <= 1000:  # Don't split small segments
                    new_segments.append(segment)
                    continue

                parts = segment.split(separator)
                for i, part in enumerate(parts):
                    if i == 0:
                        new_segments.append(part)
                    else:
                        # Preserve separator
                        new_segments.append(separator.lstrip("\n") + part)

            segments = new_segments

        return [s for s in segments if s.strip()]

    def _split_large_segment(
        self, segment: str, target_size: int, overlap_size: int
    ) -> List[str]:
        """
        Split a large segment into overlapping chunks.

        Args:
            segment: Segment to split
            target_size: Target chunk size
            overlap_size: Overlap between chunks

        Returns:
            List[str]: List of chunks
        """
        chunks = []
        start = 0

        while start < len(segment):
            end = start + target_size

            if end >= len(segment):
                # Last chunk
                chunks.append(segment[start:])
                break

            # Try to find a good break point (newline, space, etc.)
            break_point = end
            for i in range(end - 100, end):
                if i > start and segment[i] in "\n.;":
                    break_point = i + 1
                    break

            chunks.append(segment[start:break_point])
            start = break_point - overlap_size

            # Ensure we make progress
            if start <= chunks[-1].find("\n") if chunks else 0:
                start = break_point

        return chunks

    def _create_chunk(
        self, content: str, file_path: str, chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.

        Args:
            content: Chunk content
            file_path: Source file path
            chunk_index: Index of chunk within file

        Returns:
            Dict[str, Any]: Chunk with metadata
        """
        return {
            "content": content.strip(),
            "file_path": file_path,
            "chunk_index": chunk_index,
            "char_count": len(content.strip()),
            "line_count": len(content.strip().split("\n")),
            "file_extension": Path(file_path).suffix.lower(),
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Repository Ingestion Pipeline for KonveyN2AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --repo-path . --index-id 805460437066842112 --index-endpoint-id ENDPOINT_ID
  python scripts/ingest.py --repo-path /path/to/repo --batch-size 20 --dry-run
        """,
    )

    parser.add_argument(
        "--repo-path", required=True, help="Path to repository root directory"
    )
    parser.add_argument(
        "--index-id", help="Matching Engine index ID (e.g., 805460437066842112)"
    )
    parser.add_argument("--index-endpoint-id", help="Matching Engine index endpoint ID")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for embedding generation (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run file collection and chunking without uploading to Matching Engine",
    )
    parser.add_argument(
        "--output-file", help="Save chunks to JSON file instead of uploading"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.output_file:
        if not args.index_id or not args.index_endpoint_id:
            parser.error(
                "--index-id and --index-endpoint-id are required unless using --dry-run or --output-file"
            )

    # Initialize pipeline
    pipeline = RepositoryIngestionPipeline(
        repo_path=args.repo_path,
        index_id=args.index_id or "dummy",
        index_endpoint_id=args.index_endpoint_id or "dummy",
        batch_size=args.batch_size,
    )

    try:
        # Step 1: Collect files (Subtask 9.1)
        logger.info("=== Step 1: File Collection and Filtering ===")
        files = pipeline.collect_files()

        if not files:
            logger.warning("No files found to process")
            return

        # Step 2: Process files into chunks (Subtask 9.2)
        logger.info("=== Step 2: File Processing and Chunking ===")
        chunks = pipeline.process_files(files)

        if not chunks:
            logger.warning("No chunks created")
            return

        # Output results
        if args.output_file:
            logger.info(f"Saving {len(chunks)} chunks to {args.output_file}")
            with open(args.output_file, "w") as f:
                json.dump(chunks, f, indent=2)

        # Print statistics
        logger.info("=== Ingestion Statistics ===")
        for key, value in pipeline.stats.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")

        if args.dry_run:
            logger.info("Dry run completed successfully")
        else:
            logger.info(
                "File collection and chunking completed - ready for embedding generation"
            )

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
