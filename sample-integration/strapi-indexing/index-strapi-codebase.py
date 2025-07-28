#!/usr/bin/env python3
"""
Vector indexing script for Strapi codebase
Creates embeddings for all relevant code files and uploads to Vertex AI Matching Engine
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import after path modification
try:
    from janapada_memory.config import JanapadaConfig
    from janapada_memory.embedding_service import EmbeddingService
except ImportError:
    # Fallback for testing
    JanapadaConfig = None
    EmbeddingService = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """Represents a code snippet to be indexed"""

    file_path: str
    content: str
    language: str
    size_bytes: int
    function_name: str = ""
    class_name: str = ""


class StrapiCodeIndexer:
    """Indexes Strapi codebase for vector search"""

    def __init__(self, strapi_path: str = ".", output_dir: str = "strapi_vectors"):
        self.strapi_path = Path(strapi_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize Janapada config and embedding service
        self.config = JanapadaConfig()
        self.embedding_service = None

        # File extensions to process
        self.code_extensions = {
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".py": "python",
            ".json": "json",
            ".md": "markdown",
        }

        # Directories to skip
        self.skip_dirs = {
            "node_modules",
            ".git",
            "dist",
            "build",
            ".next",
            "coverage",
            ".nyc_output",
            "tmp",
            "temp",
            "__pycache__",
            ".yarn",
            ".cache",
            ".strapi",
        }

        # Patterns to exclude
        self.skip_patterns = [
            r"\.min\.js$",
            r"\.test\.js$",
            r"\.spec\.js$",
            r"\.d\.ts$",
            r"\.map$",
            r"package-lock\.json$",
            r"yarn\.lock$",
        ]

    async def initialize_embedding_service(self):
        """Initialize the embedding service"""
        try:
            self.embedding_service = EmbeddingService(self.config)
            await self.embedding_service.initialize()
            logger.info("✅ Embedding service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {e}")
            raise

    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        # Skip if in excluded directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return True

        # Skip if matches exclude pattern
        for pattern in self.skip_patterns:
            if re.search(pattern, str(file_path)):
                return True

        return False

    def extract_functions_and_classes(
        self, content: str, language: str
    ) -> list[dict[str, str]]:
        """Extract function and class definitions from code"""
        extracts = []

        if language in ["javascript", "typescript"]:
            # Extract functions
            func_pattern = (
                r"(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}"
            )
            for match in re.finditer(func_pattern, content, re.MULTILINE | re.DOTALL):
                if len(match.group(0)) < 2000:  # Reasonable size limit
                    extracts.append(
                        {
                            "type": "function",
                            "name": match.group(3),
                            "content": match.group(0).strip(),
                        }
                    )

            # Extract arrow functions
            arrow_pattern = r"(const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\}"
            for match in re.finditer(arrow_pattern, content, re.MULTILINE | re.DOTALL):
                if len(match.group(0)) < 2000:
                    extracts.append(
                        {
                            "type": "function",
                            "name": match.group(2),
                            "content": match.group(0).strip(),
                        }
                    )

            # Extract classes
            class_pattern = r"(export\s+)?(class\s+(\w+).*?\{(?:[^{}]*|\{[^{}]*\})*\})"
            for match in re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL):
                if len(match.group(2)) < 3000:
                    extracts.append(
                        {
                            "type": "class",
                            "name": match.group(3),
                            "content": match.group(2).strip(),
                        }
                    )

        elif language == "python":
            # Extract Python functions
            func_pattern = r"def\s+(\w+)\s*\([^)]*\):[^def]*?(?=\ndef|\nclass|\Z)"
            for match in re.finditer(func_pattern, content, re.MULTILINE | re.DOTALL):
                if len(match.group(0)) < 2000:
                    extracts.append(
                        {
                            "type": "function",
                            "name": match.group(1),
                            "content": match.group(0).strip(),
                        }
                    )

            # Extract Python classes
            class_pattern = r"class\s+(\w+).*?:\s*\n((?:\s+.*\n)*)"
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                if len(match.group(0)) < 3000:
                    extracts.append(
                        {
                            "type": "class",
                            "name": match.group(1),
                            "content": match.group(0).strip(),
                        }
                    )

        return extracts

    async def process_file(self, file_path: Path) -> list[CodeSnippet]:
        """Process a single file and extract code snippets"""
        snippets = []

        if self.should_skip_file(file_path):
            return snippets

        extension = file_path.suffix.lower()
        if extension not in self.code_extensions:
            return snippets

        try:
            async with aiofiles.open(file_path, encoding="utf-8", errors="ignore") as f:
                content = await f.read()

            # Skip very large files (>100KB)
            if len(content) > 100000:
                logger.warning(
                    f"⚠️  Skipping large file: {file_path} ({len(content)} bytes)"
                )
                return snippets

            # Skip binary or mostly non-text files
            if len(content.strip()) == 0:
                return snippets

            language = self.code_extensions[extension]
            relative_path = str(file_path.relative_to(self.strapi_path))

            # Extract functions and classes
            extracts = self.extract_functions_and_classes(content, language)

            # Create snippets for extracted code elements
            for extract in extracts:
                snippet = CodeSnippet(
                    file_path=relative_path,
                    content=extract["content"],
                    language=language,
                    size_bytes=len(extract["content"]),
                    function_name=(
                        extract["name"] if extract["type"] == "function" else ""
                    ),
                    class_name=extract["name"] if extract["type"] == "class" else "",
                )
                snippets.append(snippet)

            # Also create a snippet for the entire file if it's not too large
            if len(content) < 5000 and len(extracts) == 0:
                snippet = CodeSnippet(
                    file_path=relative_path,
                    content=content[:2000] + ("..." if len(content) > 2000 else ""),
                    language=language,
                    size_bytes=len(content),
                )
                snippets.append(snippet)

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")

        return snippets

    async def collect_code_snippets(self) -> list[CodeSnippet]:
        """Collect all code snippets from Strapi repository"""
        logger.info(f"🔍 Scanning Strapi repository: {self.strapi_path}")

        all_snippets = []
        files_processed = 0

        # Get all code files
        code_files = []
        for ext in self.code_extensions.keys():
            pattern = f"**/*{ext}"
            files = list(self.strapi_path.glob(pattern))
            code_files.extend(files)

        logger.info(f"📊 Found {len(code_files)} potential code files")

        # Process files in batches
        batch_size = 50
        for i in range(0, len(code_files), batch_size):
            batch = code_files[i : i + batch_size]
            batch_tasks = [self.process_file(file_path) for file_path in batch]

            batch_results = await asyncio.gather(*batch_tasks)

            for snippets in batch_results:
                all_snippets.extend(snippets)

            files_processed += len(batch)
            logger.info(
                f"📈 Processed {files_processed}/{len(code_files)} files, collected {len(all_snippets)} snippets"
            )

        return all_snippets

    async def generate_embeddings(
        self, snippets: list[CodeSnippet]
    ) -> list[dict[str, Any]]:
        """Generate embeddings for code snippets"""
        logger.info(f"🧠 Generating embeddings for {len(snippets)} code snippets")

        embeddings_data = []
        batch_size = 10  # Small batches for API limits

        for i in range(0, len(snippets), batch_size):
            batch = snippets[i : i + batch_size]

            for snippet in batch:
                try:
                    # Create enhanced text for embedding
                    enhanced_text = f"""
File: {snippet.file_path}
Language: {snippet.language}
{f"Function: {snippet.function_name}" if snippet.function_name else ""}
{f"Class: {snippet.class_name}" if snippet.class_name else ""}

Code:
{snippet.content}
"""

                    # Generate embedding
                    embedding = await self.embedding_service.generate_embedding(
                        enhanced_text
                    )

                    # Store embedding data
                    embedding_data = {
                        "id": f"strapi_{len(embeddings_data)}",
                        "file_path": snippet.file_path,
                        "content": snippet.content,
                        "language": snippet.language,
                        "function_name": snippet.function_name,
                        "class_name": snippet.class_name,
                        "size_bytes": snippet.size_bytes,
                        "embedding": (
                            embedding.tolist()
                            if hasattr(embedding, "tolist")
                            else embedding
                        ),
                        "enhanced_text": enhanced_text,
                    }

                    embeddings_data.append(embedding_data)

                except Exception as e:
                    logger.error(
                        f"❌ Failed to generate embedding for {snippet.file_path}: {e}"
                    )
                    continue

            logger.info(
                f"📈 Generated embeddings: {len(embeddings_data)}/{len(snippets)}"
            )

            # Brief pause to respect API limits
            await asyncio.sleep(0.1)

        return embeddings_data

    async def save_embeddings(self, embeddings_data: list[dict[str, Any]]):
        """Save embeddings to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full embeddings data
        embeddings_file = self.output_dir / f"strapi_embeddings_{timestamp}.json"
        async with aiofiles.open(embeddings_file, "w") as f:
            await f.write(json.dumps(embeddings_data, indent=2))

        logger.info(f"💾 Saved embeddings to: {embeddings_file}")

        # Save metadata summary
        metadata = {
            "timestamp": timestamp,
            "total_snippets": len(embeddings_data),
            "languages": list({item["language"] for item in embeddings_data}),
            "total_files": len({item["file_path"] for item in embeddings_data}),
            "embedding_dimensions": (
                len(embeddings_data[0]["embedding"]) if embeddings_data else 0
            ),
        }

        metadata_file = self.output_dir / f"strapi_metadata_{timestamp}.json"
        async with aiofiles.open(metadata_file, "w") as f:
            await f.write(json.dumps(metadata, indent=2))

        logger.info(f"📊 Saved metadata to: {metadata_file}")
        return embeddings_file, metadata_file

    async def upload_to_vector_index(self, embeddings_data: list[dict[str, Any]]):
        """Upload embeddings to Vertex AI Matching Engine"""
        logger.info(f"☁️  Uploading {len(embeddings_data)} embeddings to Vertex AI")

        try:
            # Use the embedding service's index update functionality
            # This would need to be implemented in the actual embedding service
            logger.warning(
                "⚠️  Vector index upload not implemented yet - embeddings saved locally"
            )

        except Exception as e:
            logger.error(f"❌ Failed to upload to vector index: {e}")

    async def run_indexing(self):
        """Run the complete indexing process"""
        logger.info("🚀 Starting Strapi codebase indexing")

        try:
            # Initialize services
            await self.initialize_embedding_service()

            # Collect code snippets
            snippets = await self.collect_code_snippets()

            if not snippets:
                logger.error("❌ No code snippets collected")
                return

            logger.info(f"✅ Collected {len(snippets)} code snippets")

            # Generate embeddings
            embeddings_data = await self.generate_embeddings(snippets)

            if not embeddings_data:
                logger.error("❌ No embeddings generated")
                return

            logger.info(f"✅ Generated {len(embeddings_data)} embeddings")

            # Save embeddings
            embeddings_file, metadata_file = await self.save_embeddings(embeddings_data)

            # Upload to vector index (if implemented)
            await self.upload_to_vector_index(embeddings_data)

            logger.info("🎉 Strapi indexing completed successfully!")

            # Print summary
            print(
                f"""
📊 Strapi Indexing Summary
==========================
✅ Total code snippets processed: {len(embeddings_data)}
✅ Languages indexed: {len({item['language'] for item in embeddings_data})}
✅ Files processed: {len({item['file_path'] for item in embeddings_data})}
✅ Embedding dimensions: {len(embeddings_data[0]['embedding']) if embeddings_data else 0}

📁 Output files:
   📄 Embeddings: {embeddings_file}
   📊 Metadata: {metadata_file}

🎯 Ready for testing with KonveyN2AI!
"""
            )

        except Exception as e:
            logger.error(f"❌ Indexing failed: {e}")
            raise


async def main():
    """Main entry point"""
    indexer = StrapiCodeIndexer(strapi_path=".")
    await indexer.run_indexing()


if __name__ == "__main__":
    asyncio.run(main())
