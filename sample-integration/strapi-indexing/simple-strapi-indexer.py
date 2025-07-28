#!/usr/bin/env python3
"""
Simple Strapi codebase indexer - creates JSON file with code snippets
for testing with KonveyN2AI production services
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class SimpleStrapiIndexer:
    """Simple indexer that extracts meaningful code snippets from Strapi"""

    def __init__(self, strapi_path: str = "."):
        self.strapi_path = Path(strapi_path)

        # File extensions to process
        self.code_extensions = {".js", ".ts", ".jsx", ".tsx"}

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
            "tests",
            "test",
        }

        # Patterns to exclude
        self.skip_patterns = [
            r"\.min\.js$",
            r"\.test\.js$",
            r"\.spec\.js$",
            r"\.d\.ts$",
            r"\.map$",
        ]

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

    def extract_meaningful_snippets(
        self, content: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract meaningful code snippets from content"""
        snippets = []

        # Extract functions (both regular and arrow functions)
        function_patterns = [
            r"(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            r"(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            r"(\w+):\s*(?:async\s+)?function\s*\([^)]*\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]

        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                func_content = match.group(0).strip()
                if 50 < len(func_content) < 2000:  # Reasonable size
                    # Extract function name
                    name_match = re.search(
                        r"function\s+(\w+)|(\w+)\s*[=:].*?(?:function|\(.*?\)\s*=>)",
                        func_content,
                    )
                    func_name = ""
                    if name_match:
                        func_name = (
                            name_match.group(1) or name_match.group(2) or "anonymous"
                        )

                    snippets.append(
                        {
                            "file_path": file_path,
                            "content": func_content,
                            "type": "function",
                            "name": func_name,
                            "size": len(func_content),
                        }
                    )

        # Extract class definitions
        class_pattern = r"(export\s+)?(class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{(?:[^{}]*|\{[^{}]*\})*\})"
        for match in re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL):
            class_content = match.group(2).strip()
            class_name = match.group(3)
            if 100 < len(class_content) < 3000:
                snippets.append(
                    {
                        "file_path": file_path,
                        "content": class_content,
                        "type": "class",
                        "name": class_name,
                        "size": len(class_content),
                    }
                )

        # Extract important object/config definitions
        config_pattern = r"(const|let|var)\s+(\w*[Cc]onfig\w*|\w*[Ss]chema\w*|\w*[Mm]odel\w*)\s*=\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        for match in re.finditer(config_pattern, content, re.MULTILINE | re.DOTALL):
            config_content = match.group(0).strip()
            config_name = match.group(2)
            if 50 < len(config_content) < 1500:
                snippets.append(
                    {
                        "file_path": file_path,
                        "content": config_content,
                        "type": "config",
                        "name": config_name,
                        "size": len(config_content),
                    }
                )

        # If no specific patterns found, extract the first meaningful chunk
        if not snippets and len(content.strip()) > 100:
            # Take first 1000 characters if it's a substantial file
            chunk = content.strip()[:1000]
            if len(chunk) > 100:
                snippets.append(
                    {
                        "file_path": file_path,
                        "content": chunk + ("..." if len(content) > 1000 else ""),
                        "type": "file_chunk",
                        "name": Path(file_path).stem,
                        "size": len(chunk),
                    }
                )

        return snippets

    def process_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Process a single file"""
        if self.should_skip_file(file_path):
            return []

        if file_path.suffix.lower() not in self.code_extensions:
            return []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Skip very large files
            if len(content) > 50000:
                return []

            # Skip mostly empty files
            if len(content.strip()) < 50:
                return []

            relative_path = str(file_path.relative_to(self.strapi_path))
            return self.extract_meaningful_snippets(content, relative_path)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    def collect_all_snippets(self, limit_files: int = None) -> list[dict[str, Any]]:
        """Collect all code snippets from Strapi"""
        print("🔍 Scanning Strapi repository...")

        all_snippets = []
        files_processed = 0

        # Get all code files
        all_files = []
        for ext in self.code_extensions:
            pattern = f"**/*{ext}"
            files = list(self.strapi_path.glob(pattern))
            all_files.extend(files)

        # Limit files for testing if specified
        if limit_files:
            all_files = all_files[:limit_files]
            print(f"🎯 Limited to first {limit_files} files for testing")

        for file_path in all_files:
            snippets = self.process_file(file_path)
            all_snippets.extend(snippets)
            files_processed += 1

            if files_processed % 50 == 0:
                print(
                    f"📈 Processed {files_processed}/{len(all_files)} files, collected {len(all_snippets)} snippets"
                )

        print(
            f"✅ Final result: {files_processed} files processed, {len(all_snippets)} snippets collected"
        )
        return all_snippets

    def save_snippets(self, snippets: list[dict[str, Any]]) -> str:
        """Save snippets to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"strapi_code_snippets_{timestamp}.json"

        # Add metadata
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "total_snippets": len(snippets),
                "languages": ["javascript", "typescript"],
                "source": "strapi_repository",
                "types": list({s["type"] for s in snippets}),
            },
            "snippets": snippets,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(snippets)} snippets to: {output_file}")
        return output_file

    def run(self, limit_files: int = None):
        """Run the indexing process"""
        print("🚀 Starting Strapi codebase indexing")

        snippets = self.collect_all_snippets(limit_files)

        if not snippets:
            print("❌ No code snippets found")
            return

        output_file = self.save_snippets(snippets)

        # Print summary
        types_count = {}
        for snippet in snippets:
            types_count[snippet["type"]] = types_count.get(snippet["type"], 0) + 1

        print(
            f"""
📊 Strapi Indexing Summary
==========================
✅ Total snippets: {len(snippets)}
✅ Code types found: {', '.join(types_count.keys())}
✅ Breakdown: {types_count}

📁 Output file: {output_file}

🎯 Ready for testing with KonveyN2AI production services!

Sample snippet types collected:
{json.dumps(types_count, indent=2)}
"""
        )

        return output_file


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    indexer = SimpleStrapiIndexer()
    indexer.run(limit)
