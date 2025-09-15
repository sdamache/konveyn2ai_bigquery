#!/usr/bin/env python3
"""
Performance validation script for M1 Multi-Source Ingestion system

T037: Process 100+ files per source type within 5 minutes
Validates ingestion performance across all 5 parser types using template files
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest.k8s.parser import KubernetesParserImpl
from ingest.fastapi.parser import FastAPIParserImpl
from ingest.cobol.parser import COBOLParserImpl
from ingest.irs.parser import IRSParserImpl
from ingest.mumps.parser import MUMPSParserImpl

# Import BigQuery writer for ingestion
from common.bq_writer import BigQueryWriter


class TemplateBasedPerformanceValidator:
    """Performance validation using external template files"""

    def __init__(self, dry_run: bool = True, with_bigquery: bool = False):
        self.dry_run = dry_run
        self.with_bigquery = with_bigquery
        self.results = {}
        self.temp_dir = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize BigQuery writer for ingestion if needed
        if self.with_bigquery:
            self.bq_writer = BigQueryWriter(
                project_id='konveyn2ai',
                dataset_id='source_ingestion'
            )

        # Performance targets
        self.TARGET_FILES_PER_SOURCE = 100
        self.TARGET_TOTAL_TIME_MINUTES = 5
        self.TARGET_TOTAL_TIME_SECONDS = self.TARGET_TOTAL_TIME_MINUTES * 60

        # Template directory
        self.template_dir = Path(__file__).parent.parent / "examples" / "performance_templates"

        # Verify templates exist
        self._verify_templates()

    def _verify_templates(self):
        """Verify that all template files exist"""
        required_templates = [
            "kubernetes_template.yaml",
            "fastapi_template.py",
            "cobol_template.cbl",
            "irs_template.txt",
            "mumps_template.m"
        ]

        missing = []
        for template in required_templates:
            template_path = self.template_dir / template
            if not template_path.exists():
                missing.append(template)

        if missing:
            raise FileNotFoundError(f"Missing template files: {missing}")

        self.logger.info(f"All template files found in {self.template_dir}")

    def setup_test_environment(self):
        """Create temporary directory and test files"""
        self.temp_dir = tempfile.mkdtemp(prefix="performance_test_")
        self.logger.info(f"Created test directory: {self.temp_dir}")

        # Create subdirectories for each source type
        self.test_dirs = {
            "kubernetes": Path(self.temp_dir) / "k8s",
            "fastapi": Path(self.temp_dir) / "fastapi",
            "cobol": Path(self.temp_dir) / "cobol",
            "irs": Path(self.temp_dir) / "irs",
            "mumps": Path(self.temp_dir) / "mumps"
        }

        for source_type, dir_path in self.test_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_test_files(self):
        """Generate test files for each source type using templates"""
        self.logger.info(f"Generating {self.TARGET_FILES_PER_SOURCE} files per source type...")

        generators = {
            "kubernetes": self._generate_from_k8s_template,
            "fastapi": self._generate_from_fastapi_template,
            "cobol": self._generate_from_cobol_template,
            "irs": self._generate_from_irs_template,
            "mumps": self._generate_from_mumps_template
        }

        for source_type, generator in generators.items():
            start_time = time.time()
            generator(self.test_dirs[source_type])
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {source_type} files in {generation_time:.2f}s")

    def _load_template(self, template_name: str) -> str:
        """Load template content from file"""
        template_path = self.template_dir / template_name
        return template_path.read_text()

    def _generate_from_k8s_template(self, output_dir: Path):
        """Generate Kubernetes files from template"""
        template = self._load_template("kubernetes_template.yaml")

        for i in range(self.TARGET_FILES_PER_SOURCE):
            content = template.format(
                ID=str(i).zfill(3),
                REPLICAS=3 + (i % 5),
                IMAGE_TAG=i % 10,
                PORT=8080 + (i % 100),
                MEMORY_REQUEST=64 + (i % 128),
                CPU_REQUEST=250 + (i % 500),
                MEMORY_LIMIT=128 + (i % 256),
                CPU_LIMIT=500 + (i % 1000),
                LIVENESS_DELAY=30 + (i % 60),
                LIVENESS_PERIOD=10 + (i % 20),
                READINESS_DELAY=5 + (i % 15),
                READINESS_PERIOD=5 + (i % 10)
            )

            file_path = output_dir / f"deployment_{str(i).zfill(3)}.yaml"
            file_path.write_text(content)

    def _generate_from_fastapi_template(self, output_dir: Path):
        """Generate FastAPI files from template"""
        template = self._load_template("fastapi_template.py")

        for i in range(self.TARGET_FILES_PER_SOURCE):
            content = template.format(
                ID=str(i).zfill(3),  # Use zfill instead of f-string formatting
                ITEM_ID="{item_id}",  # Keep as placeholder for FastAPI path parameter
                MAX_ITEM_ID=10000,
                ID_OFFSET=i * 1000,
                PORT=8000 + i
            )

            file_path = output_dir / f"api_module_{str(i).zfill(3)}.py"
            file_path.write_text(content)

    def _generate_from_cobol_template(self, output_dir: Path):
        """Generate COBOL files from template"""
        template = self._load_template("cobol_template.cbl")

        for i in range(self.TARGET_FILES_PER_SOURCE):
            content = template.format(
                ID=str(i).zfill(3),
                FILLER_SIZE=50 + (i % 50)
            )

            file_path = output_dir / f"record_{str(i).zfill(3)}.cbl"
            file_path.write_text(content)

    def _generate_from_irs_template(self, output_dir: Path):
        """Generate IRS files from template"""
        template = self._load_template("irs_template.txt")

        for i in range(self.TARGET_FILES_PER_SOURCE):
            tax_year = 2020 + (i % 5)
            content = template.format(
                ID=str(i).zfill(3),
                TAX_YEAR=tax_year,
                VERSION=i % 10 + 1,
                AGI_AMOUNT="999999999999",
                TAX_AMOUNT="999999999999",
                WITHHELD_AMOUNT="999999999999",
                PROCESS_DATE=f"{tax_year}1231",
                DLN_NUMBER=f"{tax_year}{str(i % 365).zfill(3)}{str(i % 999).zfill(3)}",
                CYCLE_CODE=f"{tax_year}{str(i % 52 + 1).zfill(2)}01",
                AGI_LOW=f"{i * 1000:,}",
                AGI_HIGH=f"{i * 1000 + 50000:,}"
            )

            file_path = output_dir / f"imf_layout_{str(i).zfill(3)}.txt"
            file_path.write_text(content)

    def _generate_from_mumps_template(self, output_dir: Path):
        """Generate MUMPS files from template"""
        template = self._load_template("mumps_template.m")

        for i in range(self.TARGET_FILES_PER_SOURCE):
            file_number = 10000 + i
            content = template.format(
                ID=str(i).zfill(3),
                FILE_NUMBER=file_number,
                TEST_ID_1=1000 + i,
                TEST_ID_2=2000 + i,
                TEST_ID_3=3000 + i,
                SCORE_1=85.5 + (i % 15),
                SCORE_2=78.3 + (i % 22),
                SCORE_3=95.7 + (i % 5)
            )

            file_path = output_dir / f"vista_dd_{str(i).zfill(3)}.m"
            file_path.write_text(content)

    def run_performance_test(self) -> Dict:
        """Run the full performance test"""
        self.logger.info("Starting performance validation test...")

        total_start_time = time.time()

        # Test each source type
        for source_type in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]:
            self.logger.info(f"Testing {source_type} ingestion...")

            source_start_time = time.time()

            # Get parser and run ingestion
            parser = self._get_parser(source_type)
            files = list(self.test_dirs[source_type].glob("*"))

            chunks_processed = 0
            successful_files = 0

            if self.with_bigquery:
                # Parse files and write chunks directly to BigQuery
                try:
                    all_chunks = []
                    for file_path in files:
                        try:
                            # Parse file to chunks
                            parse_result = parser.parse_file(str(file_path))
                            all_chunks.extend(parse_result.chunks)
                            successful_files += 1
                        except Exception as e:
                            self.logger.warning(f"Error processing {file_path}: {e}")

                    chunks_processed = len(all_chunks)

                    # Write chunks to BigQuery
                    if all_chunks:
                        try:
                            # Generate run_id for this ingestion
                            run_id = f"perf_test_{source_type}_{int(time.time())}"
                            result = self.bq_writer.write_chunks(all_chunks, run_id)
                            self.logger.info(f"  Ingested to BigQuery: {successful_files} files, {chunks_processed} chunks, {result.rows_written} rows written")
                        except Exception as e:
                            self.logger.warning(f"Error writing to BigQuery for {source_type}: {e}")

                except Exception as e:
                    self.logger.warning(f"Error with BigQuery ingestion for {source_type}: {e}")
            else:
                # Original parsing-only logic
                for file_path in files:
                    try:
                        # Parse file to chunks
                        parse_result = parser.parse_file(str(file_path))
                        chunks_processed += len(parse_result.chunks)
                        successful_files += 1
                    except Exception as e:
                        self.logger.warning(f"Error processing {file_path}: {e}")

            source_end_time = time.time()
            source_duration = source_end_time - source_start_time

            self.results[source_type] = {
                "files_processed": successful_files,
                "files_attempted": len(files),
                "chunks_created": chunks_processed,
                "duration_seconds": source_duration,
                "files_per_second": successful_files / source_duration if source_duration > 0 else 0,
                "chunks_per_second": chunks_processed / source_duration if source_duration > 0 else 0,
                "success_rate": successful_files / len(files) if files else 0
            }

            self.logger.info(f"{source_type}: {successful_files}/{len(files)} files, {chunks_processed} chunks in {source_duration:.2f}s")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Calculate totals
        total_files = sum(r["files_processed"] for r in self.results.values())
        total_chunks = sum(r["chunks_created"] for r in self.results.values())

        self.results["totals"] = {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "total_duration_seconds": total_duration,
            "total_duration_minutes": total_duration / 60,
            "overall_files_per_second": total_files / total_duration if total_duration > 0 else 0,
            "overall_chunks_per_second": total_chunks / total_duration if total_duration > 0 else 0,
            "overall_success_rate": sum(r["success_rate"] for r in self.results.values() if r["success_rate"] is not None) / 5
        }

        return self.results

    def _get_parser(self, source_type: str):
        """Get appropriate parser for source type"""
        parsers = {
            "kubernetes": KubernetesParserImpl(),
            "fastapi": FastAPIParserImpl(),
            "cobol": COBOLParserImpl(),
            "irs": IRSParserImpl(),
            "mumps": MUMPSParserImpl()
        }
        return parsers[source_type]

    def evaluate_performance(self) -> Dict:
        """Evaluate if performance targets were met"""
        totals = self.results["totals"]

        evaluation = {
            "target_files_per_source": self.TARGET_FILES_PER_SOURCE,
            "target_total_time_minutes": self.TARGET_TOTAL_TIME_MINUTES,
            "actual_files_per_source": {
                source: self.results[source]["files_processed"]
                for source in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]
            },
            "actual_total_time_minutes": totals["total_duration_minutes"],
            "success_rates": {
                source: self.results[source]["success_rate"]
                for source in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]
            },
            "targets_met": {
                "files_per_source": all(
                    self.results[source]["files_processed"] >= self.TARGET_FILES_PER_SOURCE
                    for source in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]
                ),
                "total_time": totals["total_duration_minutes"] <= self.TARGET_TOTAL_TIME_MINUTES,
                "success_rates": all(
                    self.results[source]["success_rate"] >= 0.95  # 95% success rate
                    for source in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]
                )
            }
        }

        evaluation["overall_success"] = all(evaluation["targets_met"].values())

        return evaluation

    def cleanup(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up test directory: {self.temp_dir}")

    def print_results(self):
        """Print performance test results"""
        print("\n" + "="*80)
        print("PERFORMANCE VALIDATION RESULTS")
        print("="*80)

        # Per-source results
        for source_type in ["kubernetes", "fastapi", "cobol", "irs", "mumps"]:
            result = self.results[source_type]
            print(f"\n{source_type.upper()}:")
            print(f"  Files processed: {result['files_processed']}/{result['files_attempted']} ({result['success_rate']:.1%})")
            print(f"  Chunks created: {result['chunks_created']}")
            print(f"  Duration: {result['duration_seconds']:.2f}s")
            print(f"  Files/sec: {result['files_per_second']:.2f}")
            print(f"  Chunks/sec: {result['chunks_per_second']:.2f}")

        # Totals
        totals = self.results["totals"]
        print(f"\nTOTALS:")
        print(f"  Total files: {totals['total_files']}")
        print(f"  Total chunks: {totals['total_chunks']}")
        print(f"  Total duration: {totals['total_duration_minutes']:.2f} minutes")
        print(f"  Overall files/sec: {totals['overall_files_per_second']:.2f}")
        print(f"  Overall chunks/sec: {totals['overall_chunks_per_second']:.2f}")
        print(f"  Overall success rate: {totals['overall_success_rate']:.1%}")

        # Evaluation
        evaluation = self.evaluate_performance()
        print(f"\nPERFORMANCE EVALUATION:")
        print(f"  Target: {self.TARGET_FILES_PER_SOURCE} files per source in {self.TARGET_TOTAL_TIME_MINUTES} minutes")
        print(f"  Files per source target met: {evaluation['targets_met']['files_per_source']}")
        print(f"  Time target met: {evaluation['targets_met']['total_time']}")
        print(f"  Success rate target met: {evaluation['targets_met']['success_rates']}")
        print(f"  Overall success: {evaluation['overall_success']}")

        if evaluation['overall_success']:
            print(f"\n🎉 PERFORMANCE VALIDATION PASSED!")
        else:
            print(f"\n❌ PERFORMANCE VALIDATION FAILED")

            # Print specific failures
            if not evaluation['targets_met']['files_per_source']:
                print("  - File count targets not met")
            if not evaluation['targets_met']['total_time']:
                print("  - Time target exceeded")
            if not evaluation['targets_met']['success_rates']:
                print("  - Success rate targets not met")

        print("="*80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Performance validation for M1 Multi-Source Ingestion")
    parser.add_argument("--no-dry-run", action="store_true", help="Run with actual BigQuery writes")
    parser.add_argument("--with-bigquery", action="store_true", help="Ingest data into BigQuery (for T038 validation)")
    parser.add_argument("--files-per-source", type=int, default=100, help="Number of files per source type")
    parser.add_argument("--target-minutes", type=int, default=5, help="Target completion time in minutes")

    args = parser.parse_args()

    validator = TemplateBasedPerformanceValidator(
        dry_run=not args.no_dry_run,
        with_bigquery=args.with_bigquery
    )
    validator.TARGET_FILES_PER_SOURCE = args.files_per_source
    validator.TARGET_TOTAL_TIME_MINUTES = args.target_minutes
    validator.TARGET_TOTAL_TIME_SECONDS = args.target_minutes * 60

    try:
        # Setup and generate test data
        validator.setup_test_environment()
        validator.generate_test_files()

        # Run performance test
        results = validator.run_performance_test()

        # Print results
        validator.print_results()

        # Return appropriate exit code
        evaluation = validator.evaluate_performance()
        exit_code = 0 if evaluation["overall_success"] else 1

        return exit_code

    except Exception as e:
        validator.logger.error(f"Performance validation failed: {e}")
        return 1

    finally:
        validator.cleanup()


if __name__ == "__main__":
    exit(main())