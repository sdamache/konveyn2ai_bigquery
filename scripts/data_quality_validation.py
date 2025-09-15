#!/usr/bin/env python3
"""
Data Quality Validation Script for M1 Multi-Source Ingestion system

T038: Validates data quality in BigQuery - verify ≥100 rows per source type
"""

import logging
import os
import sys
from datetime import datetime

from google.cloud import bigquery

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from specs.contracts.parser_interfaces import SourceType
except ImportError:
    # Fallback: define minimal types
    from enum import Enum

    class SourceType(Enum):
        KUBERNETES = "kubernetes"
        FASTAPI = "fastapi"
        COBOL = "cobol"
        IRS = "irs"
        MUMPS = "mumps"


class DataQualityValidator:
    """Validates data quality metrics in BigQuery tables"""

    def __init__(
        self, project_id: str = "konveyn2ai", dataset_id: str = "source_ingestion"
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.MIN_ROWS_PER_SOURCE = 100
        self.MIN_CHUNKS_PER_SOURCE = 50
        self.MIN_SUCCESS_RATE = 0.95  # 95%

        # Expected source types
        self.expected_sources = [
            SourceType.KUBERNETES.value,
            SourceType.FASTAPI.value,
            SourceType.COBOL.value,
            SourceType.IRS.value,
            SourceType.MUMPS.value,
        ]

    def validate_table_existence(self) -> dict[str, bool]:
        """Validate that all required tables exist"""
        required_tables = ["source_metadata", "source_metadata_errors", "ingestion_log"]
        results = {}

        self.logger.info("Validating table existence...")

        try:
            dataset_ref = self.client.dataset(self.dataset_id)
            dataset = self.client.get_dataset(dataset_ref)
            existing_tables = {
                table.table_id for table in self.client.list_tables(dataset)
            }

            for table_name in required_tables:
                exists = table_name in existing_tables
                results[table_name] = exists
                status = "✅" if exists else "❌"
                self.logger.info(
                    f"{status} Table {table_name}: {'EXISTS' if exists else 'MISSING'}"
                )

        except Exception as e:
            self.logger.error(f"Error checking table existence: {e}")
            for table_name in required_tables:
                results[table_name] = False

        return results

    def get_source_metadata_counts(self) -> dict[str, dict[str, int]]:
        """Get row counts by source type from source_metadata table"""
        self.logger.info("Querying source_metadata for row counts by source type...")

        query = f"""
        SELECT
            source_type,
            COUNT(*) as total_rows,
            COUNT(DISTINCT artifact_id) as unique_artifacts,
            COUNT(DISTINCT parent_id) as unique_parents,
            AVG(content_tokens) as avg_tokens_per_chunk,
            MIN(collected_at) as earliest_ingestion,
            MAX(collected_at) as latest_ingestion
        FROM `{self.project_id}.{self.dataset_id}.source_metadata`
        GROUP BY source_type
        ORDER BY source_type
        """

        try:
            query_job = self.client.query(query)
            results = {}

            for row in query_job:
                source_type = row.source_type
                results[source_type] = {
                    "total_rows": row.total_rows,
                    "unique_artifacts": row.unique_artifacts,
                    "unique_parents": row.unique_parents,
                    "avg_tokens_per_chunk": (
                        round(row.avg_tokens_per_chunk, 2)
                        if row.avg_tokens_per_chunk
                        else 0
                    ),
                    "earliest_ingestion": row.earliest_ingestion,
                    "latest_ingestion": row.latest_ingestion,
                }

            return results

        except Exception as e:
            self.logger.error(f"Error querying source_metadata: {e}")
            return {}

    def get_error_statistics(self) -> dict[str, dict[str, int]]:
        """Get error statistics by source type"""
        self.logger.info("Querying source_metadata_errors for error statistics...")

        query = f"""
        SELECT
            source_type,
            error_class,
            COUNT(*) as error_count
        FROM `{self.project_id}.{self.dataset_id}.source_metadata_errors`
        GROUP BY source_type, error_class
        ORDER BY source_type, error_count DESC
        """

        try:
            query_job = self.client.query(query)
            results = {}

            for row in query_job:
                source_type = row.source_type
                if source_type not in results:
                    results[source_type] = {}
                results[source_type][row.error_class] = row.error_count

            return results

        except Exception as e:
            self.logger.error(f"Error querying source_metadata_errors: {e}")
            return {}

    def get_ingestion_statistics(self) -> dict[str, any]:
        """Get overall ingestion statistics"""
        self.logger.info("Querying ingestion_log for processing statistics...")

        query = f"""
        SELECT
            COUNT(*) as total_runs,
            COUNT(DISTINCT DATE(started_at)) as unique_dates,
            SUM(files_processed) as total_files_processed,
            SUM(chunks_created) as total_chunks_created,
            AVG(processing_duration_ms) as avg_processing_time_ms,
            MIN(started_at) as first_run,
            MAX(completed_at) as last_run,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*) as success_rate
        FROM `{self.project_id}.{self.dataset_id}.ingestion_log`
        """

        try:
            query_job = self.client.query(query)
            row = next(iter(query_job))

            return {
                "total_runs": row.total_runs,
                "unique_dates": row.unique_dates,
                "total_files_processed": row.total_files_processed,
                "total_chunks_created": row.total_chunks_created,
                "avg_processing_time_ms": (
                    round(row.avg_processing_time_ms, 2)
                    if row.avg_processing_time_ms
                    else 0
                ),
                "first_run": row.first_run,
                "last_run": row.last_run,
                "success_rate": round(row.success_rate, 4) if row.success_rate else 0,
            }

        except Exception as e:
            self.logger.error(f"Error querying ingestion_log: {e}")
            return {}

    def validate_data_quality(self) -> dict[str, any]:
        """Comprehensive data quality validation"""
        self.logger.info("Starting comprehensive data quality validation...")

        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "table_existence": self.validate_table_existence(),
            "source_counts": self.get_source_metadata_counts(),
            "error_stats": self.get_error_statistics(),
            "ingestion_stats": self.get_ingestion_statistics(),
            "quality_checks": {},
        }

        # Perform quality checks
        source_counts = validation_results["source_counts"]
        quality_checks = validation_results["quality_checks"]

        # Check 1: Minimum rows per source type (T038 requirement)
        quality_checks["min_rows_per_source"] = {}
        for source_type in self.expected_sources:
            if source_type in source_counts:
                row_count = source_counts[source_type]["total_rows"]
                meets_threshold = row_count >= self.MIN_ROWS_PER_SOURCE
                quality_checks["min_rows_per_source"][source_type] = {
                    "row_count": row_count,
                    "threshold": self.MIN_ROWS_PER_SOURCE,
                    "meets_threshold": meets_threshold,
                }
            else:
                quality_checks["min_rows_per_source"][source_type] = {
                    "row_count": 0,
                    "threshold": self.MIN_ROWS_PER_SOURCE,
                    "meets_threshold": False,
                }

        # Check 2: Data diversity (unique artifacts vs total rows)
        quality_checks["data_diversity"] = {}
        for source_type, stats in source_counts.items():
            total_rows = stats["total_rows"]
            unique_artifacts = stats["unique_artifacts"]
            diversity_ratio = unique_artifacts / total_rows if total_rows > 0 else 0

            quality_checks["data_diversity"][source_type] = {
                "total_rows": total_rows,
                "unique_artifacts": unique_artifacts,
                "diversity_ratio": round(diversity_ratio, 4),
                "healthy_diversity": diversity_ratio
                > 0.1,  # At least 10% unique artifacts
            }

        # Check 3: Error rate analysis
        ingestion_stats = validation_results["ingestion_stats"]
        if ingestion_stats:
            success_rate = ingestion_stats.get("success_rate", 0)
            quality_checks["error_rate"] = {
                "success_rate": success_rate,
                "meets_threshold": success_rate >= self.MIN_SUCCESS_RATE,
                "threshold": self.MIN_SUCCESS_RATE,
            }

        # Check 4: Recent activity (data freshness)
        quality_checks["data_freshness"] = {}
        current_time = datetime.utcnow()
        for source_type, stats in source_counts.items():
            latest_ingestion = stats.get("latest_ingestion")
            if latest_ingestion:
                # Handle timezone differences
                if latest_ingestion.tzinfo is None:
                    # latest_ingestion is naive, assume UTC
                    latest_ingestion_utc = latest_ingestion
                else:
                    # latest_ingestion is timezone-aware, convert to UTC
                    latest_ingestion_utc = latest_ingestion.replace(tzinfo=None)

                hours_since_last = (
                    current_time - latest_ingestion_utc
                ).total_seconds() / 3600
                quality_checks["data_freshness"][source_type] = {
                    "latest_ingestion": (
                        latest_ingestion.isoformat()
                        if hasattr(latest_ingestion, "isoformat")
                        else str(latest_ingestion)
                    ),
                    "hours_since_last": round(hours_since_last, 2),
                    "is_recent": hours_since_last < 24,  # Within last 24 hours
                }

        # Overall quality assessment
        quality_checks["overall_assessment"] = self._assess_overall_quality(
            quality_checks
        )

        return validation_results

    def _assess_overall_quality(self, quality_checks: dict) -> dict[str, any]:
        """Assess overall data quality based on individual checks"""

        # Count passing checks
        source_threshold_passes = sum(
            1
            for check in quality_checks.get("min_rows_per_source", {}).values()
            if check["meets_threshold"]
        )
        total_sources = len(self.expected_sources)

        diversity_passes = sum(
            1
            for check in quality_checks.get("data_diversity", {}).values()
            if check["healthy_diversity"]
        )

        error_rate_pass = quality_checks.get("error_rate", {}).get(
            "meets_threshold", False
        )

        # Calculate overall score
        source_score = (source_threshold_passes / total_sources) * 40  # 40% weight
        diversity_score = (
            diversity_passes / max(1, len(quality_checks.get("data_diversity", {})))
        ) * 30  # 30% weight
        error_score = 20 if error_rate_pass else 0  # 20% weight
        freshness_score = (
            10
            if any(
                check.get("is_recent", False)
                for check in quality_checks.get("data_freshness", {}).values()
            )
            else 0
        )  # 10% weight

        overall_score = source_score + diversity_score + error_score + freshness_score

        return {
            "overall_score": round(overall_score, 2),
            "max_score": 100,
            "source_threshold_passes": f"{source_threshold_passes}/{total_sources}",
            "diversity_passes": f"{diversity_passes}/{len(quality_checks.get('data_diversity', {}))}",
            "error_rate_acceptable": error_rate_pass,
            "data_is_fresh": freshness_score > 0,
            "quality_grade": self._get_quality_grade(overall_score),
        }

    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to quality grade"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"

    def print_validation_report(self, results: dict):
        """Print comprehensive data quality validation report"""
        print("\n" + "=" * 80)
        print("DATA QUALITY VALIDATION REPORT")
        print("=" * 80)
        print(f"Validation Time: {results['timestamp']}")
        print()

        # Table existence
        print("📋 TABLE EXISTENCE:")
        print("-" * 40)
        for table, exists in results["table_existence"].items():
            status = "✅ EXISTS" if exists else "❌ MISSING"
            print(f"  {table}: {status}")
        print()

        # Source counts (T038 main requirement)
        print("📊 SOURCE TYPE DATA COUNTS (T038 Validation):")
        print("-" * 60)
        source_counts = results["source_counts"]
        quality_checks = results["quality_checks"]

        for source_type in self.expected_sources:
            if source_type in source_counts:
                stats = source_counts[source_type]
                threshold_check = quality_checks["min_rows_per_source"].get(
                    source_type, {}
                )

                status = (
                    "✅ PASS"
                    if threshold_check.get("meets_threshold", False)
                    else "❌ FAIL"
                )
                print(f"  {source_type.upper()}:")
                print(
                    f"    Rows: {stats['total_rows']:,} (threshold: {self.MIN_ROWS_PER_SOURCE}) {status}"
                )
                print(f"    Unique artifacts: {stats['unique_artifacts']:,}")
                print(f"    Avg tokens/chunk: {stats['avg_tokens_per_chunk']}")
                if stats["latest_ingestion"]:
                    print(f"    Latest ingestion: {stats['latest_ingestion']}")
            else:
                print(f"  {source_type.upper()}: ❌ NO DATA FOUND")
            print()

        # Error statistics
        error_stats = results["error_stats"]
        if error_stats:
            print("⚠️  ERROR STATISTICS:")
            print("-" * 40)
            for source_type, errors in error_stats.items():
                print(f"  {source_type.upper()}:")
                for error_class, count in errors.items():
                    print(f"    {error_class}: {count:,} errors")
            print()

        # Ingestion statistics
        ingestion_stats = results["ingestion_stats"]
        if ingestion_stats:
            print("🔄 INGESTION STATISTICS:")
            print("-" * 40)
            print(f"  Total runs: {ingestion_stats['total_runs']:,}")
            print(f"  Files processed: {ingestion_stats['total_files_processed']:,}")
            print(f"  Chunks created: {ingestion_stats['total_chunks_created']:,}")
            print(f"  Success rate: {ingestion_stats['success_rate']:.2%}")
            print(
                f"  Avg processing time: {ingestion_stats['avg_processing_time_ms']:.2f}ms"
            )
            if ingestion_stats["first_run"]:
                print(f"  First run: {ingestion_stats['first_run']}")
            if ingestion_stats["last_run"]:
                print(f"  Last run: {ingestion_stats['last_run']}")
            print()

        # Overall assessment
        overall = quality_checks.get("overall_assessment", {})
        if overall:
            print("🎯 OVERALL QUALITY ASSESSMENT:")
            print("-" * 40)
            print(f"  Overall Score: {overall['overall_score']}/100")
            print(f"  Quality Grade: {overall['quality_grade']}")
            print(f"  Source thresholds met: {overall['source_threshold_passes']}")
            print(f"  Diversity checks passed: {overall['diversity_passes']}")
            print(
                f"  Error rate acceptable: {'✅' if overall['error_rate_acceptable'] else '❌'}"
            )
            print(f"  Data freshness: {'✅' if overall['data_is_fresh'] else '❌'}")
            print()

        # T038 specific validation result
        min_rows_checks = quality_checks.get("min_rows_per_source", {})
        all_sources_pass = all(
            check.get("meets_threshold", False) for check in min_rows_checks.values()
        )

        print("🏆 T038 VALIDATION RESULT:")
        print("-" * 40)
        if all_sources_pass:
            print("✅ PASSED: All source types have ≥100 rows in BigQuery")
            print(
                "   The M1 Multi-Source Ingestion system meets data quality requirements!"
            )
        else:
            print("❌ FAILED: Some source types have <100 rows in BigQuery")
            failing_sources = [
                source
                for source, check in min_rows_checks.items()
                if not check.get("meets_threshold", False)
            ]
            print(f"   Failing sources: {', '.join(failing_sources)}")

        print("=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Quality Validation for M1 Multi-Source Ingestion"
    )
    parser.add_argument("--project", default="konveyn2ai", help="BigQuery project ID")
    parser.add_argument(
        "--dataset", default="source_ingestion", help="BigQuery dataset ID"
    )
    parser.add_argument(
        "--min-rows", type=int, default=100, help="Minimum rows per source type"
    )

    args = parser.parse_args()

    try:
        validator = DataQualityValidator(
            project_id=args.project, dataset_id=args.dataset
        )
        validator.MIN_ROWS_PER_SOURCE = args.min_rows

        # Run validation
        results = validator.validate_data_quality()

        # Print report
        validator.print_validation_report(results)

        # Determine exit code based on T038 requirement
        min_rows_checks = results.get("quality_checks", {}).get(
            "min_rows_per_source", {}
        )
        all_sources_pass = all(
            check.get("meets_threshold", False) for check in min_rows_checks.values()
        )

        return 0 if all_sources_pass else 1

    except Exception as e:
        print(f"❌ Data quality validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
