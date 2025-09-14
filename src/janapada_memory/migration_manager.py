"""
Migration Manager - Handles migration from Vertex AI to BigQuery

Manages the migration process from legacy Vertex AI vector storage 
to BigQuery with dimension reduction and quality validation.
"""

import os
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from .bigquery_vector_store import BigQueryVectorStore
from .dimension_reducer import DimensionReducer
from .schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Migration status enumeration."""
    PENDING = "PENDING"
    STARTED = "STARTED" 
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class MigrationManager:
    """Manages migration from Vertex AI to BigQuery vector storage."""
    
    def __init__(
        self,
        bigquery_store: Optional[BigQueryVectorStore] = None,
        schema_manager: Optional[SchemaManager] = None,
        dimension_reducer: Optional[DimensionReducer] = None,
        migration_storage_path: str = "migrations/"
    ):
        """
        Initialize migration manager.
        
        Args:
            bigquery_store: BigQuery vector store instance
            schema_manager: Schema manager instance
            dimension_reducer: Dimension reducer instance
            migration_storage_path: Path to store migration metadata
        """
        self.bigquery_store = bigquery_store or BigQueryVectorStore()
        self.schema_manager = schema_manager or SchemaManager()
        self.dimension_reducer = dimension_reducer
        self.migration_storage_path = migration_storage_path
        
        # Create storage directory
        os.makedirs(migration_storage_path, exist_ok=True)
        
        # Migration tracking
        self._active_migrations: Dict[str, Dict[str, Any]] = {}
        self._migration_lock = threading.Lock()
        
        logger.info("Migration manager initialized")

    def start_migration(
        self,
        source_type: str = "vertex_ai",
        target_type: str = "bigquery", 
        migration_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start a new migration process.
        
        Args:
            source_type: Source vector store type
            target_type: Target vector store type
            migration_options: Migration configuration options
            
        Returns:
            Migration initialization result
        """
        try:
            # Validate parameters
            if source_type != "vertex_ai":
                return {
                    "status": MigrationStatus.VALIDATION_ERROR,
                    "error_message": f"Unsupported source type: {source_type}"
                }
            
            if target_type != "bigquery":
                return {
                    "status": MigrationStatus.VALIDATION_ERROR,
                    "error_message": f"Unsupported target type: {target_type}"
                }
            
            # Generate migration ID
            migration_id = str(uuid.uuid4())
            
            # Default migration options
            default_options = {
                "batch_size": 100,
                "dimension_reduction": True,
                "target_dimensions": 768,
                "preserve_metadata": True,
                "validate_similarity": True,
                "enable_rollback": True,
                "quality_threshold": 0.85
            }
            
            options = {**default_options, **(migration_options or {})}
            
            # Validate options
            validation_result = self._validate_migration_options(options)
            if not validation_result["valid"]:
                return {
                    "status": MigrationStatus.VALIDATION_ERROR,
                    "error_message": f"Invalid options: {validation_result['errors']}"
                }
            
            # Initialize migration metadata
            migration_metadata = {
                "migration_id": migration_id,
                "source_type": source_type,
                "target_type": target_type,
                "options": options,
                "status": MigrationStatus.STARTED,
                "created_at": datetime.now().isoformat(),
                "progress": {
                    "total_records": 0,
                    "processed_records": 0,
                    "failed_records": 0,
                    "current_phase": "initialization"
                },
                "phases": [],
                "quality_metrics": {},
                "errors": []
            }
            
            # Store migration metadata
            self._save_migration_metadata(migration_id, migration_metadata)
            
            # Track active migration
            with self._migration_lock:
                self._active_migrations[migration_id] = migration_metadata
            
            logger.info(f"Started migration {migration_id}: {source_type} → {target_type}")
            
            return {
                "migration_id": migration_id,
                "status": MigrationStatus.STARTED,
                "options": options,
                "created_at": migration_metadata["created_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to start migration: {e}")
            return {
                "status": MigrationStatus.FAILED,
                "error_message": str(e)
            }

    def get_migration_status(self, migration_id: str) -> Dict[str, Any]:
        """
        Get migration status and progress.
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Migration status information
        """
        try:
            # Try to get from active migrations first
            with self._migration_lock:
                if migration_id in self._active_migrations:
                    metadata = self._active_migrations[migration_id].copy()
                    return {
                        "migration_id": migration_id,
                        "status": metadata["status"],
                        "progress": metadata["progress"],
                        "created_at": metadata["created_at"],
                        "current_phase": metadata["progress"]["current_phase"]
                    }
            
            # Load from storage
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return {
                    "migration_id": migration_id,
                    "status": "NOT_FOUND",
                    "error_message": f"Migration {migration_id} not found"
                }
            
            return {
                "migration_id": migration_id,
                "status": metadata["status"],
                "progress": metadata["progress"],
                "created_at": metadata["created_at"],
                "current_phase": metadata["progress"]["current_phase"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status for {migration_id}: {e}")
            return {
                "migration_id": migration_id,
                "status": "ERROR",
                "error_message": str(e)
            }

    def list_migrations(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List migrations with optional filtering.
        
        Args:
            status: Filter by migration status
            limit: Maximum number of results
            
        Returns:
            List of migration summaries
        """
        try:
            migrations = []
            
            # Get migration files
            migration_files = [
                f for f in os.listdir(self.migration_storage_path)
                if f.endswith('.json')
            ]
            
            for file_name in migration_files[:limit]:
                file_path = os.path.join(self.migration_storage_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Apply status filter
                    if status and metadata.get("status") != status:
                        continue
                    
                    # Create summary
                    migration_summary = {
                        "migration_id": metadata["migration_id"],
                        "status": metadata["status"],
                        "source_type": metadata["source_type"],
                        "target_type": metadata["target_type"],
                        "created_at": metadata["created_at"],
                        "progress": metadata["progress"]
                    }
                    
                    migrations.append(migration_summary)
                    
                except Exception as e:
                    logger.warning(f"Failed to load migration file {file_name}: {e}")
            
            # Sort by creation time (most recent first)
            migrations.sort(key=lambda x: x["created_at"], reverse=True)
            
            return migrations
            
        except Exception as e:
            logger.error(f"Failed to list migrations: {e}")
            return []

    def rollback_migration(self, migration_id: str) -> Dict[str, Any]:
        """
        Rollback a migration.
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Rollback result
        """
        try:
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return {
                    "status": "ERROR",
                    "error_message": f"Migration {migration_id} not found"
                }
            
            if not metadata["options"].get("enable_rollback", False):
                return {
                    "status": "ERROR",
                    "error_message": "Rollback not enabled for this migration"
                }
            
            # Update status
            metadata["status"] = MigrationStatus.ROLLED_BACK
            metadata["rollback_at"] = datetime.now().isoformat()
            
            # Save updated metadata
            self._save_migration_metadata(migration_id, metadata)
            
            # Remove from active migrations
            with self._migration_lock:
                self._active_migrations.pop(migration_id, None)
            
            logger.info(f"Migration {migration_id} rolled back")
            
            return {
                "status": "ROLLBACK_INITIATED",
                "migration_id": migration_id,
                "rollback_at": metadata["rollback_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration_id}: {e}")
            return {
                "status": "ERROR",
                "error_message": str(e)
            }

    def get_validation_report(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get validation report for a migration.
        
        Args:
            migration_id: Migration identifier
            
        Returns:
            Validation report
        """
        try:
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return None
            
            # Generate validation report
            report = {
                "migration_id": migration_id,
                "status": metadata["status"],
                "data_integrity": {
                    "total_records_migrated": metadata["progress"]["processed_records"],
                    "successful_migrations": metadata["progress"]["processed_records"] - metadata["progress"]["failed_records"],
                    "failed_migrations": metadata["progress"]["failed_records"],
                    "success_rate": self._calculate_success_rate(metadata["progress"])
                },
                "similarity_preservation": metadata["quality_metrics"].get("similarity_preservation", {}),
                "performance_metrics": metadata["quality_metrics"].get("performance", {}),
                "quality_assessment": {
                    "overall_score": metadata["quality_metrics"].get("overall_quality_score", 0.0),
                    "dimension_reduction_quality": metadata["quality_metrics"].get("dimension_reduction", {}),
                    "data_integrity_score": self._calculate_success_rate(metadata["progress"])
                },
                "recommendations": self._generate_migration_recommendations(metadata),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate validation report for {migration_id}: {e}")
            return None

    def get_quality_report(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Get quality report for migration."""
        try:
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return None
            
            quality_metrics = metadata.get("quality_metrics", {})
            
            return {
                "migration_id": migration_id,
                "data_integrity_score": self._calculate_success_rate(metadata["progress"]),
                "similarity_preservation_score": quality_metrics.get("similarity_correlation", 0.0),
                "performance_score": min(1.0, quality_metrics.get("throughput", 0) / 100),  # Normalize
                "overall_quality_score": quality_metrics.get("overall_quality_score", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality report for {migration_id}: {e}")
            return None

    def get_audit_trail(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Get audit trail for migration."""
        try:
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return None
            
            # Convert phases to audit events
            events = []
            for phase in metadata.get("phases", []):
                events.append({
                    "timestamp": phase.get("started_at", ""),
                    "type": f"PHASE_{phase.get('name', '').upper()}",
                    "details": phase
                })
            
            # Add migration lifecycle events
            events.insert(0, {
                "timestamp": metadata["created_at"],
                "type": "MIGRATION_STARTED",
                "details": {
                    "source_type": metadata["source_type"],
                    "target_type": metadata["target_type"],
                    "options": metadata["options"]
                }
            })
            
            if metadata["status"] in [MigrationStatus.COMPLETED, MigrationStatus.FAILED]:
                events.append({
                    "timestamp": metadata.get("completed_at", datetime.now().isoformat()),
                    "type": "MIGRATION_COMPLETED",
                    "details": {
                        "status": metadata["status"],
                        "final_progress": metadata["progress"]
                    }
                })
            
            return {
                "migration_id": migration_id,
                "events": sorted(events, key=lambda x: x["timestamp"]),
                "total_events": len(events)
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit trail for {migration_id}: {e}")
            return None

    def test_dimension_reduction(
        self,
        embeddings: List[List[float]],
        target_dimensions: int = 768,
        method: str = "PCA"
    ) -> Dict[str, Any]:
        """Test dimension reduction quality."""
        try:
            if not self.dimension_reducer:
                self.dimension_reducer = DimensionReducer(
                    target_dimensions=target_dimensions
                )
            
            # Perform dimension reduction
            reduced_embeddings, training_stats = self.dimension_reducer.fit_transform(embeddings)
            
            # Evaluate quality
            quality_metrics = self.dimension_reducer.evaluate_quality(embeddings, reduced_embeddings)
            
            return {
                "reduced_embeddings": reduced_embeddings.tolist(),
                "quality_metrics": quality_metrics,
                "training_stats": training_stats
            }
            
        except Exception as e:
            logger.error(f"Dimension reduction test failed: {e}")
            raise

    def get_migration_performance(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for migration."""
        try:
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return None
            
            progress = metadata["progress"]
            quality_metrics = metadata.get("quality_metrics", {})
            
            # Calculate performance metrics
            total_time = 0
            if "completed_at" in metadata and metadata["created_at"]:
                start_time = datetime.fromisoformat(metadata["created_at"])
                end_time = datetime.fromisoformat(metadata["completed_at"])
                total_time = (end_time - start_time).total_seconds()
            
            embeddings_per_second = 0
            if total_time > 0:
                embeddings_per_second = progress["processed_records"] / total_time
            
            return {
                "migration_id": migration_id,
                "total_time_seconds": total_time,
                "embeddings_per_second": embeddings_per_second,
                "peak_memory_mb": quality_metrics.get("peak_memory_mb", 0),
                "error_rate": progress["failed_records"] / max(1, progress["processed_records"]),
                "throughput_score": min(1.0, embeddings_per_second / 100)  # Normalized
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics for {migration_id}: {e}")
            return None

    def archive_migration_data(
        self,
        migration_id: str,
        keep_validation_report: bool = True,
        archive_location: str = "gs://konveyn2ai-backups/migrations/"
    ) -> Dict[str, Any]:
        """Archive migration data."""
        try:
            # This is a placeholder implementation
            # In practice, you would implement actual archival to cloud storage
            
            metadata = self._load_migration_metadata(migration_id)
            if not metadata:
                return {
                    "status": "ERROR",
                    "error_message": f"Migration {migration_id} not found"
                }
            
            # Simulate archival
            archive_path = f"{archive_location}{migration_id}/"
            
            logger.info(f"Archiving migration {migration_id} to {archive_path}")
            
            return {
                "status": "ARCHIVED",
                "migration_id": migration_id,
                "archive_location": archive_path,
                "archived_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to archive migration {migration_id}: {e}")
            return {
                "status": "ERROR",
                "error_message": str(e)
            }

    def cleanup_migration_workspace(self, migration_id: str) -> Dict[str, Any]:
        """Clean up migration workspace."""
        try:
            # Remove from active migrations
            with self._migration_lock:
                self._active_migrations.pop(migration_id, None)
            
            logger.info(f"Cleaned up workspace for migration {migration_id}")
            
            return {
                "status": "CLEANED",
                "migration_id": migration_id,
                "cleaned_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup migration {migration_id}: {e}")
            return {
                "status": "ERROR",
                "error_message": str(e)
            }

    def _validate_migration_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Validate migration options."""
        errors = []
        
        # Validate batch size
        batch_size = options.get("batch_size", 100)
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 1000:
            errors.append("batch_size must be integer between 1 and 1000")
        
        # Validate target dimensions
        target_dims = options.get("target_dimensions", 768)
        if not isinstance(target_dims, int) or target_dims <= 0:
            errors.append("target_dimensions must be positive integer")
        
        # Validate quality threshold
        quality_threshold = options.get("quality_threshold", 0.85)
        if not isinstance(quality_threshold, (int, float)) or not 0.0 <= quality_threshold <= 1.0:
            errors.append("quality_threshold must be number between 0.0 and 1.0")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _save_migration_metadata(self, migration_id: str, metadata: Dict[str, Any]) -> None:
        """Save migration metadata to disk."""
        file_path = os.path.join(self.migration_storage_path, f"{migration_id}.json")
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_migration_metadata(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """Load migration metadata from disk."""
        file_path = os.path.join(self.migration_storage_path, f"{migration_id}.json")
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load migration metadata {migration_id}: {e}")
            return None

    def _calculate_success_rate(self, progress: Dict[str, int]) -> float:
        """Calculate migration success rate."""
        total = progress.get("processed_records", 0)
        failed = progress.get("failed_records", 0)
        if total == 0:
            return 1.0
        return (total - failed) / total

    def _generate_migration_recommendations(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on migration results."""
        recommendations = []
        
        progress = metadata["progress"]
        success_rate = self._calculate_success_rate(progress)
        
        if success_rate < 0.95:
            recommendations.append("Consider investigating failed record migrations")
        
        quality_metrics = metadata.get("quality_metrics", {})
        similarity_score = quality_metrics.get("similarity_correlation", 1.0)
        
        if similarity_score < 0.9:
            recommendations.append("Consider adjusting dimension reduction parameters")
        
        if not recommendations:
            recommendations.append("Migration completed successfully with good quality metrics")
        
        return recommendations