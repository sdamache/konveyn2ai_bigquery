"""
Integration test: BigQuery setup workflow

This test validates the complete setup workflow:
`make setup` → Tables created → Vector index active → Ready for use

IMPORTANT: This test MUST FAIL initially (TDD Red phase)
Implementation will be done in T015-T028.
"""

import pytest
import subprocess
import time
import os


class TestSetupWorkflow:
    """Integration tests for BigQuery setup workflow."""
    
    @pytest.fixture(scope="class")
    def schema_manager(self):
        """Schema manager - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.schema_manager import SchemaManager
        return SchemaManager()
    
    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """BigQuery client - will fail until implementation exists."""
        # This import will fail until implementation is done
        from src.janapada_memory.bigquery_vector_store import BigQueryVectorStore
        return BigQueryVectorStore()

    def test_make_setup_command_success(self):
        """Test `make setup` command executes successfully."""
        
        # Set required environment variables for testing
        test_env = os.environ.copy()
        test_env.update({
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
            "BIGQUERY_DATASET_ID": "semantic_gap_detector"
        })
        
        # Run make setup command
        result = subprocess.run(
            ["make", "setup"],
            capture_output=True,
            text=True,
            env=test_env,
            timeout=60  # 60 second timeout
        )
        
        # Should execute without error
        assert result.returncode == 0, f"make setup failed with error: {result.stderr}"
        
        # Should contain success indicators in output
        output = result.stdout.lower()
        assert "bigquery setup completed" in output or "setup completed" in output
        print(f"✅ make setup completed successfully")

    def test_dataset_creation(self, schema_manager):
        """Test BigQuery dataset is created correctly."""
        
        # Verify dataset exists
        dataset_info = schema_manager.get_dataset_info()
        assert dataset_info is not None
        assert dataset_info["dataset_id"] == "semantic_gap_detector"
        assert dataset_info["project_id"] == "konveyn2ai"
        assert "location" in dataset_info
        
        print(f"✅ Dataset created: {dataset_info['project_id']}.{dataset_info['dataset_id']}")

    def test_tables_creation(self, schema_manager):
        """Test all required tables are created with correct schema."""
        
        # Check table existence
        tables_info = schema_manager.list_tables()
        table_names = [table["name"] for table in tables_info["tables"]]
        
        required_tables = ["source_metadata", "source_embeddings", "gap_metrics"]
        for table_name in required_tables:
            assert table_name in table_names, f"Required table {table_name} not found"
        
        print(f"✅ All {len(required_tables)} required tables created")
        
        # Validate table schemas
        for table_name in required_tables:
            table_schema = schema_manager.get_table_schema(table_name)
            assert table_schema is not None
            
            # Check specific schema requirements
            if table_name == "source_metadata":
                self._validate_metadata_table_schema(table_schema)
            elif table_name == "source_embeddings":
                self._validate_embeddings_table_schema(table_schema)
            elif table_name == "gap_metrics":
                self._validate_metrics_table_schema(table_schema)

    def _validate_metadata_table_schema(self, schema):
        """Validate source_metadata table schema."""
        field_names = [field["name"] for field in schema]
        
        required_fields = [
            "chunk_id", "source", "artifact_type", "text_content",
            "kind", "api_path", "record_name", "created_at", "partition_date"
        ]
        
        for field in required_fields:
            assert field in field_names, f"Missing field {field} in source_metadata"

    def _validate_embeddings_table_schema(self, schema):
        """Validate source_embeddings table schema."""
        field_names = [field["name"] for field in schema]
        
        required_fields = [
            "chunk_id", "embedding", "embedding_model", "created_at", "partition_date"
        ]
        
        for field in required_fields:
            assert field in field_names, f"Missing field {field} in source_embeddings"
        
        # Validate embedding field is array type
        embedding_field = next(f for f in schema if f["name"] == "embedding")
        assert embedding_field["type"] == "ARRAY<FLOAT64>"

    def _validate_metrics_table_schema(self, schema):
        """Validate gap_metrics table schema."""
        field_names = [field["name"] for field in schema]
        
        required_fields = [
            "analysis_id", "chunk_id", "metric_type", "metric_value",
            "metadata", "created_at", "partition_date"
        ]
        
        for field in required_fields:
            assert field in field_names, f"Missing field {field} in gap_metrics"

    def test_table_partitioning(self, schema_manager):
        """Test tables are properly partitioned."""
        
        required_tables = ["source_metadata", "source_embeddings", "gap_metrics"]
        
        for table_name in required_tables:
            partition_info = schema_manager.get_table_partition_info(table_name)
            assert partition_info is not None
            
            # Should be partitioned by date
            assert partition_info["type"] == "TIME"
            assert partition_info["field"] == "partition_date"
            assert partition_info["expiration_days"] == 365
        
        print("✅ All tables properly partitioned by date")

    def test_table_clustering(self, schema_manager):
        """Test tables are properly clustered for performance."""
        
        clustering_config = {
            "source_metadata": ["artifact_type", "source", "chunk_id"],
            "source_embeddings": ["chunk_id"],
            "gap_metrics": ["analysis_id", "metric_type", "chunk_id"]
        }
        
        for table_name, expected_clustering in clustering_config.items():
            cluster_info = schema_manager.get_table_clustering_info(table_name)
            assert cluster_info is not None
            
            # Verify clustering fields
            assert cluster_info["clustering_fields"] == expected_clustering
        
        print("✅ All tables properly clustered for performance")

    def test_vector_index_creation(self, schema_manager):
        """Test vector index is created and becomes active."""
        
        # Check if vector index exists
        indexes_info = schema_manager.list_indexes()
        index_names = [index["name"] for index in indexes_info["indexes"]]
        
        assert "embedding_similarity_index" in index_names, "Vector index not created"
        
        # Get index details
        index_info = next(
            idx for idx in indexes_info["indexes"] 
            if idx["name"] == "embedding_similarity_index"
        )
        
        # Validate index configuration
        assert index_info["table"] == "source_embeddings"
        assert index_info["column"] == "embedding"
        assert index_info["index_type"] == "IVF"
        assert index_info["distance_type"] == "COSINE"
        
        # Index should eventually become active (may take time)
        max_wait_time = 60  # 60 seconds
        wait_time = 0
        
        while wait_time < max_wait_time:
            current_status = schema_manager.get_index_status("embedding_similarity_index")
            if current_status["status"] == "ACTIVE":
                break
            elif current_status["status"] == "ERROR":
                pytest.fail(f"Vector index creation failed: {current_status.get('error_message', 'Unknown error')}")
            
            time.sleep(5)
            wait_time += 5
        
        # Final status check
        final_status = schema_manager.get_index_status("embedding_similarity_index")
        assert final_status["status"] in ["ACTIVE", "CREATING"], f"Index status: {final_status['status']}"
        
        print(f"✅ Vector index created with status: {final_status['status']}")

    def test_setup_idempotency(self):
        """Test setup command is idempotent (can be run multiple times)."""
        
        test_env = os.environ.copy()
        test_env.update({
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai", 
            "BIGQUERY_DATASET_ID": "semantic_gap_detector"
        })
        
        # Run setup command twice
        result1 = subprocess.run(
            ["make", "setup"],
            capture_output=True,
            text=True,
            env=test_env,
            timeout=60
        )
        
        result2 = subprocess.run(
            ["make", "setup"],
            capture_output=True,
            text=True,
            env=test_env,
            timeout=60
        )
        
        # Both should succeed (idempotent)
        assert result1.returncode == 0, "First setup run failed"
        assert result2.returncode == 0, "Second setup run failed (not idempotent)"
        
        print("✅ Setup command is idempotent")

    def test_environment_variable_validation(self):
        """Test setup fails gracefully with missing environment variables."""
        
        # Test with missing GOOGLE_CLOUD_PROJECT
        incomplete_env = os.environ.copy()
        incomplete_env.pop("GOOGLE_CLOUD_PROJECT", None)
        
        result = subprocess.run(
            ["make", "setup"],
            capture_output=True,
            text=True,
            env=incomplete_env,
            timeout=30
        )
        
        # Should fail with clear error message
        assert result.returncode != 0
        assert "GOOGLE_CLOUD_PROJECT" in result.stderr
        
        print("✅ Environment variable validation working")

    def test_health_check_after_setup(self, bigquery_client):
        """Test health check passes after successful setup."""
        
        # Run health check
        health_status = bigquery_client.health_check()
        assert health_status is not None
        
        # Should indicate healthy status
        assert health_status["status"] == "healthy"
        assert health_status["bigquery_connection"] is True
        assert "vector_index_status" in health_status
        
        print(f"✅ Health check passed: {health_status['status']}")

    def test_basic_operations_after_setup(self, bigquery_client):
        """Test basic operations work after setup."""
        
        # Test basic embedding insertion
        test_data = {
            "chunk_id": "setup_test_001",
            "text_content": "This is a test chunk for setup validation",
            "source": "test/setup_validation.py",
            "artifact_type": "test",
            "embedding": [0.1] * 768
        }
        
        # Should be able to insert without error
        result = bigquery_client.insert_embedding(
            chunk_data=test_data,
            embedding=test_data["embedding"]
        )
        
        assert result is not None
        assert result["chunk_id"] == test_data["chunk_id"]
        
        # Should be able to retrieve
        retrieved = bigquery_client.get_embedding_by_id(test_data["chunk_id"])
        assert retrieved is not None
        assert retrieved["chunk_id"] == test_data["chunk_id"]
        
        # Should be able to search
        search_results = bigquery_client.search_similar_text(
            query_text="test chunk",
            limit=5
        )
        assert isinstance(search_results, list)
        
        print("✅ Basic operations working after setup")

    def test_make_diagnose_after_setup(self):
        """Test `make diagnose` command works after setup."""
        
        test_env = os.environ.copy()
        test_env.update({
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
            "BIGQUERY_DATASET_ID": "semantic_gap_detector"
        })
        
        # Run diagnostics
        result = subprocess.run(
            ["make", "diagnose"],
            capture_output=True,
            text=True,
            env=test_env,
            timeout=30
        )
        
        # Should succeed and show system status
        assert result.returncode == 0
        
        output = result.stdout.lower()
        # Should contain diagnostic information
        assert any(keyword in output for keyword in ["bigquery", "connection", "status", "tables"])
        
        print("✅ Diagnostics command working")

    def test_setup_completion_indicators(self):
        """Test setup shows proper completion indicators."""
        
        test_env = os.environ.copy()
        test_env.update({
            "GOOGLE_CLOUD_PROJECT": "konveyn2ai",
            "BIGQUERY_DATASET_ID": "semantic_gap_detector"
        })
        
        result = subprocess.run(
            ["make", "setup"],
            capture_output=True,
            text=True,
            env=test_env,
            timeout=60
        )
        
        assert result.returncode == 0
        
        output = result.stdout
        
        # Should show completion indicators
        success_indicators = [
            "dataset created", "tables created", "index created", 
            "setup completed", "✓", "✅"
        ]
        
        # At least some success indicators should be present
        found_indicators = [indicator for indicator in success_indicators if indicator in output.lower()]
        assert len(found_indicators) > 0, f"No success indicators found in output: {output}"
        
        print(f"✅ Setup shows completion indicators: {found_indicators}")

    def test_logs_after_setup(self):
        """Test logs are properly generated during setup."""
        
        # Check if log files are created
        log_files = ["logs/bigquery_vector.log", "logs/schema_manager.log"]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                # Log file should not be empty
                assert os.path.getsize(log_file) > 0, f"Log file {log_file} is empty"
                
                # Should contain setup-related entries
                with open(log_file, 'r') as f:
                    log_content = f.read().lower()
                    assert any(keyword in log_content for keyword in ["setup", "create", "bigquery"])
        
        print("✅ Logs generated during setup")