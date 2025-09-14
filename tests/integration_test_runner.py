#!/usr/bin/env python3
"""
Integration Test Runner

Simple test runner to validate implementation without external dependencies.
Tests basic functionality and identifies issues.
"""

import sys
import os
import warnings

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Suppress warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            from janapada_memory import (
                BigQueryConnection, 
                BigQueryVectorStore, 
                SchemaManager,
                DimensionReducer,
                MigrationManager
            )
        
        print("✅ All core modules can be imported")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_imports():
    """Test API module imports."""
    print("Testing API imports...")
    
    try:
        from api import models
        from api.models import (
            EmbeddingInsertRequest,
            EmbeddingResponse,
            VectorSearchRequest,
            CreateTablesRequest
        )
        
        print("✅ API models can be imported")
        return True
        
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False

def test_model_validation():
    """Test Pydantic model validation."""
    print("Testing model validation...")
    
    try:
        from api.models import EmbeddingInsertRequest, VectorSearchRequest
        
        # Test valid embedding request
        valid_request = EmbeddingInsertRequest(
            chunk_id="test_001",
            source="test.py",
            artifact_type="code",
            text_content="def test(): pass",
            embedding=[0.1] * 768
        )
        
        assert valid_request.chunk_id == "test_001"
        assert len(valid_request.embedding) == 768
        
        # Test validation errors
        try:
            invalid_request = EmbeddingInsertRequest(
                chunk_id="test_002",
                source="test.py", 
                artifact_type="code",
                text_content="def test(): pass",
                embedding=[]  # Empty embedding should fail
            )
            print("❌ Empty embedding validation failed")
            return False
        except ValueError:
            pass  # Expected validation error
        
        # Test vector search request
        search_request = VectorSearchRequest(
            query_embedding=[0.1] * 768,
            limit=10,
            similarity_threshold=0.7
        )
        
        assert len(search_request.query_embedding) == 768
        assert search_request.limit == 10
        
        print("✅ Model validation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def test_dimension_reducer_standalone():
    """Test dimension reducer without dependencies."""
    print("Testing dimension reducer (standalone)...")
    
    try:
        # This should fail gracefully with missing dependencies
        from janapada_memory import DimensionReducer
        
        try:
            reducer = DimensionReducer()
            print("❌ Should have failed with missing dependencies")
            return False
        except ImportError as e:
            if "scikit-learn" in str(e):
                print("✅ Dimension reducer properly reports missing dependencies")
                return True
            else:
                print(f"❌ Unexpected import error: {e}")
                return False
        
    except Exception as e:
        print(f"❌ Dimension reducer test failed: {e}")
        return False

def test_bigquery_connection_standalone():
    """Test BigQuery connection without dependencies."""
    print("Testing BigQuery connection (standalone)...")
    
    try:
        from janapada_memory import BigQueryConnection
        
        try:
            connection = BigQueryConnection()
            print("❌ Should have failed with missing dependencies")
            return False
        except ImportError as e:
            if "google-cloud-bigquery" in str(e):
                print("✅ BigQuery connection properly reports missing dependencies") 
                return True
            else:
                print(f"❌ Unexpected import error: {e}")
                return False
        
    except Exception as e:
        print(f"❌ BigQuery connection test failed: {e}")
        return False

def test_api_structure():
    """Test API endpoint structure."""
    print("Testing API structure...")
    
    try:
        # Test that we can import the endpoints
        from api.vector_endpoints import router as vector_router
        from api.schema_endpoints import router as schema_router
        from api.main import app
        
        # Check that routers are properly configured
        assert vector_router.prefix == "/vector-store"
        assert schema_router.prefix == "/schema"
        
        # Check that main app exists
        assert app.title == "BigQuery Vector Store API"
        
        print("✅ API structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_makefile_commands():
    """Test that Makefile commands are syntactically correct."""
    print("Testing Makefile...")
    
    try:
        # Check if Makefile exists
        makefile_path = os.path.join(os.path.dirname(__file__), '..', 'Makefile')
        
        if not os.path.exists(makefile_path):
            print("❌ Makefile not found")
            return False
        
        # Read and check basic structure
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        required_targets = ['setup', 'migrate', 'run', 'test', 'diagnose']
        
        for target in required_targets:
            if f"{target}:" not in content:
                print(f"❌ Makefile missing target: {target}")
                return False
        
        print("✅ Makefile has required targets")
        return True
        
    except Exception as e:
        print(f"❌ Makefile test failed: {e}")
        return False

def test_requirements_file():
    """Test requirements.txt file."""
    print("Testing requirements.txt...")
    
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
        
        if not os.path.exists(requirements_path):
            print("❌ requirements.txt not found")
            return False
        
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'google-cloud-bigquery',
            'scikit-learn',
            'fastapi',
            'uvicorn',
            'pydantic'
        ]
        
        for package in required_packages:
            if package not in requirements:
                print(f"❌ requirements.txt missing package: {package}")
                return False
        
        print("✅ requirements.txt has required packages")
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("KONVEYN2AI BIGQUERY - INTEGRATION TEST RUNNER")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_api_imports,
        test_model_validation,
        test_dimension_reducer_standalone,
        test_bigquery_connection_standalone,
        test_api_structure,
        test_makefile_commands,
        test_requirements_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
        
        print()  # Add spacing between tests
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())