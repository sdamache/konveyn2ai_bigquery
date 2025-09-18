#!/usr/bin/env python3
"""
Real-time test for Issue #4 focusing on fallback functionality
"""

import sys
import logging
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fallback_functionality():
    """Test the fallback functionality specifically."""
    
    logger.info("🧪 REAL-TIME TEST: Fallback Functionality")
    
    try:
        from janapada_memory.fallback.local_vector_index import LocalVectorIndex
        from janapada_memory.models.vector_search_config import VectorSearchConfig, DistanceType
        
        # Test 1: Local Vector Index standalone
        logger.info("🔧 Testing LocalVectorIndex standalone...")
        local_index = LocalVectorIndex(max_entries=100)
        
        # Add some test vectors
        test_vectors = [
            {
                "chunk_id": "test_1",
                "embedding": [0.1, 0.2, 0.3] * 256,  # 768 dimensions
                "source": "test_file_1.py",
                "artifact_type": "python",
                "text_content": "def hello_world(): return 'Hello'",
            },
            {
                "chunk_id": "test_2", 
                "embedding": [0.2, 0.3, 0.4] * 256,  # 768 dimensions
                "source": "test_file_2.py",
                "artifact_type": "python",
                "text_content": "def goodbye_world(): return 'Goodbye'",
            }
        ]
        
        for vector in test_vectors:
            local_index.add_vector(**vector)
        
        logger.info(f"✅ Added {len(test_vectors)} vectors to local index")
        
        # Test search
        config = VectorSearchConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            top_k=2,
            distance_type=DistanceType.COSINE
        )
        
        query_embedding = [0.15, 0.25, 0.35] * 256  # 768 dimensions
        
        results = local_index.search_similar_vectors(
            query_embedding=query_embedding,
            config=config
        )
        
        logger.info(f"✅ Local search returned {len(results)} results")
        
        if results:
            logger.info(f"   - Best match: {results[0].chunk_id} (score: {results[0].similarity_score:.3f})")
        
        # Test 2: BigQuery Vector Index with forced fallback
        logger.info("🔄 Testing BigQueryVectorIndex with forced fallback...")
        
        from janapada_memory.bigquery_vector_index import BigQueryVectorIndex, SearchMode
        from janapada_memory.connections.bigquery_connection import BigQueryConnectionManager
        from janapada_memory.config import BigQueryConfig
        
        # Create config and connection (will fail auth)
        bq_config = BigQueryConfig(project_id="test-project", dataset_id="test_dataset")
        connection = BigQueryConnectionManager(config=bq_config)
        
        vector_index = BigQueryVectorIndex(
            connection=connection,
            config=config,
            enable_fallback=True,
            search_mode=SearchMode.FALLBACK_ONLY  # Force fallback mode
        )
        
        # Add vectors to fallback index
        for vector in test_vectors:
            vector_index.local_index.add_vector(**vector)
        
        # Test search with fallback only
        fallback_results = vector_index.similarity_search(
            query_embedding=query_embedding,
            k=2
        )
        
        logger.info(f"✅ Fallback search returned {len(fallback_results)} results")
        
        if fallback_results:
            logger.info(f"   - Best match: {fallback_results[0].chunk_id} (score: {fallback_results[0].similarity_score:.3f})")
        
        # Test 3: Memory Service with fallback
        logger.info("🎯 Testing JanapadaMemoryService with fallback...")
        
        from janapada_memory.memory_service import JanapadaMemoryService
        
        memory_service = JanapadaMemoryService(
            connection=connection,
            config=config,
            enable_fallback=True
        )
        
        # Add vectors to fallback
        for vector in test_vectors:
            memory_service.vector_index.local_index.add_vector(**vector)
        
        # Configure to use fallback only for this test
        memory_service.configure_search_mode(SearchMode.FALLBACK_ONLY)
        
        # Test the main similarity_search method (Issue #4 requirement)
        service_results = memory_service.similarity_search(
            query_embedding=query_embedding,
            k=2,
            artifact_types=["python"]
        )
        
        logger.info(f"✅ Memory service search returned {len(service_results)} results")
        
        if service_results:
            logger.info(f"   - Best match: {service_results[0]['chunk_id']} (score: {service_results[0]['similarity_score']:.3f})")
            logger.info(f"   - Metadata contains fallback flag: {service_results[0]['metadata'].get('fallback_search', False)}")
        
        # Test stats
        stats = memory_service.get_stats()
        logger.info(f"📊 Service statistics: {stats['total_searches']} total searches")
        
        memory_service.close()
        
        logger.info("🎉 FALLBACK FUNCTIONALITY TEST PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fallback_functionality()
    exit(0 if success else 1)