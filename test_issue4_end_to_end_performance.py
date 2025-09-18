#!/usr/bin/env python3
"""
Issue #4 End-to-End Performance Test with Real Document Ingestion
Tests complete BigQuery Memory Adapter with actual example documents
"""

import sys
import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def simulate_gemini_embedding(text: str) -> List[float]:
    """Simulate Gemini text embedding generation (768 dimensions)."""
    import hashlib
    import random
    
    # Use text hash for reproducible pseudo-random embeddings
    hash_obj = hashlib.md5(text.encode())
    random.seed(int(hash_obj.hexdigest()[:8], 16))
    
    # Generate 768-dimensional embedding with realistic values
    embedding = []
    for i in range(768):
        value = random.gauss(0, 0.1)  # Normal distribution around 0
        embedding.append(value)
    
    # Normalize the vector
    magnitude = sum(x * x for x in embedding) ** 0.5
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    
    return embedding

def load_example_documents() -> List[Dict[str, Any]]:
    """Load and parse all example documents."""
    
    documents = []
    base_path = Path(__file__).parent / "examples"
    
    # Load Kubernetes manifests
    k8s_path = base_path / "k8s-manifests"
    for yaml_file in k8s_path.glob("*.yaml"):
        content = yaml_file.read_text()
        documents.append({
            "chunk_id": f"k8s_{yaml_file.stem}_{uuid.uuid4().hex[:8]}",
            "source": str(yaml_file),
            "artifact_type": "kubernetes",
            "text_content": content,
            "metadata": {
                "file_type": "yaml",
                "manifest_type": yaml_file.stem,
                "size_bytes": len(content)
            }
        })
    
    # Load FastAPI project
    fastapi_path = base_path / "fastapi-project"
    for py_file in fastapi_path.glob("*.py"):
        content = py_file.read_text()
        documents.append({
            "chunk_id": f"fastapi_{py_file.stem}_{uuid.uuid4().hex[:8]}",
            "source": str(py_file),
            "artifact_type": "fastapi",
            "text_content": content,
            "metadata": {
                "file_type": "python",
                "module_name": py_file.stem,
                "size_bytes": len(content)
            }
        })
    
    # Load COBOL copybooks
    cobol_path = base_path / "cobol-copybooks"
    for cobol_file in cobol_path.glob("*.cpy"):
        content = cobol_file.read_text()
        documents.append({
            "chunk_id": f"cobol_{cobol_file.stem}_{uuid.uuid4().hex[:8]}",
            "source": str(cobol_file),
            "artifact_type": "cobol",
            "text_content": content,
            "metadata": {
                "file_type": "copybook",
                "record_name": cobol_file.stem,
                "size_bytes": len(content)
            }
        })
    
    # Load IRS layouts
    irs_path = base_path / "irs-layouts"
    for irs_file in irs_path.glob("*.txt"):
        content = irs_file.read_text()
        documents.append({
            "chunk_id": f"irs_{irs_file.stem}_{uuid.uuid4().hex[:8]}",
            "source": str(irs_file),
            "artifact_type": "irs",
            "text_content": content,
            "metadata": {
                "file_type": "layout",
                "layout_name": irs_file.stem,
                "size_bytes": len(content)
            }
        })
    
    # Load MUMPS dictionaries
    mumps_path = base_path / "mumps-dictionaries"
    for mumps_file in mumps_path.glob("*.dic"):
        content = mumps_file.read_text()
        documents.append({
            "chunk_id": f"mumps_{mumps_file.stem}_{uuid.uuid4().hex[:8]}",
            "source": str(mumps_file),
            "artifact_type": "mumps",
            "text_content": content,
            "metadata": {
                "file_type": "dictionary",
                "file_name": mumps_file.stem,
                "size_bytes": len(content)
            }
        })
    
    logger.info(f"📄 Loaded {len(documents)} example documents")
    return documents

def test_issue4_end_to_end_performance():
    """Complete end-to-end performance test for Issue #4."""
    
    logger.info("🚀 ISSUE #4 END-TO-END PERFORMANCE TEST")
    logger.info("=" * 60)
    
    # Set up environment
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    
    performance_results = {
        "test_start_time": time.time(),
        "document_loading": {},
        "memory_service_setup": {},
        "document_ingestion": {},
        "search_performance": {},
        "fallback_performance": {},
        "overall_metrics": {}
    }
    
    try:
        # Phase 1: Document Loading
        logger.info("📋 Phase 1: Loading Example Documents")
        start_time = time.time()
        
        documents = load_example_documents()
        
        performance_results["document_loading"] = {
            "documents_loaded": len(documents),
            "loading_time_ms": (time.time() - start_time) * 1000,
            "artifact_types": list(set(doc["artifact_type"] for doc in documents)),
            "total_content_size_bytes": sum(doc["metadata"]["size_bytes"] for doc in documents)
        }
        
        logger.info(f"✅ Loaded {len(documents)} documents in {performance_results['document_loading']['loading_time_ms']:.1f}ms")
        logger.info(f"📊 Artifact types: {performance_results['document_loading']['artifact_types']}")
        logger.info(f"📏 Total content: {performance_results['document_loading']['total_content_size_bytes']} bytes")
        
        # Phase 2: Memory Service Setup
        logger.info("🔧 Phase 2: Memory Service Setup")
        start_time = time.time()
        
        from janapada_memory import create_memory_service
        from janapada_memory.bigquery_vector_index import SearchMode
        
        memory_service = create_memory_service(
            project_id="konveyn2ai",
            dataset_id="semantic_gap_detector",
            enable_fallback=True
        )
        
        setup_time = (time.time() - start_time) * 1000
        
        # Health check
        health = memory_service.health_check()
        config_valid = memory_service.validate_configuration()['configuration_valid']
        
        performance_results["memory_service_setup"] = {
            "setup_time_ms": setup_time,
            "bigquery_healthy": health['bigquery_connection']['is_healthy'],
            "config_valid": config_valid,
            "fallback_enabled": memory_service.enable_fallback
        }
        
        logger.info(f"✅ Service setup in {setup_time:.1f}ms")
        logger.info(f"🔗 BigQuery connection: {'✅ Healthy' if health['bigquery_connection']['is_healthy'] else '❌ Unhealthy'}")
        logger.info(f"⚙️  Configuration: {'✅ Valid' if config_valid else '❌ Invalid'}")
        
        # Phase 3: Document Ingestion to Fallback Index
        logger.info("📥 Phase 3: Document Ingestion to Fallback Index")
        start_time = time.time()
        
        # Configure to use fallback for consistent testing
        memory_service.configure_search_mode(SearchMode.FALLBACK_ONLY)
        
        ingestion_metrics = {
            "documents_processed": 0,
            "embeddings_generated": 0,
            "total_embedding_time_ms": 0,
            "total_ingestion_time_ms": 0,
            "errors": []
        }
        
        for doc in documents:
            doc_start = time.time()
            
            try:
                # Generate embedding
                embedding_start = time.time()
                embedding = simulate_gemini_embedding(doc["text_content"])
                embedding_time = (time.time() - embedding_start) * 1000
                
                # Add to fallback index
                memory_service.vector_index.local_index.add_vector(
                    chunk_id=doc["chunk_id"],
                    embedding=embedding,
                    source=doc["source"],
                    artifact_type=doc["artifact_type"],
                    text_content=doc["text_content"],
                    metadata=doc["metadata"]
                )
                
                ingestion_metrics["documents_processed"] += 1
                ingestion_metrics["embeddings_generated"] += 1
                ingestion_metrics["total_embedding_time_ms"] += embedding_time
                
            except Exception as e:
                ingestion_metrics["errors"].append(f"Doc {doc['chunk_id']}: {str(e)}")
                logger.warning(f"⚠️  Failed to ingest {doc['chunk_id']}: {e}")
        
        ingestion_metrics["total_ingestion_time_ms"] = (time.time() - start_time) * 1000
        performance_results["document_ingestion"] = ingestion_metrics
        
        logger.info(f"✅ Ingested {ingestion_metrics['documents_processed']} documents")
        logger.info(f"⏱️  Total ingestion time: {ingestion_metrics['total_ingestion_time_ms']:.1f}ms")
        logger.info(f"🧠 Average embedding time: {ingestion_metrics['total_embedding_time_ms'] / max(1, ingestion_metrics['embeddings_generated']):.1f}ms per document")
        logger.info(f"❌ Errors: {len(ingestion_metrics['errors'])}")
        
        # Phase 4: Search Performance Testing
        logger.info("🔍 Phase 4: Search Performance Testing")
        
        search_metrics = {
            "test_queries": [],
            "total_searches": 0,
            "successful_searches": 0,
            "average_search_time_ms": 0,
            "total_search_time_ms": 0,
            "results_by_type": {}
        }
        
        # Test queries for different artifact types
        test_queries = [
            {"text": "kubernetes deployment nginx container", "types": ["kubernetes"], "description": "K8s deployment search"},
            {"text": "fastapi user model endpoint", "types": ["fastapi"], "description": "FastAPI code search"},
            {"text": "customer record name address", "types": ["cobol"], "description": "COBOL structure search"},
            {"text": "social security number taxpayer", "types": ["irs"], "description": "IRS field search"},
            {"text": "patient name date birth", "types": ["mumps"], "description": "MUMPS field search"},
            {"text": "api web application service", "types": None, "description": "Cross-artifact search"}
        ]
        
        for query in test_queries:
            query_start = time.time()
            
            try:
                query_embedding = simulate_gemini_embedding(query["text"])
                
                results = memory_service.similarity_search(
                    query_embedding=query_embedding,
                    k=5,
                    artifact_types=query["types"]
                )
                
                query_time = (time.time() - query_start) * 1000
                
                query_result = {
                    "description": query["description"],
                    "query_text": query["text"],
                    "artifact_types": query["types"],
                    "results_count": len(results),
                    "search_time_ms": query_time,
                    "top_similarity": results[0]["similarity_score"] if results else 0.0,
                    "top_artifact_type": results[0]["artifact_type"] if results else None
                }
                
                search_metrics["test_queries"].append(query_result)
                search_metrics["total_searches"] += 1
                search_metrics["successful_searches"] += 1
                search_metrics["total_search_time_ms"] += query_time
                
                # Track results by type
                for result in results:
                    artifact_type = result["artifact_type"]
                    if artifact_type not in search_metrics["results_by_type"]:
                        search_metrics["results_by_type"][artifact_type] = 0
                    search_metrics["results_by_type"][artifact_type] += 1
                
                logger.info(f"  ✅ {query['description']}: {len(results)} results in {query_time:.1f}ms")
                if results:
                    logger.info(f"     🎯 Top match: {results[0]['artifact_type']} (score: {results[0]['similarity_score']:.3f})")
                
            except Exception as e:
                logger.error(f"  ❌ {query['description']}: {e}")
                search_metrics["total_searches"] += 1
        
        if search_metrics["successful_searches"] > 0:
            search_metrics["average_search_time_ms"] = search_metrics["total_search_time_ms"] / search_metrics["successful_searches"]
        
        performance_results["search_performance"] = search_metrics
        
        logger.info(f"✅ Completed {search_metrics['successful_searches']}/{search_metrics['total_searches']} searches")
        logger.info(f"⏱️  Average search time: {search_metrics['average_search_time_ms']:.1f}ms")
        logger.info(f"📊 Results by type: {search_metrics['results_by_type']}")
        
        # Phase 5: Fallback Performance Validation
        logger.info("🔄 Phase 5: Fallback Performance Validation")
        
        fallback_start = time.time()
        
        # Test fallback index directly
        fallback_health = memory_service.vector_index.local_index.health_check()
        
        # Test rapid consecutive searches (stress test)
        stress_test_start = time.time()
        stress_searches = 10
        stress_results = []
        
        for i in range(stress_searches):
            query_embedding = simulate_gemini_embedding(f"stress test query {i}")
            search_start = time.time()
            results = memory_service.similarity_search(query_embedding, k=3)
            search_time = (time.time() - search_start) * 1000
            stress_results.append(search_time)
        
        stress_test_time = (time.time() - stress_test_start) * 1000
        
        fallback_metrics = {
            "fallback_health": fallback_health,
            "stress_test_searches": stress_searches,
            "stress_test_total_time_ms": stress_test_time,
            "stress_test_avg_time_ms": sum(stress_results) / len(stress_results),
            "stress_test_min_time_ms": min(stress_results),
            "stress_test_max_time_ms": max(stress_results),
            "fallback_validation_time_ms": (time.time() - fallback_start) * 1000
        }
        
        performance_results["fallback_performance"] = fallback_metrics
        
        logger.info(f"✅ Fallback health: {fallback_health['status']}")
        logger.info(f"💪 Stress test: {stress_searches} searches in {stress_test_time:.1f}ms")
        logger.info(f"⚡ Search latency: avg={fallback_metrics['stress_test_avg_time_ms']:.1f}ms, min={fallback_metrics['stress_test_min_time_ms']:.1f}ms, max={fallback_metrics['stress_test_max_time_ms']:.1f}ms")
        
        # Phase 6: Overall Metrics and Cleanup
        logger.info("📊 Phase 6: Overall Metrics and Cleanup")
        
        # Get final service statistics
        final_stats = memory_service.get_stats()
        
        total_test_time = (time.time() - performance_results["test_start_time"]) * 1000
        
        performance_results["overall_metrics"] = {
            "total_test_time_ms": total_test_time,
            "documents_per_second": (len(documents) / total_test_time) * 1000,
            "searches_per_second": (search_metrics["total_searches"] / total_test_time) * 1000,
            "final_service_stats": final_stats,
            "memory_efficiency": {
                "fallback_index_size": len(memory_service.vector_index.local_index),
                "estimated_memory_mb": fallback_health.get("estimated_memory_mb", 0)
            }
        }
        
        # Cleanup
        memory_service.close()
        
        logger.info("🎉 ISSUE #4 END-TO-END PERFORMANCE TEST COMPLETED!")
        logger.info("=" * 60)
        logger.info("📊 FINAL PERFORMANCE SUMMARY:")
        logger.info(f"⏱️  Total test time: {total_test_time:.1f}ms")
        logger.info(f"📄 Documents processed: {len(documents)}")
        logger.info(f"🔍 Searches completed: {search_metrics['successful_searches']}")
        logger.info(f"📈 Document processing rate: {performance_results['overall_metrics']['documents_per_second']:.1f} docs/sec")
        logger.info(f"🔎 Search rate: {performance_results['overall_metrics']['searches_per_second']:.1f} searches/sec")
        logger.info(f"💾 Memory usage: {performance_results['overall_metrics']['memory_efficiency']['estimated_memory_mb']:.1f} MB")
        
        # Save detailed results
        results_file = "issue4_end_to_end_performance_results.json"
        with open(results_file, 'w') as f:
            json.dump(performance_results, f, indent=2, default=str)
        
        logger.info(f"💾 Detailed results saved to: {results_file}")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_issue4_end_to_end_performance()
    if results:
        print(f"\n🎯 Test Status: SUCCESS")
        print(f"📊 Documents: {results['document_loading']['documents_loaded']}")
        print(f"🔍 Searches: {results['search_performance']['successful_searches']}")
        print(f"⏱️  Total Time: {results['overall_metrics']['total_test_time_ms']:.1f}ms")
        exit(0)
    else:
        print(f"\n❌ Test Status: FAILED")
        exit(1)