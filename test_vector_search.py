"""
Test script for BigQuery vector search using approximate_neighbors function.

This script tests the vector similarity search functionality after embeddings
have been generated and stored in the source_embeddings table.

Usage:
    python test_vector_search.py --project PROJECT_ID --dataset DATASET_ID [options]
"""

import argparse
import logging
import os
from typing import Any, Dict, List

from src.janapada_memory.bigquery_connection import BigQueryConnection


logger = logging.getLogger(__name__)


class VectorSearchTester:
    """Tests BigQuery vector search functionality."""
    
    def __init__(self, project_id: str, dataset_id: str):
        """
        Initialize vector search tester.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.connection = BigQueryConnection(project_id=project_id, dataset_id=dataset_id)
        
        logger.info(f"Vector search tester initialized for {project_id}.{dataset_id}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        query = f"""
        SELECT 
            COUNT(*) as total_embeddings,
            COUNT(DISTINCT chunk_id) as unique_chunks,
            COUNT(DISTINCT model) as unique_models,
            MIN(created_at) as first_embedding,
            MAX(created_at) as last_embedding,
            ARRAY_LENGTH(embedding) as embedding_dimensions
        FROM `{self.project_id}.{self.dataset_id}.source_embeddings`
        LIMIT 1
        """
        
        try:
            results = list(self.connection.execute_query(query))
            if results:
                row = results[0]
                return {
                    'total_embeddings': row.total_embeddings,
                    'unique_chunks': row.unique_chunks,
                    'unique_models': row.unique_models,
                    'first_embedding': row.first_embedding.isoformat() if row.first_embedding else None,
                    'last_embedding': row.last_embedding.isoformat() if row.last_embedding else None,
                    'embedding_dimensions': row.embedding_dimensions
                }
            else:
                return {'total_embeddings': 0}
        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {'error': str(e)}
    
    def get_sample_chunk(self) -> Dict[str, Any]:
        """Get a sample chunk with embedding for testing."""
        query = f"""
        SELECT 
            se.chunk_id,
            se.embedding,
            sm.text_content,
            sm.source,
            sm.artifact_type
        FROM `{self.project_id}.{self.dataset_id}.source_embeddings` se
        JOIN `{self.project_id}.{self.dataset_id}.source_metadata` sm
            ON se.chunk_id = sm.chunk_id
        ORDER BY se.created_at
        LIMIT 1
        """
        
        try:
            results = list(self.connection.execute_query(query))
            if results:
                row = results[0]
                return {
                    'chunk_id': row.chunk_id,
                    'embedding': row.embedding,
                    'text_content': row.text_content[:200] + '...' if len(row.text_content) > 200 else row.text_content,
                    'source': row.source,
                    'artifact_type': row.artifact_type
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get sample chunk: {e}")
            return None
    
    def test_vector_similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        chunk_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Test vector similarity search using approximate_neighbors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of nearest neighbors to return
            chunk_id: Optional chunk_id for reference
            
        Returns:
            List of similar chunks with distances
        """
        
        # Convert embedding to string format for BigQuery
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        query = f"""
        SELECT 
            base.chunk_id,
            base.distance,
            sm.text_content,
            sm.source,
            sm.artifact_type,
            se.model
        FROM VECTOR_SEARCH(
            TABLE `{self.project_id}.{self.dataset_id}.source_embeddings`,
            'embedding',
            {embedding_str},
            top_k => {top_k},
            distance_type => 'COSINE'
        ) AS base
        JOIN `{self.project_id}.{self.dataset_id}.source_metadata` sm
            ON base.chunk_id = sm.chunk_id
        JOIN `{self.project_id}.{self.dataset_id}.source_embeddings` se
            ON base.chunk_id = se.chunk_id
        ORDER BY base.distance
        """
        
        try:
            results = list(self.connection.execute_query(query))
            similar_chunks = []
            
            for row in results:
                similar_chunks.append({
                    'chunk_id': row.chunk_id,
                    'distance': float(row.distance),
                    'similarity_score': 1.0 - float(row.distance),  # Convert cosine distance to similarity
                    'text_content': row.text_content[:200] + '...' if len(row.text_content) > 200 else row.text_content,
                    'source': row.source,
                    'artifact_type': row.artifact_type,
                    'model': row.model
                })
            
            return similar_chunks
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise
    
    def test_approximate_neighbors_legacy(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Test vector similarity search using legacy approximate_neighbors function.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of nearest neighbors to return
            
        Returns:
            List of similar chunks with distances
        """
        
        # Convert embedding to string format for BigQuery
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        query = f"""
        WITH neighbors AS (
            SELECT 
                chunk_id,
                distance
            FROM ML.APPROXIMATE_NEIGHBORS(
                TABLE `{self.project_id}.{self.dataset_id}.source_embeddings`,
                STRUCT({embedding_str} AS embedding),
                'COSINE',
                {top_k}
            )
        )
        SELECT 
            n.chunk_id,
            n.distance,
            sm.text_content,
            sm.source,
            sm.artifact_type,
            se.model
        FROM neighbors n
        JOIN `{self.project_id}.{self.dataset_id}.source_metadata` sm
            ON n.chunk_id = sm.chunk_id
        JOIN `{self.project_id}.{self.dataset_id}.source_embeddings` se
            ON n.chunk_id = se.chunk_id
        ORDER BY n.distance
        """
        
        try:
            results = list(self.connection.execute_query(query))
            similar_chunks = []
            
            for row in results:
                similar_chunks.append({
                    'chunk_id': row.chunk_id,
                    'distance': float(row.distance),
                    'similarity_score': 1.0 - float(row.distance),  # Convert cosine distance to similarity
                    'text_content': row.text_content[:200] + '...' if len(row.text_content) > 200 else row.text_content,
                    'source': row.source,
                    'artifact_type': row.artifact_type,
                    'model': row.model
                })
            
            return similar_chunks
        except Exception as e:
            logger.error(f"Approximate neighbors search failed: {e}")
            raise
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive vector search test."""
        test_results = {
            'embedding_stats': {},
            'sample_chunk': None,
            'vector_search_results': [],
            'approximate_neighbors_results': [],
            'test_status': 'FAILED'
        }
        
        try:
            # Get embedding statistics
            logger.info("Getting embedding statistics...")
            test_results['embedding_stats'] = self.get_embedding_stats()
            
            if test_results['embedding_stats'].get('total_embeddings', 0) == 0:
                logger.warning("No embeddings found in the database")
                test_results['test_status'] = 'NO_DATA'
                return test_results
            
            # Get sample chunk for testing
            logger.info("Getting sample chunk...")
            sample_chunk = self.get_sample_chunk()
            if not sample_chunk:
                logger.warning("No sample chunk found")
                test_results['test_status'] = 'NO_SAMPLE'
                return test_results
            
            test_results['sample_chunk'] = sample_chunk
            query_embedding = sample_chunk['embedding']
            
            # Test VECTOR_SEARCH function
            logger.info("Testing VECTOR_SEARCH function...")
            try:
                vector_search_results = self.test_vector_similarity_search(
                    query_embedding=query_embedding,
                    top_k=5,
                    chunk_id=sample_chunk['chunk_id']
                )
                test_results['vector_search_results'] = vector_search_results
                logger.info(f"VECTOR_SEARCH returned {len(vector_search_results)} results")
            except Exception as e:
                logger.warning(f"VECTOR_SEARCH failed: {e}")
                test_results['vector_search_error'] = str(e)
            
            # Test ML.APPROXIMATE_NEIGHBORS function (legacy)
            logger.info("Testing ML.APPROXIMATE_NEIGHBORS function...")
            try:
                approximate_neighbors_results = self.test_approximate_neighbors_legacy(
                    query_embedding=query_embedding,
                    top_k=5
                )
                test_results['approximate_neighbors_results'] = approximate_neighbors_results
                logger.info(f"ML.APPROXIMATE_NEIGHBORS returned {len(approximate_neighbors_results)} results")
            except Exception as e:
                logger.warning(f"ML.APPROXIMATE_NEIGHBORS failed: {e}")
                test_results['approximate_neighbors_error'] = str(e)
            
            # Determine test status
            if (test_results.get('vector_search_results') or 
                test_results.get('approximate_neighbors_results')):
                test_results['test_status'] = 'SUCCESS'
            else:
                test_results['test_status'] = 'FAILED'
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            test_results['error'] = str(e)
            test_results['test_status'] = 'ERROR'
        
        return test_results


def main():
    """Command-line interface for vector search testing."""
    parser = argparse.ArgumentParser(description="Test BigQuery vector search functionality")
    
    # Required arguments
    parser.add_argument("--project", required=True, help="Google Cloud project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    
    # Optional arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize tester
        tester = VectorSearchTester(
            project_id=args.project,
            dataset_id=args.dataset
        )
        
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Print results
        print(f"\nVector Search Test Results:")
        print(f"  Test Status: {results['test_status']}")
        
        stats = results['embedding_stats']
        print(f"\nEmbedding Statistics:")
        print(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"  Unique chunks: {stats.get('unique_chunks', 0)}")
        print(f"  Embedding dimensions: {stats.get('embedding_dimensions', 'N/A')}")
        print(f"  First embedding: {stats.get('first_embedding', 'N/A')}")
        print(f"  Last embedding: {stats.get('last_embedding', 'N/A')}")
        
        if results.get('sample_chunk'):
            chunk = results['sample_chunk']
            print(f"\nSample Chunk:")
            print(f"  Chunk ID: {chunk['chunk_id']}")
            print(f"  Source: {chunk['source']}")
            print(f"  Type: {chunk['artifact_type']}")
            print(f"  Content: {chunk['text_content']}")
        
        # VECTOR_SEARCH results
        if results.get('vector_search_results'):
            print(f"\nVECTOR_SEARCH Results ({len(results['vector_search_results'])} found):")
            for i, result in enumerate(results['vector_search_results'][:3], 1):
                print(f"  {i}. Chunk ID: {result['chunk_id']}")
                print(f"     Similarity: {result['similarity_score']:.4f}")
                print(f"     Content: {result['text_content']}")
                print()
        elif results.get('vector_search_error'):
            print(f"\nVECTOR_SEARCH Error: {results['vector_search_error']}")
        
        # ML.APPROXIMATE_NEIGHBORS results
        if results.get('approximate_neighbors_results'):
            print(f"\nML.APPROXIMATE_NEIGHBORS Results ({len(results['approximate_neighbors_results'])} found):")
            for i, result in enumerate(results['approximate_neighbors_results'][:3], 1):
                print(f"  {i}. Chunk ID: {result['chunk_id']}")
                print(f"     Similarity: {result['similarity_score']:.4f}")
                print(f"     Content: {result['text_content']}")
                print()
        elif results.get('approximate_neighbors_error'):
            print(f"\nML.APPROXIMATE_NEIGHBORS Error: {results['approximate_neighbors_error']}")
        
        # Return appropriate exit code
        if results['test_status'] in ['SUCCESS', 'NO_DATA']:
            return 0
        else:
            return 1
        
    except Exception as e:
        logger.error(f"Vector search test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())