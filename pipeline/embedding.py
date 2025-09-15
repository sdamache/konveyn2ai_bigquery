"""
Embedding Generation Pipeline for BigQuery Vector Backend

This module generates 768-dimensional embeddings for text chunks using Google Gemini API
and stores them in BigQuery with batching, caching, and idempotent behavior.

Usage:
    python -m pipeline.embedding [options]
    
Environment Variables:
    GOOGLE_CLOUD_PROJECT: Google Cloud project ID
    BIGQUERY_DATASET_ID: BigQuery dataset ID  
    GOOGLE_API_KEY: Google API key for Gemini
    EMBED_BATCH_SIZE: Batch size for API calls (default: 32)
    EMBED_MAX_RETRIES: Max retries for failed requests (default: 3)
    EMBED_CACHE_DIR: Cache directory (default: .cache/embeddings)
"""

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from google.cloud import bigquery
# EmbedContentResponse import not needed - using embed_content directly

from src.janapada_memory.bigquery_connection import BigQueryConnection
from src.janapada_memory.schema_manager import SchemaManager


logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Disk-based cache for embeddings keyed by content hash."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Embedding cache initialized at {self.cache_dir}")
    
    def _cache_key(self, content: str, model: str) -> str:
        """Generate cache key from content and model."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"{model}_{content_hash}"
    
    def get(self, content: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        cache_key = self._cache_key(content, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    return cached['vector']
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
                return None
        return None
    
    def set(self, content: str, model: str, vector: List[float]) -> None:
        """Cache embedding vector."""
        cache_key = self._cache_key(content, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'model': model,
                'vector': vector,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'content_hash': hashlib.sha256(content.encode('utf-8')).hexdigest()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")


class EmbeddingGenerator:
    """Generates embeddings using Google Gemini API with batching and caching."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        batch_size: int = 32,
        max_retries: int = 3,
        cache_dir: str = ".cache/embeddings"
    ):
        """
        Initialize embedding generator.
        
        Args:
            api_key: Google API key for Gemini
            model: Embedding model name
            batch_size: Number of texts to process in each batch
            max_retries: Maximum retries for failed requests
            cache_dir: Directory for caching embeddings
        """
        genai.configure(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache = EmbeddingCache(cache_dir)
        
        # Statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failed_requests': 0,
            'total_latency_ms': 0
        }
        
        logger.info(f"Embedding generator initialized with model {model}, batch_size {batch_size}")
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent caching."""
        return content.strip().replace('\r\n', '\n').replace('\r', '\n')
    
    def _exponential_backoff(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = delay * 0.1 * (0.5 - hash(str(time.time())) % 100 / 100)
        return max(0.1, delay + jitter)
    
    def generate_embedding(self, content: str) -> Optional[List[float]]:
        """Generate embedding for single text content."""
        normalized_content = self._normalize_content(content)
        
        # Check cache first
        cached_embedding = self.cache.get(normalized_content, self.model)
        if cached_embedding:
            self.stats['cache_hits'] += 1
            return cached_embedding
        
        self.stats['cache_misses'] += 1
        
        # Generate embedding via API
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = genai.embed_content(
                    model=self.model,
                    content=normalized_content,
                    task_type="retrieval_document"
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                self.stats['total_latency_ms'] += latency_ms
                self.stats['api_calls'] += 1
                
                if response and hasattr(response, 'embedding') and response.embedding:
                    embedding = response.embedding
                    # Cache the result
                    self.cache.set(normalized_content, self.model, embedding)
                    return embedding
                else:
                    logger.warning(f"Empty embedding response for content: {content[:100]}...")
                    
            except Exception as e:
                logger.warning(f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    self.stats['failed_requests'] += 1
                    logger.error(f"Failed to generate embedding after {self.max_retries} attempts")
        
        return None
    
    def generate_embeddings_batch(self, contents: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for batch of content."""
        results = []
        
        for content in contents:
            embedding = self.generate_embedding(content)
            results.append(embedding)
            
            # Small delay between requests to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.stats.copy()
        if self.stats['api_calls'] > 0:
            stats['avg_latency_ms'] = self.stats['total_latency_ms'] / self.stats['api_calls']
        else:
            stats['avg_latency_ms'] = 0
        return stats


class EmbeddingPipeline:
    """Main pipeline for generating and storing embeddings in BigQuery."""
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        api_key: str,
        embedding_model: str = "models/text-embedding-004",
        batch_size: int = 32,
        cache_dir: str = ".cache/embeddings"
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            api_key: Google API key for Gemini
            embedding_model: Model name for embeddings
            batch_size: Batch size for processing
            cache_dir: Cache directory for embeddings
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.embedding_model = embedding_model
        
        # Initialize components
        self.connection = BigQueryConnection(project_id=project_id, dataset_id=dataset_id)
        self.schema_manager = SchemaManager(connection=self.connection)
        self.generator = EmbeddingGenerator(
            api_key=api_key,
            model=embedding_model,
            batch_size=batch_size,
            cache_dir=cache_dir
        )
        
        # Ensure tables exist
        self._ensure_tables_exist()
        
        logger.info(f"Embedding pipeline initialized for {project_id}.{dataset_id}")
    
    def _ensure_tables_exist(self) -> None:
        """Ensure required tables exist."""
        try:
            # Create dataset if needed
            self.schema_manager.create_dataset()
            
            # Create tables if needed  
            result = self.schema_manager.create_tables(
                tables=["source_metadata", "source_embeddings"],
                force_recreate=False
            )
            
            if result.get("errors"):
                logger.warning(f"Table creation warnings: {result['errors']}")
                
        except Exception as e:
            logger.error(f"Failed to ensure tables exist: {e}")
            raise
    
    def _get_pending_chunks(self, limit: Optional[int] = None, where_clause: str = "") -> List[Dict[str, Any]]:
        """Get chunks from source_metadata that don't have embeddings yet."""
        limit_clause = f"LIMIT {limit}" if limit else ""
        where_clause = f"AND ({where_clause})" if where_clause else ""
        
        query = f"""
        SELECT 
            sm.chunk_id,
            sm.text_content,
            sm.source,
            sm.artifact_type
        FROM `{self.project_id}.{self.dataset_id}.source_metadata` sm
        LEFT JOIN `{self.project_id}.{self.dataset_id}.source_embeddings` se
            ON sm.chunk_id = se.chunk_id 
            AND se.model = '{self.embedding_model}'
        WHERE se.chunk_id IS NULL
        {where_clause}
        ORDER BY sm.created_at
        {limit_clause}
        """
        
        try:
            results = list(self.connection.execute_query(query))
            chunks = []
            for row in results:
                chunks.append({
                    'chunk_id': row.chunk_id,
                    'text_content': row.text_content,
                    'source': row.source,
                    'artifact_type': row.artifact_type
                })
            return chunks
        except Exception as e:
            logger.error(f"Failed to get pending chunks: {e}")
            raise
    
    def _content_hash(self, content: str) -> str:
        """Generate content hash for idempotency."""
        normalized = self.generator._normalize_content(content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _store_embeddings_batch(self, chunk_embeddings: List[Dict[str, Any]]) -> int:
        """Store batch of embeddings in BigQuery."""
        if not chunk_embeddings:
            return 0
        
        # Prepare rows for insertion
        rows_to_insert = []
        now = datetime.now(timezone.utc)
        partition_date = now.date()
        
        for item in chunk_embeddings:
            if item['embedding'] is not None:
                row = {
                    'chunk_id': item['chunk_id'],
                    'model': self.embedding_model,
                    'content_hash': item['content_hash'],
                    'embedding': item['embedding'],
                    'created_at': now.isoformat(),
                    'source_type': item.get('artifact_type'),
                    'artifact_id': item.get('source'),
                    'partition_date': partition_date.isoformat()
                }
                rows_to_insert.append(row)
        
        if not rows_to_insert:
            return 0
        
        try:
            # Insert rows
            table_id = f"{self.project_id}.{self.dataset_id}.source_embeddings"
            table = self.connection.client.get_table(table_id)
            
            errors = self.connection.client.insert_rows_json(table, rows_to_insert)
            
            if errors:
                logger.error(f"BigQuery insertion errors: {errors}")
                return 0
            
            logger.info(f"Successfully inserted {len(rows_to_insert)} embeddings")
            return len(rows_to_insert)
            
        except Exception as e:
            logger.error(f"Failed to store embeddings batch: {e}")
            return 0
    
    def generate_embeddings(
        self,
        limit: Optional[int] = None,
        where_clause: str = "",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings for pending chunks.
        
        Args:
            limit: Maximum number of chunks to process
            where_clause: Additional WHERE clause for filtering
            dry_run: If True, don't actually store embeddings
            
        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting embedding generation (limit={limit}, dry_run={dry_run})")
        
        # Get pending chunks
        pending_chunks = self._get_pending_chunks(limit=limit, where_clause=where_clause)
        
        if not pending_chunks:
            logger.info("No pending chunks found")
            return {
                'chunks_scanned': 0,
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'processing_time_ms': 0,
                'generator_stats': self.generator.get_stats()
            }
        
        logger.info(f"Found {len(pending_chunks)} pending chunks")
        
        # Process in batches
        embeddings_generated = 0
        embeddings_stored = 0
        
        for i in range(0, len(pending_chunks), self.generator.batch_size):
            batch = pending_chunks[i:i + self.generator.batch_size]
            batch_num = (i // self.generator.batch_size) + 1
            total_batches = (len(pending_chunks) + self.generator.batch_size - 1) // self.generator.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            # Generate embeddings for batch
            batch_embeddings = []
            for chunk in batch:
                embedding = self.generator.generate_embedding(chunk['text_content'])
                if embedding:
                    embeddings_generated += 1
                
                batch_embeddings.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_hash': self._content_hash(chunk['text_content']),
                    'embedding': embedding,
                    'artifact_type': chunk['artifact_type'],
                    'source': chunk['source']
                })
            
            # Store embeddings (unless dry run)
            if not dry_run:
                stored_count = self._store_embeddings_batch(batch_embeddings)
                embeddings_stored += stored_count
            else:
                logger.info(f"DRY RUN: Would store {len([e for e in batch_embeddings if e['embedding']])} embeddings")
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Final statistics
        result = {
            'chunks_scanned': len(pending_chunks),
            'embeddings_generated': embeddings_generated,
            'embeddings_stored': embeddings_stored,
            'processing_time_ms': processing_time_ms,
            'generator_stats': self.generator.get_stats()
        }
        
        logger.info(f"Embedding generation completed: {result}")
        return result


def main():
    """Command-line interface for embedding generation."""
    parser = argparse.ArgumentParser(description="Generate embeddings for BigQuery vector backend")
    
    # Required arguments
    parser.add_argument("--project", required=True, help="Google Cloud project ID")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    # Optional arguments
    parser.add_argument("--model", default="models/text-embedding-004", help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for API calls")
    parser.add_argument("--limit", type=int, help="Maximum number of chunks to process")
    parser.add_argument("--where", default="", help="Additional WHERE clause for filtering")
    parser.add_argument("--cache-dir", default=".cache/embeddings", help="Cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually store embeddings")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Google API key required. Use --api-key or set GOOGLE_API_KEY environment variable")
        return 1
    
    try:
        # Initialize pipeline
        pipeline = EmbeddingPipeline(
            project_id=args.project,
            dataset_id=args.dataset,
            api_key=api_key,
            embedding_model=args.model,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir
        )
        
        # Generate embeddings
        result = pipeline.generate_embeddings(
            limit=args.limit,
            where_clause=args.where,
            dry_run=args.dry_run
        )
        
        # Print results
        print(f"\nEmbedding Generation Results:")
        print(f"  Chunks scanned: {result['chunks_scanned']}")
        print(f"  Embeddings generated: {result['embeddings_generated']}")
        print(f"  Embeddings stored: {result['embeddings_stored']}")
        print(f"  Processing time: {result['processing_time_ms']}ms")
        
        stats = result['generator_stats']
        print(f"\nAPI Statistics:")
        print(f"  API calls: {stats['api_calls']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Failed requests: {stats['failed_requests']}")
        print(f"  Average latency: {stats.get('avg_latency_ms', 0):.1f}ms")
        
        return 0
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return 1


# Data Models for API compatibility
class EmbeddingRequest:
    """Request model for embedding generation."""
    def __init__(self, text: str, model: str = "text-embedding-004"):
        self.text = text
        self.model = model

class EmbeddingResponse:
    """Response model for embedding generation."""
    def __init__(self, embedding: List[float], model: str, content_hash: str):
        self.embedding = embedding
        self.model = model
        self.content_hash = content_hash

class ProcessingStats:
    """Statistics for embedding processing."""
    def __init__(self, api_calls: int = 0, cache_hits: int = 0, cache_misses: int = 0, 
                 failed_requests: int = 0, avg_latency_ms: float = 0.0):
        self.api_calls = api_calls
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.failed_requests = failed_requests
        self.avg_latency_ms = avg_latency_ms


if __name__ == "__main__":
    exit(main())