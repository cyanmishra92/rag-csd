"""
Async interface module for RAG-CSD.
This module provides asynchronous and parallel processing capabilities.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import threading

import numpy as np

from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.augmentation.augmentor import Augmentor
from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class AsyncRAGCSD:
    """Asynchronous RAG-CSD interface for concurrent query processing."""
    
    def __init__(self, config: Optional[Dict] = None, max_workers: Optional[int] = None):
        """
        Initialize the async RAG-CSD interface.
        
        Args:
            config: Configuration dictionary.
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.config = config or {}
        self.max_workers = max_workers or min(32, (threading.cpu_count() or 1) + 4)
        
        # Initialize components
        self.encoder = None
        self.vector_store = None
        self.augmentor = None
        self.vector_db_path = None
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Initialize flag
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        logger.info(f"AsyncRAGCSD initialized with {self.max_workers} workers")
    
    async def initialize(self, vector_db_path: str) -> None:
        """
        Initialize the RAG-CSD components asynchronously.
        
        Args:
            vector_db_path: Path to the vector database.
        """
        async with self._init_lock:
            if self._initialized:
                return
            
            logger.info("Initializing RAG-CSD components asynchronously...")
            start_time = time.time()
            
            # Initialize components in parallel using thread pool
            loop = asyncio.get_event_loop()
            
            # Create initialization tasks
            encoder_task = loop.run_in_executor(
                self.executor, self._init_encoder
            )
            vector_store_task = loop.run_in_executor(
                self.executor, self._init_vector_store, vector_db_path
            )
            augmentor_task = loop.run_in_executor(
                self.executor, self._init_augmentor
            )
            
            # Wait for all components to initialize
            self.encoder, self.vector_store, self.augmentor = await asyncio.gather(
                encoder_task, vector_store_task, augmentor_task
            )
            
            self.vector_db_path = vector_db_path
            self._initialized = True
            
            init_time = time.time() - start_time
            logger.info(f"RAG-CSD components initialized in {init_time:.2f}s")
    
    def _init_encoder(self) -> Encoder:
        """Initialize the encoder in a thread."""
        return Encoder(self.config)
    
    def _init_vector_store(self, vector_db_path: str) -> VectorStore:
        """Initialize the vector store in a thread."""
        return VectorStore(vector_db_path, self.config)
    
    def _init_augmentor(self) -> Augmentor:
        """Initialize the augmentor in a thread."""
        return Augmentor(self.config)
    
    async def encode_async(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        Encode query asynchronously.
        
        Args:
            query: Input query or list of queries.
            
        Returns:
            Vector embedding(s).
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.encoder.encode, query
        )
    
    async def search_async(
        self, 
        query_embedding: np.ndarray, 
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for similar documents asynchronously.
        
        Args:
            query_embedding: Query vector embedding.
            top_k: Number of results to return.
            
        Returns:
            List of retrieved documents.
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.vector_store.search, query_embedding, top_k
        )
    
    async def augment_async(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Augment query with retrieved documents asynchronously.
        
        Args:
            query: Original query.
            retrieved_docs: Retrieved documents.
            
        Returns:
            Augmented query.
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.augmentor.augment, query, retrieved_docs
        )
    
    async def process_query_async(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Process a single query through the full RAG pipeline asynchronously.
        
        Args:
            query: Input query.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with query results and metadata.
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Step 1: Encode query
        encode_start = time.time()
        query_embedding = await self.encode_async(query)
        encode_time = time.time() - encode_start
        
        # Step 2: Search for similar documents
        search_start = time.time()
        retrieved_docs = await self.search_async(query_embedding, top_k)
        search_time = time.time() - search_start
        
        # Step 3: Augment query
        augment_start = time.time()
        augmented_query = await self.augment_async(query, retrieved_docs)
        augment_time = time.time() - augment_start
        
        total_time = time.time() - start_time
        
        return {
            "original_query": query,
            "query_embedding": query_embedding.tolist(),
            "retrieved_docs": retrieved_docs,
            "augmented_query": augmented_query,
            "timings": {
                "encode": encode_time,
                "search": search_time,
                "augment": augment_time,
                "total": total_time,
            },
            "metadata": {
                "num_retrieved": len(retrieved_docs),
                "top_k": top_k or self.config.get("retrieval", {}).get("top_k", 5),
            }
        }
    
    async def process_queries_parallel(
        self, 
        queries: List[str], 
        top_k: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple queries in parallel.
        
        Args:
            queries: List of input queries.
            top_k: Number of documents to retrieve per query.
            max_concurrent: Maximum number of concurrent queries. If None, no limit.
            
        Returns:
            List of query results.
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        max_concurrent = max_concurrent or len(queries)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(query: str) -> Dict:
            async with semaphore:
                return await self.process_query_async(query, top_k)
        
        # Process all queries concurrently
        tasks = [process_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def process_queries_batch_optimized(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple queries using optimized batch processing.
        
        Args:
            queries: List of input queries.
            top_k: Number of documents to retrieve per query.
            batch_size: Batch size for encoding.
            
        Returns:
            List of query results.
        """
        if not self._initialized:
            raise RuntimeError("AsyncRAGCSD not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Step 1: Batch encode all queries
        loop = asyncio.get_event_loop()
        encode_start = time.time()
        query_embeddings = await loop.run_in_executor(
            self.executor, self.encoder.encode_batch, queries, batch_size
        )
        encode_time = time.time() - encode_start
        
        # Step 2: Batch search
        search_start = time.time()
        all_retrieved_docs = await loop.run_in_executor(
            self.executor, self.vector_store.search_batch, query_embeddings, top_k
        )
        search_time = time.time() - search_start
        
        # Step 3: Parallel augmentation
        augment_start = time.time()
        augmentation_tasks = []
        for i, (query, retrieved_docs) in enumerate(zip(queries, all_retrieved_docs)):
            task = self.augment_async(query, retrieved_docs)
            augmentation_tasks.append(task)
        
        augmented_queries = await asyncio.gather(*augmentation_tasks)
        augment_time = time.time() - augment_start
        
        total_time = time.time() - start_time
        
        # Construct results
        results = []
        for i, (query, embedding, retrieved_docs, augmented_query) in enumerate(
            zip(queries, query_embeddings, all_retrieved_docs, augmented_queries)
        ):
            result = {
                "original_query": query,
                "query_embedding": embedding.tolist(),
                "retrieved_docs": retrieved_docs,
                "augmented_query": augmented_query,
                "timings": {
                    "encode": encode_time / len(queries),  # Amortized
                    "search": search_time / len(queries),  # Amortized
                    "augment": augment_time / len(queries),  # Amortized
                    "total": total_time / len(queries),  # Amortized
                },
                "metadata": {
                    "num_retrieved": len(retrieved_docs),
                    "top_k": top_k or self.config.get("retrieval", {}).get("top_k", 5),
                    "batch_processing": True,
                }
            }
            results.append(result)
        
        logger.info(f"Batch processed {len(queries)} queries in {total_time:.3f}s "
                   f"(encode: {encode_time:.3f}s, search: {search_time:.3f}s, "
                   f"augment: {augment_time:.3f}s)")
        
        return results
    
    async def close(self) -> None:
        """Close the async interface and cleanup resources."""
        logger.info("Closing AsyncRAGCSD interface...")
        self.executor.shutdown(wait=True)
        self._initialized = False
        logger.info("AsyncRAGCSD interface closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def get_stats(self) -> Dict:
        """Get statistics about the async interface."""
        return {
            "max_workers": self.max_workers,
            "initialized": self._initialized,
            "vector_db_path": self.vector_db_path,
            "config_keys": list(self.config.keys()) if self.config else [],
        }