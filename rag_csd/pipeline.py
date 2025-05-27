"""
Complete RAG-CSD pipeline implementation.
This module provides a high-level interface for the entire RAG-CSD system.
"""

import time
from typing import Dict, List, Optional, Union, Any
import logging

from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.augmentation.augmentor import Augmentor
from rag_csd.async_interface import AsyncRAGCSD
from rag_csd.utils.logger import get_logger, performance_logger
from rag_csd.utils.metrics import metrics_collector, monitor_operation
from rag_csd.utils.error_handling import handle_exceptions, retry_with_backoff, RAGCSDError

logger = get_logger(__name__)


class RAGCSDPipeline:
    """
    Complete RAG-CSD pipeline with advanced features.
    
    This class provides a high-level interface to the RAG-CSD system with:
    - Automatic optimization selection
    - Performance monitoring
    - Error handling and recovery
    - Multiple processing modes (sync, async, batch)
    """
    
    def __init__(self, config: Optional[Dict] = None, vector_db_path: str = None):
        """
        Initialize the RAG-CSD pipeline.
        
        Args:
            config: Configuration dictionary.
            vector_db_path: Path to the vector database.
        """
        self.config = config or {}
        self.vector_db_path = vector_db_path
        
        # Components
        self.encoder = None
        self.vector_store = None
        self.augmentor = None
        self.async_interface = None
        
        # State
        self._initialized = False
        self._performance_stats = {}
        
        logger.info("RAGCSDPipeline initialized")
    
    @handle_exceptions(reraise=True)
    @retry_with_backoff(max_retries=3)
    def initialize(self, vector_db_path: Optional[str] = None) -> None:
        """
        Initialize all pipeline components.
        
        Args:
            vector_db_path: Path to vector database. Uses constructor value if None.
        """
        if self._initialized:
            logger.info("Pipeline already initialized")
            return
        
        vector_db_path = vector_db_path or self.vector_db_path
        if not vector_db_path:
            raise RAGCSDError("Vector database path must be provided")
        
        logger.info("Initializing RAG-CSD pipeline components...")
        
        with monitor_operation("pipeline_initialization"):
            # Initialize components
            self.encoder = Encoder(self.config)
            self.vector_store = VectorStore(vector_db_path, self.config)
            self.augmentor = Augmentor(self.config)
            
            # Initialize async interface
            max_workers = self.config.get("performance", {}).get("num_threads", 8)
            self.async_interface = AsyncRAGCSD(self.config, max_workers=max_workers)
            
            self._initialized = True
        
        logger.info("Pipeline initialization completed")
    
    def query(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        mode: str = "auto",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single query through the RAG pipeline.
        
        Args:
            query: Input query string.
            top_k: Number of documents to retrieve.
            mode: Processing mode ("sync", "async", "auto").
            include_metadata: Whether to include detailed metadata.
            
        Returns:
            Dictionary with query results and metadata.
        """
        if not self._initialized:
            self.initialize()
        
        # Auto-select mode based on query characteristics
        if mode == "auto":
            mode = self._select_optimal_mode(query)
        
        with monitor_operation("query_processing", context=mode):
            if mode == "async":
                return self._query_async(query, top_k, include_metadata)
            else:
                return self._query_sync(query, top_k, include_metadata)
    
    def query_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        mode: str = "auto",
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries efficiently.
        
        Args:
            queries: List of query strings.
            top_k: Number of documents to retrieve per query.
            mode: Processing mode ("sync", "async", "batch").
            max_concurrent: Maximum concurrent queries for async mode.
            
        Returns:
            List of query results.
        """
        if not self._initialized:
            self.initialize()
        
        if not queries:
            return []
        
        # Auto-select mode based on batch characteristics
        if mode == "auto":
            mode = self._select_optimal_batch_mode(queries)
        
        with monitor_operation("batch_query_processing", context=f"{mode}_{len(queries)}"):
            if mode == "batch":
                return self._query_batch_optimized(queries, top_k)
            elif mode == "async":
                return self._query_batch_async(queries, top_k, max_concurrent)
            else:
                return self._query_batch_sync(queries, top_k)
    
    def _query_sync(self, query: str, top_k: Optional[int], include_metadata: bool) -> Dict[str, Any]:
        """Process query synchronously."""
        start_time = time.time()
        
        # Encode query
        with monitor_operation("encode"):
            query_embedding = self.encoder.encode(query)
        
        # Retrieve documents
        with monitor_operation("retrieve"):
            retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Augment query
        with monitor_operation("augment"):
            augmented_query = self.augmentor.augment(query, retrieved_docs)
        
        total_time = time.time() - start_time
        
        result = {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": retrieved_docs,
            "num_retrieved": len(retrieved_docs),
        }
        
        if include_metadata:
            result.update({
                "processing_mode": "sync",
                "total_time": total_time,
                "top_k": top_k or self.config.get("retrieval", {}).get("top_k", 5),
                "embedding_dim": len(query_embedding),
            })
        
        return result
    
    async def _query_async(self, query: str, top_k: Optional[int], include_metadata: bool) -> Dict[str, Any]:
        """Process query asynchronously."""
        if not self.async_interface._initialized:
            await self.async_interface.initialize(self.vector_db_path)
        
        result = await self.async_interface.process_query_async(query, top_k)
        
        if not include_metadata:
            # Remove metadata if not requested
            result = {
                "query": result["original_query"],
                "augmented_query": result["augmented_query"],
                "retrieved_docs": result["retrieved_docs"],
                "num_retrieved": result["metadata"]["num_retrieved"],
            }
        else:
            result["processing_mode"] = "async"
        
        return result
    
    def _query_batch_sync(self, queries: List[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """Process batch synchronously."""
        results = []
        for query in queries:
            result = self._query_sync(query, top_k, include_metadata=True)
            result["processing_mode"] = "batch_sync"
            results.append(result)
        return results
    
    async def _query_batch_async(self, queries: List[str], top_k: Optional[int], max_concurrent: Optional[int]) -> List[Dict[str, Any]]:
        """Process batch asynchronously."""
        if not self.async_interface._initialized:
            await self.async_interface.initialize(self.vector_db_path)
        
        results = await self.async_interface.process_queries_parallel(
            queries, top_k, max_concurrent
        )
        
        for result in results:
            result["processing_mode"] = "batch_async"
        
        return results
    
    async def _query_batch_optimized(self, queries: List[str], top_k: Optional[int]) -> List[Dict[str, Any]]:
        """Process batch with optimization."""
        if not self.async_interface._initialized:
            await self.async_interface.initialize(self.vector_db_path)
        
        results = await self.async_interface.process_queries_batch_optimized(
            queries, top_k
        )
        
        for result in results:
            result["processing_mode"] = "batch_optimized"
        
        return results
    
    def _select_optimal_mode(self, query: str) -> str:
        """Select optimal processing mode for a single query."""
        # Simple heuristics - can be enhanced with ML
        query_length = len(query)
        
        if query_length > 500:  # Long queries might benefit from async
            return "async"
        else:
            return "sync"
    
    def _select_optimal_batch_mode(self, queries: List[str]) -> str:
        """Select optimal processing mode for batch queries."""
        num_queries = len(queries)
        avg_length = sum(len(q) for q in queries) / num_queries
        
        if num_queries >= 10:  # Large batches benefit from optimization
            return "batch"
        elif num_queries >= 3 and avg_length > 200:  # Medium batches with long queries
            return "async"
        else:
            return "sync"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "metrics": metrics_collector.get_all_metrics_summary(),
            "cache_stats": {
                "model_cache": getattr(self.encoder, 'model_cache', {}).get_cache_info() if self.encoder else {},
                "embedding_cache": getattr(self.encoder, 'embedding_cache', {}).get_stats() if self.encoder else {},
            },
            "system_performance": performance_logger.metrics,
        }
    
    def optimize_for_workload(self, workload_type: str = "balanced") -> None:
        """
        Optimize pipeline configuration for specific workload types.
        
        Args:
            workload_type: Type of workload ("latency", "throughput", "balanced").
        """
        logger.info(f"Optimizing pipeline for {workload_type} workload")
        
        if workload_type == "latency":
            # Optimize for low latency
            if "embedding" not in self.config:
                self.config["embedding"] = {}
            self.config["embedding"]["batch_size"] = 1
            self.config["embedding"]["cache"]["enabled"] = True
            
        elif workload_type == "throughput":
            # Optimize for high throughput
            if "embedding" not in self.config:
                self.config["embedding"] = {}
            self.config["embedding"]["batch_size"] = 64
            
        elif workload_type == "balanced":
            # Balanced optimization
            if "embedding" not in self.config:
                self.config["embedding"] = {}
            self.config["embedding"]["batch_size"] = 32
        
        logger.info(f"Pipeline optimized for {workload_type}")
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all pipeline components.
        
        Returns:
            Dictionary with health status of each component.
        """
        health = {
            "initialized": self._initialized,
            "encoder": False,
            "vector_store": False,
            "augmentor": False,
            "async_interface": False,
        }
        
        if self._initialized:
            try:
                # Test encoder
                if self.encoder:
                    test_embedding = self.encoder.encode("test")
                    health["encoder"] = len(test_embedding) > 0
                
                # Test vector store
                if self.vector_store and health["encoder"]:
                    test_results = self.vector_store.search(test_embedding, top_k=1)
                    health["vector_store"] = isinstance(test_results, list)
                
                # Test augmentor
                if self.augmentor:
                    test_aug = self.augmentor.augment("test", [])
                    health["augmentor"] = isinstance(test_aug, str)
                
                # Test async interface
                if self.async_interface:
                    health["async_interface"] = hasattr(self.async_interface, 'initialize')
                
            except Exception as e:
                logger.warning(f"Health check error: {e}")
        
        return health
    
    def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        logger.info("Shutting down RAG-CSD pipeline...")
        
        if self.async_interface:
            try:
                import asyncio
                asyncio.run(self.async_interface.close())
            except Exception as e:
                logger.warning(f"Error shutting down async interface: {e}")
        
        self._initialized = False
        logger.info("Pipeline shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()