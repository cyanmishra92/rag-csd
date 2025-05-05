"""
CSD simulator module for RAG-CSD.
This module simulates the behavior of a Computational Storage Device (CSD).
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class CSDSimulator:
    """Simulator for Computational Storage Device (CSD) operations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CSD simulator with configuration.
        
        Args:
            config: Configuration dictionary.
                If None, default configuration will be used.
        """
        self.config = config or {}
        self.latency = self.config.get("csd", {}).get("latency", 5)  # in ms
        self.bandwidth = self.config.get("csd", {}).get("bandwidth", 2000)  # in MB/s
        self.memory = self.config.get("csd", {}).get("memory", 4)  # in GB
        self.parallel_operations = self.config.get("csd", {}).get("parallel_operations", 8)
        
        # Semaphore to limit parallel operations
        self.semaphore = threading.Semaphore(self.parallel_operations)
        
        # Embedding model configuration
        self.embedding_dimension = 384  # Default for the small model
        
        logger.info(f"CSD simulator initialized with {self.parallel_operations} "
                   f"parallel operations, {self.latency}ms latency, "
                   f"{self.bandwidth}MB/s bandwidth, {self.memory}GB memory")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simulate tokenization of text on the CSD.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of tokens.
        """
        # Simulate CSD operation latency
        self._simulate_latency()
        
        # Simple whitespace tokenization for simulation
        tokens = text.split()
        
        return tokens
    
    def encode(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        Simulate encoding of query to vector embedding on the CSD.
        
        Args:
            query: Input query text or list of query texts.
            
        Returns:
            Vector embedding(s) as numpy array.
        """
        # Acquire semaphore for parallel operation limit
        with self.semaphore:
            # Simulate CSD operation latency
            self._simulate_latency()
            
            # Handle single query or list of queries
            if isinstance(query, str):
                queries = [query]
            else:
                queries = query
            
            # Generate random embeddings for simulation
            embeddings = np.random.randn(len(queries), self.embedding_dimension)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Return single embedding for single query
            if isinstance(query, str):
                return embeddings[0]
            
            return embeddings
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        database_embeddings: np.ndarray, 
        top_k: int = 5,
        similarity_metric: str = "cosine"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate similarity search on the CSD.
        
        Args:
            query_embedding: The query vector.
            database_embeddings: The database vectors.
            top_k: Number of results to return.
            similarity_metric: Similarity metric to use.
            
        Returns:
            Tuple of (scores, indices).
        """
        # Acquire semaphore for parallel operation limit
        with self.semaphore:
            # Simulate CSD operation latency
            data_size_mb = database_embeddings.nbytes / (1024 * 1024)
            transfer_time_ms = (data_size_mb / self.bandwidth) * 1000
            computation_time_ms = self.latency
            total_time_ms = transfer_time_ms + computation_time_ms
            
            self._simulate_latency(latency_ms=total_time_ms)
            
            # Ensure query_embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Compute similarity based on metric
            if similarity_metric == "cosine" or similarity_metric == "dot":
                # Normalize for cosine similarity
                if similarity_metric == "cosine":
                    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
                    db_norm = np.linalg.norm(database_embeddings, axis=1, keepdims=True)
                    query_embedding = query_embedding / query_norm
                    database_embeddings = database_embeddings / db_norm
                
                # Compute dot product
                scores = np.dot(query_embedding, database_embeddings.T)[0]
                
            elif similarity_metric == "euclidean":
                # Compute euclidean distance
                distances = np.linalg.norm(
                    query_embedding - database_embeddings[:, np.newaxis], axis=2
                )[0]
                # Convert distances to scores (1 / (1 + distance))
                scores = 1.0 / (1.0 + distances)
            
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
            
            # Get top-k indices
            if len(scores) <= top_k:
                indices = np.argsort(scores)[::-1]
            else:
                indices = np.argpartition(scores, -top_k)[-top_k:]
                indices = indices[np.argsort(scores[indices])[::-1]]
            
            # Get corresponding scores
            top_scores = scores[indices]
            
            return top_scores, indices
    
    def _simulate_latency(self, latency_ms: Optional[float] = None):
        """
        Simulate operation latency.
        
        Args:
            latency_ms: Custom latency in milliseconds.
                If None, uses the default.
        """
        latency = latency_ms if latency_ms is not None else self.latency
        time.sleep(latency / 1000.0)  # Convert ms to seconds