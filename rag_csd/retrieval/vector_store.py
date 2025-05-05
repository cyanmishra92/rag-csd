"""
Vector store module for RAG-CSD.
This module handles the storage and retrieval of vectors.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Vector store class for storing and searching vectors."""
    
    def __init__(self, vector_db_path: str, config: Optional[Dict] = None):
        """
        Initialize the vector store with the database path and configuration.
        
        Args:
            vector_db_path: Path to the vector database.
            config: Configuration dictionary.
                If None, default configuration will be used.
        """
        self.vector_db_path = vector_db_path
        self.config = config or {}
        self.vector_db_type = self.config.get("retrieval", {}).get("vector_db_type", "faiss")
        self.similarity_metric = self.config.get("retrieval", {}).get("similarity_metric", "cosine")
        self.top_k = self.config.get("retrieval", {}).get("top_k", 5)
        self.min_similarity_score = self.config.get("retrieval", {}).get("min_similarity_score", 0.7)
        self.csd_enabled = self.config.get("csd", {}).get("enabled", False)
        self.csd_offload_search = self.config.get("csd", {}).get("offload_search", True)
        
        # If CSD is enabled and offloading is enabled for search,
        # we'll simulate the CSD behavior
        if self.csd_enabled and self.csd_offload_search:
            self.device = "csd"
            logger.info("Vector store will use CSD simulation")
            self._init_csd_simulation()
        else:
            # Initialize the database
            self.device = "cpu"  # FAISS uses CPU by default
            logger.info(f"Initializing vector store: {self.vector_db_type}")
            self._load_vector_db()
    
    def _load_vector_db(self) -> None:
        """Load the vector database."""
        try:
            # Load embeddings
            embeddings_path = os.path.join(self.vector_db_path, "embeddings.npy")
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Loaded {self.embeddings.shape[0]} embeddings "
                       f"with dimension {self.embeddings.shape[1]}")
            
            # Load chunks
            chunks_path = os.path.join(self.vector_db_path, "chunks.json")
            with open(chunks_path, "r") as f:
                self.chunks = json.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.vector_db_path, "metadata.json")
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            
            # Initialize FAISS index
            self._init_faiss_index()
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            raise RuntimeError(f"Failed to load vector database: {e}")
    
    def _init_faiss_index(self) -> None:
        """Initialize the FAISS index based on the similarity metric."""
        try:
            d = self.embeddings.shape[1]  # Dimension
            
            if self.similarity_metric == "cosine":
                # Normalize embeddings for cosine similarity
                self.faiss_index = faiss.IndexFlatIP(d)  # Inner product
                # Normalize the vectors to use inner product for cosine similarity
                faiss.normalize_L2(self.embeddings)
            elif self.similarity_metric == "dot":
                self.faiss_index = faiss.IndexFlatIP(d)  # Inner product
            elif self.similarity_metric == "euclidean":
                self.faiss_index = faiss.IndexFlatL2(d)  # L2 distance
            else:
                raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
            # Add vectors to the index
            self.faiss_index.add(self.embeddings)
            logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    
    def _init_csd_simulation(self) -> None:
        """Initialize the CSD simulation for vector search."""
        # In a real implementation, this would initialize the connection to the CSD
        # For simulation, we'll load the database but with added latency
        self.csd_latency = self.config.get("csd", {}).get("latency", 5)  # in ms
        logger.info(f"CSD simulation initialized with latency: {self.csd_latency}ms")
        
        # Load the database for simulation
        self._load_vector_db()
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: Optional[int] = None, 
        min_score: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar vectors in the database.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return. If None, uses the default.
            min_score: Minimum similarity score. If None, uses the default.
            
        Returns:
            List of dictionaries containing the retrieved results.
        """
        start_time = time.time()
        
        # Set defaults if not provided
        if top_k is None:
            top_k = self.top_k
        if min_score is None:
            min_score = self.min_similarity_score
        
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Handle different devices for search
        if self.device == "csd":
            results = self._search_on_csd(query_embedding, top_k)
        else:
            results = self._search_with_faiss(query_embedding, top_k)
        
        # Filter by minimum score if needed
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]
        
        elapsed = (time.time() - start_time) * 1000  # convert to ms
        logger.debug(f"Search completed in {elapsed:.2f}ms, found {len(results)} results")
        
        return results
    
    def _search_with_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Search for similar vectors using FAISS.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing the retrieved results.
        """
        # Normalize query for cosine similarity if needed
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search the index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Convert to scores (higher is better)
        if self.similarity_metric == "euclidean":
            # Convert distances to scores (1 / (1 + distance))
            scores = 1.0 / (1.0 + distances[0])
        else:
            # For cosine and dot product, higher is better
            scores = distances[0]
        
        # Construct results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores)):
            if idx != -1:  # FAISS returns -1 for not enough results
                result = {
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score),
                    "index": int(idx),
                }
                results.append(result)
        
        return results
    
    def _search_on_csd(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Simulate searching for similar vectors on the CSD.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing the retrieved results.
        """
        # Simulate CSD latency
        time.sleep(self.csd_latency / 1000.0)  # Convert ms to seconds
        
        # Use the regular search method but with the added latency
        return self._search_with_faiss(query_embedding, top_k)