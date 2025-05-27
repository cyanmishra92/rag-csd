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
        """Initialize the FAISS index based on the similarity metric and dataset size."""
        try:
            d = self.embeddings.shape[1]  # Dimension
            n = self.embeddings.shape[0]  # Number of vectors
            
            # Choose index type based on dataset size and configuration
            index_type = self.config.get("retrieval", {}).get("index_type", "auto")
            
            if index_type == "auto":
                # Automatically choose index type based on dataset size
                if n < 500:  # Small datasets: use flat
                    index_type = "flat"
                elif n < 10000:  # Medium datasets: use IVF if we have enough points
                    if n >= 39 * 4:  # Minimum points for 4 clusters
                        index_type = "ivf"
                    else:
                        index_type = "flat"
                else:  # Large datasets: use HNSW
                    index_type = "hnsw"
            
            logger.info(f"Using FAISS index type: {index_type} for {n} vectors")
            
            # Prepare embeddings based on similarity metric
            embeddings_to_index = self.embeddings.copy().astype(np.float32)
            if self.similarity_metric == "cosine":
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_to_index)
            
            # Create the appropriate index
            if index_type == "flat":
                self.faiss_index = self._create_flat_index(d)
            elif index_type == "ivf":
                self.faiss_index = self._create_ivf_index(d, n, embeddings_to_index)
            elif index_type == "hnsw":
                self.faiss_index = self._create_hnsw_index(d, embeddings_to_index)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Add vectors to the index
            if index_type != "hnsw":  # HNSW adds vectors during creation
                self.faiss_index.add(embeddings_to_index)
            
            logger.info(f"FAISS {index_type} index created with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    
    def _create_flat_index(self, d: int) -> faiss.Index:
        """Create a flat (brute force) FAISS index."""
        if self.similarity_metric in ["cosine", "dot"]:
            return faiss.IndexFlatIP(d)  # Inner product
        else:  # euclidean
            return faiss.IndexFlatL2(d)  # L2 distance
    
    def _create_ivf_index(self, d: int, n: int, embeddings: np.ndarray) -> faiss.Index:
        """Create an IVF (Inverted File) FAISS index."""
        # Choose number of clusters based on dataset size
        # FAISS requires at least 39*nlist training points for good clustering
        min_clusters = max(4, int(np.sqrt(n)))
        max_clusters = min(n // 39, 4096)  # Ensure we have enough training points
        nlist = max(min_clusters, min(max_clusters, 100))
        
        if self.similarity_metric in ["cosine", "dot"]:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
        else:  # euclidean
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
        
        # Train the index
        logger.info(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings)
        
        # Set search parameters
        index.nprobe = min(max(nlist // 8, 1), 64)  # Search 1/8 of clusters, max 64
        
        return index
    
    def _create_hnsw_index(self, d: int, embeddings: np.ndarray) -> faiss.Index:
        """Create an HNSW (Hierarchical Navigable Small World) FAISS index."""
        M = self.config.get("retrieval", {}).get("hnsw_M", 16)  # Number of connections
        
        if self.similarity_metric in ["cosine", "dot"]:
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        else:  # euclidean
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
        
        # Set construction parameters
        index.hnsw.efConstruction = self.config.get("retrieval", {}).get("hnsw_efConstruction", 200)
        
        # Add vectors during construction for HNSW
        logger.info("Building HNSW index...")
        index.add(embeddings)
        
        # Set search parameter
        index.hnsw.efSearch = self.config.get("retrieval", {}).get("hnsw_efSearch", 64)
        
        return index
    
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
    
    def search_batch(
        self, 
        query_embeddings: np.ndarray, 
        top_k: Optional[int] = None, 
        min_score: Optional[float] = None
    ) -> List[List[Dict]]:
        """
        Search for similar vectors for multiple query embeddings.
        
        Args:
            query_embeddings: Array of query vectors with shape (n_queries, embedding_dim).
            top_k: Number of results to return per query. If None, uses the default.
            min_score: Minimum similarity score. If None, uses the default.
            
        Returns:
            List of result lists, one for each query.
        """
        start_time = time.time()
        
        # Set defaults if not provided
        if top_k is None:
            top_k = self.top_k
        if min_score is None:
            min_score = self.min_similarity_score
        
        # Ensure query_embeddings is 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        all_results = []
        
        # Handle different devices for search
        if self.device == "csd":
            # For CSD, process one by one to simulate latency properly
            for i in range(query_embeddings.shape[0]):
                query_embedding = query_embeddings[i:i+1]
                results = self._search_on_csd(query_embedding, top_k)
                
                # Filter by minimum score if needed
                if min_score > 0:
                    results = [r for r in results if r["score"] >= min_score]
                
                all_results.append(results)
        else:
            # For regular processing, use batch FAISS search
            results = self._search_batch_with_faiss(query_embeddings, top_k)
            
            # Filter by minimum score if needed for each query
            if min_score > 0:
                for i in range(len(results)):
                    results[i] = [r for r in results[i] if r["score"] >= min_score]
            
            all_results = results
        
        elapsed = (time.time() - start_time) * 1000  # convert to ms
        total_results = sum(len(r) for r in all_results)
        logger.debug(f"Batch search completed in {elapsed:.2f}ms, found {total_results} total results")
        
        return all_results
    
    def _search_batch_with_faiss(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Dict]]:
        """
        Search for similar vectors using FAISS batch processing.
        
        Args:
            query_embeddings: Array of query vectors.
            top_k: Number of results to return per query.
            
        Returns:
            List of result lists, one for each query.
        """
        # Make a copy and ensure it's the right type
        query_copy = query_embeddings.copy().astype(np.float32)
        
        # Normalize queries for cosine similarity if needed
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_copy)
        
        # Batch search the index
        distances, indices = self.faiss_index.search(query_copy, top_k)
        
        # Convert to scores (higher is better)
        if self.similarity_metric == "euclidean":
            # Convert distances to scores (1 / (1 + distance))
            scores = 1.0 / (1.0 + distances)
        else:
            # For cosine and dot product, higher is better
            scores = distances
        
        # Construct results for each query
        all_results = []
        for query_idx in range(query_embeddings.shape[0]):
            query_results = []
            for i, (idx, score) in enumerate(zip(indices[query_idx], scores[query_idx])):
                if idx != -1:  # FAISS returns -1 for not enough results
                    result = {
                        "chunk": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": float(score),
                        "index": int(idx),
                    }
                    query_results.append(result)
            all_results.append(query_results)
        
        return all_results
    
    def _search_with_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Search for similar vectors using FAISS.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing the retrieved results.
        """
        # Make a copy and ensure it's the right type
        query_copy = query_embedding.copy().astype(np.float32)
        
        # Normalize query for cosine similarity if needed
        if self.similarity_metric == "cosine":
            # Ensure query is 2D for FAISS normalization
            if query_copy.ndim == 1:
                query_copy = query_copy.reshape(1, -1)
            faiss.normalize_L2(query_copy)
        else:
            # Ensure query is 2D for FAISS search
            if query_copy.ndim == 1:
                query_copy = query_copy.reshape(1, -1)
        
        # Search the index
        distances, indices = self.faiss_index.search(query_copy, top_k)
        
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