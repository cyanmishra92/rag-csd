"""
Baseline RAG system implementations for comparison.
This module provides simplified implementations of other RAG systems for benchmarking.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
import faiss

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class BaseRAGSystem(ABC):
    """Abstract base class for RAG systems."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def initialize(self, vector_db_path: str) -> None:
        """Initialize the RAG system."""
        pass
    
    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a single query."""
        pass
    
    @abstractmethod
    def query_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries."""
        pass


class VanillaRAG(BaseRAGSystem):
    """
    Vanilla RAG implementation (baseline).
    Simple, unoptimized RAG system for comparison.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
    
    def initialize(self, vector_db_path: str) -> None:
        """Initialize vanilla RAG components."""
        logger.info(f"Initializing {self.name}")
        
        # Load model fresh each time (no caching)
        model_name = self.config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        
        # Load vector database
        import os
        import json
        
        embeddings = np.load(os.path.join(vector_db_path, "embeddings.npy"))
        
        with open(os.path.join(vector_db_path, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        with open(os.path.join(vector_db_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # Create simple flat index (no optimization)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process single query without optimization."""
        start_time = time.time()
        
        # Encode query (no caching)
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score)
                })
        
        # Simple concatenation augmentation
        context = " ".join([r["chunk"] for r in results])
        augmented_query = f"{query} Context: {context}"
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": results,
            "processing_time": total_time,
            "system": self.name
        }
    
    def query_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process batch sequentially (no optimization)."""
        results = []
        for query in queries:
            results.append(self.query(query, top_k))
        return results


class PipeRAGLike(BaseRAGSystem):
    """
    PipeRAG-inspired implementation.
    Focuses on pipeline optimization and parallel processing.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
    
    def initialize(self, vector_db_path: str) -> None:
        """Initialize PipeRAG-like system."""
        logger.info(f"Initializing {self.name}")
        
        # Load model with basic caching
        model_name = self.config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        
        # Load vector database
        import os
        import json
        
        embeddings = np.load(os.path.join(vector_db_path, "embeddings.npy")).astype(np.float32)
        
        with open(os.path.join(vector_db_path, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        with open(os.path.join(vector_db_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # Use IVF index for better performance
        d = embeddings.shape[1]
        n = embeddings.shape[0]
        
        if n > 100:
            nlist = min(int(np.sqrt(n)), 100)
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            # Normalize and train
            faiss.normalize_L2(embeddings)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(nlist // 4, 16)
        else:
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process single query with pipeline optimization."""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score)
                })
        
        # Template-based augmentation
        context = "\n\n".join([r["chunk"] for r in results])
        augmented_query = f"Query: {query}\n\nContext: {context}"
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": results,
            "processing_time": total_time,
            "system": self.name
        }
    
    def query_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process batch with basic optimization."""
        start_time = time.time()
        
        # Batch encode
        query_embeddings = self.model.encode(queries).astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        # Batch search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Build results
        results = []
        for i, query in enumerate(queries):
            query_results = []
            for idx, score in zip(indices[i], scores[i]):
                if idx != -1:
                    query_results.append({
                        "chunk": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": float(score)
                    })
            
            # Augment
            context = "\n\n".join([r["chunk"] for r in query_results])
            augmented_query = f"Query: {query}\n\nContext: {context}"
            
            results.append({
                "query": query,
                "augmented_query": augmented_query,
                "retrieved_docs": query_results,
                "processing_time": (time.time() - start_time) / len(queries),  # Amortized
                "system": self.name
            })
        
        return results


class EdgeRAGLike(BaseRAGSystem):
    """
    EdgeRAG-inspired implementation.
    Focuses on edge computing optimizations and resource efficiency.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.query_cache = {}  # Simple query cache
    
    def initialize(self, vector_db_path: str) -> None:
        """Initialize EdgeRAG-like system."""
        logger.info(f"Initializing {self.name}")
        
        # Use smaller model for edge deployment
        model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller model
        self.model = SentenceTransformer(model_name)
        
        # Load vector database
        import os
        import json
        
        embeddings = np.load(os.path.join(vector_db_path, "embeddings.npy")).astype(np.float32)
        
        with open(os.path.join(vector_db_path, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        with open(os.path.join(vector_db_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        
        # Use simple index for edge efficiency
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process single query with edge optimizations."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key].copy()
            cached_result["from_cache"] = True
            cached_result["processing_time"] = time.time() - start_time
            return cached_result
        
        # Encode query
        query_embedding = self.model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search with reduced precision for speed
        scores, indices = self.index.search(query_embedding, min(top_k, 3))  # Limit results for edge
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score)
                })
        
        # Minimal augmentation for efficiency
        if results:
            context = results[0]["chunk"]  # Use only top result
            augmented_query = f"{query} {context}"
        else:
            augmented_query = query
        
        total_time = time.time() - start_time
        
        result = {
            "query": query,
            "augmented_query": augmented_query,
            "retrieved_docs": results,
            "processing_time": total_time,
            "system": self.name,
            "from_cache": False
        }
        
        # Cache result (limited cache size)
        if len(self.query_cache) < 100:
            self.query_cache[cache_key] = result.copy()
        
        return result
    
    def query_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process batch with edge constraints."""
        results = []
        
        # Process in smaller batches for memory efficiency
        batch_size = 4  # Small batch size for edge
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            for query in batch:
                results.append(self.query(query, top_k))
        
        return results


def get_baseline_systems() -> Dict[str, BaseRAGSystem]:
    """Get all available baseline systems."""
    return {
        "vanilla_rag": VanillaRAG,
        "piperag_like": PipeRAGLike,
        "edgerag_like": EdgeRAGLike,
    }