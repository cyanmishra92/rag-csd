"""
Query encoder module for RAG-CSD.
This module handles the encoding of queries to vector embeddings.
"""

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag_csd.utils.logger import get_logger
from rag_csd.utils.model_cache import model_cache
from rag_csd.utils.embedding_cache import get_embedding_cache
from rag_csd.utils.text_processor import get_text_processor

logger = get_logger(__name__)


class Encoder:
    """Query encoder class for generating vector embeddings."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the encoder with configuration.
        
        Args:
            config: Configuration dictionary.
                If None, default configuration will be used.
        """
        self.config = config or {}
        self.model_name = self.config.get("embedding", {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.batch_size = self.config.get("embedding", {}).get("batch_size", 32)
        self.normalize = self.config.get("embedding", {}).get("normalize", True)
        self.pooling_strategy = self.config.get("embedding", {}).get("pooling_strategy", "mean")
        self.use_amp = self.config.get("embedding", {}).get("use_amp", False)
        self.csd_enabled = self.config.get("csd", {}).get("enabled", False)
        self.csd_offload_embedding = self.config.get("csd", {}).get("offload_embedding", True)
        
        # Initialize embedding cache
        cache_config = self.config.get("embedding", {}).get("cache", {})
        self.use_cache = cache_config.get("enabled", True)
        if self.use_cache:
            cache_size = cache_config.get("max_size", 1000)
            cache_ttl = cache_config.get("ttl", 3600)  # 1 hour default
            self.embedding_cache = get_embedding_cache(
                max_size=cache_size,
                ttl=cache_ttl
            )
            logger.info(f"Embedding cache enabled: max_size={cache_size}, ttl={cache_ttl}")
        else:
            self.embedding_cache = None
        
        # Initialize text processor
        self.text_processor = get_text_processor(self.config)
        
        # If CSD is enabled and offloading is enabled for embedding,
        # we'll simulate the CSD behavior
        if self.csd_enabled and self.csd_offload_embedding:
            self.device = "csd"
            logger.info("Encoder will use CSD simulation")
            self._init_csd_simulation()
        else:
            # Initialize the model
            self.device = self._get_device()
            logger.info(f"Initializing encoder model: {self.model_name}")
            self._init_model()
    
    def _get_device(self) -> str:
        """
        Determine the device to use for encoding.
        
        Returns:
            str: Device name ('cpu', 'cuda', or 'csd')
        """
        device = self.config.get("general", {}).get("device", "auto")
        
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        return device
    
    def _init_model(self) -> None:
        """Initialize the encoding model using the model cache."""
        try:
            # Use model cache to get or load the model
            self.model = model_cache.get_model(self.model_name, self.device)
            
            # Set the dimension from the model
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Encoder dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error initializing encoder model: {e}")
            raise RuntimeError(f"Failed to initialize encoder model: {e}")
    
    def _init_csd_simulation(self) -> None:
        """Initialize the CSD simulation for encoding."""
        # In a real implementation, this would initialize the connection to the CSD
        # For simulation, we'll just set the dimension and latency
        self.dimension = 384  # Default dimension for the selected model
        self.csd_latency = self.config.get("csd", {}).get("latency", 5)  # in ms
        logger.info(f"CSD simulation initialized with latency: {self.csd_latency}ms")
        
        # Optionally load the model for simulation purposes
        if self.config.get("csd", {}).get("simulator", True):
            try:
                # Use model cache for CSD simulation too
                self.model = model_cache.get_model(self.model_name, "cpu")  # Use CPU for simulation
                logger.info(f"Model loaded for CSD simulation: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load model for CSD simulation: {e}")
                self.model = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of tokens.
        """
        if hasattr(self, 'model') and self.model is not None:
            # Use the model's tokenizer
            tokens = self.model.tokenize([text])[0]
            return tokens
        else:
            # Use optimized text processor
            return self.text_processor.tokenize(text)
    
    def encode(self, query: Union[str, List[str]]) -> np.ndarray:
        """
        Encode the query to a vector embedding.
        
        Args:
            query: Input query text or list of query texts.
            
        Returns:
            Vector embedding(s) as numpy array.
        """
        start_time = time.time()
        
        # Handle single query or list of queries
        if isinstance(query, str):
            queries = [query]
            single_query = True
        else:
            queries = query
            single_query = False
        
        # Preprocess queries using the text processor
        queries = [self.text_processor.preprocess_query(q) for q in queries]
        
        # Try to get embeddings from cache first
        cached_embeddings = []
        uncached_queries = []
        uncached_indices = []
        
        if self.use_cache and self.embedding_cache:
            for i, q in enumerate(queries):
                cached_embedding = self.embedding_cache.get(q)
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    uncached_queries.append(q)
                    uncached_indices.append(i)
        else:
            uncached_queries = queries
            uncached_indices = list(range(len(queries)))
        
        # Encode uncached queries
        if uncached_queries:
            if self.device == "csd":
                # Simulate CSD encoding
                new_embeddings = self._encode_on_csd(uncached_queries)
            else:
                # Regular encoding using the model
                new_embeddings = self._encode_with_model(uncached_queries)
            
            # Cache the new embeddings
            if self.use_cache and self.embedding_cache:
                for q, embedding in zip(uncached_queries, new_embeddings):
                    self.embedding_cache.put(q, embedding)
        else:
            new_embeddings = np.array([])
        
        # Combine cached and new embeddings in the correct order
        if cached_embeddings and len(uncached_queries) > 0:
            all_embeddings = [None] * len(queries)
            
            # Place cached embeddings
            for idx, embedding in cached_embeddings:
                all_embeddings[idx] = embedding
            
            # Place new embeddings
            for i, idx in enumerate(uncached_indices):
                all_embeddings[idx] = new_embeddings[i]
            
            embeddings = np.array(all_embeddings)
        elif cached_embeddings:
            # All from cache
            embeddings = np.array([emb for _, emb in cached_embeddings])
        else:
            # All newly computed
            embeddings = new_embeddings
        
        elapsed = (time.time() - start_time) * 1000  # convert to ms
        cache_hits = len(cached_embeddings)
        cache_misses = len(uncached_queries)
        logger.debug(f"Encoding {len(queries)} queries completed in {elapsed:.2f}ms "
                    f"(cache hits: {cache_hits}, misses: {cache_misses})")
        
        # Return single embedding for single query
        if single_query:
            return embeddings[0]
        
        return embeddings
    
    def encode_batch(self, queries: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode multiple queries in optimized batches.
        
        Args:
            queries: List of query texts to encode.
            batch_size: Batch size for encoding. If None, uses the configured batch size.
            
        Returns:
            Array of embeddings with shape (len(queries), embedding_dim).
        """
        if not queries:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        
        # If batch size is larger than queries, encode all at once
        if len(queries) <= batch_size:
            return self.encode(queries)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _encode_with_model(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries using the loaded model.
        
        Args:
            queries: List of query texts.
            
        Returns:
            Vector embeddings as numpy array.
        """
        with torch.no_grad():
            if self.use_amp and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(
                        queries,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=self.normalize,
                    )
            else:
                embeddings = self.model.encode(
                    queries,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize,
                )
        
        return embeddings
    
    def _encode_on_csd(self, queries: List[str]) -> np.ndarray:
        """
        Simulate encoding queries on the CSD.
        
        Args:
            queries: List of query texts.
            
        Returns:
            Vector embeddings as numpy array.
        """
        # Simulate CSD latency
        time.sleep(self.csd_latency / 1000.0)  # Convert ms to seconds
        
        if hasattr(self, 'model') and self.model is not None:
            # Use the model for simulation, but add the latency
            return self._encode_with_model(queries)
        else:
            # Generate random embeddings of the correct dimension for simulation
            embeddings = np.random.randn(len(queries), self.dimension)
            
            # Normalize if required
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            return embeddings