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
        """Initialize the encoding model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info(f"Encoder model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Encoder model loaded on CPU")
            
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
                # Load the model but don't use it for actual encoding in CSD mode
                self.model = SentenceTransformer(self.model_name)
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
            # Fallback to simple whitespace tokenization
            return text.split()
    
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
        else:
            queries = query
        
        if self.device == "csd":
            # Simulate CSD encoding
            embeddings = self._encode_on_csd(queries)
        else:
            # Regular encoding using the model
            embeddings = self._encode_with_model(queries)
        
        elapsed = (time.time() - start_time) * 1000  # convert to ms
        logger.debug(f"Encoding completed in {elapsed:.2f}ms")
        
        # Return single embedding for single query
        if isinstance(query, str):
            return embeddings[0]
        
        return embeddings
    
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