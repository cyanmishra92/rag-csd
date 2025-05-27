"""
Model cache module for RAG-CSD.
This module provides caching for embedding models to reduce initialization overhead.
"""

import logging
import os
import threading
import time
from typing import Dict, Optional

import torch
from sentence_transformers import SentenceTransformer

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCache:
    """Singleton model cache for managing embedding models."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.models: Dict[str, SentenceTransformer] = {}
            self.model_locks: Dict[str, threading.Lock] = {}
            self.cache_dir = os.path.expanduser("~/.cache/rag_csd/models")
            os.makedirs(self.cache_dir, exist_ok=True)
            self._initialized = True
            logger.info("ModelCache initialized")
    
    def get_model(self, model_name: str, device: str = "auto") -> SentenceTransformer:
        """
        Get a cached model or load it if not cached.
        
        Args:
            model_name: Name of the model to load.
            device: Device to load the model on.
            
        Returns:
            The loaded SentenceTransformer model.
        """
        cache_key = f"{model_name}_{device}"
        
        # Check if model is already cached
        if cache_key in self.models:
            logger.debug(f"Using cached model: {cache_key}")
            return self.models[cache_key]
        
        # Get or create lock for this model
        if cache_key not in self.model_locks:
            self.model_locks[cache_key] = threading.Lock()
        
        # Load model with lock to prevent duplicate loading
        with self.model_locks[cache_key]:
            # Double check after acquiring lock
            if cache_key in self.models:
                logger.debug(f"Using cached model (after lock): {cache_key}")
                return self.models[cache_key]
            
            logger.info(f"Loading model: {model_name} on {device}")
            start_time = time.time()
            
            try:
                # Load the model
                model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
                
                # Move to appropriate device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if device == "cuda" and torch.cuda.is_available():
                    model = model.to("cuda")
                    logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("Model loaded on CPU")
                
                # Cache the model
                self.models[cache_key] = model
                
                load_time = time.time() - start_time
                logger.info(f"Model {model_name} loaded and cached in {load_time:.2f}s")
                
                return model
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def warm_up_model(self, model_name: str, device: str = "auto", test_text: str = "test") -> None:
        """
        Warm up a model by running a test encoding.
        
        Args:
            model_name: Name of the model to warm up.
            device: Device to load the model on.
            test_text: Test text to encode for warm-up.
        """
        logger.info(f"Warming up model: {model_name}")
        start_time = time.time()
        
        model = self.get_model(model_name, device)
        
        # Run a test encoding to warm up the model
        with torch.no_grad():
            _ = model.encode([test_text], show_progress_bar=False)
        
        warmup_time = time.time() - start_time
        logger.info(f"Model {model_name} warmed up in {warmup_time:.2f}s")
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        logger.info("Clearing model cache")
        self.models.clear()
        self.model_locks.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about cached models."""
        return {
            "cached_models": list(self.models.keys()),
            "num_models": len(self.models),
            "cache_dir": self.cache_dir
        }


# Global instance
model_cache = ModelCache()