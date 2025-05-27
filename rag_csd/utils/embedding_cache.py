"""
Embedding cache module for RAG-CSD.
This module provides caching for query embeddings to avoid recomputation.
"""

import hashlib
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """LRU cache for query embeddings with persistence support."""
    
    def __init__(self, max_size: int = 1000, cache_dir: Optional[str] = None, ttl: Optional[int] = None):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache.
            cache_dir: Directory to persist cache. If None, cache is memory-only.
            ttl: Time-to-live for cache entries in seconds. If None, no expiration.
        """
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.ttl = ttl
        
        # Internal cache storage
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}  # key -> (embedding, timestamp)
        self._access_order: Dict[str, int] = {}  # key -> access_counter
        self._access_counter = 0
        self._lock = threading.RLock()
        
        # Setup persistence
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_persistent_cache()
        
        logger.info(f"EmbeddingCache initialized: max_size={max_size}, cache_dir={cache_dir}, ttl={ttl}")
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash key for a query."""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - timestamp > self.ttl
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self._cache:
            return
        
        # Find the key with the smallest access counter
        lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_order[lru_key]
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        if self.ttl is None:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            del self._access_order[key]
            logger.debug(f"Removed expired cache entry: {key}")
    
    def get(self, query: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            query: Query text.
            
        Returns:
            Cached embedding or None if not found.
        """
        key = self._hash_query(query)
        
        with self._lock:
            if key not in self._cache:
                logger.debug(f"Cache miss for query: {query[:50]}...")
                return None
            
            embedding, timestamp = self._cache[key]
            
            # Check if expired
            if self._is_expired(timestamp):
                del self._cache[key]
                del self._access_order[key]
                logger.debug(f"Cache entry expired for query: {query[:50]}...")
                return None
            
            # Update access order
            self._access_counter += 1
            self._access_order[key] = self._access_counter
            
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return embedding.copy()
    
    def put(self, query: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            query: Query text.
            embedding: Query embedding.
        """
        key = self._hash_query(query)
        
        with self._lock:
            # Clean up expired entries
            self._cleanup_expired()
            
            # Evict LRU if at capacity and adding new entry
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store the embedding
            timestamp = time.time()
            self._cache[key] = (embedding.copy(), timestamp)
            
            # Update access order
            self._access_counter += 1
            self._access_order[key] = self._access_counter
            
            logger.debug(f"Cached embedding for query: {query[:50]}...")
            
            # Persist if enabled
            if self.cache_dir:
                self._persist_entry(key, embedding, timestamp)
    
    def _persist_entry(self, key: str, embedding: np.ndarray, timestamp: float) -> None:
        """Persist a cache entry to disk."""
        try:
            entry_path = os.path.join(self.cache_dir, f"{key}.npz")
            np.savez_compressed(entry_path, embedding=embedding, timestamp=timestamp)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {key}: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load cache entries from disk."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
        
        loaded_count = 0
        
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.npz'):
                    continue
                
                key = filename[:-4]  # Remove .npz extension
                entry_path = os.path.join(self.cache_dir, filename)
                
                try:
                    data = np.load(entry_path)
                    embedding = data['embedding']
                    timestamp = float(data['timestamp'])
                    
                    # Check if expired
                    if not self._is_expired(timestamp):
                        self._cache[key] = (embedding, timestamp)
                        self._access_order[key] = 0  # Will be updated on access
                        loaded_count += 1
                    else:
                        # Remove expired file
                        os.remove(entry_path)
                
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
                    # Remove corrupted file
                    try:
                        os.remove(entry_path)
                    except:
                        pass
        
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} cache entries from disk")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_counter = 0
            
            # Clear persistent cache
            if self.cache_dir and os.path.exists(self.cache_dir):
                try:
                    for filename in os.listdir(self.cache_dir):
                        if filename.endswith('.npz'):
                            os.remove(os.path.join(self.cache_dir, filename))
                except Exception as e:
                    logger.warning(f"Failed to clear persistent cache: {e}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
                "cache_dir": self.cache_dir,
                "ttl": self.ttl,
            }


# Global embedding cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(max_size: int = 1000, cache_dir: Optional[str] = None, ttl: Optional[int] = None) -> EmbeddingCache:
    """
    Get or create the global embedding cache instance.
    
    Args:
        max_size: Maximum number of embeddings to cache.
        cache_dir: Directory to persist cache.
        ttl: Time-to-live for cache entries in seconds.
        
    Returns:
        The global embedding cache instance.
    """
    global _embedding_cache
    
    if _embedding_cache is None:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/rag_csd/embeddings")
        _embedding_cache = EmbeddingCache(max_size, cache_dir, ttl)
    
    return _embedding_cache