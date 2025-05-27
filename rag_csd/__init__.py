"""
RAG-CSD: Retrieval-Augmented Generation with Computational Storage Devices.

This package implements the Embedding (E), Retrieval (R), and Augmentation (A)
components of RAG, with a focus on offloading vector operations to Computational
Storage Devices (CSDs).
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.augmentation.augmentor import Augmentor
from rag_csd.csd.simulator import CSDSimulator
from rag_csd.utils.model_cache import model_cache


def warm_up_system(config=None):
    """
    Warm up the RAG-CSD system by pre-loading models.
    
    Args:
        config: Configuration dictionary. If None, uses default.
    """
    if config is None:
        config = {}
    
    model_name = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    device = config.get("general", {}).get("device", "auto")
    
    # Warm up the embedding model
    model_cache.warm_up_model(model_name, device)