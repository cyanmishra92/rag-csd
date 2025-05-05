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