"""
Augmentation module for RAG-CSD.
This module handles the augmentation of queries with retrieved documents.
"""

import logging
from typing import Dict, List, Optional, Set, Union

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class Augmentor:
    """Augmentor class for enhancing queries with retrieved documents."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the augmentor with configuration.
        
        Args:
            config: Configuration dictionary.
                If None, default configuration will be used.
        """
        self.config = config or {}
        self.strategy = self.config.get("augmentation", {}).get("strategy", "concatenate")
        self.max_tokens_per_chunk = self.config.get("augmentation", {}).get(
            "max_tokens_per_chunk", 256
        )
        self.include_metadata = self.config.get("augmentation", {}).get("include_metadata", True)
        self.template = self.config.get("augmentation", {}).get(
            "template", "Query: {query}\n\nContext: {context}"
        )
        self.separator = self.config.get("augmentation", {}).get("separator", "\n\n")
        self.deduplicate = self.config.get("augmentation", {}).get("deduplicate", True)
    
    def augment(
        self, query: str, retrieved_docs: List[Dict], strategy: Optional[str] = None
    ) -> str:
        """
        Augment the query with retrieved documents.
        
        Args:
            query: The original query.
            retrieved_docs: List of retrieved documents.
            strategy: Augmentation strategy. If None, uses the default.
            
        Returns:
            Augmented query string.
        """
        if not retrieved_docs:
            logger.warning("No documents retrieved for augmentation")
            return query
        
        # Use provided strategy or default
        strategy = strategy or self.strategy
        
        if strategy == "concatenate":
            return self._augment_concatenate(query, retrieved_docs)
        elif strategy == "template":
            return self._augment_template(query, retrieved_docs)
        elif strategy == "weighted":
            return self._augment_weighted(query, retrieved_docs)
        else:
            logger.warning(f"Unknown augmentation strategy: {strategy}, using concatenate")
            return self._augment_concatenate(query, retrieved_docs)
    
    def _augment_concatenate(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Augment the query by concatenating retrieved documents.
        
        Args:
            query: The original query.
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Augmented query string.
        """
        # Extract chunks from retrieved documents
        chunks = [doc["chunk"] for doc in retrieved_docs]
        
        # Deduplicate if needed
        if self.deduplicate:
            chunks = self._deduplicate_chunks(chunks)
        
        # Truncate chunks if needed
        chunks = self._truncate_chunks(chunks)
        
        # Join chunks with separator
        context = self.separator.join(chunks)
        
        # Combine query and context
        augmented_query = f"{query}{self.separator}{context}"
        
        return augmented_query
    
    def _augment_template(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Augment the query using a template.
        
        Args:
            query: The original query.
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Augmented query string.
        """
        # Extract chunks from retrieved documents
        chunks = [doc["chunk"] for doc in retrieved_docs]
        
        # Deduplicate if needed
        if self.deduplicate:
            chunks = self._deduplicate_chunks(chunks)
        
        # Truncate chunks if needed
        chunks = self._truncate_chunks(chunks)
        
        # Join chunks with separator
        context = self.separator.join(chunks)
        
        # Fill the template
        augmented_query = self.template.format(query=query, context=context)
        
        return augmented_query
    
    def _augment_weighted(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Augment the query with retrieved documents, weighted by their similarity score.
        
        Args:
            query: The original query.
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Augmented query string.
        """
        # Sort documents by score in descending order
        sorted_docs = sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)
        
        # Extract chunks and weights
        chunks = [doc["chunk"] for doc in sorted_docs]
        weights = [doc["score"] for doc in sorted_docs]
        
        # Normalize weights
        max_weight = max(weights)
        if max_weight > 0:
            weights = [w / max_weight for w in weights]
        
        # Deduplicate if needed
        if self.deduplicate:
            # More complex deduplication with weights
            unique_chunks = []
            unique_weights = []
            seen_chunks = set()
            
            for chunk, weight in zip(chunks, weights):
                if chunk not in seen_chunks:
                    unique_chunks.append(chunk)
                    unique_weights.append(weight)
                    seen_chunks.add(chunk)
            
            chunks = unique_chunks
            weights = unique_weights
        
        # Truncate chunks if needed, considering weights
        chunks, weights = self._truncate_weighted_chunks(chunks, weights)
        
        # Combine context with weights
        context_parts = []
        for i, (chunk, weight) in enumerate(zip(chunks, weights)):
            # Add relevance indicator based on weight
            if self.include_metadata:
                relevance = f"[Relevance: {weight:.2f}]"
                context_parts.append(f"{relevance} {chunk}")
            else:
                context_parts.append(chunk)
        
        context = self.separator.join(context_parts)
        
        # Combine query and context
        augmented_query = f"{query}{self.separator}{context}"
        
        return augmented_query
    
    def _deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Remove duplicate chunks while preserving order.
        
        Args:
            chunks: List of text chunks.
            
        Returns:
            Deduplicated list of chunks.
        """
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)
        
        return unique_chunks
    
    def _truncate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Truncate chunks to the maximum token limit.
        
        Args:
            chunks: List of text chunks.
            
        Returns:
            Truncated list of chunks.
        """
        # Simple whitespace tokenization for simulation
        # In a real implementation, use a proper tokenizer
        truncated_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            tokens = chunk.split()
            token_count = len(tokens)
            
            if total_tokens + token_count <= self.max_tokens_per_chunk:
                truncated_chunks.append(chunk)
                total_tokens += token_count
            else:
                # Take as many tokens as possible
                remaining = self.max_tokens_per_chunk - total_tokens
                if remaining > 0:
                    partial_chunk = " ".join(tokens[:remaining])
                    truncated_chunks.append(partial_chunk)
                break
        
        return truncated_chunks
    
    def _truncate_weighted_chunks(
        self, chunks: List[str], weights: List[float]
    ) -> tuple[List[str], List[float]]:
        """
        Truncate chunks based on weights and token limit.
        
        Args:
            chunks: List of text chunks.
            weights: List of weights for each chunk.
            
        Returns:
            Tuple of (truncated chunks, truncated weights).
        """
        # Sort chunks by weight
        sorted_items = sorted(zip(chunks, weights), key=lambda x: x[1], reverse=True)
        sorted_chunks, sorted_weights = zip(*sorted_items) if sorted_items else ([], [])
        
        # Truncate based on token limit
        truncated_chunks = []
        truncated_weights = []
        total_tokens = 0
        
        for chunk, weight in zip(sorted_chunks, sorted_weights):
            tokens = chunk.split()
            token_count = len(tokens)
            
            if total_tokens + token_count <= self.max_tokens_per_chunk:
                truncated_chunks.append(chunk)
                truncated_weights.append(weight)
                total_tokens += token_count
            else:
                # Take as many tokens as possible
                remaining = self.max_tokens_per_chunk - total_tokens
                if remaining > 0:
                    partial_chunk = " ".join(tokens[:remaining])
                    truncated_chunks.append(partial_chunk)
                    truncated_weights.append(weight)
                break
        
        return truncated_chunks, truncated_weights