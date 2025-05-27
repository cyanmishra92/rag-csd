"""
Text preprocessing module for RAG-CSD.
This module provides optimized text processing utilities.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Set

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """Optimized text processor for query and document preprocessing."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the text processor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        data_config = self.config.get("data", {})
        
        # Text processing settings
        self.lowercase = data_config.get("lowercase", True)
        self.normalize_unicode = data_config.get("normalize_unicode", True)
        self.remove_stopwords = data_config.get("remove_stopwords", False)
        self.min_token_length = data_config.get("min_token_length", 1)
        self.max_token_length = data_config.get("max_token_length", 100)
        
        # Precompile regex patterns for better performance
        self._compile_patterns()
        
        # Load stopwords if needed
        if self.remove_stopwords:
            self.stopwords = self._load_stopwords()
        else:
            self.stopwords = set()
        
        logger.info(f"TextProcessor initialized: lowercase={self.lowercase}, "
                   f"normalize_unicode={self.normalize_unicode}, "
                   f"remove_stopwords={self.remove_stopwords}")
    
    def _compile_patterns(self) -> None:
        """Precompile regex patterns for better performance."""
        # Pattern to normalize whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Pattern to remove special characters (keep alphanumeric and basic punctuation)
        self.special_char_pattern = re.compile(r'[^\w\s\-\.\,\!\?\:\;]')
        
        # Pattern to identify sentence boundaries
        self.sentence_boundary_pattern = re.compile(r'[.!?]+\s+')
        
        # Pattern to split on word boundaries
        self.word_boundary_pattern = re.compile(r'\b\w+\b')
    
    def _load_stopwords(self) -> Set[str]:
        """Load a basic set of English stopwords."""
        # Basic English stopwords - can be extended
        basic_stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if', 'up',
            'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
            'make', 'like', 'into', 'him', 'could', 'two', 'more', 'very', 'after',
            'first', 'been', 'than', 'its', 'who', 'did', 'get', 'may', 'way', 'use'
        }
        return basic_stopwords
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove special characters (but keep basic punctuation)
        text = self.special_char_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str, remove_stopwords: Optional[bool] = None) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize.
            remove_stopwords: Whether to remove stopwords. If None, uses config setting.
            
        Returns:
            List of tokens.
        """
        if not text:
            return []
        
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Extract words using regex
        tokens = self.word_boundary_pattern.findall(cleaned_text)
        
        # Filter tokens by length
        tokens = [
            token for token in tokens 
            if self.min_token_length <= len(token) <= self.max_token_length
        ]
        
        # Remove stopwords if configured
        should_remove_stopwords = remove_stopwords if remove_stopwords is not None else self.remove_stopwords
        if should_remove_stopwords and self.stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]
        
        return tokens
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text to split.
            
        Returns:
            List of sentences.
        """
        if not text:
            return []
        
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Split on sentence boundaries
        sentences = self.sentence_boundary_pattern.split(cleaned_text)
        
        # Clean up sentences and filter empty ones
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        
        return sentences
    
    def chunk_text_optimized(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int = 0,
        min_chunk_size: int = 50,
        prefer_sentence_boundaries: bool = True
    ) -> List[str]:
        """
        Optimized text chunking that respects sentence boundaries when possible.
        
        Args:
            text: Input text to chunk.
            chunk_size: Target size for each chunk (in characters).
            chunk_overlap: Number of characters to overlap between chunks.
            min_chunk_size: Minimum chunk size.
            prefer_sentence_boundaries: Whether to prefer splitting at sentence boundaries.
            
        Returns:
            List of text chunks.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        
        if prefer_sentence_boundaries:
            # Split into sentences first
            sentences = self.split_sentences(text)
            
            current_chunk = ""
            
            for sentence in sentences:
                # If adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) + 1 > chunk_size:
                    # Save current chunk if it meets minimum size
                    if len(current_chunk) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                        
                        # Start new chunk with overlap
                        if chunk_overlap > 0 and chunks:
                            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                            current_chunk = overlap_text + " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        # Current chunk is too small, just add the sentence
                        current_chunk += " " + sentence if current_chunk else sentence
                else:
                    # Add sentence to current chunk
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
        
        else:
            # Fall back to character-based chunking
            chunks = self._chunk_by_characters(text, chunk_size, chunk_overlap, min_chunk_size)
        
        return chunks
    
    def _chunk_by_characters(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int,
        min_chunk_size: int
    ) -> List[str]:
        """Character-based text chunking fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to find a good breaking point (space, punctuation)
            if end < len(text):
                # Look for a space or punctuation in the last 20% of the chunk
                search_start = max(start + int(chunk_size * 0.8), start + min_chunk_size)
                
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in ' \t\n.!?;:':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            
            # Move start position considering overlap
            start = max(start + chunk_size - chunk_overlap, end - chunk_overlap)
        
        return chunks
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a query for optimal encoding.
        
        Args:
            query: Input query text.
            
        Returns:
            Preprocessed query.
        """
        # Clean and normalize the query
        cleaned_query = self.clean_text(query)
        
        # For queries, we typically don't remove stopwords as they can be important for context
        # But we can do additional query-specific preprocessing here
        
        return cleaned_query
    
    def preprocess_document(self, document: str) -> str:
        """
        Preprocess a document for optimal chunking and indexing.
        
        Args:
            document: Input document text.
            
        Returns:
            Preprocessed document.
        """
        # Clean and normalize the document
        cleaned_doc = self.clean_text(document)
        
        return cleaned_doc
    
    def get_stats(self) -> Dict:
        """Get text processor statistics and configuration."""
        return {
            "lowercase": self.lowercase,
            "normalize_unicode": self.normalize_unicode,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "stopwords_count": len(self.stopwords),
        }


# Global text processor instance
_text_processor: Optional[TextProcessor] = None


def get_text_processor(config: Optional[Dict] = None) -> TextProcessor:
    """
    Get or create the global text processor instance.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        The global text processor instance.
    """
    global _text_processor
    
    if _text_processor is None:
        _text_processor = TextProcessor(config)
    
    return _text_processor