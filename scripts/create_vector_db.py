#!/usr/bin/env python
"""
Script to create a vector database from text documents.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.utils.logger import setup_logger


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, min_chunk_size: int
) -> List[str]:
    """Split text into chunks with specified size and overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to find a sensible breaking point (newline or period)
        if end < len(text):
            # Look for a newline
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + min_chunk_size:
                end = newline_pos + 1
            else:
                # Look for a period followed by space
                period_pos = text.rfind(". ", start, end)
                if period_pos > start + min_chunk_size:
                    end = period_pos + 2

        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move the start position for the next chunk, considering overlap
        start = max(start + chunk_size - chunk_overlap, end - chunk_overlap)
    
    return chunks


def process_documents(
    input_dir: str,
    config: Dict,
    file_ext: str = "*.txt",
) -> Tuple[List[str], List[Dict]]:
    """Process all documents in the input directory and generate chunks with metadata."""
    file_pattern = os.path.join(input_dir, file_ext)
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        raise ValueError(f"No {file_ext} files found in {input_dir}")
    
    chunk_size = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]
    min_chunk_size = config["data"]["min_chunk_size"]
    max_chunks_per_doc = config["data"]["max_chunks_per_doc"]
    
    all_chunks = []
    all_metadata = []
    
    for file_path in tqdm(file_paths, desc="Processing documents"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Generate chunks
            chunks = chunk_text(text, chunk_size, chunk_overlap, min_chunk_size)
            
            # Limit the number of chunks per document if needed
            if max_chunks_per_doc > 0 and len(chunks) > max_chunks_per_doc:
                logging.warning(
                    f"Document {file_path} has {len(chunks)} chunks, "
                    f"limiting to {max_chunks_per_doc}"
                )
                chunks = chunks[:max_chunks_per_doc]
            
            # Create metadata for each chunk
            doc_id = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "file_path": file_path,
                    "position": i,
                    "total_chunks": len(chunks),
                })
        
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    
    return all_chunks, all_metadata


def generate_embeddings(
    chunks: List[str], config: Dict
) -> np.ndarray:
    """Generate embeddings for all chunks using the specified model."""
    model_name = config["embedding"]["model"]
    batch_size = config["embedding"]["batch_size"]
    normalize = config["embedding"]["normalize"]
    
    logging.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logging.info(f"Generating embeddings for {len(chunks)} chunks")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    
    return embeddings


def save_vector_db(
    output_dir: str,
    chunks: List[str],
    metadata: List[Dict],
    embeddings: np.ndarray,
    config: Dict,
) -> None:
    """Save the vector database to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks
    chunks_path = os.path.join(output_dir, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    
    # Save configuration used
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    logging.info(f"Vector database saved to {output_dir}")
    logging.info(f"Total chunks: {len(chunks)}")
    logging.info(f"Embedding dimensions: {embeddings.shape}")


def create_vector_db(
    input_dir: str,
    output_dir: str,
    config_path: str,
    file_ext: str = "*.txt",
) -> None:
    """Create a vector database from text documents."""
    # Load configuration
    config = load_config(config_path)
    
    # Process documents
    chunks, metadata = process_documents(input_dir, config, file_ext)
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, config)
    
    # Save vector database
    save_vector_db(output_dir, chunks, metadata, embeddings, config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create a vector database from text documents.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing text documents.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for the vector database.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--file-ext", "-e", type=str, default="*.txt", help="File extension pattern to process.")
    parser.add_argument("--log-level", "-l", type=str, default="INFO", help="Logging level.")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    
    start_time = time.time()
    
    try:
        create_vector_db(args.input, args.output, args.config, args.file_ext)
        elapsed_time = time.time() - start_time
        logging.info(f"Vector database creation completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()