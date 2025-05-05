#!/usr/bin/env python
"""
Script to process PDF files and create a vector database.
Supports both single PDFs and a list of PDFs from a file.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.utils.logger import setup_logger
from rag_csd.embedding.encoder import Encoder

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 is required. Install it with: pip install PyPDF2")
    sys.exit(1)


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_pdf_paths(input_path: str) -> List[str]:
    """
    Get list of PDF paths from a file or a directory.
    
    Args:
        input_path: Path to a PDF file, directory, or a text file with PDF paths.
        
    Returns:
        List of PDF file paths.
    """
    pdf_paths = []
    
    if os.path.isfile(input_path):
        # Check if it's a PDF file
        if input_path.lower().endswith('.pdf'):
            pdf_paths.append(input_path)
        # Check if it's a file containing PDF paths
        else:
            with open(input_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path and os.path.exists(path) and path.lower().endswith('.pdf'):
                        pdf_paths.append(path)
    
    elif os.path.isdir(input_path):
        # It's a directory, find all PDFs
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
    
    return pdf_paths


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Extracted text.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n\n"
    
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    
    return text


def chunk_text(
    text: str, 
    chunk_size: int, 
    chunk_overlap: int, 
    min_chunk_size: int
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


def process_pdfs(
    pdf_paths: List[str],
    config: Dict,
) -> Tuple[List[str], List[Dict]]:
    """
    Process PDF files and generate chunks with metadata.
    
    Args:
        pdf_paths: List of PDF file paths.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (chunks, metadata).
    """
    chunk_size = config["data"]["chunk_size"]
    chunk_overlap = config["data"]["chunk_overlap"]
    min_chunk_size = config["data"]["min_chunk_size"]
    max_chunks_per_doc = config["data"]["max_chunks_per_doc"]
    
    all_chunks = []
    all_metadata = []
    
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            # Extract text from PDF
            logging.info(f"Extracting text from {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                logging.warning(f"No text extracted from {pdf_path}")
                continue
            
            # Generate chunks
            chunks = chunk_text(text, chunk_size, chunk_overlap, min_chunk_size)
            
            # Limit the number of chunks per document if needed
            if max_chunks_per_doc > 0 and len(chunks) > max_chunks_per_doc:
                logging.warning(
                    f"Document {pdf_path} has {len(chunks)} chunks, "
                    f"limiting to {max_chunks_per_doc}"
                )
                chunks = chunks[:max_chunks_per_doc]
            
            # Create metadata for each chunk
            doc_id = os.path.basename(pdf_path)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "file_path": pdf_path,
                    "position": i,
                    "total_chunks": len(chunks),
                })
            
            logging.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
    
    return all_chunks, all_metadata


def generate_embeddings(
    chunks: List[str], config: Dict
) -> np.ndarray:
    """Generate embeddings for all chunks using the specified model."""
    encoder = Encoder(config)
    
    # Get batch size from config
    batch_size = config["embedding"].get("batch_size", 32)
    
    logging.info(f"Generating embeddings for {len(chunks)} chunks with batch size {batch_size}")
    
    # Process in batches to avoid memory issues with large corpora
    all_embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Batch encoding"):
        batch = chunks[i:i+batch_size]
        batch_embeddings = encoder.encode(batch)
        
        # Handle both single embedding and batch of embeddings
        if len(batch) == 1:
            all_embeddings.append(batch_embeddings.reshape(1, -1))
        else:
            all_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process PDF files and create a vector database.")
    parser.add_argument(
        "--input", "-i", type=str, required=True, 
        help="Input PDF file, directory, or text file with PDF paths."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, 
        help="Output directory for the vector database."
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True, 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO", 
        help="Logging level."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int,
        help="Batch size for embedding generation. Overrides config value."
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override batch size if specified
        if args.batch_size:
            config["embedding"]["batch_size"] = args.batch_size
            logging.info(f"Overriding batch size to {args.batch_size}")
        
        # Get PDF paths
        pdf_paths = get_pdf_paths(args.input)
        if not pdf_paths:
            logging.error(f"No PDF files found in {args.input}")
            sys.exit(1)
        
        logging.info(f"Found {len(pdf_paths)} PDF files to process")
        
        # Process PDFs
        chunks, metadata = process_pdfs(pdf_paths, config)
        if not chunks:
            logging.error("No text chunks extracted from PDFs")
            sys.exit(1)
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, config)
        
        # Save vector database
        save_vector_db(args.output, chunks, metadata, embeddings, config)
        
        elapsed_time = time.time() - start_time
        logging.info(f"PDF processing and vector database creation completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logging.error(f"Error processing PDFs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
