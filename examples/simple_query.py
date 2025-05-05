#!/usr/bin/env python
"""
Simple example of using the RAG-CSD system with a single query.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict

import yaml

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.augmentation.augmentor import Augmentor
from rag_csd.utils.logger import setup_logger


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run a single query through the RAG-CSD system.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", help="Path to the configuration file."
    )
    parser.add_argument("--vector-db", "-v", type=str, required=True, help="Path to the vector database.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Query to process.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of documents to retrieve.")
    parser.add_argument("--output", "-o", type=str, help="Path to save the output JSON.")
    parser.add_argument("--log-level", "-l", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--use-csd", action="store_true", help="Use CSD simulation.")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override CSD setting if specified
    if args.use_csd:
        config["csd"]["enabled"] = True
    
    # Initialize components
    start_time = time.time()
    
    logger.info("Initializing components...")
    
    # Initialize encoder
    encoder = Encoder(config)
    
    # Initialize vector store
    vector_store = VectorStore(args.vector_db, config)
    
    # Initialize augmentor
    augmentor = Augmentor(config)
    
    init_time = time.time() - start_time
    logger.info(f"Components initialized in {init_time:.2f} seconds")
    
    # Process query
    logger.info(f"Processing query: {args.query}")
    
    # Step 1: Encode the query
    logger.info("Encoding query...")
    encode_start = time.time()
    query_embedding = encoder.encode(args.query)
    encode_time = time.time() - encode_start
    logger.info(f"Query encoded in {encode_time:.2f} seconds")
    
    # Step 2: Retrieve similar documents
    logger.info(f"Retrieving top-{args.top_k} documents...")
    retrieve_start = time.time()
    retrieved_docs = vector_store.search(query_embedding, top_k=args.top_k)
    retrieve_time = time.time() - retrieve_start
    logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieve_time:.2f} seconds")
    
    # Log retrieved documents
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Document {i+1}: score={doc['score']:.4f}, metadata={doc['metadata']['doc_id']}")
    
    # Step 3: Augment the query
    logger.info("Augmenting query...")
    augment_start = time.time()
    augmented_query = augmentor.augment(args.query, retrieved_docs)
    augment_time = time.time() - augment_start
    logger.info(f"Query augmented in {augment_time:.2f} seconds")
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Prepare output
    output = {
        "original_query": args.query,
        "query_embedding": query_embedding.tolist(),
        "retrieved_docs": retrieved_docs,
        "augmented_query": augmented_query,
        "timings": {
            "init": init_time,
            "encode": encode_time,
            "retrieve": retrieve_time,
            "augment": augment_time,
            "total": total_time,
        },
    }
    
    # Save output if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Output saved to {args.output}")
    
    # Print summary
    print("\n===== RAG-CSD Query Processing Summary =====")
    print(f"Query: {args.query}")
    print(f"Retrieved {len(retrieved_docs)} documents")
    print("\nTop documents:")
    for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
        print(f"  {i+1}. Score: {doc['score']:.4f}, Doc: {doc['metadata']['doc_id']}")
    print("\nAugmented Query Preview:")
    preview_length = min(500, len(augmented_query))
    print(f"{augmented_query[:preview_length]}{'...' if len(augmented_query) > preview_length else ''}")
    print("\nTimings:")
    print(f"  Initialization: {init_time:.2f}s")
    print(f"  Encode:         {encode_time:.2f}s")
    print(f"  Retrieve:       {retrieve_time:.2f}s")
    print(f"  Augment:        {augment_time:.2f}s")
    print(f"  Total:          {total_time:.2f}s")
    print("============================================\n")


if __name__ == "__main__":
    main()