#!/usr/bin/env python
"""
Script to run batch queries against the RAG-CSD system.
Loads queries from a file and processes them with customizable parameters.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union

import yaml
from tqdm import tqdm

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


def load_queries(query_file: str) -> List[Dict]:
    """
    Load queries from a file. 
    Supports JSON, CSV, or plain text format.
    
    Args:
        query_file: Path to the query file.
        
    Returns:
        List of query dictionaries.
    """
    queries = []
    file_ext = os.path.splitext(query_file)[1].lower()
    
    try:
        if file_ext == '.json':
            # Load from JSON file
            with open(query_file, 'r') as f:
                data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, str):
                            queries.append({
                                "id": f"q{i+1}",
                                "text": item
                            })
                        elif isinstance(item, dict) and "text" in item:
                            if "id" not in item:
                                item["id"] = f"q{i+1}"
                            queries.append(item)
                
                elif isinstance(data, dict) and "queries" in data:
                    for i, item in enumerate(data["queries"]):
                        if isinstance(item, str):
                            queries.append({
                                "id": f"q{i+1}",
                                "text": item
                            })
                        elif isinstance(item, dict) and "text" in item:
                            if "id" not in item:
                                item["id"] = f"q{i+1}"
                            queries.append(item)
        
        elif file_ext == '.csv':
            # Load from CSV file
            with open(query_file, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                
                if headers and 'text' in headers:
                    # CSV with headers
                    text_idx = headers.index('text')
                    id_idx = headers.index('id') if 'id' in headers else -1
                    
                    for i, row in enumerate(reader):
                        if len(row) > text_idx:
                            query_id = row[id_idx] if id_idx >= 0 and id_idx < len(row) else f"q{i+1}"
                            queries.append({
                                "id": query_id,
                                "text": row[text_idx]
                            })
                else:
                    # CSV without headers or headers don't include 'text'
                    # Assume first column is the query text
                    for i, row in enumerate(reader):
                        if row:  # Skip empty rows
                            queries.append({
                                "id": f"q{i+1}",
                                "text": row[0]
                            })
        
        else:
            # Assume it's a plain text file with one query per line
            with open(query_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:  # Skip empty lines
                        queries.append({
                            "id": f"q{i+1}",
                            "text": line
                        })
    
    except Exception as e:
        logging.error(f"Error loading queries from {query_file}: {e}")
        raise
    
    return queries


def process_query(
    query: Dict,
    encoder: Encoder,
    vector_store: VectorStore,
    augmentor: Augmentor,
    top_k: int,
) -> Dict:
    """
    Process a single query through the RAG-CSD pipeline.
    
    Args:
        query: Query dictionary with 'id' and 'text' keys.
        encoder: Encoder instance.
        vector_store: VectorStore instance.
        augmentor: Augmentor instance.
        top_k: Number of documents to retrieve.
        
    Returns:
        Dictionary with query results.
    """
    query_id = query["id"]
    query_text = query["text"]
    
    result = {
        "query_id": query_id,
        "query_text": query_text,
        "top_k": top_k,
    }
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Encode the query
        encode_start = time.time()
        query_embedding = encoder.encode(query_text)
        encode_time = time.time() - encode_start
        
        # Step 2: Retrieve similar documents
        retrieve_start = time.time()
        retrieved_docs = vector_store.search(query_embedding, top_k=top_k)
        retrieve_time = time.time() - retrieve_start
        
        # Step 3: Augment the query
        augment_start = time.time()
        augmented_query = augmentor.augment(query_text, retrieved_docs)
        augment_time = time.time() - augment_start
        
        # Total processing time
        total_time = time.time() - start_time
        
        # Record retrieved documents and timings
        result["retrieved_docs"] = retrieved_docs
        result["augmented_query"] = augmented_query
        result["timings"] = {
            "encode": encode_time,
            "retrieve": retrieve_time,
            "augment": augment_time,
            "total": total_time,
        }
        result["success"] = True
    
    except Exception as e:
        logging.error(f"Error processing query {query_id}: {e}")
        result["error"] = str(e)
        result["success"] = False
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run batch queries with RAG-CSD.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--vector-db", "-v", type=str, required=True, 
        help="Path to the vector database."
    )
    parser.add_argument(
        "--queries", "-q", type=str, required=True, 
        help="Path to the query file (JSON, CSV, or text)."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, 
        help="Path to save the results (JSON)."
    )
    parser.add_argument(
        "--top-k", "-k", type=int, 
        help="Number of documents to retrieve. Overrides config value."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, 
        help="Batch size for embedding. Overrides config value."
    )
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO", 
        help="Logging level."
    )
    parser.add_argument(
        "--use-csd", action="store_true", 
        help="Use CSD simulation."
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config values if specified
        if args.top_k:
            config["retrieval"]["top_k"] = args.top_k
            logger.info(f"Overriding top_k to {args.top_k}")
        
        if args.batch_size:
            config["embedding"]["batch_size"] = args.batch_size
            logger.info(f"Overriding batch size to {args.batch_size}")
        
        if args.use_csd:
            config["csd"]["enabled"] = True
            logger.info("Enabling CSD simulation")
        
        # Get the actual top_k value to use
        top_k = config["retrieval"]["top_k"]
        
        # Load queries
        logger.info(f"Loading queries from {args.queries}")
        queries = load_queries(args.queries)
        logger.info(f"Loaded {len(queries)} queries")
        
        # Initialize components
        logger.info("Initializing RAG-CSD components")
        encoder = Encoder(config)
        vector_store = VectorStore(args.vector_db, config)
        augmentor = Augmentor(config)
        
        # Process queries
        logger.info(f"Processing {len(queries)} queries with top_k={top_k}")
        results = []
        
        for query in tqdm(queries, desc="Processing queries"):
            result = process_query(query, encoder, vector_store, augmentor, top_k)
            results.append(result)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                "config": {
                    "top_k": top_k,
                    "use_csd": config["csd"]["enabled"],
                    "batch_size": config["embedding"]["batch_size"],
                },
                "queries": results,
            }, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        success_count = sum(1 for r in results if r.get("success", False))
        logger.info(f"Processed {len(queries)} queries: {success_count} succeeded, "
                   f"{len(queries) - success_count} failed")
    
    except Exception as e:
        logger.error(f"Error in batch query processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
