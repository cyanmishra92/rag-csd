#!/usr/bin/env python
"""
Test script for batch processing capabilities in RAG-CSD.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import yaml

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.utils.logger import setup_logger


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_queries(queries_path: str) -> List[Dict]:
    """Load test queries from a JSON file."""
    with open(queries_path, "r") as f:
        queries = json.load(f)
    return queries


def test_batch_processing(
    config: Dict,
    vector_db_path: str,
    queries: List[str],
    batch_size: int = 32,
) -> Dict:
    """
    Test batch processing performance.
    
    Args:
        config: Configuration dictionary.
        vector_db_path: Path to the vector database.
        queries: List of query texts.
        batch_size: Batch size for processing.
        
    Returns:
        Dictionary with performance results.
    """
    logger = logging.getLogger(__name__)
    
    # Disable CSD simulation for cleaner testing
    config["csd"]["enabled"] = False
    
    # Initialize components
    logger.info("Initializing components...")
    start_time = time.time()
    
    encoder = Encoder(config)
    vector_store = VectorStore(vector_db_path, config)
    
    init_time = time.time() - start_time
    logger.info(f"Components initialized in {init_time:.2f}s")
    
    # Test 1: Single query processing (one at a time)
    logger.info("Testing single query processing...")
    single_start = time.time()
    single_results = []
    
    for query in queries:
        embedding = encoder.encode(query)
        results = vector_store.search(embedding, top_k=5)
        single_results.append(results)
    
    single_time = time.time() - single_start
    logger.info(f"Single processing: {single_time:.3f}s for {len(queries)} queries")
    
    # Test 2: Batch encoding + individual search
    logger.info("Testing batch encoding + individual search...")
    batch_encode_start = time.time()
    
    # Batch encode all queries
    batch_embeddings = encoder.encode_batch(queries, batch_size=batch_size)
    
    # Individual searches
    batch_encode_results = []
    for i, embedding in enumerate(batch_embeddings):
        results = vector_store.search(embedding, top_k=5)
        batch_encode_results.append(results)
    
    batch_encode_time = time.time() - batch_encode_start
    logger.info(f"Batch encoding + individual search: {batch_encode_time:.3f}s")
    
    # Test 3: Batch encoding + batch search
    logger.info("Testing full batch processing...")
    full_batch_start = time.time()
    
    # Batch encode all queries
    batch_embeddings = encoder.encode_batch(queries, batch_size=batch_size)
    
    # Batch search
    batch_results = vector_store.search_batch(batch_embeddings, top_k=5)
    
    full_batch_time = time.time() - full_batch_start
    logger.info(f"Full batch processing: {full_batch_time:.3f}s")
    
    # Calculate speedups
    encode_speedup = single_time / batch_encode_time
    full_speedup = single_time / full_batch_time
    
    # Verify results are similar (check first query)
    def compare_results(results1, results2):
        """Compare if two result sets are similar."""
        if len(results1) != len(results2):
            return False
        for r1, r2 in zip(results1[:3], results2[:3]):  # Check top 3
            if abs(r1["score"] - r2["score"]) > 0.001:
                return False
        return True
    
    results_match = (
        compare_results(single_results[0], batch_encode_results[0]) and
        compare_results(single_results[0], batch_results[0])
    )
    
    return {
        "num_queries": len(queries),
        "batch_size": batch_size,
        "init_time": init_time,
        "single_time": single_time,
        "batch_encode_time": batch_encode_time,
        "full_batch_time": full_batch_time,
        "encode_speedup": encode_speedup,
        "full_speedup": full_speedup,
        "results_match": results_match,
        "throughput": {
            "single_qps": len(queries) / single_time,
            "batch_encode_qps": len(queries) / batch_encode_time,
            "full_batch_qps": len(queries) / full_batch_time,
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test batch processing capabilities.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--vector-db", "-v", type=str, default="data/vectors_large", help="Path to the vector database."
    )
    parser.add_argument(
        "--queries", "-q", type=str, default="data/test_queries.json", help="Path to test queries JSON."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for processing."
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=3, help="Number of times to repeat each query for testing."
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to save benchmark results JSON."
    )
    parser.add_argument(
        "--log-level", "-l", type=str, default="INFO", help="Logging level."
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Load test queries
    query_data = load_queries(args.queries)
    query_texts = [q["text"] for q in query_data]
    
    # Repeat queries for more robust testing
    test_queries = query_texts * args.repeat
    
    logger.info(f"Testing with {len(test_queries)} queries (original: {len(query_texts)}, repeat: {args.repeat})")
    
    # Run batch processing test
    results = test_batch_processing(
        config, args.vector_db, test_queries, args.batch_size
    )
    
    # Print results
    print("\n===== Batch Processing Test Results =====")
    print(f"Queries processed: {results['num_queries']}")
    print(f"Batch size: {results['batch_size']}")
    print(f"Results match: {results['results_match']}")
    print("\nTimings:")
    print(f"  Single processing:    {results['single_time']:.3f}s")
    print(f"  Batch encoding:       {results['batch_encode_time']:.3f}s")
    print(f"  Full batch:           {results['full_batch_time']:.3f}s")
    print("\nSpeedups:")
    print(f"  Batch encoding:       {results['encode_speedup']:.2f}x")
    print(f"  Full batch:           {results['full_speedup']:.2f}x")
    print("\nThroughput (queries/second):")
    print(f"  Single processing:    {results['throughput']['single_qps']:.1f}")
    print(f"  Batch encoding:       {results['throughput']['batch_encode_qps']:.1f}")
    print(f"  Full batch:           {results['throughput']['full_batch_qps']:.1f}")
    print("==========================================\n")
    
    # Save results if output specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()