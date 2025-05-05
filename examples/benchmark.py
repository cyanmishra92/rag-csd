#!/usr/bin/env python
"""
Benchmark script for RAG-CSD system.
This script compares standard execution with CSD simulation.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
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


def load_queries(queries_path: str) -> List[Dict]:
    """Load test queries from a JSON file."""
    with open(queries_path, "r") as f:
        queries = json.load(f)
    return queries


def run_benchmark(
    config: Dict,
    vector_db_path: str,
    queries: List[Dict],
    use_csd: bool,
    top_k: int,
    runs_per_query: int,
) -> Dict:
    """
    Run benchmark on the RAG-CSD system.
    
    Args:
        config: Configuration dictionary.
        vector_db_path: Path to the vector database.
        queries: List of query dictionaries.
        use_csd: Whether to use CSD simulation.
        top_k: Number of documents to retrieve.
        runs_per_query: Number of runs per query for averaging.
        
    Returns:
        Dictionary with benchmark results.
    """
    # Override CSD setting if specified
    if use_csd:
        config["csd"]["enabled"] = True
    else:
        config["csd"]["enabled"] = False
    
    # Initialize components
    start_time = time.time()
    
    logger = logging.getLogger(__name__)
    logger.info("Initializing components...")
    
    # Initialize encoder
    encoder = Encoder(config)
    
    # Initialize vector store
    vector_store = VectorStore(vector_db_path, config)
    
    # Initialize augmentor
    augmentor = Augmentor(config)
    
    init_time = time.time() - start_time
    logger.info(f"Components initialized in {init_time:.2f} seconds")
    
    # Prepare result containers
    results = {
        "config": {
            "use_csd": use_csd,
            "top_k": top_k,
            "runs_per_query": runs_per_query,
        },
        "init_time": init_time,
        "query_results": [],
        "summary": {},
    }
    
    # Run benchmarks for each query
    encode_times = []
    retrieve_times = []
    augment_times = []
    total_times = []
    
    for query_data in tqdm(queries, desc="Processing queries"):
        query_id = query_data["id"]
        query_text = query_data["text"]
        query_topic = query_data.get("topic", "unknown")
        
        query_results = []
        
        # Run multiple times to get average performance
        for run in range(runs_per_query):
            # Process query
            query_start_time = time.time()
            
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
            
            # Calculate total time
            total_time = time.time() - query_start_time
            
            # Store times
            encode_times.append(encode_time)
            retrieve_times.append(retrieve_time)
            augment_times.append(augment_time)
            total_times.append(total_time)
            
            # Record results for this run
            run_results = {
                "run": run + 1,
                "encode_time": encode_time,
                "retrieve_time": retrieve_time,
                "augment_time": augment_time,
                "total_time": total_time,
                "num_docs_retrieved": len(retrieved_docs),
            }
            
            query_results.append(run_results)
        
        # Calculate average results for this query
        avg_encode_time = sum(r["encode_time"] for r in query_results) / runs_per_query
        avg_retrieve_time = sum(r["retrieve_time"] for r in query_results) / runs_per_query
        avg_augment_time = sum(r["augment_time"] for r in query_results) / runs_per_query
        avg_total_time = sum(r["total_time"] for r in query_results) / runs_per_query
        
        # Store query results
        results["query_results"].append({
            "query_id": query_id,
            "query_text": query_text,
            "query_topic": query_topic,
            "avg_encode_time": avg_encode_time,
            "avg_retrieve_time": avg_retrieve_time,
            "avg_augment_time": avg_augment_time,
            "avg_total_time": avg_total_time,
            "runs": query_results,
        })
    
    # Calculate overall summary statistics
    results["summary"] = {
        "encode_time": {
            "mean": float(np.mean(encode_times)),
            "median": float(np.median(encode_times)),
            "min": float(np.min(encode_times)),
            "max": float(np.max(encode_times)),
            "std": float(np.std(encode_times)),
        },
        "retrieve_time": {
            "mean": float(np.mean(retrieve_times)),
            "median": float(np.median(retrieve_times)),
            "min": float(np.min(retrieve_times)),
            "max": float(np.max(retrieve_times)),
            "std": float(np.std(retrieve_times)),
        },
        "augment_time": {
            "mean": float(np.mean(augment_times)),
            "median": float(np.median(augment_times)),
            "min": float(np.min(augment_times)),
            "max": float(np.max(augment_times)),
            "std": float(np.std(augment_times)),
        },
        "total_time": {
            "mean": float(np.mean(total_times)),
            "median": float(np.median(total_times)),
            "min": float(np.min(total_times)),
            "max": float(np.max(total_times)),
            "std": float(np.std(total_times)),
        },
    }
    
    return results


def compare_benchmarks(standard_results: Dict, csd_results: Dict) -> Dict:
    """
    Compare benchmark results between standard execution and CSD simulation.
    
    Args:
        standard_results: Results from standard execution.
        csd_results: Results from CSD simulation.
        
    Returns:
        Dictionary with comparison results.
    """
    comparison = {
        "encode_time_ratio": csd_results["summary"]["encode_time"]["mean"] / 
                            standard_results["summary"]["encode_time"]["mean"],
        "retrieve_time_ratio": csd_results["summary"]["retrieve_time"]["mean"] /
                               standard_results["summary"]["retrieve_time"]["mean"],
        "augment_time_ratio": csd_results["summary"]["augment_time"]["mean"] /
                              standard_results["summary"]["augment_time"]["mean"],
        "total_time_ratio": csd_results["summary"]["total_time"]["mean"] /
                            standard_results["summary"]["total_time"]["mean"],
    }
    
    return comparison


def print_results(standard_results: Dict, csd_results: Dict, comparison: Dict) -> None:
    """
    Print benchmark results in a nice format.
    
    Args:
        standard_results: Results from standard execution.
        csd_results: Results from CSD simulation.
        comparison: Comparison results.
    """
    print("\n===== RAG-CSD Benchmark Results =====\n")
    
    print("Standard Execution Summary:")
    print(f"  Encode Time:   {standard_results['summary']['encode_time']['mean']:.4f}s "
          f"(±{standard_results['summary']['encode_time']['std']:.4f}s)")
    print(f"  Retrieve Time: {standard_results['summary']['retrieve_time']['mean']:.4f}s "
          f"(±{standard_results['summary']['retrieve_time']['std']:.4f}s)")
    print(f"  Augment Time:  {standard_results['summary']['augment_time']['mean']:.4f}s "
          f"(±{standard_results['summary']['augment_time']['std']:.4f}s)")
    print(f"  Total Time:    {standard_results['summary']['total_time']['mean']:.4f}s "
          f"(±{standard_results['summary']['total_time']['std']:.4f}s)")
    
    print("\nCSD Simulation Summary:")
    print(f"  Encode Time:   {csd_results['summary']['encode_time']['mean']:.4f}s "
          f"(±{csd_results['summary']['encode_time']['std']:.4f}s)")
    print(f"  Retrieve Time: {csd_results['summary']['retrieve_time']['mean']:.4f}s "
          f"(±{csd_results['summary']['retrieve_time']['std']:.4f}s)")
    print(f"  Augment Time:  {csd_results['summary']['augment_time']['mean']:.4f}s "
          f"(±{csd_results['summary']['augment_time']['std']:.4f}s)")
    print(f"  Total Time:    {csd_results['summary']['total_time']['mean']:.4f}s "
          f"(±{csd_results['summary']['total_time']['std']:.4f}s)")
    
    print("\nComparison (CSD / Standard):")
    print(f"  Encode Time Ratio:   {comparison['encode_time_ratio']:.2f}x")
    print(f"  Retrieve Time Ratio: {comparison['retrieve_time_ratio']:.2f}x")
    print(f"  Augment Time Ratio:  {comparison['augment_time_ratio']:.2f}x")
    print(f"  Total Time Ratio:    {comparison['total_time_ratio']:.2f}x")
    
    print("\nNote: Ratios > 1 mean CSD simulation is slower due to simulated latency.")
    print("      In real CSD hardware, we expect these ratios to be < 1 (faster).")
    print("\n=================================\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark the RAG-CSD system.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--vector-db", "-v", type=str, default="data/vectors", help="Path to the vector database."
    )
    parser.add_argument(
        "--queries", "-q", type=str, default="data/test_queries.json", help="Path to test queries JSON."
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3, help="Number of runs per query for averaging."
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
    queries = load_queries(args.queries)
    logger.info(f"Loaded {len(queries)} test queries from {args.queries}")
    
    # Run standard benchmark
    logger.info("Running standard benchmark (without CSD simulation)...")
    standard_results = run_benchmark(
        config, args.vector_db, queries, False, args.top_k, args.runs
    )
    
    # Run CSD simulation benchmark
    logger.info("Running benchmark with CSD simulation...")
    csd_results = run_benchmark(
        config, args.vector_db, queries, True, args.top_k, args.runs
    )
    
    # Compare results
    comparison = compare_benchmarks(standard_results, csd_results)
    
    # Print results
    print_results(standard_results, csd_results, comparison)
    
    # Save results if output specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results = {
            "standard": standard_results,
            "csd": csd_results,
            "comparison": comparison,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()