#!/usr/bin/env python
"""
Test script for async processing capabilities in RAG-CSD.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List

import yaml

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.async_interface import AsyncRAGCSD
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


async def test_async_processing(
    config: Dict,
    vector_db_path: str,
    queries: List[str],
    max_workers: int = 4,
) -> Dict:
    """
    Test async processing performance.
    
    Args:
        config: Configuration dictionary.
        vector_db_path: Path to the vector database.
        queries: List of query texts.
        max_workers: Maximum number of worker threads.
        
    Returns:
        Dictionary with performance results.
    """
    logger = logging.getLogger(__name__)
    
    # Disable CSD simulation for cleaner testing
    config["csd"]["enabled"] = False
    
    # Test 1: Synchronous processing (baseline)
    logger.info("Testing synchronous processing...")
    sync_start = time.time()
    
    encoder = Encoder(config)
    vector_store = VectorStore(vector_db_path, config)
    augmentor = Augmentor(config)
    
    sync_results = []
    for query in queries:
        embedding = encoder.encode(query)
        retrieved_docs = vector_store.search(embedding, top_k=5)
        augmented_query = augmentor.augment(query, retrieved_docs)
        sync_results.append({
            "query": query,
            "retrieved_count": len(retrieved_docs),
            "augmented_length": len(augmented_query)
        })
    
    sync_time = time.time() - sync_start
    logger.info(f"Synchronous processing: {sync_time:.3f}s for {len(queries)} queries")
    
    # Test 2: Async parallel processing
    logger.info("Testing async parallel processing...")
    
    async with AsyncRAGCSD(config, max_workers=max_workers) as async_rag:
        await async_rag.initialize(vector_db_path)
        
        parallel_start = time.time()
        parallel_results = await async_rag.process_queries_parallel(queries, top_k=5)
        parallel_time = time.time() - parallel_start
    
    logger.info(f"Async parallel processing: {parallel_time:.3f}s")
    
    # Test 3: Batch optimized async processing
    logger.info("Testing batch optimized async processing...")
    
    async with AsyncRAGCSD(config, max_workers=max_workers) as async_rag:
        await async_rag.initialize(vector_db_path)
        
        batch_start = time.time()
        batch_results = await async_rag.process_queries_batch_optimized(queries, top_k=5)
        batch_time = time.time() - batch_start
    
    logger.info(f"Batch optimized async processing: {batch_time:.3f}s")
    
    # Verify results consistency
    def verify_results(sync_res, async_res):
        """Simple verification that results are consistent."""
        if len(sync_res) != len(async_res):
            return False
        
        for sync_r, async_r in zip(sync_res, async_res):
            if sync_r["retrieved_count"] != len(async_r["retrieved_docs"]):
                return False
        
        return True
    
    parallel_consistent = verify_results(sync_results, parallel_results)
    batch_consistent = verify_results(sync_results, batch_results)
    
    return {
        "num_queries": len(queries),
        "max_workers": max_workers,
        "sync_time": sync_time,
        "parallel_time": parallel_time,
        "batch_time": batch_time,
        "parallel_speedup": sync_time / parallel_time,
        "batch_speedup": sync_time / batch_time,
        "parallel_consistent": parallel_consistent,
        "batch_consistent": batch_consistent,
        "throughput": {
            "sync_qps": len(queries) / sync_time,
            "parallel_qps": len(queries) / parallel_time,
            "batch_qps": len(queries) / batch_time,
        }
    }


async def test_concurrent_load(
    config: Dict,
    vector_db_path: str,
    queries: List[str],
    concurrent_levels: List[int] = [1, 2, 4, 8],
) -> Dict:
    """
    Test system performance under different concurrency levels.
    
    Args:
        config: Configuration dictionary.
        vector_db_path: Path to the vector database.
        queries: List of query texts.
        concurrent_levels: List of concurrency levels to test.
        
    Returns:
        Dictionary with concurrency test results.
    """
    logger = logging.getLogger(__name__)
    
    # Disable CSD simulation for cleaner testing
    config["csd"]["enabled"] = False
    
    results = {}
    
    for concurrency in concurrent_levels:
        logger.info(f"Testing concurrency level: {concurrency}")
        
        async with AsyncRAGCSD(config, max_workers=concurrency * 2) as async_rag:
            await async_rag.initialize(vector_db_path)
            
            start_time = time.time()
            
            # Process queries with limited concurrency
            concurrent_results = await async_rag.process_queries_parallel(
                queries, top_k=5, max_concurrent=concurrency
            )
            
            elapsed_time = time.time() - start_time
            
            results[concurrency] = {
                "time": elapsed_time,
                "qps": len(queries) / elapsed_time,
                "results_count": len(concurrent_results),
            }
            
            logger.info(f"  Concurrency {concurrency}: {elapsed_time:.3f}s, "
                       f"{results[concurrency]['qps']:.1f} QPS")
    
    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test async processing capabilities.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--vector-db", "-v", type=str, default="data/vectors_optimized", help="Path to the vector database."
    )
    parser.add_argument(
        "--queries", "-q", type=str, default="data/test_queries.json", help="Path to test queries JSON."
    )
    parser.add_argument(
        "--max-workers", "-w", type=int, default=4, help="Maximum number of worker threads."
    )
    parser.add_argument(
        "--repeat", "-r", type=int, default=2, help="Number of times to repeat each query for testing."
    )
    parser.add_argument(
        "--test-concurrency", action="store_true", help="Run concurrency load test."
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to save test results JSON."
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
    
    # Run async processing test
    logger.info("=== Testing Async Processing Performance ===")
    async_results = await test_async_processing(
        config, args.vector_db, test_queries, args.max_workers
    )
    
    # Print results
    print("\n===== Async Processing Test Results =====")
    print(f"Queries processed: {async_results['num_queries']}")
    print(f"Max workers: {async_results['max_workers']}")
    print(f"Results consistent: Parallel={async_results['parallel_consistent']}, "
          f"Batch={async_results['batch_consistent']}")
    print("\nTimings:")
    print(f"  Synchronous:       {async_results['sync_time']:.3f}s")
    print(f"  Async parallel:    {async_results['parallel_time']:.3f}s")
    print(f"  Batch optimized:   {async_results['batch_time']:.3f}s")
    print("\nSpeedups:")
    print(f"  Async parallel:    {async_results['parallel_speedup']:.2f}x")
    print(f"  Batch optimized:   {async_results['batch_speedup']:.2f}x")
    print("\nThroughput (queries/second):")
    print(f"  Synchronous:       {async_results['throughput']['sync_qps']:.1f}")
    print(f"  Async parallel:    {async_results['throughput']['parallel_qps']:.1f}")
    print(f"  Batch optimized:   {async_results['throughput']['batch_qps']:.1f}")
    print("==========================================\n")
    
    all_results = {"async_performance": async_results}
    
    # Run concurrency test if requested
    if args.test_concurrency:
        logger.info("=== Testing Concurrency Levels ===")
        concurrency_results = await test_concurrent_load(
            config, args.vector_db, test_queries
        )
        
        print("===== Concurrency Test Results =====")
        for concurrency, result in concurrency_results.items():
            print(f"Concurrency {concurrency:2d}: {result['time']:.3f}s, "
                  f"{result['qps']:6.1f} QPS")
        print("=====================================\n")
        
        all_results["concurrency"] = concurrency_results
    
    # Save results if output specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())