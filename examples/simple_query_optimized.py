#!/usr/bin/env python
"""
Enhanced example showcasing RAG-CSD's high-performance optimizations.
This example demonstrates the significant performance improvements achieved
through model caching, embedding cache, and optimized processing.
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

from rag_csd.pipeline import RAGCSDPipeline
from rag_csd.embedding.encoder import Encoder
from rag_csd.retrieval.vector_store import VectorStore
from rag_csd.augmentation.augmentor import Augmentor
from rag_csd.utils.logger import setup_logger


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def demonstrate_legacy_vs_optimized(vector_db_path: str, queries: List[str], top_k: int = 5):
    """Demonstrate performance difference between legacy and optimized approaches."""
    print("🔥 RAG-CSD Performance Comparison: Legacy vs Optimized")
    print("=" * 70)
    
    # Legacy approach (original implementation)
    print("\n📊 Testing Legacy Implementation...")
    legacy_times = []
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}: {query[:50]}...")
        
        start_time = time.time()
        
        # Initialize components each time (simulating cold start)
        encoder = Encoder()
        vector_store = VectorStore(vector_db_path)
        augmentor = Augmentor()
        
        # Process query
        query_embedding = encoder.encode(query)
        retrieved_docs = vector_store.search(query_embedding, top_k=top_k)
        augmented_query = augmentor.augment(query, retrieved_docs)
        
        end_time = time.time()
        query_time = end_time - start_time
        legacy_times.append(query_time)
        print(f"    Time: {query_time:.3f}s")
    
    # Optimized approach (new pipeline)
    print("\n🚀 Testing Optimized Pipeline...")
    optimized_times = []
    
    # Initialize optimized pipeline once
    pipeline = RAGCSDPipeline(vector_db_path)
    
    for i, query in enumerate(queries):
        print(f"  Query {i+1}: {query[:50]}...")
        
        start_time = time.time()
        result = pipeline.query(query, top_k=top_k)
        end_time = time.time()
        
        query_time = end_time - start_time
        optimized_times.append(query_time)
        print(f"    Time: {query_time:.3f}s (Cache hit: {result.get('cache_hit', False)})")
    
    # Calculate and display results
    print("\n📈 Performance Analysis:")
    print("-" * 40)
    
    avg_legacy = sum(legacy_times) / len(legacy_times)
    avg_optimized = sum(optimized_times) / len(optimized_times)
    speedup = avg_legacy / avg_optimized if avg_optimized > 0 else float('inf')
    
    print(f"Legacy Average:     {avg_legacy:.3f}s")
    print(f"Optimized Average:  {avg_optimized:.3f}s")
    print(f"🎯 Speedup:         {speedup:.1f}x faster!")
    
    # Show individual improvements
    print("\nPer-Query Improvements:")
    for i, (legacy, optimized) in enumerate(zip(legacy_times, optimized_times)):
        individual_speedup = legacy / optimized if optimized > 0 else float('inf')
        print(f"  Query {i+1}: {individual_speedup:.1f}x faster ({legacy:.3f}s → {optimized:.3f}s)")
    
    return avg_legacy, avg_optimized, speedup


def demonstrate_cache_effectiveness(vector_db_path: str, repeated_query: str, num_runs: int = 5):
    """Demonstrate the effectiveness of embedding cache with repeated queries."""
    print(f"\n💾 Cache Effectiveness Test - Running same query {num_runs} times")
    print("=" * 70)
    
    pipeline = RAGCSDPipeline(vector_db_path)
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        result = pipeline.query(repeated_query, top_k=5)
        end_time = time.time()
        
        query_time = end_time - start_time
        times.append(query_time)
        cache_hit = result.get('cache_hit', False)
        
        print(f"  Run {i+1}: {query_time:.4f}s (Cache hit: {cache_hit})")
    
    # Calculate cache speedup
    first_run = times[0]  # Cache miss
    cached_avg = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    cache_speedup = first_run / cached_avg if cached_avg > 0 else 1
    
    print(f"\n📊 Cache Performance:")
    print(f"  First run (cache miss): {first_run:.4f}s")
    print(f"  Cached runs average:    {cached_avg:.4f}s")
    print(f"  🎯 Cache speedup:       {cache_speedup:.0f}x faster!")
    
    return first_run, cached_avg, cache_speedup


def demonstrate_batch_processing(vector_db_path: str, queries: List[str]):
    """Demonstrate batch processing optimization."""
    print(f"\n⚡ Batch Processing Test - {len(queries)} queries")
    print("=" * 70)
    
    pipeline = RAGCSDPipeline(vector_db_path)
    
    # Sequential processing
    print("🔄 Sequential Processing:")
    sequential_start = time.time()
    sequential_results = []
    for i, query in enumerate(queries):
        result = pipeline.query(query, top_k=5)
        sequential_results.append(result)
        print(f"  Query {i+1}: {result['processing_time']:.3f}s")
    sequential_total = time.time() - sequential_start
    
    # Batch processing
    print("\n🚀 Batch Processing:")
    batch_start = time.time()
    batch_results = pipeline.query_batch(queries, top_k=5)
    batch_total = time.time() - batch_start
    
    for i, result in enumerate(batch_results):
        print(f"  Query {i+1}: {result['processing_time']:.3f}s")
    
    # Calculate batch speedup
    batch_speedup = sequential_total / batch_total if batch_total > 0 else 1
    
    print(f"\n📊 Batch Performance:")
    print(f"  Sequential total: {sequential_total:.3f}s")
    print(f"  Batch total:      {batch_total:.3f}s")
    print(f"  🎯 Batch speedup: {batch_speedup:.1f}x faster!")
    
    return sequential_total, batch_total, batch_speedup


def main():
    """Main entry point showcasing RAG-CSD optimizations."""
    parser = argparse.ArgumentParser(description="Demonstrate RAG-CSD performance optimizations.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", help="Path to the configuration file."
    )
    parser.add_argument("--vector-db", "-v", type=str, required=True, help="Path to the vector database.")
    parser.add_argument("--log-level", "-l", type=str, default="WARNING", help="Logging level.")
    parser.add_argument("--output", "-o", type=str, help="Path to save the benchmark results JSON.")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    
    print("🚀 RAG-CSD High-Performance Optimization Demo")
    print("=" * 80)
    print("This demo showcases the significant performance improvements achieved")
    print("through intelligent caching, optimized indexing, and batch processing.")
    print("=" * 80)
    
    # Test queries
    test_queries = [
        "What is computational storage and how does it work?",
        "Explain vector similarity search algorithms",
        "Benefits of retrieval-augmented generation systems",
        "How do embedding models process text?",
        "What are the advantages of FAISS indexing?"
    ]
    
    # Demonstrate legacy vs optimized
    avg_legacy, avg_optimized, overall_speedup = demonstrate_legacy_vs_optimized(
        args.vector_db, test_queries[:3]
    )
    
    # Demonstrate cache effectiveness
    cache_first, cache_avg, cache_speedup = demonstrate_cache_effectiveness(
        args.vector_db, test_queries[0]
    )
    
    # Demonstrate batch processing
    seq_time, batch_time, batch_speedup = demonstrate_batch_processing(
        args.vector_db, test_queries
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"🚀 Overall Pipeline Speedup:     {overall_speedup:.1f}x")
    print(f"💾 Embedding Cache Speedup:      {cache_speedup:.0f}x")
    print(f"⚡ Batch Processing Speedup:     {batch_speedup:.1f}x")
    print(f"🏆 Total Optimization Impact:    Up to {max(overall_speedup, cache_speedup, batch_speedup):.0f}x faster!")
    print("=" * 80)
    
    # Save results if requested
    if args.output:
        results = {
            "performance_summary": {
                "overall_speedup": overall_speedup,
                "cache_speedup": cache_speedup,
                "batch_speedup": batch_speedup,
                "max_speedup": max(overall_speedup, cache_speedup, batch_speedup)
            },
            "detailed_results": {
                "legacy_vs_optimized": {
                    "avg_legacy_time": avg_legacy,
                    "avg_optimized_time": avg_optimized,
                    "speedup": overall_speedup
                },
                "cache_effectiveness": {
                    "first_run_time": cache_first,
                    "cached_avg_time": cache_avg,
                    "speedup": cache_speedup
                },
                "batch_processing": {
                    "sequential_time": seq_time,
                    "batch_time": batch_time,
                    "speedup": batch_speedup
                }
            },
            "test_queries": test_queries
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()