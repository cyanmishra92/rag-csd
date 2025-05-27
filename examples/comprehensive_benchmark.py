#!/usr/bin/env python
"""
Comprehensive benchmark script comparing RAG-CSD against baseline RAG systems.
This script demonstrates the performance advantages of RAG-CSD optimizations
against VanillaRAG, PipeRAG-like, and EdgeRAG-like implementations.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any
import traceback

import numpy as np
import yaml
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.pipeline import RAGCSDPipeline
from rag_csd.benchmarks.baseline_systems import VanillaRAG, PipeRAGLike, EdgeRAGLike
from rag_csd.benchmarks.visualizer import PerformanceVisualizer
from rag_csd.utils.logger import setup_logger


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_test_queries() -> List[str]:
    """Generate test queries for benchmarking."""
    return [
        "What is computational storage and how does it differ from traditional storage?",
        "Explain the benefits of retrieval-augmented generation in AI systems",
        "How do vector similarity search algorithms work in practice?",
        "What are the key advantages of using FAISS for vector indexing?",
        "Describe the process of embedding text documents for semantic search",
        "How can batch processing improve RAG system performance?",
        "What role does caching play in optimizing embedding computations?",
        "Compare different approaches to text preprocessing in NLP pipelines",
        "How do transformer models generate contextual embeddings?",
        "What are the trade-offs between accuracy and speed in vector search?"
    ]


class BenchmarkRunner:
    """Handles running benchmarks across multiple RAG systems."""
    
    def __init__(self, vector_db_path: str, config: Dict):
        self.vector_db_path = vector_db_path
        self.config = config
        self.systems = self._initialize_systems()
        
    def _initialize_systems(self) -> Dict[str, Any]:
        """Initialize all RAG systems for comparison."""
        systems = {}
        
        try:
            systems["RAG-CSD"] = RAGCSDPipeline(self.vector_db_path)
            print("✅ RAG-CSD initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAG-CSD: {e}")
            
        try:
            systems["VanillaRAG"] = VanillaRAG(self.vector_db_path)
            print("✅ VanillaRAG initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize VanillaRAG: {e}")
            
        try:
            systems["PipeRAG-like"] = PipeRAGLike(self.vector_db_path)
            print("✅ PipeRAG-like initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize PipeRAG-like: {e}")
            
        try:
            systems["EdgeRAG-like"] = EdgeRAGLike(self.vector_db_path)
            print("✅ EdgeRAG-like initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize EdgeRAG-like: {e}")
            
        return systems
    
    def run_single_query_benchmark(
        self, 
        queries: List[str], 
        top_k: int = 5, 
        runs_per_query: int = 3
    ) -> Dict[str, List[Dict]]:
        """Run single query benchmarks across all systems."""
        results = {}
        
        for system_name, system in self.systems.items():
            print(f"\n🔄 Benchmarking {system_name}...")
            system_results = []
            
            for query in tqdm(queries, desc=f"Processing {system_name}"):
                query_results = []
                
                # Multiple runs for statistical significance
                for run in range(runs_per_query):
                    try:
                        start_time = time.time()
                        result = system.query(query, top_k=top_k)
                        end_time = time.time()
                        
                        latency = end_time - start_time
                        
                        run_result = {
                            'query': query,
                            'latency': latency,
                            'run': run + 1,
                            'relevant_docs': len(result.get('retrieved_docs', [])),
                            'cache_hit': result.get('cache_hit', False),
                            'processing_time': result.get('processing_time', latency)
                        }
                        
                        query_results.append(run_result)
                        
                    except Exception as e:
                        print(f"❌ Error in {system_name} query '{query[:50]}...': {e}")
                        traceback.print_exc()
                        
                # Calculate average metrics for this query
                if query_results:
                    avg_latency = np.mean([r['latency'] for r in query_results])
                    min_latency = np.min([r['latency'] for r in query_results])
                    max_latency = np.max([r['latency'] for r in query_results])
                    std_latency = np.std([r['latency'] for r in query_results])
                    
                    system_results.append({
                        'query': query,
                        'avg_latency': avg_latency,
                        'min_latency': min_latency,
                        'max_latency': max_latency,
                        'std_latency': std_latency,
                        'runs': query_results
                    })
            
            results[system_name] = system_results
            
        return results
    
    def run_batch_benchmark(
        self, 
        queries: List[str], 
        top_k: int = 5
    ) -> Dict[str, Dict]:
        """Run batch processing benchmarks (where supported)."""
        results = {}
        
        for system_name, system in self.systems.items():
            print(f"\n⚡ Testing batch processing for {system_name}...")
            
            try:
                # Check if system supports batch processing
                if hasattr(system, 'query_batch'):
                    start_time = time.time()
                    batch_results = system.query_batch(queries, top_k=top_k)
                    end_time = time.time()
                    
                    batch_time = end_time - start_time
                    avg_per_query = batch_time / len(queries)
                    
                    results[system_name] = {
                        'supports_batch': True,
                        'total_time': batch_time,
                        'avg_per_query': avg_per_query,
                        'num_queries': len(queries),
                        'results': batch_results
                    }
                    
                    print(f"  ✅ Batch processing: {batch_time:.3f}s total ({avg_per_query:.3f}s per query)")
                else:
                    # Fall back to sequential processing
                    start_time = time.time()
                    sequential_results = []
                    for query in queries:
                        result = system.query(query, top_k=top_k)
                        sequential_results.append(result)
                    end_time = time.time()
                    
                    sequential_time = end_time - start_time
                    avg_per_query = sequential_time / len(queries)
                    
                    results[system_name] = {
                        'supports_batch': False,
                        'total_time': sequential_time,
                        'avg_per_query': avg_per_query,
                        'num_queries': len(queries),
                        'results': sequential_results
                    }
                    
                    print(f"  ⏳ Sequential processing: {sequential_time:.3f}s total ({avg_per_query:.3f}s per query)")
                    
            except Exception as e:
                print(f"  ❌ Error in batch processing for {system_name}: {e}")
                results[system_name] = {
                    'supports_batch': False,
                    'error': str(e)
                }
        
        return results
    
    def run_cache_effectiveness_test(
        self, 
        repeated_query: str, 
        num_runs: int = 5, 
        top_k: int = 5
    ) -> Dict[str, Dict]:
        """Test cache effectiveness with repeated queries."""
        results = {}
        
        for system_name, system in self.systems.items():
            print(f"\n💾 Testing cache effectiveness for {system_name}...")
            
            try:
                run_times = []
                cache_hits = []
                
                for run in range(num_runs):
                    start_time = time.time()
                    result = system.query(repeated_query, top_k=top_k)
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    cache_hit = result.get('cache_hit', False)
                    
                    run_times.append(latency)
                    cache_hits.append(cache_hit)
                    
                    print(f"  Run {run+1}: {latency:.4f}s (Cache hit: {cache_hit})")
                
                # Calculate cache effectiveness
                first_run_time = run_times[0]
                if len(run_times) > 1:
                    cached_times = run_times[1:]
                    avg_cached_time = np.mean(cached_times)
                    cache_speedup = first_run_time / avg_cached_time if avg_cached_time > 0 else 1
                else:
                    avg_cached_time = first_run_time
                    cache_speedup = 1
                
                results[system_name] = {
                    'first_run_time': first_run_time,
                    'avg_cached_time': avg_cached_time,
                    'cache_speedup': cache_speedup,
                    'cache_hit_rate': sum(cache_hits) / len(cache_hits),
                    'run_times': run_times,
                    'cache_hits': cache_hits
                }
                
                print(f"  📊 Cache speedup: {cache_speedup:.1f}x")
                
            except Exception as e:
                print(f"  ❌ Error in cache test for {system_name}: {e}")
                results[system_name] = {'error': str(e)}
        
        return results


def print_comprehensive_results(
    single_query_results: Dict,
    batch_results: Dict,
    cache_results: Dict
) -> None:
    """Print comprehensive benchmark results."""
    print("\n" + "=" * 80)
    print("🏆 COMPREHENSIVE RAG SYSTEM BENCHMARK RESULTS")
    print("=" * 80)
    
    # Single query performance
    print("\n📊 SINGLE QUERY PERFORMANCE")
    print("-" * 50)
    
    system_averages = {}
    for system_name, results in single_query_results.items():
        if results:
            avg_latency = np.mean([r['avg_latency'] for r in results])
            std_latency = np.std([r['avg_latency'] for r in results])
            system_averages[system_name] = avg_latency
            
            print(f"{system_name:15}: {avg_latency:.3f}s ± {std_latency:.3f}s")
    
    # Calculate speedups
    if "VanillaRAG" in system_averages:
        baseline = system_averages["VanillaRAG"]
        print(f"\n🚀 Speedup vs VanillaRAG:")
        for system_name, avg_time in system_averages.items():
            speedup = baseline / avg_time if avg_time > 0 else float('inf')
            print(f"  {system_name:15}: {speedup:.1f}x")
    
    # Batch processing performance
    print(f"\n⚡ BATCH PROCESSING PERFORMANCE")
    print("-" * 50)
    
    for system_name, result in batch_results.items():
        if 'error' not in result:
            supports = "✅" if result['supports_batch'] else "❌"
            print(f"{system_name:15}: {supports} {result['avg_per_query']:.3f}s per query")
    
    # Cache effectiveness
    print(f"\n💾 CACHE EFFECTIVENESS")
    print("-" * 50)
    
    for system_name, result in cache_results.items():
        if 'error' not in result:
            speedup = result['cache_speedup']
            hit_rate = result['cache_hit_rate'] * 100
            print(f"{system_name:15}: {speedup:.1f}x speedup, {hit_rate:.0f}% hit rate")
    
    print("=" * 80)


def main():
    """Main entry point for comprehensive benchmarking."""
    parser = argparse.ArgumentParser(description="Comprehensive RAG system benchmark.")
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml", 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--vector-db", "-v", type=str, required=True, 
        help="Path to the vector database."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="benchmark_results/", 
        help="Directory to save benchmark results and visualizations."
    )
    parser.add_argument(
        "--runs", "-r", type=int, default=3, 
        help="Number of runs per query for averaging."
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, 
        help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--log-level", "-l", type=str, default="WARNING", 
        help="Logging level."
    )
    parser.add_argument(
        "--skip-visualizations", action="store_true", 
        help="Skip generating visualization plots."
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(level=args.log_level)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 RAG System Comprehensive Benchmark")
    print("=" * 80)
    print("This benchmark compares RAG-CSD against baseline RAG implementations")
    print("including VanillaRAG, PipeRAG-like, and EdgeRAG-like systems.")
    print("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate test queries
    test_queries = generate_test_queries()
    print(f"📝 Generated {len(test_queries)} test queries")
    
    # Initialize benchmark runner
    print(f"\n🔧 Initializing benchmark systems...")
    benchmark_runner = BenchmarkRunner(args.vector_db, config)
    
    if not benchmark_runner.systems:
        print("❌ No systems initialized successfully. Exiting.")
        return
    
    print(f"✅ Initialized {len(benchmark_runner.systems)} systems: {list(benchmark_runner.systems.keys())}")
    
    # Run single query benchmarks
    print(f"\n🎯 Running single query benchmarks...")
    single_query_results = benchmark_runner.run_single_query_benchmark(
        test_queries[:5],  # Use first 5 queries for single query test
        top_k=args.top_k,
        runs_per_query=args.runs
    )
    
    # Run batch processing benchmarks
    print(f"\n⚡ Running batch processing benchmarks...")
    batch_results = benchmark_runner.run_batch_benchmark(
        test_queries[:5],  # Same queries for batch test
        top_k=args.top_k
    )
    
    # Run cache effectiveness test
    print(f"\n💾 Running cache effectiveness test...")
    cache_results = benchmark_runner.run_cache_effectiveness_test(
        test_queries[0],  # Use first query for cache test
        num_runs=5,
        top_k=args.top_k
    )
    
    # Print results
    print_comprehensive_results(single_query_results, batch_results, cache_results)
    
    # Save results
    all_results = {
        'single_query': single_query_results,
        'batch_processing': batch_results,
        'cache_effectiveness': cache_results,
        'test_queries': test_queries,
        'config': {
            'top_k': args.top_k,
            'runs_per_query': args.runs,
            'vector_db_path': args.vector_db
        }
    }
    
    results_file = os.path.join(args.output_dir, "comprehensive_benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_file}")
    
    # Generate visualizations
    if not args.skip_visualizations:
        print(f"\n📊 Generating performance visualizations...")
        try:
            visualizer = PerformanceVisualizer()
            
            # Latency comparison
            latency_plot = os.path.join(args.output_dir, "latency_comparison.png")
            visualizer.plot_latency_comparison(single_query_results, save_path=latency_plot)
            print(f"  ✅ Latency comparison saved to {latency_plot}")
            
            # Throughput analysis
            throughput_plot = os.path.join(args.output_dir, "throughput_analysis.png")
            visualizer.plot_throughput_analysis(single_query_results, save_path=throughput_plot)
            print(f"  ✅ Throughput analysis saved to {throughput_plot}")
            
            # System comparison radar
            radar_plot = os.path.join(args.output_dir, "system_comparison_radar.png")
            visualizer.plot_system_comparison_radar(single_query_results, save_path=radar_plot)
            print(f"  ✅ System comparison radar saved to {radar_plot}")
            
        except Exception as e:
            print(f"  ❌ Error generating visualizations: {e}")
    
    print(f"\n✅ Comprehensive benchmark completed!")
    print(f"📁 All results and visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()