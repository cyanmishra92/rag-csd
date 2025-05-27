#!/usr/bin/env python
"""
Demonstration of RAG-CSD's async and parallel processing capabilities.
This example shows how to leverage async interfaces for maximum throughput.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Dict

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.async_interface import AsyncRAGCSD
from rag_csd.pipeline import RAGCSDPipeline


async def demonstrate_async_vs_sync(vector_db_path: str, queries: List[str]):
    """Demonstrate the performance difference between async and sync processing."""
    print("🔄 Async vs Sync Processing Comparison")
    print("=" * 60)
    
    # Sync processing
    print("\n📊 Synchronous Processing:")
    sync_pipeline = RAGCSDPipeline(vector_db_path)
    
    sync_start = time.time()
    sync_results = []
    for i, query in enumerate(queries):
        result = sync_pipeline.query(query, top_k=5)
        sync_results.append(result)
        print(f"  Query {i+1}: {result['processing_time']:.3f}s")
    sync_total = time.time() - sync_start
    
    print(f"  📈 Total sync time: {sync_total:.3f}s")
    
    # Async processing
    print("\n🚀 Asynchronous Processing:")
    async_rag = AsyncRAGCSD(vector_db_path)
    
    async_start = time.time()
    async_results = await async_rag.process_queries_parallel(queries, top_k=5)
    async_total = time.time() - async_start
    
    for i, result in enumerate(async_results):
        print(f"  Query {i+1}: {result['processing_time']:.3f}s")
    
    print(f"  📈 Total async time: {async_total:.3f}s")
    
    # Calculate speedup
    speedup = sync_total / async_total if async_total > 0 else float('inf')
    print(f"\n🎯 Async Speedup: {speedup:.1f}x faster!")
    
    return sync_total, async_total, speedup


async def demonstrate_batch_optimization(vector_db_path: str, queries: List[str]):
    """Demonstrate batch optimization capabilities."""
    print("\n⚡ Batch Optimization Demonstration")
    print("=" * 60)
    
    async_rag = AsyncRAGCSD(vector_db_path)
    
    # Individual queries
    print("\n📊 Individual Query Processing:")
    individual_start = time.time()
    individual_results = []
    
    for i, query in enumerate(queries):
        result = await async_rag.process_query(query, top_k=5)
        individual_results.append(result)
        print(f"  Query {i+1}: {result['processing_time']:.3f}s")
    
    individual_total = time.time() - individual_start
    print(f"  📈 Total individual time: {individual_total:.3f}s")
    
    # Batch processing
    print("\n🚀 Optimized Batch Processing:")
    batch_start = time.time()
    batch_results = await async_rag.process_queries_batch_optimized(queries, top_k=5)
    batch_total = time.time() - batch_start
    
    for i, result in enumerate(batch_results):
        cache_hit = result.get('cache_hit', False)
        print(f"  Query {i+1}: {result['processing_time']:.3f}s (Cache: {cache_hit})")
    
    print(f"  📈 Total batch time: {batch_total:.3f}s")
    
    # Calculate speedup
    batch_speedup = individual_total / batch_total if batch_total > 0 else float('inf')
    print(f"\n🎯 Batch Speedup: {batch_speedup:.1f}x faster!")
    
    return individual_total, batch_total, batch_speedup


async def demonstrate_concurrent_users(vector_db_path: str, queries: List[str], num_users: int = 3):
    """Simulate concurrent users accessing the system."""
    print(f"\n👥 Concurrent Users Simulation ({num_users} users)")
    print("=" * 60)
    
    async_rag = AsyncRAGCSD(vector_db_path)
    
    async def simulate_user(user_id: int, user_queries: List[str]):
        """Simulate a single user's query session."""
        user_results = []
        user_start = time.time()
        
        for query in user_queries:
            result = await async_rag.process_query(query, top_k=5)
            user_results.append(result)
        
        user_total = time.time() - user_start
        return user_id, user_results, user_total
    
    # Create user tasks
    user_tasks = []
    queries_per_user = len(queries) // num_users + (1 if len(queries) % num_users else 0)
    
    for user_id in range(num_users):
        start_idx = user_id * queries_per_user
        end_idx = min(start_idx + queries_per_user, len(queries))
        user_queries = queries[start_idx:end_idx]
        
        if user_queries:  # Only create task if user has queries
            task = simulate_user(user_id, user_queries)
            user_tasks.append(task)
    
    # Run concurrent users
    concurrent_start = time.time()
    user_results = await asyncio.gather(*user_tasks)
    concurrent_total = time.time() - concurrent_start
    
    # Print results
    total_queries = sum(len(results[1]) for results in user_results)
    avg_query_time = concurrent_total / total_queries if total_queries > 0 else 0
    
    for user_id, results, user_time in user_results:
        avg_user_time = user_time / len(results) if results else 0
        print(f"  User {user_id+1}: {len(results)} queries in {user_time:.3f}s (avg: {avg_user_time:.3f}s)")
    
    print(f"\n📈 Concurrent Performance:")
    print(f"  Total time: {concurrent_total:.3f}s")
    print(f"  Total queries: {total_queries}")
    print(f"  Avg per query: {avg_query_time:.3f}s")
    print(f"  Effective throughput: {total_queries/concurrent_total:.1f} queries/sec")
    
    return concurrent_total, total_queries, avg_query_time


async def demonstrate_cache_sharing(vector_db_path: str, shared_queries: List[str]):
    """Demonstrate cache sharing between concurrent operations."""
    print(f"\n💾 Cache Sharing Demonstration")
    print("=" * 60)
    
    async_rag = AsyncRAGCSD(vector_db_path)
    
    # First, prime the cache
    print("🔄 Priming cache with initial queries...")
    for query in shared_queries[:2]:
        await async_rag.process_query(query, top_k=5)
        print(f"  Cached: {query[:50]}...")
    
    # Now run multiple concurrent requests for the same queries
    print(f"\n🚀 Running {len(shared_queries)} concurrent requests...")
    
    async def timed_query(query: str, request_id: int):
        start_time = time.time()
        result = await async_rag.process_query(query, top_k=5)
        end_time = time.time()
        return request_id, query, end_time - start_time, result.get('cache_hit', False)
    
    # Create concurrent tasks for the same queries
    tasks = []
    for i, query in enumerate(shared_queries):
        task = timed_query(query, i)
        tasks.append(task)
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Print results
    cache_hits = sum(1 for _, _, _, cache_hit in results if cache_hit)
    cache_hit_rate = cache_hits / len(results) * 100
    
    for request_id, query, query_time, cache_hit in results:
        status = "🎯 CACHE HIT" if cache_hit else "📁 CACHE MISS"
        print(f"  Request {request_id+1}: {query_time:.4f}s {status}")
    
    print(f"\n📊 Cache Performance:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Cache hit rate: {cache_hit_rate:.1f}%")
    print(f"  Avg query time: {sum(t for _, _, t, _ in results) / len(results):.4f}s")
    
    return total_time, cache_hit_rate


def generate_test_queries() -> List[str]:
    """Generate test queries for async demonstration."""
    return [
        "What is computational storage?",
        "How does vector similarity search work?", 
        "Benefits of retrieval-augmented generation",
        "Explain embedding models for text",
        "FAISS indexing advantages",
        "Batch processing in RAG systems",
        "Cache optimization strategies",
        "Transformer model architectures",
        "Semantic search implementation",
        "RAG system performance tuning"
    ]


async def main():
    """Main async demonstration function."""
    parser = argparse.ArgumentParser(description="Demonstrate RAG-CSD async capabilities.")
    parser.add_argument(
        "--vector-db", "-v", type=str, required=True, 
        help="Path to the vector database."
    )
    parser.add_argument(
        "--output", "-o", type=str, 
        help="Path to save async benchmark results JSON."
    )
    parser.add_argument(
        "--num-users", "-u", type=int, default=3,
        help="Number of concurrent users to simulate."
    )
    
    args = parser.parse_args()
    
    print("🚀 RAG-CSD Async Processing Demonstration")
    print("=" * 80)
    print("This demo showcases the async and parallel processing capabilities")
    print("that make RAG-CSD significantly faster than traditional RAG systems.")
    print("=" * 80)
    
    # Generate test queries
    queries = generate_test_queries()
    print(f"📝 Generated {len(queries)} test queries")
    
    # Run demonstrations
    results = {}
    
    try:
        # Async vs Sync comparison
        sync_time, async_time, async_speedup = await demonstrate_async_vs_sync(
            args.vector_db, queries[:5]
        )
        results['async_vs_sync'] = {
            'sync_time': sync_time,
            'async_time': async_time,
            'speedup': async_speedup
        }
        
        # Batch optimization
        individual_time, batch_time, batch_speedup = await demonstrate_batch_optimization(
            args.vector_db, queries[:5]
        )
        results['batch_optimization'] = {
            'individual_time': individual_time,
            'batch_time': batch_time,
            'speedup': batch_speedup
        }
        
        # Concurrent users
        concurrent_time, total_queries, avg_query_time = await demonstrate_concurrent_users(
            args.vector_db, queries, args.num_users
        )
        results['concurrent_users'] = {
            'total_time': concurrent_time,
            'total_queries': total_queries,
            'avg_query_time': avg_query_time,
            'throughput': total_queries / concurrent_time if concurrent_time > 0 else 0
        }
        
        # Cache sharing
        cache_time, cache_hit_rate = await demonstrate_cache_sharing(
            args.vector_db, queries[:4]
        )
        results['cache_sharing'] = {
            'total_time': cache_time,
            'cache_hit_rate': cache_hit_rate
        }
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 ASYNC PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"🚀 Async vs Sync Speedup:        {async_speedup:.1f}x")
    print(f"⚡ Batch Processing Speedup:     {batch_speedup:.1f}x")
    print(f"👥 Concurrent Users Throughput:  {results['concurrent_users']['throughput']:.1f} queries/sec")
    print(f"💾 Cache Hit Rate:               {cache_hit_rate:.1f}%")
    print(f"🏆 Max Demonstrated Speedup:     {max(async_speedup, batch_speedup):.1f}x")
    print("=" * 80)
    
    # Save results
    if args.output:
        results['summary'] = {
            'max_speedup': max(async_speedup, batch_speedup),
            'total_demonstrations': 4,
            'vector_db_path': args.vector_db
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Async benchmark results saved to {args.output}")


def sync_main():
    """Synchronous wrapper for the async main function."""
    asyncio.run(main())


if __name__ == "__main__":
    sync_main()