#!/usr/bin/env python
"""
Performance dashboard demonstration and CLI tool.
This script shows how to use the RAG-CSD performance dashboard for
real-time monitoring and analysis.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import random
from typing import List, Dict

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_csd.benchmarks.dashboard import PerformanceDashboard
from rag_csd.pipeline import RAGCSDPipeline
from rag_csd.benchmarks.baseline_systems import VanillaRAG, PipeRAGLike
from rag_csd.async_interface import AsyncRAGCSD


class DashboardDemo:
    """Demonstration of the performance dashboard with live data."""
    
    def __init__(self, vector_db_path: str):
        self.vector_db_path = vector_db_path
        self.dashboard = PerformanceDashboard(history_size=500)
        self.systems = {}
        self.test_queries = self._generate_test_queries()
        
    def _generate_test_queries(self) -> List[str]:
        """Generate diverse test queries for simulation."""
        return [
            "What is computational storage?",
            "How does vector similarity search work?",
            "Benefits of retrieval-augmented generation",
            "Explain FAISS indexing algorithms",
            "Text embedding model architectures",
            "RAG system optimization strategies",
            "Cache efficiency in NLP systems",
            "Transformer attention mechanisms",
            "Semantic search implementation details",
            "Performance tuning for vector databases",
            "Batch processing optimization techniques",
            "Memory management in ML systems",
            "Distributed computing for AI workloads",
            "Real-time inference optimization",
            "Model serving infrastructure"
        ]
    
    def initialize_systems(self) -> None:
        """Initialize RAG systems for monitoring."""
        print("🔧 Initializing systems for dashboard monitoring...")
        
        try:
            self.systems["RAG-CSD"] = RAGCSDPipeline(self.vector_db_path)
            print("  ✅ RAG-CSD initialized")
        except Exception as e:
            print(f"  ❌ RAG-CSD failed: {e}")
            
        try:
            self.systems["VanillaRAG"] = VanillaRAG(self.vector_db_path)
            print("  ✅ VanillaRAG initialized")
        except Exception as e:
            print(f"  ❌ VanillaRAG failed: {e}")
            
        try:
            self.systems["PipeRAG-like"] = PipeRAGLike(self.vector_db_path)
            print("  ✅ PipeRAG-like initialized")
        except Exception as e:
            print(f"  ❌ PipeRAG-like failed: {e}")
    
    def simulate_workload(self, duration: int = 300, query_rate: float = 1.0) -> None:
        """Simulate realistic workload for dashboard demonstration."""
        print(f"🚀 Simulating workload for {duration} seconds at {query_rate} queries/sec...")
        
        start_time = time.time()
        query_count = 0
        
        while time.time() - start_time < duration:
            # Select random query and system
            query = random.choice(self.test_queries)
            system_name = random.choice(list(self.systems.keys()))
            system = self.systems[system_name]
            
            try:
                # Execute query and measure performance
                query_start = time.time()
                result = system.query(query, top_k=5)
                query_end = time.time()
                
                latency = query_end - query_start
                cache_hit = result.get('cache_hit', False)
                
                # Record metrics
                self.dashboard.record_query_performance(
                    system_name=system_name,
                    latency=latency,
                    cache_hit=cache_hit,
                    error=False,
                    query_type="simulation"
                )
                
                query_count += 1
                
                # Add some realistic variation
                if random.random() < 0.05:  # 5% chance of longer query
                    time.sleep(random.uniform(0.1, 0.3))
                    
            except Exception as e:
                # Record error
                self.dashboard.record_query_performance(
                    system_name=system_name,
                    latency=1.0,  # Default penalty
                    cache_hit=False,
                    error=True,
                    query_type="simulation"
                )
                print(f"  ❌ Error in {system_name}: {e}")
            
            # Control query rate
            time.sleep(1.0 / query_rate)
            
            # Print progress every 30 seconds
            if query_count % (30 * query_rate) == 0:
                elapsed = time.time() - start_time
                print(f"  📊 Progress: {elapsed:.0f}s elapsed, {query_count} queries executed")
        
        print(f"✅ Simulation complete: {query_count} queries in {duration} seconds")
    
    def generate_dashboard_reports(self, output_dir: str) -> None:
        """Generate comprehensive dashboard reports and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        system_names = list(self.systems.keys())
        
        print(f"📊 Generating dashboard reports in {output_dir}...")
        
        # Real-time metrics plot
        print("  📈 Generating real-time latency plot...")
        self.dashboard.plot_real_time_metrics(
            systems=system_names,
            metric_name="latency",
            time_window=300,
            save_path=os.path.join(output_dir, "real_time_latency.png")
        )
        
        # System comparison plot
        print("  📊 Generating system comparison plot...")
        self.dashboard.plot_system_comparison(
            systems=system_names,
            time_window=600,
            save_path=os.path.join(output_dir, "system_comparison.png")
        )
        
        # Performance report
        print("  📄 Generating performance report...")
        report = self.dashboard.generate_performance_report(
            systems=system_names,
            time_window=600,
            output_file=os.path.join(output_dir, "performance_report.json")
        )
        
        # Export raw metrics
        print("  💾 Exporting raw metrics...")
        self.dashboard.export_metrics(
            output_file=os.path.join(output_dir, "raw_metrics.json"),
            systems=system_names,
            time_window=600
        )
        
        # Print summary
        self._print_dashboard_summary(report, system_names)
        
        print(f"✅ Dashboard reports generated in {output_dir}")
    
    def _print_dashboard_summary(self, report: Dict, system_names: List[str]) -> None:
        """Print dashboard summary to console."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE DASHBOARD SUMMARY")
        print("=" * 80)
        
        for system_name in system_names:
            if system_name not in report["systems"]:
                continue
                
            system_data = report["systems"][system_name]
            stats = system_data["stats"]
            health_score = system_data["health_score"]
            alerts = system_data["alerts"]
            
            print(f"\n🔍 {system_name}:")
            print(f"  Health Score:     {health_score:.1f}/100")
            print(f"  Avg Latency:      {stats['avg_latency']:.3f}s")
            print(f"  P95 Latency:      {stats['p95_latency']:.3f}s")
            print(f"  Throughput:       {stats['throughput']:.1f} queries/sec")
            print(f"  Cache Hit Rate:   {stats['cache_hit_rate']*100:.1f}%")
            print(f"  Error Rate:       {stats['error_rate']*100:.1f}%")
            print(f"  Total Queries:    {stats['total_queries']}")
            
            if alerts:
                print(f"  🚨 Alerts: {len(alerts)}")
                for alert in alerts:
                    print(f"    - {alert['type']}: {alert['current_value']:.3f}")
        
        # Comparative metrics
        if "comparisons" in report:
            comparisons = report["comparisons"]
            print(f"\n🏆 Performance Comparisons (vs {comparisons['baseline']}):")
            
            for system, speedup in comparisons["speedups"].items():
                if speedup == 1.0:
                    print(f"  {system:15}: Baseline")
                elif speedup > 1.0:
                    print(f"  {system:15}: {speedup:.1f}x faster")
                else:
                    print(f"  {system:15}: {1/speedup:.1f}x slower")
        
        print("=" * 80)
    
    def monitor_live_performance(self, duration: int = 60) -> None:
        """Monitor live performance with periodic updates."""
        print(f"👁️  Starting live performance monitoring for {duration} seconds...")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Run a test query on each system
                query = random.choice(self.test_queries)
                
                print(f"🔍 Testing query: {query[:50]}...")
                
                for system_name, system in self.systems.items():
                    try:
                        query_start = time.time()
                        result = system.query(query, top_k=5)
                        latency = time.time() - query_start
                        
                        cache_hit = result.get('cache_hit', False)
                        
                        # Record metrics
                        self.dashboard.record_query_performance(
                            system_name=system_name,
                            latency=latency,
                            cache_hit=cache_hit,
                            error=False,
                            query_type="live_monitoring"
                        )
                        
                        # Check for alerts
                        alerts = self.dashboard.check_alerts(system_name)
                        alert_status = f" 🚨({len(alerts)})" if alerts else ""
                        
                        print(f"  {system_name:15}: {latency:.3f}s (cache: {cache_hit}){alert_status}")
                        
                    except Exception as e:
                        print(f"  {system_name:15}: ERROR - {e}")
                        self.dashboard.record_query_performance(
                            system_name=system_name,
                            latency=1.0,
                            cache_hit=False,
                            error=True,
                            query_type="live_monitoring"
                        )
                
                print("")
                time.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️  Live monitoring stopped by user")
        
        print("✅ Live monitoring completed")


def main():
    """Main entry point for dashboard demonstration."""
    parser = argparse.ArgumentParser(description="RAG-CSD Performance Dashboard Demo")
    parser.add_argument(
        "--vector-db", "-v", type=str, required=True,
        help="Path to the vector database."
    )
    parser.add_argument(
        "--mode", "-m", type=str, choices=["simulate", "live", "both"], default="both",
        help="Dashboard demonstration mode."
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=120,
        help="Duration for simulation or live monitoring (seconds)."
    )
    parser.add_argument(
        "--query-rate", "-r", type=float, default=2.0,
        help="Query rate for simulation (queries per second)."
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="dashboard_output/",
        help="Output directory for reports and visualizations."
    )
    
    args = parser.parse_args()
    
    print("📊 RAG-CSD Performance Dashboard Demonstration")
    print("=" * 80)
    print("This demo showcases the comprehensive performance monitoring")
    print("and analysis capabilities of the RAG-CSD dashboard system.")
    print("=" * 80)
    
    # Initialize demo
    demo = DashboardDemo(args.vector_db)
    demo.initialize_systems()
    
    if not demo.systems:
        print("❌ No systems initialized. Cannot proceed with demonstration.")
        return
    
    # Run demonstration based on mode
    if args.mode in ["simulate", "both"]:
        print(f"\n🔄 Running workload simulation...")
        demo.simulate_workload(duration=args.duration, query_rate=args.query_rate)
        
        print(f"\n📊 Generating dashboard reports...")
        demo.generate_dashboard_reports(args.output_dir)
    
    if args.mode in ["live", "both"]:
        if args.mode == "both":
            print(f"\n⏸️  Pausing 10 seconds before live monitoring...")
            time.sleep(10)
            
        print(f"\n👁️  Starting live performance monitoring...")
        demo.monitor_live_performance(duration=args.duration // 2)
    
    print(f"\n🎯 Dashboard demonstration completed!")
    print(f"📁 Check {args.output_dir} for generated reports and visualizations")


if __name__ == "__main__":
    main()