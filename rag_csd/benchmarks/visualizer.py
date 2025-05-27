"""
Performance visualization and analysis tools for RAG-CSD benchmarks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
import time

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceVisualizer:
    """Visualization tools for RAG system performance comparison."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots.
        """
        self.figsize = figsize
        self.results_cache = {}
    
    def plot_latency_comparison(
        self, 
        results: Dict[str, List[Dict]], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot latency comparison between different RAG systems.
        
        Args:
            results: Dictionary with system names as keys and result lists as values.
            save_path: Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        systems = []
        latencies = []
        query_counts = []
        
        for system_name, system_results in results.items():
            processing_times = [r.get("processing_time", 0) for r in system_results]
            systems.extend([system_name] * len(processing_times))
            latencies.extend(processing_times)
            query_counts.append(len(processing_times))
        
        df = pd.DataFrame({
            "System": systems,
            "Latency (s)": latencies
        })
        
        # Box plot for latency distribution
        sns.boxplot(data=df, x="System", y="Latency (s)", ax=ax1)
        ax1.set_title("Latency Distribution by System")
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar plot for average latency
        avg_latencies = df.groupby("System")["Latency (s)"].mean()
        avg_latencies.plot(kind='bar', ax=ax2, color=sns.color_palette("husl", len(avg_latencies)))
        ax2.set_title("Average Latency by System")
        ax2.set_ylabel("Average Latency (s)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latency comparison saved to {save_path}")
        
        plt.show()
    
    def plot_throughput_comparison(
        self,
        results: Dict[str, List[Dict]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot throughput comparison between different RAG systems.
        
        Args:
            results: Dictionary with system names as keys and result lists as values.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate throughput for each system
        throughputs = {}
        
        for system_name, system_results in results.items():
            if not system_results:
                continue
            
            total_time = sum(r.get("processing_time", 0) for r in system_results)
            num_queries = len(system_results)
            
            if total_time > 0:
                throughput = num_queries / total_time
                throughputs[system_name] = throughput
        
        # Create bar plot
        systems = list(throughputs.keys())
        values = list(throughputs.values())
        
        bars = ax.bar(systems, values, color=sns.color_palette("husl", len(systems)))
        ax.set_title("Throughput Comparison (Queries per Second)")
        ax.set_ylabel("Throughput (QPS)")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Throughput comparison saved to {save_path}")
        
        plt.show()
    
    def plot_scalability_analysis(
        self,
        scalability_results: Dict[str, Dict[int, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot scalability analysis showing performance vs. query count.
        
        Args:
            scalability_results: Dict with system names and query_count -> time mappings.
            save_path: Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot processing time vs query count
        for system_name, data in scalability_results.items():
            query_counts = sorted(data.keys())
            times = [data[count] for count in query_counts]
            
            ax1.plot(query_counts, times, marker='o', label=system_name, linewidth=2)
        
        ax1.set_xlabel("Number of Queries")
        ax1.set_ylabel("Total Processing Time (s)")
        ax1.set_title("Processing Time vs Query Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot throughput vs query count
        for system_name, data in scalability_results.items():
            query_counts = sorted(data.keys())
            throughputs = [count / data[count] for count in query_counts]
            
            ax2.plot(query_counts, throughputs, marker='s', label=system_name, linewidth=2)
        
        ax2.set_xlabel("Number of Queries")
        ax2.set_ylabel("Throughput (QPS)")
        ax2.set_title("Throughput vs Query Count")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scalability analysis saved to {save_path}")
        
        plt.show()
    
    def plot_optimization_impact(
        self,
        baseline_results: Dict[str, Any],
        optimized_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot the impact of optimizations on RAG-CSD performance.
        
        Args:
            baseline_results: Results before optimization.
            optimized_results: Results after optimization.
            save_path: Path to save the plot.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ["avg_latency", "throughput", "cache_hit_rate", "memory_usage"]
        metric_labels = ["Average Latency (s)", "Throughput (QPS)", "Cache Hit Rate (%)", "Memory Usage (MB)"]
        
        # Extract metrics
        baseline_values = []
        optimized_values = []
        
        for metric in metrics:
            baseline_val = baseline_results.get(metric, 0)
            optimized_val = optimized_results.get(metric, 0)
            
            baseline_values.append(baseline_val)
            optimized_values.append(optimized_val)
        
        # Plot comparisons
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (ax, metric, label) in enumerate(zip(axes, metrics, metric_labels)):
            baseline_val = baseline_values[i]
            optimized_val = optimized_values[i]
            
            bars = ax.bar(["Baseline", "Optimized"], [baseline_val, optimized_val],
                         color=['lightcoral', 'lightgreen'])
            
            ax.set_title(f"{label}")
            ax.set_ylabel(label)
            
            # Add improvement percentage
            if baseline_val > 0:
                if metric == "avg_latency":  # Lower is better
                    improvement = (baseline_val - optimized_val) / baseline_val * 100
                else:  # Higher is better
                    improvement = (optimized_val - baseline_val) / baseline_val * 100
                
                ax.text(0.5, max(baseline_val, optimized_val) * 1.1,
                       f'{improvement:+.1f}%', ha='center', fontweight='bold',
                       color='green' if improvement > 0 else 'red')
            
            # Add value labels
            for bar, value in zip(bars, [baseline_val, optimized_val]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{value:.2f}', ha='center', va='center', fontweight='bold')
        
        plt.suptitle("RAG-CSD Optimization Impact", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization impact plot saved to {save_path}")
        
        plt.show()
    
    def plot_system_comparison_radar(
        self,
        systems_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a radar chart comparing multiple systems across different metrics.
        
        Args:
            systems_metrics: Dict with system names and their metric values.
            save_path: Path to save the plot.
        """
        from math import pi
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Get all unique metrics
        all_metrics = set()
        for metrics in systems_metrics.values():
            all_metrics.update(metrics.keys())
        
        metrics = sorted(list(all_metrics))
        num_metrics = len(metrics)
        
        # Calculate angles for each metric
        angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each system
        colors = sns.color_palette("husl", len(systems_metrics))
        
        for i, (system_name, system_metrics) in enumerate(systems_metrics.items()):
            values = []
            for metric in metrics:
                # Normalize values to 0-1 scale
                value = system_metrics.get(metric, 0)
                # Simple normalization - can be enhanced
                normalized_value = min(value / 100, 1.0) if value > 0 else 0
                values.append(normalized_value)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=system_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("RAG Systems Comparison (Normalized Metrics)", size=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Radar chart saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Complete benchmark results.
            output_path: Directory to save the report.
        """
        import os
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Generate individual plots
        if "latency_results" in results:
            self.plot_latency_comparison(
                results["latency_results"],
                os.path.join(output_path, "latency_comparison.png")
            )
        
        if "throughput_results" in results:
            self.plot_throughput_comparison(
                results["throughput_results"],
                os.path.join(output_path, "throughput_comparison.png")
            )
        
        if "scalability_results" in results:
            self.plot_scalability_analysis(
                results["scalability_results"],
                os.path.join(output_path, "scalability_analysis.png")
            )
        
        # Generate summary report
        report_path = os.path.join(output_path, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Performance report generated in {output_path}")
    
    def create_performance_dashboard(
        self,
        live_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a live performance dashboard.
        
        Args:
            live_results: Real-time performance data.
            save_path: Path to save the dashboard.
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a complex dashboard layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # System metrics over time
        ax1 = fig.add_subplot(gs[0, :2])
        if "timeline_metrics" in live_results:
            timeline = live_results["timeline_metrics"]
            times = list(timeline.keys())
            latencies = [metrics.get("latency", 0) for metrics in timeline.values()]
            throughputs = [metrics.get("throughput", 0) for metrics in timeline.values()]
            
            ax1_twin = ax1.twinx()
            ax1.plot(times, latencies, 'b-', label="Latency (s)")
            ax1_twin.plot(times, throughputs, 'r-', label="Throughput (QPS)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Latency (s)", color='b')
            ax1_twin.set_ylabel("Throughput (QPS)", color='r')
            ax1.set_title("Performance Metrics Over Time")
        
        # Current system status
        ax2 = fig.add_subplot(gs[0, 2:])
        if "current_status" in live_results:
            status = live_results["current_status"]
            metrics = list(status.keys())
            values = list(status.values())
            
            bars = ax2.barh(metrics, values)
            ax2.set_title("Current System Status")
            
            # Color bars based on performance
            for bar, value in zip(bars, values):
                if value > 0.8:  # Good performance
                    bar.set_color('green')
                elif value > 0.5:  # Medium performance
                    bar.set_color('orange')
                else:  # Poor performance
                    bar.set_color('red')
        
        plt.suptitle("RAG-CSD Performance Dashboard", fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance dashboard saved to {save_path}")
        
        plt.show()