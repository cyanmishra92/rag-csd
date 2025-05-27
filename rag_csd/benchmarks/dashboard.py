#!/usr/bin/env python
"""
Comprehensive performance dashboard for RAG-CSD system monitoring and analysis.
This module provides real-time performance tracking, historical analysis, and
comparative benchmarking capabilities.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    system_name: str


@dataclass
class SystemStats:
    """System performance statistics."""
    avg_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    cache_hit_rate: float
    error_rate: float
    total_queries: int


class PerformanceDashboard:
    """
    Comprehensive performance dashboard for monitoring RAG-CSD systems.
    
    Features:
    - Real-time performance tracking
    - Historical trend analysis
    - Multi-system comparison
    - Alert thresholds
    - Export capabilities
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.system_stats = {}
        self.alert_thresholds = {}
        self.lock = threading.Lock()
        
        # Set default alert thresholds
        self.set_alert_threshold("latency", max_value=1.0)  # 1 second
        self.set_alert_threshold("error_rate", max_value=0.05)  # 5%
        self.set_alert_threshold("cache_hit_rate", min_value=0.8)  # 80%
        
    def record_metric(
        self, 
        system_name: str, 
        metric_name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a performance metric."""
        if tags is None:
            tags = {}
            
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=metric_name,
            value=value,
            tags=tags,
            system_name=system_name
        )
        
        with self.lock:
            key = f"{system_name}:{metric_name}"
            self.metrics_history[key].append(metric)
            
    def record_query_performance(
        self, 
        system_name: str, 
        latency: float, 
        cache_hit: bool = False,
        error: bool = False,
        query_type: str = "standard"
    ) -> None:
        """Record query performance metrics."""
        self.record_metric(system_name, "latency", latency, {"type": query_type})
        self.record_metric(system_name, "cache_hit", 1.0 if cache_hit else 0.0)
        self.record_metric(system_name, "error", 1.0 if error else 0.0)
        self.record_metric(system_name, "throughput", 1.0 / latency if latency > 0 else 0.0)
        
    def get_system_stats(self, system_name: str, time_window: Optional[float] = None) -> SystemStats:
        """Get comprehensive statistics for a system."""
        if time_window is None:
            time_window = 3600  # 1 hour default
            
        cutoff_time = time.time() - time_window
        
        # Collect metrics within time window
        latencies = []
        cache_hits = []
        errors = []
        
        with self.lock:
            for key, metrics in self.metrics_history.items():
                if not key.startswith(f"{system_name}:"):
                    continue
                    
                _, metric_name = key.split(":", 1)
                
                for metric in metrics:
                    if metric.timestamp < cutoff_time:
                        continue
                        
                    if metric_name == "latency":
                        latencies.append(metric.value)
                    elif metric_name == "cache_hit":
                        cache_hits.append(metric.value)
                    elif metric_name == "error":
                        errors.append(metric.value)
        
        # Calculate statistics
        if latencies:
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput = len(latencies) / time_window
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = throughput = 0.0
            
        cache_hit_rate = np.mean(cache_hits) if cache_hits else 0.0
        error_rate = np.mean(errors) if errors else 0.0
        total_queries = len(latencies)
        
        return SystemStats(
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=throughput,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate,
            total_queries=total_queries
        )
        
    def set_alert_threshold(
        self, 
        metric_name: str, 
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None
    ) -> None:
        """Set alert thresholds for a metric."""
        self.alert_thresholds[metric_name] = {
            "min": min_value,
            "max": max_value
        }
        
    def check_alerts(self, system_name: str) -> List[Dict[str, Any]]:
        """Check for threshold violations and return alerts."""
        alerts = []
        stats = self.get_system_stats(system_name, time_window=300)  # 5 minute window
        
        # Check latency threshold
        if "latency" in self.alert_thresholds:
            threshold = self.alert_thresholds["latency"]
            if threshold["max"] and stats.avg_latency > threshold["max"]:
                alerts.append({
                    "type": "HIGH_LATENCY",
                    "system": system_name,
                    "current_value": stats.avg_latency,
                    "threshold": threshold["max"],
                    "severity": "WARNING"
                })
                
        # Check error rate threshold
        if "error_rate" in self.alert_thresholds:
            threshold = self.alert_thresholds["error_rate"]
            if threshold["max"] and stats.error_rate > threshold["max"]:
                alerts.append({
                    "type": "HIGH_ERROR_RATE",
                    "system": system_name,
                    "current_value": stats.error_rate,
                    "threshold": threshold["max"],
                    "severity": "ERROR"
                })
                
        # Check cache hit rate threshold
        if "cache_hit_rate" in self.alert_thresholds:
            threshold = self.alert_thresholds["cache_hit_rate"]
            if threshold["min"] and stats.cache_hit_rate < threshold["min"]:
                alerts.append({
                    "type": "LOW_CACHE_HIT_RATE",
                    "system": system_name,
                    "current_value": stats.cache_hit_rate,
                    "threshold": threshold["min"],
                    "severity": "WARNING"
                })
                
        return alerts
        
    def plot_real_time_metrics(
        self, 
        systems: List[str], 
        metric_name: str = "latency",
        time_window: int = 300,
        save_path: Optional[str] = None
    ) -> None:
        """Plot real-time metrics for specified systems."""
        plt.figure(figsize=(12, 6))
        
        cutoff_time = time.time() - time_window
        
        with self.lock:
            for system_name in systems:
                key = f"{system_name}:{metric_name}"
                if key not in self.metrics_history:
                    continue
                    
                metrics = self.metrics_history[key]
                
                # Filter by time window
                recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                
                if not recent_metrics:
                    continue
                    
                timestamps = [m.timestamp for m in recent_metrics]
                values = [m.value for m in recent_metrics]
                
                # Convert timestamps to relative time (seconds ago)
                relative_times = [(time.time() - t) for t in timestamps]
                
                plt.plot(relative_times, values, label=system_name, marker='o', alpha=0.7)
        
        plt.xlabel("Time (seconds ago)")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"Real-time {metric_name.replace('_', ' ').title()} Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_xaxis()  # Most recent on the left
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def plot_system_comparison(
        self, 
        systems: List[str], 
        time_window: float = 3600,
        save_path: Optional[str] = None
    ) -> None:
        """Create comprehensive system comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("System Performance Comparison", fontsize=16)
        
        system_data = {}
        for system in systems:
            stats = self.get_system_stats(system, time_window)
            system_data[system] = stats
            
        # Latency comparison
        ax1 = axes[0, 0]
        latency_metrics = ["avg_latency", "p95_latency", "p99_latency"]
        x = np.arange(len(systems))
        width = 0.25
        
        for i, metric in enumerate(latency_metrics):
            values = [getattr(system_data[sys], metric) for sys in systems]
            ax1.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
            
        ax1.set_xlabel("Systems")
        ax1.set_ylabel("Latency (seconds)")
        ax1.set_title("Latency Metrics")
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(systems)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput comparison
        ax2 = axes[0, 1]
        throughputs = [system_data[sys].throughput for sys in systems]
        bars = ax2.bar(systems, throughputs, color='green', alpha=0.7)
        ax2.set_ylabel("Throughput (queries/sec)")
        ax2.set_title("Throughput Comparison")
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Cache hit rate comparison
        ax3 = axes[1, 0]
        cache_rates = [system_data[sys].cache_hit_rate * 100 for sys in systems]
        bars = ax3.bar(systems, cache_rates, color='blue', alpha=0.7)
        ax3.set_ylabel("Cache Hit Rate (%)")
        ax3.set_title("Cache Hit Rate")
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, cache_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Error rate comparison
        ax4 = axes[1, 1]
        error_rates = [system_data[sys].error_rate * 100 for sys in systems]
        bars = ax4.bar(systems, error_rates, color='red', alpha=0.7)
        ax4.set_ylabel("Error Rate (%)")
        ax4.set_title("Error Rate")
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, error_rates):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def generate_performance_report(
        self, 
        systems: List[str], 
        time_window: float = 3600,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "time_window_hours": time_window / 3600,
            "systems": {}
        }
        
        for system in systems:
            stats = self.get_system_stats(system, time_window)
            alerts = self.check_alerts(system)
            
            report["systems"][system] = {
                "stats": asdict(stats),
                "alerts": alerts,
                "health_score": self._calculate_health_score(stats)
            }
            
        # Calculate comparative metrics
        if len(systems) > 1:
            report["comparisons"] = self._calculate_comparisons(
                [report["systems"][sys]["stats"] for sys in systems], 
                systems
            )
            
        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def _calculate_health_score(self, stats: SystemStats) -> float:
        """Calculate overall health score (0-100) for a system."""
        score = 100.0
        
        # Penalize high latency (target: < 0.1s)
        if stats.avg_latency > 0.1:
            score -= min(50, (stats.avg_latency - 0.1) * 500)
            
        # Penalize low cache hit rate (target: > 80%)
        if stats.cache_hit_rate < 0.8:
            score -= (0.8 - stats.cache_hit_rate) * 100
            
        # Penalize errors (target: 0%)
        score -= stats.error_rate * 100
        
        # Penalize low throughput (relative)
        if stats.throughput < 1.0:  # Less than 1 query/sec
            score -= (1.0 - stats.throughput) * 20
            
        return max(0.0, min(100.0, score))
        
    def _calculate_comparisons(
        self, 
        stats_list: List[Dict], 
        system_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate comparative metrics between systems."""
        comparisons = {}
        
        # Find baseline (system with median performance)
        latencies = [s["avg_latency"] for s in stats_list]
        baseline_idx = np.argsort(latencies)[len(latencies) // 2]
        baseline_name = system_names[baseline_idx]
        baseline_stats = stats_list[baseline_idx]
        
        comparisons["baseline"] = baseline_name
        comparisons["speedups"] = {}
        
        for i, (system, stats) in enumerate(zip(system_names, stats_list)):
            if system == baseline_name:
                speedup = 1.0
            else:
                speedup = baseline_stats["avg_latency"] / stats["avg_latency"] if stats["avg_latency"] > 0 else float('inf')
                
            comparisons["speedups"][system] = speedup
            
        return comparisons
        
    def export_metrics(
        self, 
        output_file: str, 
        systems: Optional[List[str]] = None,
        time_window: Optional[float] = None
    ) -> None:
        """Export raw metrics data to JSON file."""
        if time_window is None:
            cutoff_time = 0
        else:
            cutoff_time = time.time() - time_window
            
        export_data = {
            "export_timestamp": time.time(),
            "time_window": time_window,
            "metrics": {}
        }
        
        with self.lock:
            for key, metrics in self.metrics_history.items():
                system_name, metric_name = key.split(":", 1)
                
                if systems and system_name not in systems:
                    continue
                    
                filtered_metrics = [
                    asdict(m) for m in metrics 
                    if m.timestamp >= cutoff_time
                ]
                
                if filtered_metrics:
                    if system_name not in export_data["metrics"]:
                        export_data["metrics"][system_name] = {}
                    export_data["metrics"][system_name][metric_name] = filtered_metrics
                    
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)
            
    def clear_history(self, older_than: Optional[float] = None) -> None:
        """Clear metrics history older than specified time."""
        if older_than is None:
            older_than = time.time() - 86400  # 24 hours
            
        with self.lock:
            for key in self.metrics_history:
                # Filter out old metrics
                self.metrics_history[key] = deque(
                    [m for m in self.metrics_history[key] if m.timestamp >= older_than],
                    maxlen=self.history_size
                )