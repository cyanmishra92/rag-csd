"""
Performance monitoring and metrics module for RAG-CSD.
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Deque
import statistics

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricDataPoint:
    """Data structure for a single metric measurement."""
    timestamp: float
    value: float
    context: Optional[str] = None


class MetricsCollector:
    """Centralized metrics collection and monitoring."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize the metrics collector."""
        self.max_history = max_history
        self.metrics: Dict[str, Deque[MetricDataPoint]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        logger.info(f"MetricsCollector initialized with max_history={max_history}")
    
    def record_timing(self, metric_name: str, duration: float, context: str = None) -> None:
        """Record a timing metric."""
        with self._lock:
            data_point = MetricDataPoint(
                timestamp=time.time(),
                value=duration,
                context=context
            )
            self.metrics[f"timing.{metric_name}"].append(data_point)
    
    def record_count(self, metric_name: str, value: int = 1) -> None:
        """Record a count metric."""
        with self._lock:
            self.counters[metric_name] += value
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        with self._lock:
            if metric_name not in self.metrics:
                return {}
            
            values = [dp.value for dp in self.metrics[metric_name]]
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
            }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "timing_stats": {},
            }
            
            # Get stats for timing metrics
            for metric_name in self.metrics:
                if metric_name.startswith("timing."):
                    clean_name = metric_name[7:]  # Remove "timing." prefix
                    summary["timing_stats"][clean_name] = self.get_metric_stats(metric_name)
            
            return summary


class PerformanceMonitor:
    """Context manager for monitoring operation performance."""
    
    def __init__(self, metrics_collector: MetricsCollector, operation_name: str, context: str = None):
        """Initialize the performance monitor."""
        self.metrics_collector = metrics_collector
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and record metrics."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timing(
                self.operation_name, 
                duration, 
                self.context
            )
            
            # Count the operation
            self.metrics_collector.record_count(f"operations.{self.operation_name}")
            
            # If there was an exception, count it
            if exc_type is not None:
                self.metrics_collector.record_count(f"errors.{self.operation_name}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def monitor_operation(operation_name: str, context: str = None):
    """Convenience function to create a performance monitor."""
    return PerformanceMonitor(metrics_collector, operation_name, context)