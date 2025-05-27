"""
Logging module for RAG-CSD.
This module provides a consistent logging interface.
"""

import logging
import os
import sys
from typing import Dict, Optional

# Define log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    formatter: Optional[str] = None,
) -> logging.Logger:
    """
    Setup the root logger with the specified configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, no file logging.
        console_output: Whether to output logs to console.
        formatter: Custom formatter string. If None, uses default.
        
    Returns:
        Configured logger.
    """
    # Convert string level to logging level
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if formatter is None:
        formatter = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_formatter = logging.Formatter(formatter)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
        self.metrics = {}
    
    def log_timing(self, operation: str, duration: float, context: str = "") -> None:
        """
        Log timing information for an operation.
        
        Args:
            operation: Name of the operation.
            duration: Duration in seconds.
            context: Additional context information.
        """
        context_str = f" ({context})" if context else ""
        self.logger.info(f"TIMING: {operation}{context_str}: {duration:.3f}s")
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def log_throughput(self, operation: str, count: int, duration: float, unit: str = "items") -> None:
        """
        Log throughput information.
        
        Args:
            operation: Name of the operation.
            count: Number of items processed.
            duration: Duration in seconds.
            unit: Unit name for the items.
        """
        throughput = count / duration if duration > 0 else 0
        self.logger.info(f"THROUGHPUT: {operation}: {throughput:.1f} {unit}/s "
                        f"({count} {unit} in {duration:.3f}s)")
    
    def log_memory_usage(self, operation: str, memory_mb: float) -> None:
        """
        Log memory usage information.
        
        Args:
            operation: Name of the operation.
            memory_mb: Memory usage in MB.
        """
        self.logger.info(f"MEMORY: {operation}: {memory_mb:.1f} MB")
    
    def get_avg_timing(self, operation: str) -> float:
        """
        Get average timing for an operation.
        
        Args:
            operation: Name of the operation.
            
        Returns:
            Average timing in seconds, or 0 if no data.
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()


# Global performance logger instance
performance_logger = PerformanceLogger()