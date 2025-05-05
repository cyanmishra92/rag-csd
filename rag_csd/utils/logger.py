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