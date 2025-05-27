"""
Error handling utilities for RAG-CSD.
This module provides custom exceptions and error handling utilities.
"""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Dict, Optional, Type, Union

from rag_csd.utils.logger import get_logger

logger = get_logger(__name__)


# Custom exceptions
class RAGCSDError(Exception):
    """Base exception for RAG-CSD related errors."""
    pass


class ModelLoadError(RAGCSDError):
    """Exception raised when model loading fails."""
    pass


class EmbeddingError(RAGCSDError):
    """Exception raised when embedding generation fails."""
    pass


class VectorStoreError(RAGCSDError):
    """Exception raised when vector store operations fail."""
    pass


class ConfigurationError(RAGCSDError):
    """Exception raised when configuration is invalid."""
    pass


class CSDError(RAGCSDError):
    """Exception raised when CSD operations fail."""
    pass


class TextProcessingError(RAGCSDError):
    """Exception raised when text processing fails."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Factor by which delay increases after each retry.
        exceptions: Tuple of exceptions to catch and retry on.
    
    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, "
                                 f"retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def handle_exceptions(
    default_return: Any = None,
    log_traceback: bool = True,
    reraise: bool = True,
    exception_map: Optional[Dict[Type[Exception], Type[RAGCSDError]]] = None
) -> Callable:
    """
    Decorator that handles exceptions with logging and optional re-raising.
    
    Args:
        default_return: Default value to return if exception is caught and not re-raised.
        log_traceback: Whether to log the full traceback.
        reraise: Whether to re-raise the exception after logging.
        exception_map: Map of exceptions to custom RAG-CSD exceptions.
    
    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception
                error_msg = f"Exception in {func.__name__}: {str(e)}"
                
                if log_traceback:
                    logger.error(error_msg, exc_info=True)
                else:
                    logger.error(error_msg)
                
                # Map to custom exception if provided
                if exception_map and type(e) in exception_map:
                    custom_exception = exception_map[type(e)]
                    mapped_exception = custom_exception(f"{func.__name__}: {str(e)}")
                    
                    if reraise:
                        raise mapped_exception from e
                    else:
                        logger.warning(f"Returning default value due to exception: {mapped_exception}")
                        return default_return
                
                # Re-raise or return default
                if reraise:
                    raise
                else:
                    logger.warning(f"Returning default value due to exception: {e}")
                    return default_return
        
        return wrapper
    return decorator


def validate_config(config: Dict, required_keys: list, section: str = "") -> None:
    """
    Validate that a configuration dictionary contains required keys.
    
    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required keys.
        section: Configuration section name for error messages.
    
    Raises:
        ConfigurationError: If required keys are missing.
    """
    missing_keys = []
    
    for key in required_keys:
        if '.' in key:
            # Handle nested keys like 'embedding.model'
            keys = key.split('.')
            current = config
            
            try:
                for k in keys:
                    current = current[k]
            except (KeyError, TypeError):
                missing_keys.append(key)
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        section_prefix = f"[{section}] " if section else ""
        raise ConfigurationError(
            f"{section_prefix}Missing required configuration keys: {missing_keys}"
        )


def validate_file_path(file_path: str, must_exist: bool = True) -> None:
    """
    Validate that a file path is valid and optionally exists.
    
    Args:
        file_path: Path to validate.
        must_exist: Whether the file must exist.
    
    Raises:
        ConfigurationError: If the file path is invalid.
    """
    import os
    
    if not file_path:
        raise ConfigurationError("File path cannot be empty")
    
    if must_exist and not os.path.exists(file_path):
        raise ConfigurationError(f"File does not exist: {file_path}")
    
    if must_exist and not os.path.isfile(file_path):
        raise ConfigurationError(f"Path is not a file: {file_path}")


def validate_directory_path(dir_path: str, must_exist: bool = True, create_if_missing: bool = False) -> None:
    """
    Validate that a directory path is valid and optionally exists.
    
    Args:
        dir_path: Directory path to validate.
        must_exist: Whether the directory must exist.
        create_if_missing: Whether to create the directory if it doesn't exist.
    
    Raises:
        ConfigurationError: If the directory path is invalid.
    """
    import os
    
    if not dir_path:
        raise ConfigurationError("Directory path cannot be empty")
    
    if not os.path.exists(dir_path):
        if create_if_missing:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except OSError as e:
                raise ConfigurationError(f"Failed to create directory {dir_path}: {e}")
        elif must_exist:
            raise ConfigurationError(f"Directory does not exist: {dir_path}")
    elif not os.path.isdir(dir_path):
        raise ConfigurationError(f"Path is not a directory: {dir_path}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value to return if denominator is zero.
    
    Returns:
        Result of division or default value.
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def log_function_call(include_args: bool = False, include_result: bool = False) -> Callable:
    """
    Decorator that logs function calls.
    
    Args:
        include_args: Whether to include function arguments in the log.
        include_result: Whether to include the return value in the log.
    
    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # Log function call
            if include_args:
                args_str = f"args={args}, kwargs={kwargs}"
                logger.debug(f"Calling {func.__name__}({args_str})")
            else:
                logger.debug(f"Calling {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Log successful completion
                if include_result:
                    logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s, result={result}")
                else:
                    logger.debug(f"Completed {func.__name__} in {elapsed:.3f}s")
                
                return result
            
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


class ErrorCollector:
    """Utility class to collect and manage multiple errors."""
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, error: Union[str, Exception], context: str = "") -> None:
        """
        Add an error to the collection.
        
        Args:
            error: Error message or exception.
            context: Additional context for the error.
        """
        if isinstance(error, Exception):
            error_msg = f"{type(error).__name__}: {str(error)}"
        else:
            error_msg = str(error)
        
        if context:
            error_msg = f"[{context}] {error_msg}"
        
        self.errors.append(error_msg)
        logger.warning(f"Error collected: {error_msg}")
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def get_errors(self) -> list:
        """Get the list of collected errors."""
        return self.errors.copy()
    
    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
    
    def raise_if_errors(self, exception_class: Type[Exception] = RAGCSDError) -> None:
        """
        Raise an exception if any errors have been collected.
        
        Args:
            exception_class: Exception class to raise.
        
        Raises:
            exception_class: If errors have been collected.
        """
        if self.has_errors():
            error_summary = f"Multiple errors occurred: {'; '.join(self.errors)}"
            raise exception_class(error_summary)
    
    def log_summary(self, level: str = "error") -> None:
        """
        Log a summary of all collected errors.
        
        Args:
            level: Log level to use.
        """
        if not self.has_errors():
            return
        
        log_func = getattr(logger, level, logger.error)
        log_func(f"Error summary: {len(self.errors)} errors collected:")
        
        for i, error in enumerate(self.errors, 1):
            log_func(f"  {i}. {error}")


# Exception mapping for common errors
COMMON_EXCEPTION_MAP = {
    ImportError: ModelLoadError,
    ModuleNotFoundError: ModelLoadError,
    FileNotFoundError: ConfigurationError,
    ValueError: ConfigurationError,
    TypeError: ConfigurationError,
    KeyError: ConfigurationError,
}