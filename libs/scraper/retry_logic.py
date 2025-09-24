"""
Retry Logic with Exponential Backoff

This module provides retry decorators and utilities for handling transient failures
in web scraping operations, particularly for rate limiting and network issues.
"""
from __future__ import annotations
import time
import random
import logging
from typing import Optional, Callable, Any, Type, Union, Tuple
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetryableError(Enum):
    """Types of errors that should trigger retry logic"""
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays
    

def is_retryable_exception(exception: Exception) -> Optional[RetryableError]:
    """Determine if an exception should trigger retry logic
    
    Args:
        exception: The exception to check
        
    Returns:
        RetryableError type if retryable, None otherwise
    """
    # Import here to avoid circular dependencies and handle missing packages
    try:
        from selenium.common.exceptions import TimeoutException, WebDriverException
    except ImportError:
        TimeoutException = type('TimeoutException', (Exception,), {})
        WebDriverException = type('WebDriverException', (Exception,), {})
        
    try:
        from requests.exceptions import RequestException, Timeout, ConnectionError
    except ImportError:
        RequestException = type('RequestException', (Exception,), {})
        Timeout = type('Timeout', (Exception,), {})
        ConnectionError = type('ConnectionError', (Exception,), {})
    
    # Rate limiting (HTTP 429 or similar patterns)
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        if exception.response.status_code == 429:
            return RetryableError.RATE_LIMIT
        elif 500 <= exception.response.status_code < 600:
            return RetryableError.SERVER_ERROR
    
    # Check exception message for rate limiting patterns
    exception_str = str(exception).lower()
    if any(pattern in exception_str for pattern in ['rate limit', 'too many requests', 'quota exceeded']):
        return RetryableError.RATE_LIMIT
    
    # Network-related exceptions
    if isinstance(exception, (ConnectionError, TimeoutException, Timeout)):
        return RetryableError.NETWORK_ERROR
    
    # WebDriver exceptions that may be transient
    if isinstance(exception, WebDriverException):
        if any(pattern in exception_str for pattern in ['timeout', 'connection', 'network']):
            return RetryableError.NETWORK_ERROR
    
    # Generic request exceptions
    if isinstance(exception, RequestException):
        return RetryableError.NETWORK_ERROR
    
    return None


def calculate_delay(attempt: int, config: RetryConfig, error_type: Optional[RetryableError] = None) -> float:
    """Calculate delay for retry attempt with exponential backoff
    
    Args:
        attempt: Current attempt number (0-based)
        config: Retry configuration
        error_type: Type of error that triggered retry
        
    Returns:
        Delay in seconds
    """
    # Base exponential backoff
    delay = config.base_delay * (config.backoff_multiplier ** attempt)
    
    # Rate limiting gets longer delays
    if error_type == RetryableError.RATE_LIMIT:
        delay *= 2
    
    # Apply maximum delay limit
    delay = min(delay, config.max_delay)
    
    # Add jitter to avoid thundering herd
    if config.jitter:
        jitter_factor = 0.1  # +/- 10% jitter
        jitter = delay * jitter_factor * (2 * random.random() - 1)
        delay += jitter
    
    return max(0, delay)


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for retry logic with exponential backoff
    
    Args:
        config: Retry configuration (uses default if None)
        exceptions: Exception types that should trigger retry
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful retry
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    error_type = is_retryable_exception(e)
                    
                    # Always retry if this exception type was explicitly included
                    should_retry = True
                    
                    # But check if it's a recognized retryable error for logging
                    if error_type is None and attempt == 0:
                        logger.debug(f"{func.__name__} failed with non-retryable error: {e}")
                        # Still retry since the exception type was in the allowed list
                    
                    # Don't retry on final attempt
                    if attempt == config.max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {config.max_attempts} attempts: {e}")
                        break
                    
                    # Calculate delay and wait
                    delay = calculate_delay(attempt, config, error_type)
                    logger.warning(
                        f"{func.__name__} failed on attempt {attempt + 1} "
                        f"({error_type.value if error_type else 'unknown'}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
            
            # All retries exhausted
            raise last_exception
            
        return wrapper
    return decorator


# Common retry configurations
DEFAULT_RETRY_CONFIG = RetryConfig()
AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_multiplier=1.5
)
RATE_LIMIT_RETRY_CONFIG = RetryConfig(
    max_attempts=4,
    base_delay=2.0,
    max_delay=120.0,
    backoff_multiplier=2.5
)


# Convenience decorators
def retry_on_rate_limit(func: Callable) -> Callable:
    """Decorator specifically for rate limiting scenarios"""
    return retry_with_backoff(RATE_LIMIT_RETRY_CONFIG)(func)


def retry_on_network_error(func: Callable) -> Callable:
    """Decorator specifically for network-related errors"""
    return retry_with_backoff(DEFAULT_RETRY_CONFIG)(func)