"""Stage runner with retry logic and error handling."""
import logging
import time
import random
from pathlib import Path
from typing import Callable, Any, Optional, TypeVar
from functools import wraps

from ..core.config import get_config
from ..core.exceptions import RetryExhaustedError, SleepHistoryError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StageRunner:
    """Runs pipeline stages with retry logic."""
    
    def __init__(self):
        self.config = get_config()
        self.max_retries = self.config.get('pipeline', 'retry_attempts', default=3)
        self.base_delay = self.config.get('pipeline', 'retry_base_delay', default=15)
        self.max_delay = self.config.get('pipeline', 'max_retry_delay', default=180)
        
        # Transient error patterns
        self.transient_patterns = (
            "503", "429", "500", "502", "504", "UNAVAILABLE", "overloaded",
            "RESOURCE_EXHAUSTED", "timed out", "Connection", "DEADLINE_EXCEEDED",
            "internal error", "service unavailable", "timeout", "rate limit"
        )
    
    def _is_transient(self, error: Exception) -> bool:
        """Check if error is transient and worth retrying."""
        msg = str(error)
        return any(pattern in msg for pattern in self.transient_patterns)
    
    def run_with_retry(
        self,
        func: Callable[..., T],
        *args,
        stage_name: str = "unknown",
        **kwargs
    ) -> T:
        """Run function with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Stage '{stage_name}' attempt {attempt + 1}/{self.max_retries}")
                return func(*args, **kwargs)
                
            except SleepHistoryError as e:
                # Don't retry our own errors (they're usually not transient)
                if not self._is_transient(e):
                    logger.error(f"Stage '{stage_name}' failed with non-transient error: {e}")
                    raise
                last_error = e
                
            except Exception as e:
                if not self._is_transient(e):
                    logger.error(f"Stage '{stage_name}' failed with non-transient error: {e}")
                    raise
                last_error = e
            
            if attempt < self.max_retries - 1:
                wait = min(self.base_delay * (2 ** attempt), self.max_delay)
                wait += random.randint(0, 10)
                logger.warning(f"Stage '{stage_name}' attempt {attempt + 1} failed: {last_error}. Retrying in {wait}s...")
                time.sleep(wait)
        
        raise RetryExhaustedError(f"stage_{stage_name}", self.max_retries, last_error)
    
    def run_stage_safe(
        self,
        stage_func: Callable[..., T],
        *args,
        stage_name: str = "unknown",
        checkpoint_mgr = None,
        **kwargs
    ) -> T:
        """Run stage with retry and checkpoint saving on failure."""
        try:
            return self.run_with_retry(stage_func, *args, stage_name=stage_name, **kwargs)
        except Exception as e:
            # Save failure checkpoint if manager provided
            if checkpoint_mgr:
                checkpoint_mgr.save(stage_name, {}, success=False, error=str(e))
            raise


def with_retry(stage_name: str = "unknown"):
    """Decorator to add retry logic to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            runner = StageRunner()
            return runner.run_with_retry(func, *args, stage_name=stage_name, **kwargs)
        return wrapper
    return decorator
