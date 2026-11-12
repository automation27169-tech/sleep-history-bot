"""Pollinations.ai image generation wrapper (free, no API key)."""
import os
import time
import random
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote

from ..core.config import get_config
from ..core.exceptions import PollinationsError, RetryExhaustedError
from ..services.cache import get_image_cache

logger = logging.getLogger(__name__)

# Transient HTTP errors
TRANSIENT_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_ERRORS = ("timeout", "connection", "timed out", "unreachable")


class PollinationsClient:
    """Client for Pollinations.ai free image generation."""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.get('images', 'pollinations_base_url', 
            default="https://image.pollinations.ai/prompt/")
        self.cache = get_image_cache()
        self.timeout = self.config.get('images', 'timeout_seconds', default=60)
        self.max_retries = self.config.get('images', 'max_retries', default=3)
        
        # Retry settings
        self.base_delay = self.config.get('pipeline', 'retry_base_delay', default=15)
        self.max_delay = self.config.get('pipeline', 'max_retry_delay', default=180)
    
    def _is_transient(self, error: Exception, status_code: Optional[int] = None) -> bool:
        """Check if error is transient."""
        if status_code and status_code in TRANSIENT_STATUS:
            return True
        msg = str(error).lower()
        return any(t in msg for t in TRANSIENT_ERRORS)
    
    def _build_url(self, prompt: str, **params) -> str:
        """Build Pollinations URL with parameters."""
        encoded_prompt = quote(prompt)
        url = f"{self.base_url}{encoded_prompt}"
        
        # Add parameters
        param_strs = []
        for key, value in params.items():
            if value is not None:
                param_strs.append(f"{key}={quote(str(value))}")
        
        if param_strs:
            url += "?" + "&".join(param_strs)
        
        return url
    
    def _download_with_retry(self, url: str) -> bytes:
        """Download image with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                
                if response.status_code == 200:
                    return response.content
                
                if response.status_code in TRANSIENT_STATUS:
                    raise PollinationsError(f"HTTP {response.status_code}", 
                        Exception(f"Status {response.status_code}"))
                
                # Non-retryable error
                raise PollinationsError(f"HTTP {response.status_code}: {response.text[:200]}")
                
            except requests.exceptions.Timeout:
                last_error = PollinationsError("Request timeout")
            except requests.exceptions.ConnectionError as e:
                last_error = PollinationsError(f"Connection error: {e}")
            except PollinationsError:
                raise
            except Exception as e:
                last_error = PollinationsError(f"Unexpected error: {e}")
            
            if attempt < self.max_retries - 1:
                wait = min(self.base_delay * (2 ** attempt), self.max_delay)
                wait += random.randint(0, 5)
                logger.warning(f"Pollinations attempt {attempt + 1} failed: {last_error}. Retrying in {wait}s...")
                time.sleep(wait)
        
        raise RetryExhaustedError("pollinations_download", self.max_retries, last_error)
    
    def generate_image(
        self,
        prompt: str,
        width: int = 1920,
        height: int = 1080,
        model: str = "flux",
        seed: Optional[int] = None,
        nologo: bool = True,
        private: bool = True,
        enhance: bool = True,
        use_cache: bool = True
    ) -> bytes:
        """
        Generate image from Pollinations.
        
        Args:
            prompt: Image prompt
            width: Image width
            height: Image height
            model: Model to use (flux, gptimage, etc.)
            seed: Random seed for reproducibility
            nologo: Remove watermark
            private: Don't show in public feed
            enhance: Enhance prompt automatically
            use_cache: Use cached result if available
        """
        # Create cache key from all parameters
        cache_params = f"{prompt}|{width}x{height}|{model}|{seed}|{nologo}|{enhance}"
        cache_key = hashlib.sha256(cache_params.encode()).hexdigest()[:32]
        
        if use_cache:
            cached_path = self.cache.get_image_path(cache_key)
            if cached_path and cached_path.exists():
                logger.debug(f"Cache hit for image: {cache_key[:8]}")
                with open(cached_path, 'rb') as f:
                    return f.read()
        
        # Build URL with parameters
        params = {
            "width": width,
            "height": height,
            "model": model,
            "nologo": "true" if nologo else "false",
            "private": "true" if private else "false",
            "enhance": "true" if enhance else "false",
        }
        if seed is not None:
            params["seed"] = seed
        
        url = self._build_url(prompt, **params)
        logger.debug(f"Pollinations URL: {url[:100]}...")
        
        # Download image
        image_data = self._download_with_retry(url)
        
        # Validate image data
        if len(image_data) < 1000:
            raise PollinationsError(f"Downloaded image too small: {len(image_data)} bytes")
        
        # Save to cache
        if use_cache:
            self.cache.save_image(cache_key, image_data, ".png")
        
        logger.info(f"Generated image: {len(image_data)} bytes")
        return image_data
    
    def generate_image_async(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Return URL for async generation (not implemented, returns sync URL)."""
        params = {
            "width": kwargs.get('width', 1920),
            "height": kwargs.get('height', 1080),
            "model": kwargs.get('model', 'flux'),
            "nologo": "true" if kwargs.get('nologo', True) else "false",
            "private": "true" if kwargs.get('private', True) else "false",
            "enhance": "true" if kwargs.get('enhance', True) else "false",
        }
        if kwargs.get('seed') is not None:
            params["seed"] = kwargs['seed']
        
        return self._build_url(prompt, **params)


def get_pollinations_client() -> PollinationsClient:
    """Get Pollinations client instance."""
    return PollinationsClient()
