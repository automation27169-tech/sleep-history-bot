"""File-based caching for API responses and generated assets."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Callable, TypeVar
import logging

from ..core.config import get_config
from ..core.exceptions import CacheError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Cache:
    """Generic file-based cache with TTL support."""
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400 * 30):  # 30 days default
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._index_file = self.cache_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> dict:
        """Load cache index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save cache index."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _make_key(self, *parts: str) -> str:
        """Create a cache key from parts."""
        content = "|".join(str(p) for p in parts)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _get_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.cache"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value if not expired."""
        if key not in self._index:
            return default
        
        entry = self._index[key]
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            self.delete(key)
            return default
        
        path = self._get_path(key)
        if not path.exists():
            del self._index[key]
            self._save_index()
            return default
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            self.delete(key)
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w') as f:
                json.dump(value, f, ensure_ascii=False)
            
            self._index[key] = {
                'timestamp': time.time(),
                'size': path.stat().st_size
            }
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")
            raise CacheError(f"Failed to save cache: {e}")
    
    def delete(self, key: str) -> None:
        """Delete cache entry."""
        if key in self._index:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
            del self._index[key]
            self._save_index()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self._index.keys()):
            self.delete(key)
    
    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get from cache or compute and cache."""
        cached = self.get(key)
        if cached is not None:
            logger.debug(f"Cache hit: {key}")
            return cached
        
        logger.debug(f"Cache miss: {key}")
        value = compute_fn()
        self.set(key, value)
        return value


class ImageCache(Cache):
    """Specialized cache for image files."""
    
    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir, ttl_seconds=86400 * 90)  # 90 days for images
    
    def get_image_path(self, key: str) -> Optional[Path]:
        """Get cached image path if exists and not expired."""
        if key not in self._index:
            return None
        
        entry = self._index[key]
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            self.delete(key)
            return None
        
        path = self._get_path(key)
        if path.exists():
            return path
        return None
    
    def save_image(self, key: str, image_data: bytes, extension: str = ".png") -> Path:
        """Save image data to cache."""
        path = self._get_path(key).with_suffix(extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(image_data)
        
        self._index[key] = {
            'timestamp': time.time(),
            'size': path.stat().st_size,
            'extension': extension
        }
        self._save_index()
        
        return path


# Global cache instances
_general_cache: Optional[Cache] = None
_image_cache: Optional[ImageCache] = None


def get_cache() -> Cache:
    """Get global general cache instance."""
    global _general_cache
    if _general_cache is None:
        config = get_config()
        cache_dir = Path(config.get('pipeline', 'checkpoint_dir', default='data/cache')).parent / "general"
        _general_cache = Cache(cache_dir)
    return _general_cache


def get_image_cache() -> ImageCache:
    """Get global image cache instance."""
    global _image_cache
    if _image_cache is None:
        config = get_config()
        cache_dir = Path(config.get('images', 'cache_dir', default='data/cache/images'))
        _image_cache = ImageCache(cache_dir)
    return _image_cache
