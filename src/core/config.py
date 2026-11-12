"""Configuration loader for Sleep History Bot."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


class Config:
    """Configuration manager with YAML file and environment variable support."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load()
    
    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file and environment variables."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            'GEMINI_API_KEY': ('gemini', 'api_key'),
            'TOPIC': ('runtime', 'topic'),
            'MODE': ('runtime', 'mode'),
            'SECTIONS': ('script', 'sections'),
            'WORDS_PER_SECTION': ('script', 'words_per_section'),
            'IMAGE_SECONDS': ('visual', 'image_seconds'),
            'FPS': ('visual', 'fps'),
            'OUT_W': ('visual', 'resolution', 0),
            'OUT_H': ('visual', 'resolution', 1),
            'CROP_RIGHT': ('visual', 'crop_right'),
            'CROP_BOTTOM': ('visual', 'crop_bottom'),
            'VOICE': ('audio', 'gemini_voice'),
            'EDGE_VOICE': ('audio', 'edge_voice'),
            'STORY_STYLE': ('script', 'story_style'),
            'EMBERS': ('visual', 'embers_enabled'),
            'THREE_D': ('visual', 'three_d_enabled'),
            'BROLL_EVERY': ('visual', 'broll_every'),
            'BURN_SUBS': ('runtime', 'burn_subs'),
            'WHISPER_MODEL': ('timestamps', 'whisper_model'),
        }
        
        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(path, self._parse_env_value(value))
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]
        return value
    
    def _set_nested(self, path: tuple, value: Any) -> None:
        """Set nested dictionary value using tuple path."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get(self, *path: str, default: Any = None) -> Any:
        """Get configuration value by path."""
        current = self._config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    config.load(config_path)
    return config
