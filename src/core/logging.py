"""Structured logging setup for Sleep History Bot."""
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


def setup_logging(config_path: Optional[str] = None) -> logging.Logger:
    """Initialize logging from YAML configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "logging.yaml"
    
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    return logging.getLogger("sleep_history")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(f"sleep_history.{name}")


# Initialize on import
_logger = None


def get_root_logger() -> logging.Logger:
    """Get the root sleep_history logger, initializing if needed."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger
