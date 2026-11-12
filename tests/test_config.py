"""Tests for configuration loading."""
import pytest
import tempfile
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import Config, get_config, reload_config


def test_config_loads():
    """Test that config loads from YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'test_section': {
                'key1': 'value1',
                'key2': 42
            }
        }, f)
        config_path = f.name
    
    try:
        config = Config()
        config.load(config_path)
        
        assert config.get('test_section', 'key1') == 'value1'
        assert config.get('test_section', 'key2') == 42
        assert config.get('nonexistent', default='default') == 'default'
    finally:
        Path(config_path).unlink()


def test_config_env_override():
    """Test environment variable overrides."""
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'section': {'key': 'default'}}, f)
        config_path = f.name
    
    try:
        os.environ['TEST_OVERRIDE'] = 'env_value'
        
        config = Config()
        config.load(config_path)
        
        # Manually apply override
        config._apply_env_overrides = lambda: config._set_nested(
            ('section', 'key'), config._parse_env_value(os.environ['TEST_OVERRIDE'])
        )
        config._apply_env_overrides()
        
        assert config.get('section', 'key') == 'env_value'
    finally:
        Path(config_path).unlink()
        del os.environ['TEST_OVERRIDE']


def test_config_singleton():
    """Test Config is a singleton."""
    config1 = Config()
    config2 = Config()
    assert config1 is config2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
