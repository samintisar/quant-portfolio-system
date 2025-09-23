"""
Simple configuration management for the portfolio system.
"""

import yaml
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'portfolio': {
            'risk_free_rate': 0.02,
            'max_position_size': 0.05,
            'max_sector_concentration': 0.20,
            'trading_days_per_year': 252,
            'default_period': '5y',
            'min_sharpe_ratio': 1.0,
            'max_drawdown_threshold': 0.15
        },
        'data': {
            'min_data_points': 252,
            'max_missing_percentage': 0.05,
            'use_offline_data': True,
            'offline_data_dir': './data',
            'fallback_to_online': True,
            'enable_caching': False,
            'cache_dir': './cache'
        },
        'api': {
            'host': 'localhost',
            'port': 8000,
            'debug': False
        },
        'optimization': {
            'default_method': 'mean_variance',
            'risk_free_rate': 0.02,
            'risk_model': 'ledoit_wolf',  # 'sample'|'ledoit_wolf'|'oas'
            'entropy_penalty': 0.0,
            'turnover_penalty': 0.0
        },
        'backtest': {
            'max_position_cap': 0.20,
            'risk_model': 'ledoit_wolf',
            'turnover_penalty': 0.0
        }
    }


class Config:
    """Simple configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config.yaml')
        self._config = self._load_config()

        # Add convenience attributes for test compatibility
        self._add_convenience_attributes()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return self._merge_with_defaults(config)
            else:
                logger.info("No config file found, using defaults")
                return get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            return get_default_config()

    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded config with defaults."""
        defaults = get_default_config()
        merged = defaults.copy()

        def merge_dict(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dict(d1[k], v)
                else:
                    d1[k] = v

        merge_dict(merged, config)
        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values."""
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    if k in d and isinstance(d[k], dict):
                        update_dict(d[k], v)
                    else:
                        d[k] = v
                else:
                    d[k] = v

        update_dict(self._config, updates)

    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.safe_dump(self._config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _add_convenience_attributes(self):
        """Add convenience attributes for test compatibility."""
        # Add optimization section
        if 'optimization' not in self._config:
            self._config['optimization'] = {
                'default_method': 'mean_variance',
                'risk_free_rate': 0.02
            }

        # Add data section
        if 'data' not in self._config:
            self._config['data'] = {
                'min_data_points': 252,
                'max_missing_percentage': 0.05,
                'use_offline_data': True,
                'offline_data_dir': './data',
                'fallback_to_online': True,
                'enable_caching': False,
                'cache_dir': './cache'
            }

        # Add performance section
        if 'performance' not in self._config:
            self._config['performance'] = {
                'benchmark': 'SPY',
                'risk_free_rate': 0.02
            }

        # Add convenience attributes as object properties
        for section_name, section_config in self._config.items():
            if isinstance(section_config, dict):
                # Create a simple namespace object for the section
                namespace = type('ConfigSection', (), {})()
                for key, value in section_config.items():
                    setattr(namespace, key, value)
                setattr(self, section_name, namespace)


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def load_config(config_path: str) -> Config:
    """Load configuration from specific path."""
    return Config(config_path)


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    get_config().update(updates)
