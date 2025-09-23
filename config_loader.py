"""
Simple configuration loader for the portfolio optimization system.
"""

import yaml
from typing import Dict, Any
import os

def load_config(config_path: str = "config_simple.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()

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
            'enable_caching': True,
            'cache_dir': './cache'
        },
        'api': {
            'host': 'localhost',
            'port': 8000,
            'debug': False
        }
    }

# Global configuration instance
_config = load_config()

def get_config() -> Dict[str, Any]:
    """Get the global configuration instance."""
    return _config

def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
    global _config
    update_dict(_config, updates)