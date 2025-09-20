"""
API Configuration class for financial data processing services.

This module provides configuration management for API services,
including feature generation, validation, and processing limits.
"""

import json
import os
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class FeatureGenerationConfig:
    """Configuration for feature generation."""
    max_window_size: int = 252
    default_features: list = field(default_factory=lambda: ['returns', 'volatility'])
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    batch_size: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    strict_mode: bool = True
    quality_threshold: float = 0.8
    enable_auto_correction: bool = False
    max_validation_errors: int = 100
    log_validation_details: bool = True
    custom_rules_path: Optional[str] = None


@dataclass
class APILimitsConfig:
    """Configuration for API limits and throttling."""
    max_data_points: int = 100000
    max_file_size_mb: int = 10
    request_timeout_seconds: int = 30
    rate_limit_requests_per_minute: int = 100
    concurrent_requests_limit: int = 10
    max_features_per_request: int = 50


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    logs_db_path: str = "storage/preprocessing_logs.db"
    quality_db_path: str = "storage/quality_metrics.db"
    rules_db_path: str = "storage/preprocessing_rules.db"
    connection_pool_size: int = 5
    connection_timeout_seconds: int = 30


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_authentication: bool = False
    api_key_required: bool = False
    allowed_origins: list = field(default_factory=lambda: ['*'])
    max_request_size_mb: int = 10
    enable_cors: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file_path: str = "logs/api.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size_mb: int = 10
    log_retention_days: int = 30
    structured_logging: bool = True


class APIConfig:
    """Main API configuration class."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize API configuration."""
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_config.json')

        # Initialize default configurations
        self.feature_generation = FeatureGenerationConfig()
        self.validation = ValidationConfig()
        self.api_limits = APILimitsConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()

        # Load configuration if file exists
        self.load_config()

    def load_config(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Load configuration from file or dictionary.

        Args:
            config_data: Optional configuration dictionary to load
        """
        if config_data:
            self._load_from_dict(config_data)
        elif os.path.exists(self.config_path):
            self._load_from_file()

    def _load_from_file(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            self._load_from_dict(config_data)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            print("Using default configuration.")

    def _load_from_dict(self, config_data: Dict[str, Any]):
        """Load configuration from dictionary."""
        # Load feature generation config
        if 'feature_generation' in config_data:
            fg_data = config_data['feature_generation']
            for key, value in fg_data.items():
                if hasattr(self.feature_generation, key):
                    setattr(self.feature_generation, key, value)

        # Load validation config
        if 'validation' in config_data:
            val_data = config_data['validation']
            for key, value in val_data.items():
                if hasattr(self.validation, key):
                    setattr(self.validation, key, value)

        # Load API limits config
        if 'api_limits' in config_data:
            limits_data = config_data['api_limits']
            for key, value in limits_data.items():
                if hasattr(self.api_limits, key):
                    setattr(self.api_limits, key, value)

        # Load database config
        if 'database' in config_data:
            db_data = config_data['database']
            for key, value in db_data.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)

        # Load security config
        if 'security' in config_data:
            sec_data = config_data['security']
            for key, value in sec_data.items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)

        # Load logging config
        if 'logging' in config_data:
            log_data = config_data['logging']
            for key, value in log_data.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)

        # Load monitoring config
        if 'monitoring' in config_data:
            mon_data = config_data['monitoring']
            for key, value in mon_data.items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)

    def save_config(self, config_path: Optional[str] = None):
        """
        Save configuration to JSON file.

        Args:
            config_path: Optional path to save configuration
        """
        save_path = config_path or self.config_path
        config_data = self.to_dict()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'feature_generation': asdict(self.feature_generation),
            'validation': asdict(self.validation),
            'api_limits': asdict(self.api_limits),
            'database': asdict(self.database),
            'security': asdict(self.security),
            'logging': asdict(self.logging)
        }

    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Feature-specific configuration dictionary
        """
        feature_configs = {
            'returns': {
                'window_sizes': [1, 5, 21, 63, 252],
                'methods': ['simple', 'log', 'percentage'],
                'annualization_factor': 252
            },
            'volatility': {
                'methods': ['rolling', 'ewma', 'garch'],
                'default_window': 21,
                'min_window': 5,
                'max_window': 252
            },
            'momentum': {
                'indicators': ['rsi', 'macd', 'stochastic', 'roc'],
                'default_periods': [14, 21, 63],
                'overbought_threshold': 70,
                'oversold_threshold': 30
            },
            'risk': {
                'var_confidence_levels': [0.95, 0.99],
                'cvar_confidence_levels': [0.95, 0.99],
                'beta_calculation_window': 252
            }
        }

        return feature_configs.get(feature_name, {})

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration settings.

        Returns:
            Validation result dictionary
        """
        validation_errors = []
        validation_warnings = []

        # Validate feature generation config
        if self.feature_generation.max_window_size <= 0:
            validation_errors.append("max_window_size must be positive")

        if self.feature_generation.max_workers <= 0:
            validation_errors.append("max_workers must be positive")

        # Validate validation config
        if not 0 <= self.validation.quality_threshold <= 1:
            validation_errors.append("quality_threshold must be between 0 and 1")

        # Validate API limits config
        if self.api_limits.max_data_points <= 0:
            validation_errors.append("max_data_points must be positive")

        if self.api_limits.rate_limit_requests_per_minute <= 0:
            validation_errors.append("rate_limit_requests_per_minute must be positive")

        # Validate database config paths
        for db_path_attr in ['logs_db_path', 'quality_db_path', 'rules_db_path']:
            db_path = getattr(self.database, db_path_attr)
            if not db_path:
                validation_warnings.append(f"{db_path_attr} is empty")

        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.log_level not in valid_log_levels:
            validation_errors.append(f"log_level must be one of {valid_log_levels}")

        return {
            'is_valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'warnings': validation_warnings
        }

    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        Update specific configuration section.

        Args:
            section: Configuration section name
            updates: Dictionary of updates to apply
        """
        if hasattr(self, section):
            config_section = getattr(self, section)
            for key, value in updates.items():
                if hasattr(config_section, key):
                    setattr(config_section, key, value)
                else:
                    print(f"Warning: Unknown key '{key}' in section '{section}'")
        else:
            print(f"Warning: Unknown configuration section '{section}'")

    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration overrides.

        Returns:
            Environment configuration dictionary
        """
        env_config = {}

        # Override with environment variables
        env_mappings = {
            'API_MAX_DATA_POINTS': ('api_limits', 'max_data_points'),
            'API_RATE_LIMIT': ('api_limits', 'rate_limit_requests_per_minute'),
            'LOG_LEVEL': ('logging', 'log_level'),
            'FEATURE_CACHE_TTL': ('feature_generation', 'cache_ttl'),
            'VALIDATION_STRICT_MODE': ('validation', 'strict_mode')
        }

        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    # Convert to appropriate type
                    if key in ['max_data_points', 'rate_limit_requests_per_minute', 'cache_ttl']:
                        value = int(value)
                    elif key in ['strict_mode']:
                        value = value.lower() in ['true', '1', 'yes']

                    env_config.setdefault(section, {})[key] = value
                except ValueError as e:
                    print(f"Warning: Invalid value for {env_var}: {e}")

        return env_config

    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_config = self.get_environment_config()
        for section, updates in env_config.items():
            self.update_config(section, updates)

    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get effective configuration including environment overrides.

        Returns:
            Complete effective configuration
        """
        # Apply environment overrides temporarily
        original_config = self.to_dict()

        # Create a copy and apply environment overrides
        env_config = self.get_environment_config()
        effective_config = original_config.copy()

        for section, updates in env_config.items():
            if section in effective_config:
                effective_config[section].update(updates)

        return effective_config

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"APIConfig(feature_generation={self.feature_generation}, validation={self.validation})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return f"APIConfig(path={self.config_path}, sections={list(self.to_dict().keys())})"