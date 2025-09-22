"""CLI Configuration manager.

Provides lightweight configuration loading, validation, and access helpers used by
the CLI-side contract tests. The implementation intentionally keeps dependencies
minimal while supporting both JSON and YAML configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:  # YAML support is optional; fall back to JSON only when unavailable.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - best effort optional dependency.
    yaml = None


class ConfigManager:
    """Simple configuration manager used by CLI contract tests."""

    # Public feature names recognised by the CLI helpers.
    _ALLOWED_FEATURES = {
        'returns',
        'volatility',
        'momentum',
        'risk_metrics',
        'trend',
        'volume',
        'simple_returns',
        'log_returns',
        'annualized_returns',
        'rolling_volatility',
        'ewma_volatility',
        'garch_volatility',
        'rsi',
        'macd',
        'simple_momentum',
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path: Optional[str] = None
        self._config: Dict[str, Any] = {}

        if config_path:
            self.load_config(config_path)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load a configuration file and validate it."""

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        data = self._read_config_file(config_file)
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a mapping/dictionary")

        validation = self.validate_config(data)
        if not validation['is_valid']:
            joined = '; '.join(validation['errors']) or 'Unknown validation error'
            raise ValueError(f"Configuration validation failed: {joined}")

        self._config = data
        self.config_path = str(config_file)
        return self._config

    def load_from_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration directly from a dictionary (mainly for tests)."""

        validation = self.validate_config(config)
        if not validation['is_valid']:
            joined = '; '.join(validation['errors']) or 'Unknown validation error'
            raise ValueError(f"Configuration validation failed: {joined}")

        self._config = dict(config)
        self.config_path = None
        return self._config

    def _read_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Read configuration content based on file suffix."""

        suffix = config_file.suffix.lower()
        text = config_file.read_text(encoding='utf-8')

        if suffix in {'.yaml', '.yml'}:
            if yaml is None:
                raise ValueError(
                    "PyYAML is required to load YAML configuration files",
                )
            return yaml.safe_load(text) or {}

        # Default to JSON for .json or unknown extensions
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - thin wrapper
            raise ValueError(f"Invalid JSON configuration: {exc}") from exc

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dotted notation."""

        value: Any = self._config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dotted notation."""

        if not key:
            raise ValueError("Key must be a non-empty string")

        target = self._config
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Return a shallow copy of the full configuration."""

        return dict(self._config)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration content.

        Returns a dictionary containing ``is_valid`` along with ``errors`` and
        ``warnings`` lists so the caller can report validation feedback.
        """

        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(config, dict):
            return {
                'is_valid': False,
                'errors': ['Configuration must be a dictionary'],
                'warnings': warnings,
            }

        feature_generation = config.get('feature_generation', {})
        if feature_generation and not isinstance(feature_generation, dict):
            errors.append('feature_generation section must be a dictionary')
        else:
            self._validate_feature_generation(feature_generation, errors)

        validation_section = config.get('validation', {})
        if validation_section and not isinstance(validation_section, dict):
            errors.append('validation section must be a dictionary')
        else:
            self._validate_validation(validation_section, errors)

        return {
            'is_valid': not errors,
            'errors': errors,
            'warnings': warnings,
        }

    def _validate_feature_generation(
        self,
        section: Dict[str, Any],
        errors: list[str],
    ) -> None:
        """Validate the feature generation section."""

        if not section:
            return

        max_window_size = section.get('max_window_size')
        if max_window_size is not None:
            if not isinstance(max_window_size, int):
                errors.append('feature_generation.max_window_size must be an integer')
            elif max_window_size <= 0:
                errors.append('feature_generation.max_window_size must be positive')

        default_features = section.get('default_features')
        if default_features is not None:
            if not isinstance(default_features, Iterable) or isinstance(default_features, (str, bytes)):
                errors.append('feature_generation.default_features must be a list of feature names')
            else:
                unknown = [
                    feature
                    for feature in default_features
                    if feature not in self._ALLOWED_FEATURES
                ]
                if unknown:
                    errors.append(
                        'feature_generation.default_features contains unknown features: '
                        + ', '.join(sorted(set(unknown))),
                    )

    def _validate_validation(self, section: Dict[str, Any], errors: list[str]) -> None:
        """Validate the validation section of the configuration."""

        if not section:
            return

        quality_threshold = section.get('quality_threshold')
        if quality_threshold is not None:
            if not isinstance(quality_threshold, (int, float)):
                errors.append('validation.quality_threshold must be numeric')
            elif not 0 <= quality_threshold <= 1:
                errors.append('validation.quality_threshold must be between 0 and 1')

        strict_mode = section.get('strict_mode')
        if strict_mode is not None and not isinstance(strict_mode, bool):
            errors.append('validation.strict_mode must be a boolean')

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def merge_config(self, other_config: Dict[str, Any]) -> None:
        """Deep-merge another configuration mapping into the current one."""

        self._config = self._deep_merge(dict(self._config), other_config)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def save_config(self, output_path: str) -> None:
        """Persist the current configuration to disk as JSON."""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self._config, indent=2, default=str), encoding='utf-8')


__all__ = ['ConfigManager']
