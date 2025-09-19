"""
Configuration System for Preprocessing Pipelines

Manages configuration files, settings, and parameter management
for data preprocessing pipelines with validation and defaults.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import os
from datetime import datetime
import logging


@dataclass
class PreprocessingConfig:
    """Configuration for a preprocessing pipeline."""
    pipeline_id: str
    description: str
    asset_classes: List[str]
    rules: List[Dict[str, Any]]
    quality_thresholds: Dict[str, float]
    output_format: str = "parquet"
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pipeline_id': self.pipeline_id,
            'description': self.description,
            'asset_classes': self.asset_classes,
            'rules': self.rules,
            'quality_thresholds': self.quality_thresholds,
            'output_format': self.output_format,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        """Create from dictionary representation."""
        created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        updated_at = datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now()

        return cls(
            pipeline_id=data['pipeline_id'],
            description=data['description'],
            asset_classes=data['asset_classes'],
            rules=data['rules'],
            quality_thresholds=data['quality_thresholds'],
            output_format=data.get('output_format', 'parquet'),
            version=data.get('version', '1.0.0'),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get('metadata', {})
        )


class PipelineConfigManager:
    """Manages preprocessing pipeline configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._configs = {}
        self._load_default_configs()

    def _load_default_configs(self):
        """Load default configuration templates."""
        # Default quality thresholds
        self.default_quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.90,
            'accuracy': 0.85,
            'timeliness': 0.95,
            'uniqueness': 0.95
        }

        # Default preprocessing rules
        self.default_rules = [
            {
                'rule_type': 'missing_value',
                'parameters': {
                    'method': 'forward_fill',
                    'threshold': 0.1,
                    'window_size': 5
                },
                'priority': 1
            },
            {
                'rule_type': 'outlier',
                'parameters': {
                    'method': 'iqr',
                    'threshold': 1.5,
                    'action': 'flag'
                },
                'priority': 2
            },
            {
                'rule_type': 'normalization',
                'parameters': {
                    'method': 'zscore',
                    'preserve_stats': True
                },
                'priority': 3
            }
        ]

        # Default asset classes
        self.default_asset_classes = ['equity', 'fx', 'bond', 'commodity']

    def create_default_config(self, pipeline_id: str, description: str = "Default preprocessing pipeline",
                            asset_classes: Optional[List[str]] = None,
                            quality_thresholds: Optional[Dict[str, float]] = None,
                            rules: Optional[List[Dict[str, Any]]] = None) -> PreprocessingConfig:
        """Create a default preprocessing configuration.

        Args:
            pipeline_id: Unique identifier for the pipeline
            description: Description of the pipeline
            asset_classes: List of asset classes to process
            quality_thresholds: Quality thresholds for validation
            rules: Preprocessing rules to apply

        Returns:
            PreprocessingConfig object
        """
        config = PreprocessingConfig(
            pipeline_id=pipeline_id,
            description=description,
            asset_classes=asset_classes or self.default_asset_classes.copy(),
            rules=rules or self.default_rules.copy(),
            quality_thresholds=quality_thresholds or self.default_quality_thresholds.copy()
        )

        self._configs[pipeline_id] = config
        return config

    def load_config(self, config_path: str) -> PreprocessingConfig:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            PreprocessingConfig object
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            config = PreprocessingConfig.from_dict(data)
            self._configs[config.pipeline_id] = config

            self.logger.info(f"Loaded configuration '{config.pipeline_id}' from {config_path}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def save_config(self, config: PreprocessingConfig, config_path: Optional[str] = None) -> str:
        """Save configuration to file.

        Args:
            config: PreprocessingConfig to save
            config_path: Path to save configuration (optional)

        Returns:
            Path where configuration was saved
        """
        if config_path is None:
            config_path = self.config_dir / f"{config.pipeline_id}.json"
        else:
            config_path = Path(config_path)

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
                else:
                    json.dump(config.to_dict(), f, indent=2)

            self._configs[config.pipeline_id] = config
            self.logger.info(f"Saved configuration '{config.pipeline_id}' to {config_path}")
            return str(config_path)

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    def get_config(self, pipeline_id: str) -> Optional[PreprocessingConfig]:
        """Get configuration by pipeline ID.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PreprocessingConfig or None if not found
        """
        return self._configs.get(pipeline_id)

    def list_configs(self) -> List[str]:
        """List all available configuration pipeline IDs.

        Returns:
            List of pipeline IDs
        """
        return list(self._configs.keys())

    def delete_config(self, pipeline_id: str) -> bool:
        """Delete configuration.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if deleted, False if not found
        """
        if pipeline_id in self._configs:
            del self._configs[pipeline_id]

            # Delete file if it exists
            config_file = self.config_dir / f"{pipeline_id}.json"
            if config_file.exists():
                config_file.unlink()

            self.logger.info(f"Deleted configuration '{pipeline_id}'")
            return True

        return False

    def update_config(self, pipeline_id: str, updates: Dict[str, Any]) -> PreprocessingConfig:
        """Update existing configuration.

        Args:
            pipeline_id: Pipeline identifier
            updates: Dictionary of updates to apply

        Returns:
            Updated PreprocessingConfig
        """
        config = self.get_config(pipeline_id)
        if not config:
            raise ValueError(f"Configuration '{pipeline_id}' not found")

        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now()
        self._configs[pipeline_id] = config

        return config

    def validate_config(self, config: PreprocessingConfig) -> Dict[str, Any]:
        """Validate configuration for completeness and correctness.

        Args:
            config: Configuration to validate

        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate required fields
        if not config.pipeline_id:
            results['errors'].append("Pipeline ID is required")
            results['is_valid'] = False

        if not config.description:
            results['errors'].append("Description is required")
            results['is_valid'] = False

        if not config.asset_classes:
            results['errors'].append("At least one asset class is required")
            results['is_valid'] = False

        # Validate asset classes
        valid_asset_classes = ['equity', 'fx', 'bond', 'commodity', 'crypto', 'all']
        for asset_class in config.asset_classes:
            if asset_class not in valid_asset_classes:
                results['errors'].append(f"Invalid asset class: {asset_class}")
                results['is_valid'] = False

        # Validate rules
        if not config.rules:
            results['errors'].append("At least one preprocessing rule is required")
            results['is_valid'] = False
        else:
            for i, rule in enumerate(config.rules):
                if not isinstance(rule, dict):
                    results['errors'].append(f"Rule {i} must be a dictionary")
                    results['is_valid'] = False
                    continue

                if 'rule_type' not in rule:
                    results['errors'].append(f"Rule {i} missing 'rule_type' field")
                    results['is_valid'] = False

                if 'parameters' not in rule:
                    results['errors'].append(f"Rule {i} missing 'parameters' field")
                    results['is_valid'] = False

        # Validate quality thresholds
        if not config.quality_thresholds:
            results['errors'].append("Quality thresholds are required")
            results['is_valid'] = False
        else:
            for metric, threshold in config.quality_thresholds.items():
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    results['errors'].append(f"Invalid threshold for {metric}: {threshold}")
                    results['is_valid'] = False

        # Validate output format
        valid_formats = ['parquet', 'csv', 'json', 'feather']
        if config.output_format not in valid_formats:
            results['errors'].append(f"Invalid output format: {config.output_format}")
            results['is_valid'] = False

        return results

    def create_equity_config(self, pipeline_id: str) -> PreprocessingConfig:
        """Create specialized configuration for equity data.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PreprocessingConfig optimized for equity data
        """
        equity_rules = [
            {
                'rule_type': 'missing_value',
                'parameters': {
                    'method': 'forward_fill',
                    'threshold': 0.05,
                    'window_size': 3
                },
                'priority': 1
            },
            {
                'rule_type': 'outlier',
                'parameters': {
                    'method': 'iqr',
                    'threshold': 2.0,
                    'action': 'flag'
                },
                'priority': 2
            },
            {
                'rule_type': 'normalization',
                'parameters': {
                    'method': 'zscore',
                    'preserve_stats': True
                },
                'priority': 3
            }
        ]

        equity_thresholds = {
            'completeness': 0.98,
            'consistency': 0.95,
            'accuracy': 0.90,
            'timeliness': 0.99,
            'uniqueness': 0.99
        }

        return self.create_default_config(
            pipeline_id=pipeline_id,
            description="Equity data preprocessing pipeline",
            asset_classes=['equity'],
            quality_thresholds=equity_thresholds,
            rules=equity_rules
        )

    def create_fx_config(self, pipeline_id: str) -> PreprocessingConfig:
        """Create specialized configuration for FX data.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PreprocessingConfig optimized for FX data
        """
        fx_rules = [
            {
                'rule_type': 'missing_value',
                'parameters': {
                    'method': 'interpolation',
                    'threshold': 0.02,
                    'window_size': 5
                },
                'priority': 1
            },
            {
                'rule_type': 'outlier',
                'parameters': {
                    'method': 'zscore',
                    'threshold': 3.0,
                    'action': 'clip'
                },
                'priority': 2
            },
            {
                'rule_type': 'normalization',
                'parameters': {
                    'method': 'minmax',
                    'range': [0, 1],
                    'preserve_stats': True
                },
                'priority': 3
            }
        ]

        fx_thresholds = {
            'completeness': 0.99,
            'consistency': 0.98,
            'accuracy': 0.95,
            'timeliness': 0.95,
            'uniqueness': 0.99
        }

        return self.create_default_config(
            pipeline_id=pipeline_id,
            description="FX data preprocessing pipeline",
            asset_classes=['fx'],
            quality_thresholds=fx_thresholds,
            rules=fx_rules
        )

    def get_config_summary(self, pipeline_id: str) -> Dict[str, Any]:
        """Get summary of configuration.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Configuration summary
        """
        config = self.get_config(pipeline_id)
        if not config:
            raise ValueError(f"Configuration '{pipeline_id}' not found")

        return {
            'pipeline_id': config.pipeline_id,
            'description': config.description,
            'asset_classes': config.asset_classes,
            'rules_count': len(config.rules),
            'quality_thresholds': config.quality_thresholds,
            'output_format': config.output_format,
            'version': config.version,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat()
        }

    def export_config_template(self, output_path: str) -> str:
        """Export a configuration template file.

        Args:
            output_path: Path to save template

        Returns:
            Path where template was saved
        """
        template_config = self.create_default_config(
            pipeline_id="template_pipeline",
            description="Template configuration - modify as needed"
        )

        return self.save_config(template_config, output_path)

    def compare_configs(self, pipeline_id1: str, pipeline_id2: str) -> Dict[str, Any]:
        """Compare two configurations.

        Args:
            pipeline_id1: First pipeline identifier
            pipeline_id2: Second pipeline identifier

        Returns:
            Comparison results
        """
        config1 = self.get_config(pipeline_id1)
        config2 = self.get_config(pipeline_id2)

        if not config1 or not config2:
            raise ValueError("One or both configurations not found")

        comparison = {
            'pipeline1': pipeline_id1,
            'pipeline2': pipeline_id2,
            'differences': {},
            'similarities': {}
        }

        # Compare asset classes
        if set(config1.asset_classes) != set(config2.asset_classes):
            comparison['differences']['asset_classes'] = {
                'pipeline1': config1.asset_classes,
                'pipeline2': config2.asset_classes
            }
        else:
            comparison['similarities']['asset_classes'] = config1.asset_classes

        # Compare quality thresholds
        if config1.quality_thresholds != config2.quality_thresholds:
            comparison['differences']['quality_thresholds'] = {
                'pipeline1': config1.quality_thresholds,
                'pipeline2': config2.quality_thresholds
            }
        else:
            comparison['similarities']['quality_thresholds'] = config1.quality_thresholds

        # Compare rules
        if len(config1.rules) != len(config2.rules):
            comparison['differences']['rules_count'] = {
                'pipeline1': len(config1.rules),
                'pipeline2': len(config2.rules)
            }

        return comparison


# Global configuration manager instance
config_manager = PipelineConfigManager()


def get_config_manager() -> PipelineConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def create_default_pipeline_config(pipeline_id: str, **kwargs) -> PreprocessingConfig:
    """Create a default pipeline configuration.

    Args:
        pipeline_id: Pipeline identifier
        **kwargs: Additional configuration parameters

    Returns:
        PreprocessingConfig object
    """
    return config_manager.create_default_config(pipeline_id, **kwargs)


def load_pipeline_config(config_path: str) -> PreprocessingConfig:
    """Load pipeline configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        PreprocessingConfig object
    """
    return config_manager.load_config(config_path)


def save_pipeline_config(config: PreprocessingConfig, config_path: Optional[str] = None) -> str:
    """Save pipeline configuration to file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration

    Returns:
        Path where configuration was saved
    """
    return config_manager.save_config(config, config_path)