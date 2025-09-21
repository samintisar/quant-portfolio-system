"""
Configuration management for multiple model variants and parameters.

Implements comprehensive configuration management system for:
- Multiple model variants and parameters
- Environment-specific configurations
- Dynamic parameter optimization
- Configuration validation and versioning
"""

import json
import yaml
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import copy


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelType(Enum):
    """Model types supported by the system."""
    ARIMA = "arima"
    GARCH = "garch"
    HMM = "hmm"
    BAYESIAN = "bayesian"
    LSTM = "lstm"
    XGBOOST = "xgboost"
    TRANSFORMER = "transformer"


@dataclass
class ModelConfiguration:
    """Base configuration for models."""
    model_type: str
    version: str
    parameters: Dict[str, Any]
    environment: Environment
    created_at: str
    updated_at: str
    is_active: bool = True
    validation_rules: Optional[Dict[str, Any]] = None
    performance_thresholds: Optional[Dict[str, float]] = None


@dataclass
class ARIMAConfiguration(ModelConfiguration):
    """ARIMA model specific configuration."""
    model_type: str = "arima"
    order: Optional[tuple] = None
    seasonal_order: Optional[tuple] = None
    heavy_tail: bool = True
    confidence_level: float = 0.95


@dataclass
class GARCHConfiguration(ModelConfiguration):
    """GARCH model specific configuration."""
    model_type: str = "garch"
    garch_type: str = "egarch"  # garch, egarch, gjr
    p: int = 1
    q: int = 1
    include_asymmetric: bool = True
    heavy_tail: bool = True
    regime_switching: bool = False


@dataclass
class HMMConfiguration(ModelConfiguration):
    """HMM model specific configuration."""
    model_type: str = "hmm"
    n_regimes: int = 2
    emission_model: str = "student_t"  # gaussian, student_t, mixture
    include_transitions: bool = True
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000


@dataclass
class MLConfiguration(ModelConfiguration):
    """Machine learning model configuration."""
    model_type: str = "ml"
    hidden_layers: Optional[List[int]] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.2
    regularization: float = 0.01
    # GPU-specific settings
    use_gpu: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    torch_compile: bool = True
    gradient_accumulation_steps: int = 2


class ConfigurationValidator:
    """Validate configuration parameters."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_arima_config(self, config: ARIMAConfiguration) -> List[str]:
        """Validate ARIMA configuration."""
        errors = []

        if config.order and len(config.order) != 3:
            errors.append("ARIMA order must be a tuple of 3 integers (p,d,q)")

        if config.confidence_level <= 0 or config.confidence_level >= 1:
            errors.append("Confidence level must be between 0 and 1")

        return errors

    def validate_garch_config(self, config: GARCHConfiguration) -> List[str]:
        """Validate GARCH configuration."""
        errors = []

        if config.garch_type not in ["garch", "egarch", "gjr"]:
            errors.append("GARCH type must be one of: garch, egarch, gjr")

        if config.p < 0 or config.q < 0:
            errors.append("GARCH parameters p and q must be non-negative")

        if config.confidence_level <= 0 or config.confidence_level >= 1:
            errors.append("Confidence level must be between 0 and 1")

        return errors

    def validate_hmm_config(self, config: HMMConfiguration) -> List[str]:
        """Validate HMM configuration."""
        errors = []

        if config.n_regimes < 2:
            errors.append("Number of regimes must be at least 2")

        if config.emission_model not in ["gaussian", "student_t", "mixture"]:
            errors.append("Emission model must be one of: gaussian, student_t, mixture")

        if config.convergence_threshold <= 0:
            errors.append("Convergence threshold must be positive")

        if config.max_iterations <= 0:
            errors.append("Max iterations must be positive")

        return errors

    def validate_ml_config(self, config: MLConfiguration) -> List[str]:
        """Validate machine learning configuration."""
        errors = []

        if config.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        if config.batch_size <= 0:
            errors.append("Batch size must be positive")

        if config.epochs <= 0:
            errors.append("Epochs must be positive")

        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            errors.append("Dropout rate must be between 0 and 1")

        if config.regularization < 0:
            errors.append("Regularization must be non-negative")

        return errors


class ConfigurationManager:
    """Main configuration management service."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configurations = {}
        self.validator = ConfigurationValidator()
        self.logger = logging.getLogger(__name__)

        # Model configuration mappings
        self.model_config_classes = {
            "arima": ARIMAConfiguration,
            "garch": GARCHConfiguration,
            "hmm": HMMConfiguration,
            "ml": MLConfiguration
        }

        # Load existing configurations
        self._load_configurations()

    def _load_configurations(self):
        """Load all existing configurations from files."""
        try:
            for config_file in self.config_dir.glob("*.json"):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.configurations[config_data['model_type']] = config_data

            self.logger.info(f"Loaded {len(self.configurations)} configurations")
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")

    def get_configuration(self, model_type: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get configuration for a model type."""
        if model_type not in self.configurations:
            return None

        if version == "latest":
            return self.configurations[model_type]

        # Version-specific retrieval would be implemented here
        return self.configurations[model_type]

    def create_configuration(self,
                           model_type: str,
                           parameters: Dict[str, Any],
                           environment: Environment = Environment.DEVELOPMENT,
                           validation_rules: Optional[Dict[str, Any]] = None,
                           performance_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Create a new model configuration."""
        try:
            # Create configuration object
            config_class = self.model_config_classes.get(model_type)
            if not config_class:
                raise ValueError(f"Unsupported model type: {model_type}")

            config_data = {
                "model_type": model_type,
                "version": "1.0.0",
                "parameters": parameters,
                "environment": environment.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "is_active": True,
                "validation_rules": validation_rules,
                "performance_thresholds": performance_thresholds
            }

            # Merge with model-specific parameters
            config_data.update(parameters)

            # Create configuration object
            config = config_class(**config_data)

            # Validate configuration
            errors = self._validate_configuration(config)
            if errors:
                return {"error": "Validation failed", "errors": errors}

            # Save configuration
            self.configurations[model_type] = config_data
            self._save_configuration(model_type, config_data)

            return {"success": True, "configuration": config_data}

        except Exception as e:
            self.logger.error(f"Failed to create configuration: {e}")
            return {"error": str(e)}

    def _validate_configuration(self, config: ModelConfiguration) -> List[str]:
        """Validate a configuration object."""
        if isinstance(config, ARIMAConfiguration):
            return self.validator.validate_arima_config(config)
        elif isinstance(config, GARCHConfiguration):
            return self.validator.validate_garch_config(config)
        elif isinstance(config, HMMConfiguration):
            return self.validator.validate_hmm_config(config)
        elif isinstance(config, MLConfiguration):
            return self.validator.validate_ml_config(config)
        else:
            return ["Unknown configuration type"]

    def _save_configuration(self, model_type: str, config_data: Dict[str, Any]):
        """Save configuration to file."""
        config_file = self.config_dir / f"{model_type}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    def update_configuration(self,
                           model_type: str,
                           updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing configuration."""
        try:
            if model_type not in self.configurations:
                return {"error": f"Configuration not found for {model_type}"}

            # Update configuration
            config = self.configurations[model_type]
            config.update(updates)
            config["updated_at"] = datetime.now().isoformat()

            # Validate updated configuration
            config_class = self.model_config_classes.get(model_type)
            if config_class:
                config_obj = config_class(**config)
                errors = self._validate_configuration(config_obj)
                if errors:
                    return {"error": "Validation failed", "errors": errors}

            # Save updated configuration
            self._save_configuration(model_type, config)

            return {"success": True, "configuration": config}

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return {"error": str(e)}

    def delete_configuration(self, model_type: str) -> Dict[str, Any]:
        """Delete a configuration."""
        try:
            if model_type not in self.configurations:
                return {"error": f"Configuration not found for {model_type}"}

            # Remove from memory
            del self.configurations[model_type]

            # Remove file
            config_file = self.config_dir / f"{model_type}_config.json"
            if config_file.exists():
                config_file.unlink()

            return {"success": True}

        except Exception as e:
            self.logger.error(f"Failed to delete configuration: {e}")
            return {"error": str(e)}

    def get_all_configurations(self) -> Dict[str, Any]:
        """Get all configurations."""
        return {
            "configurations": self.configurations,
            "total": len(self.configurations),
            "timestamp": datetime.now().isoformat()
        }

    def get_environment_configurations(self, environment: Environment) -> Dict[str, Any]:
        """Get configurations for a specific environment."""
        env_configs = {
            model_type: config
            for model_type, config in self.configurations.items()
            if config.get("environment") == environment.value
        }

        return {
            "environment": environment.value,
            "configurations": env_configs,
            "total": len(env_configs)
        }

    def optimize_parameters(self,
                          model_type: str,
                          parameter_space: Dict[str, List[Any]],
                          optimization_metric: str = "mse") -> Dict[str, Any]:
        """
        Optimize model parameters using grid search.

        This would typically integrate with actual optimization routines.
        For now, returns mock optimization results.
        """
        try:
            if model_type not in self.configurations:
                return {"error": f"Configuration not found for {model_type}"}

            # Mock optimization - in real implementation, this would:
            # 1. Run cross-validation for each parameter combination
            # 2. Select best parameters based on optimization metric
            # 3. Update configuration with optimized parameters

            best_params = {}
            for param_name, param_values in parameter_space.items():
                best_params[param_name] = param_values[len(param_values) // 2]  # Mock selection

            # Update configuration
            update_result = self.update_configuration(model_type, {
                "parameters": best_params,
                "optimization_metric": optimization_metric,
                "optimized_at": datetime.now().isoformat()
            })

            if update_result.get("success"):
                return {
                    "success": True,
                    "best_parameters": best_params,
                    "optimization_metric": optimization_metric,
                    "optimization_time": datetime.now().isoformat()
                }
            else:
                return update_result

        except Exception as e:
            self.logger.error(f"Failed to optimize parameters: {e}")
            return {"error": str(e)}

    def export_configuration(self, model_type: str, format: str = "json") -> Dict[str, Any]:
        """Export configuration to file."""
        try:
            if model_type not in self.configurations:
                return {"error": f"Configuration not found for {model_type}"}

            config_data = self.configurations[model_type]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_type}_config_{timestamp}.{format}"

            if format == "json":
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
            elif format == "yaml":
                with open(filename, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:
                return {"error": f"Unsupported format: {format}"}

            return {
                "success": True,
                "filename": filename,
                "format": format,
                "exported_at": timestamp
            }

        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return {"error": str(e)}

    def import_configuration(self, filename: str) -> Dict[str, Any]:
        """Import configuration from file."""
        try:
            file_path = Path(filename)
            if not file_path.exists():
                return {"error": f"File not found: {filename}"}

            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    config_data = json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    return {"error": f"Unsupported file format: {file_path.suffix}"}

            model_type = config_data.get("model_type")
            if not model_type:
                return {"error": "Configuration missing model_type"}

            # Validate imported configuration
            config_class = self.model_config_classes.get(model_type)
            if config_class:
                config_obj = config_class(**config_data)
                errors = self._validate_configuration(config_obj)
                if errors:
                    return {"error": "Validation failed", "errors": errors}

            # Import configuration
            self.configurations[model_type] = config_data
            self._save_configuration(model_type, config_data)

            return {
                "success": True,
                "model_type": model_type,
                "imported_from": filename,
                "imported_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return {"error": str(e)}

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        summary = {
            "total_configurations": len(self.configurations),
            "environments": {},
            "model_types": list(self.configurations.keys()),
            "last_updated": None
        }

        # Count by environment
        for config in self.configurations.values():
            env = config.get("environment", "unknown")
            summary["environments"][env] = summary["environments"].get(env, 0) + 1

            # Track last update
            updated_at = config.get("updated_at", config.get("created_at"))
            if updated_at and (summary["last_updated"] is None or updated_at > summary["last_updated"]):
                summary["last_updated"] = updated_at

        return summary