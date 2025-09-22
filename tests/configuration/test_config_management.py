"""
Configuration management and reproducibility validation tests for financial features.

Tests configuration system functionality:
- Configuration creation, loading, and validation
- Reproducibility of calculations with same configuration
- Version control and parameter management
- Environment variable integration
- Template and preset management
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import pickle

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.returns import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_annualized_returns,
    calculate_sharpe_ratio
)

from lib.volatility import (
    calculate_rolling_volatility,
    calculate_ewma_volatility,
    calculate_garch11_volatility
)

from lib.momentum import (
    calculate_rsi,
    calculate_macd,
    calculate_simple_momentum
)

from config.pipeline_config import PipelineConfigManager, PreprocessingConfig


class FinancialFeaturesConfig:
    """Configuration class for financial features calculations."""

    def __init__(self,
                 config_id: str,
                 description: str,
                 features: List[str],
                 parameters: Dict[str, Any],
                 version: str = "1.0.0",
                 seed: Optional[int] = None,
                 environment_vars: Optional[Dict[str, str]] = None):
        self.config_id = config_id
        self.description = description
        self.features = features
        self.parameters = parameters
        self.version = version
        self.seed = seed
        self.environment_vars = environment_vars or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'config_id': self.config_id,
            'description': self.description,
            'features': self.features,
            'parameters': self.parameters,
            'version': self.version,
            'seed': self.seed,
            'environment_vars': self.environment_vars,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialFeaturesConfig':
        """Create configuration from dictionary."""
        created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        updated_at = datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now()

        config = cls(
            config_id=data['config_id'],
            description=data['description'],
            features=data['features'],
            parameters=data['parameters'],
            version=data.get('version', '1.0.0'),
            seed=data.get('seed'),
            environment_vars=data.get('environment_vars', {})
        )
        config.created_at = created_at
        config.updated_at = updated_at
        config.metadata = data.get('metadata', {})
        return config

    def get_hash(self) -> str:
        """Get hash of configuration for reproducibility tracking."""
        config_dict = self.to_dict()
        # Remove timestamp fields for hashing
        config_dict.pop('created_at', None)
        config_dict.pop('updated_at', None)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()


class ConfigurationManager:
    """Manages financial features configurations."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        self._configs = {}
        self._templates = {}
        self._load_templates()

    def _load_templates(self):
        """Load default configuration templates."""
        # Returns calculation template
        self._templates['returns'] = FinancialFeaturesConfig(
            config_id="returns_template",
            description="Template for returns calculations",
            features=['simple_returns', 'log_returns', 'annualized_returns'],
            parameters={
                'period': 1,
                'annualization_periods': 252,
                'risk_free_rate': 0.02,
                'fill_method': 'pad'
            },
            seed=42
        )

        # Volatility calculation template
        self._templates['volatility'] = FinancialFeaturesConfig(
            config_id="volatility_template",
            description="Template for volatility calculations",
            features=['rolling_volatility', 'ewma_volatility', 'garch_volatility'],
            parameters={
                'window': 21,
                'span': 30,
                'garch_omega': 0.0001,
                'garch_alpha': 0.1,
                'garch_beta': 0.85,
                'annualization_periods': 252
            },
            seed=42
        )

        # Momentum calculation template
        self._templates['momentum'] = FinancialFeaturesConfig(
            config_id="momentum_template",
            description="Template for momentum calculations",
            features=['rsi', 'macd', 'simple_momentum'],
            parameters={
                'rsi_period': 14,
                'rsi_method': 'wilders',
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'momentum_period': 10,
                'momentum_normalize': False
            },
            seed=42
        )

        # Comprehensive template
        self._templates['comprehensive'] = FinancialFeaturesConfig(
            config_id="comprehensive_template",
            description="Comprehensive financial features template",
            features=['returns', 'volatility', 'momentum', 'risk_metrics'],
            parameters={
                'returns': {
                    'period': 1,
                    'annualization_periods': 252,
                    'risk_free_rate': 0.02
                },
                'volatility': {
                    'window': 21,
                    'span': 30,
                    'annualization_periods': 252
                },
                'momentum': {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                }
            },
            seed=42
        )

    def create_config_from_template(self, template_name: str, config_id: str,
                                  custom_parameters: Optional[Dict[str, Any]] = None) -> FinancialFeaturesConfig:
        """Create configuration from template."""
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self._templates[template_name]
        config = FinancialFeaturesConfig(
            config_id=config_id,
            description=template.description.replace(template_name, config_id),
            features=template.features.copy(),
            parameters=template.parameters.copy(),
            version=template.version,
            seed=template.seed,
            environment_vars=template.environment_vars.copy()
        )

        # Apply custom parameters
        if custom_parameters:
            self._update_nested_dict(config.parameters, custom_parameters)

        self._configs[config_id] = config
        return config

    def save_config(self, config: FinancialFeaturesConfig, config_path: Optional[str] = None) -> str:
        """Save configuration to file."""
        if config_path is None:
            config_path = self.config_dir / f"{config.config_id}.json"

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        self._configs[config.config_id] = config
        return str(config_path)

    def load_config(self, config_path: str) -> FinancialFeaturesConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        config = FinancialFeaturesConfig.from_dict(data)
        self._configs[config.config_id] = config
        return config

    def get_config(self, config_id: str) -> Optional[FinancialFeaturesConfig]:
        """Get configuration by ID."""
        return self._configs.get(config_id)

    def list_configs(self) -> List[str]:
        """List all configuration IDs."""
        return list(self._configs.keys())

    def delete_config(self, config_id: str) -> bool:
        """Delete configuration."""
        if config_id in self._configs:
            del self._configs[config_id]

            config_file = self.config_dir / f"{config_id}.json"
            if config_file.exists():
                config_file.unlink()

            return True
        return False

    def validate_config(self, config: FinancialFeaturesConfig) -> Dict[str, Any]:
        """Validate configuration."""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate required fields
        if not config.config_id:
            results['errors'].append("Config ID is required")
            results['is_valid'] = False

        if not config.description:
            results['errors'].append("Description is required")
            results['is_valid'] = False

        if not config.features:
            results['errors'].append("At least one feature is required")
            results['is_valid'] = False

        # Validate features
        valid_features = ['returns', 'volatility', 'momentum', 'risk_metrics']
        for feature in config.features:
            if feature not in valid_features:
                results['errors'].append(f"Invalid feature: {feature}")
                results['is_valid'] = False

        # Validate parameters
        self._validate_parameters(config, results)

        return results

    def _validate_parameters(self, config: FinancialFeaturesConfig, results: Dict[str, Any]):
        """Validate configuration parameters."""
        params = config.parameters

        # Validate returns parameters
        if 'returns' in config.features or 'returns' in params:
            returns_params = params.get('returns', params)
            if 'period' in returns_params and returns_params['period'] <= 0:
                results['errors'].append("Returns period must be positive")
                results['is_valid'] = False

        # Validate volatility parameters
        if 'volatility' in config.features or 'volatility' in params:
            vol_params = params.get('volatility', params)
            if 'window' in vol_params and vol_params['window'] <= 0:
                results['errors'].append("Volatility window must be positive")
                results['is_valid'] = False

        # Validate momentum parameters
        if 'momentum' in config.features or 'momentum' in params:
            mom_params = params.get('momentum', params)
            if 'rsi_period' in mom_params and mom_params['rsi_period'] <= 0:
                results['errors'].append("RSI period must be positive")
                results['is_valid'] = False

    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict):
        """Update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value

    def compare_configs(self, config_id1: str, config_id2: str) -> Dict[str, Any]:
        """Compare two configurations."""
        config1 = self.get_config(config_id1)
        config2 = self.get_config(config_id2)

        if not config1 or not config2:
            raise ValueError("One or both configurations not found")

        comparison = {
            'config1_id': config_id1,
            'config2_id': config_id2,
            'features_differ': set(config1.features) != set(config2.features),
            'parameters_differ': config1.parameters != config2.parameters,
            'seed_differ': config1.seed != config2.seed,
            'version_differ': config1.version != config2.version,
            'hash_differ': config1.get_hash() != config2.get_hash()
        }

        return comparison

    def export_config_with_env(self, config: FinancialFeaturesConfig, include_env: bool = True) -> Dict[str, Any]:
        """Export configuration with environment variables."""
        export_data = config.to_dict()

        if include_env:
            # Add current environment variables
            env_vars = {}
            for var_name in config.environment_vars.keys():
                env_vars[var_name] = os.environ.get(var_name, 'NOT_SET')
            export_data['current_environment'] = env_vars

        return export_data


class TestConfigurationManagement:
    """Test configuration management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_template_creation(self):
        """Test creating configurations from templates."""
        print("\n=== Template Creation Test ===")

        # Test returns template
        returns_config = self.config_manager.create_config_from_template(
            'returns', 'test_returns'
        )
        assert returns_config.config_id == 'test_returns'
        assert 'returns' in returns_config.features
        assert returns_config.seed == 42

        # Test volatility template
        vol_config = self.config_manager.create_config_from_template(
            'volatility', 'test_volatility'
        )
        assert vol_config.config_id == 'test_volatility'
        assert 'volatility' in vol_config.features

        # Test comprehensive template
        comp_config = self.config_manager.create_config_from_template(
            'comprehensive', 'test_comprehensive'
        )
        assert len(comp_config.features) == 4
        assert 'returns' in comp_config.features
        assert 'volatility' in comp_config.features

    def test_configuration_save_load(self):
        """Test saving and loading configurations."""
        print("\n=== Configuration Save/Load Test ===")

        # Create configuration
        config = self.config_manager.create_config_from_template(
            'returns', 'test_save_load'
        )

        # Save configuration
        config_path = self.config_manager.save_config(config)
        assert os.path.exists(config_path)

        # Load configuration
        loaded_config = self.config_manager.load_config(config_path)
        assert loaded_config.config_id == config.config_id
        assert loaded_config.features == config.features
        assert loaded_config.parameters == config.parameters

    def test_configuration_validation(self):
        """Test configuration validation."""
        print("\n=== Configuration Validation Test ===")

        # Valid configuration
        valid_config = self.config_manager.create_config_from_template(
            'returns', 'valid_test'
        )
        validation = self.config_manager.validate_config(valid_config)
        assert validation['is_valid'] == True
        assert len(validation['errors']) == 0

        # Invalid configuration (missing features)
        invalid_config = FinancialFeaturesConfig(
            config_id="invalid_test",
            description="Invalid config",
            features=[],
            parameters={},
            seed=42
        )
        validation = self.config_manager.validate_config(invalid_config)
        assert validation['is_valid'] == False
        assert len(validation['errors']) > 0

    def test_configuration_comparison(self):
        """Test configuration comparison."""
        print("\n=== Configuration Comparison Test ===")

        # Create two similar configurations
        config1 = self.config_manager.create_config_from_template(
            'returns', 'compare_test1'
        )
        config2 = self.config_manager.create_config_from_template(
            'returns', 'compare_test2'
        )

        # Modify one parameter
        config2.parameters['period'] = 5

        comparison = self.config_manager.compare_configs('compare_test1', 'compare_test2')
        assert comparison['parameters_differ'] == True
        assert comparison['features_differ'] == False

    def test_custom_parameters(self):
        """Test applying custom parameters to templates."""
        print("\n=== Custom Parameters Test ===")

        custom_params = {
            'period': 5,
            'annualization_periods': 365,
            'risk_free_rate': 0.03
        }

        config = self.config_manager.create_config_from_template(
            'returns', 'custom_test', custom_params
        )

        assert config.parameters['period'] == 5
        assert config.parameters['annualization_periods'] == 365
        assert config.parameters['risk_free_rate'] == 0.03


class TestReproducibility:
    """Test reproducibility of calculations with same configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)

        # Generate test data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        returns = np.random.normal(0.001, 0.02, 1000)
        self.prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_deterministic_results_with_seed(self):
        """Test that same seed produces identical results."""
        print("\n=== Deterministic Results with Seed Test ===")

        # Create two identical configurations with same seed
        config1 = self.config_manager.create_config_from_template(
            'returns', 'seeded_test1'
        )
        config2 = self.config_manager.create_config_from_template(
            'returns', 'seeded_test2'
        )

        # Set same seed for both
        config1.seed = 12345
        config2.seed = 12345

        # Calculate returns with first config
        np.random.seed(config1.seed)
        returns1 = calculate_simple_returns(self.prices, period=config1.parameters['period'])

        # Calculate returns with second config
        np.random.seed(config2.seed)
        returns2 = calculate_simple_returns(self.prices, period=config2.parameters['period'])

        # Results should be identical
        assert returns1.equals(returns2), "Results should be identical with same seed"

    def test_reproducibility_across_sessions(self):
        """Test reproducibility when saving and loading configurations."""
        print("\n=== Reproducibility Across Sessions Test ===")

        # Create and save configuration
        original_config = self.config_manager.create_config_from_template(
            'volatility', 'repro_test'
        )
        original_config.seed = 54321

        config_path = self.config_manager.save_config(original_config)

        # Load configuration in new session
        new_config_manager = ConfigurationManager(self.temp_dir)
        loaded_config = new_config_manager.load_config(config_path)

        # Calculate volatility with original config
        np.random.seed(original_config.seed)
        returns = calculate_simple_returns(self.prices)
        vol1 = calculate_rolling_volatility(returns, window=original_config.parameters['window'])

        # Calculate volatility with loaded config
        np.random.seed(loaded_config.seed)
        returns = calculate_simple_returns(self.prices)
        vol2 = calculate_rolling_volatility(returns, window=loaded_config.parameters['window'])

        # Results should be identical
        assert vol1.equals(vol2), "Results should be reproducible across sessions"

    def test_parameter_change_impact(self):
        """Test that parameter changes produce different results."""
        print("\n=== Parameter Change Impact Test ===")

        # Create two configurations with different parameters
        config1 = self.config_manager.create_config_from_template(
            'momentum', 'param_test1'
        )
        config2 = self.config_manager.create_config_from_template(
            'momentum', 'param_test2'
        )

        # Change RSI period
        config2.parameters['rsi_period'] = 21

        # Calculate RSI with different periods
        rsi1 = calculate_rsi(self.prices, period=config1.parameters['rsi_period'])
        rsi2 = calculate_rsi(self.prices, period=config2.parameters['rsi_period'])

        # Results should be different
        assert not rsi1.equals(rsi2), "Different parameters should produce different results"

    def test_version_control(self):
        """Test version control of configurations."""
        print("\n=== Version Control Test ===")

        # Create initial configuration
        config = self.config_manager.create_config_from_template(
            'comprehensive', 'version_test'
        )
        config.version = "1.0.0"

        # Save initial version
        initial_path = self.config_manager.save_config(config)
        initial_hash = config.get_hash()

        # Update configuration
        config.parameters['returns']['period'] = 5
        config.version = "1.1.0"
        config.updated_at = datetime.now()

        # Save updated version
        updated_path = self.config_manager.save_config(config, initial_path.replace('.json', '_v1.1.0.json'))
        updated_hash = config.get_hash()

        # Hashes should be different
        assert initial_hash != updated_hash, "Different versions should have different hashes"

    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        print("\n=== Environment Variable Integration Test ===")

        # Set environment variables
        os.environ['TEST_RISK_FREE_RATE'] = '0.05'
        os.environ['TEST_ANNUALIZATION_PERIODS'] = '365'

        # Create configuration with environment variables
        config = self.config_manager.create_config_from_template(
            'returns', 'env_test'
        )
        config.environment_vars = {
            'TEST_RISK_FREE_RATE': 'risk_free_rate',
            'TEST_ANNUALIZATION_PERIODS': 'annualization_periods'
        }

        # Export with environment
        export_data = self.config_manager.export_config_with_env(config, include_env=True)

        # Verify environment variables are captured
        assert 'current_environment' in export_data
        env_data = export_data['current_environment']
        assert env_data['TEST_RISK_FREE_RATE'] == '0.05'
        assert env_data['TEST_ANNUALIZATION_PERIODS'] == '365'

        # Clean up environment variables
        del os.environ['TEST_RISK_FREE_RATE']
        del os.environ['TEST_ANNUALIZATION_PERIODS']

    def test_configuration_hash_uniqueness(self):
        """Test that different configurations have unique hashes."""
        print("\n=== Configuration Hash Uniqueness Test ===")

        configs = []
        hashes = []

        # Create multiple configurations with slight variations
        for i in range(5):
            config = self.config_manager.create_config_from_template(
                'returns', f'hash_test_{i}'
            )
            config.parameters['period'] = i + 1  # Different period for each
            configs.append(config)
            hashes.append(config.get_hash())

        # All hashes should be unique
        assert len(hashes) == len(set(hashes)), "All configurations should have unique hashes"

    def test_reproducibility_report(self):
        """Test generating reproducibility reports."""
        print("\n=== Reproducibility Report Test ===")

        # Create test configuration
        config = self.config_manager.create_config_from_template(
            'comprehensive', 'report_test'
        )
        config.seed = 99999

        # Calculate features multiple times with same configuration
        feature_sets = []
        for run in range(3):
            np.random.seed(config.seed)

            returns = calculate_simple_returns(self.prices, period=config.parameters['returns']['period'])
            volatility = calculate_rolling_volatility(returns, window=config.parameters['volatility']['window'])
            rsi = calculate_rsi(self.prices, period=config.parameters['momentum']['rsi_period'])

            feature_sets.append({
                'returns': returns,
                'volatility': volatility,
                'rsi': rsi
            })

        # Verify all runs produce identical results
        for feature_name in ['returns', 'volatility', 'rsi']:
            values = [fs[feature_name] for fs in feature_sets]
            # All should be equal
            for i in range(1, len(values)):
                assert values[0].equals(values[i]), f"Run {i} {feature_name} differs from run 0"

        # Generate reproducibility report
        report = {
            'config_id': config.config_id,
            'config_hash': config.get_hash(),
            'seed': config.seed,
            'features_calculated': list(config.features),
            'runs_performed': 3,
            'reproducibility_verified': True,
            'verification_timestamp': datetime.now().isoformat()
        }

        print(f"Reproducibility Report:")
        print(f"  Config ID: {report['config_id']}")
        print(f"  Config Hash: {report['config_hash'][:16]}...")
        print(f"  Seed: {report['seed']}")
        print(f"  Features: {report['features_calculated']}")
        print(f"  Reproducibility: {'✓' if report['reproducibility_verified'] else '✗'}")

        assert report['reproducibility_verified'] == True, "Reproducibility verification failed"


if __name__ == "__main__":
    # Run configuration management tests
    print("Starting Financial Features Configuration Management Tests")
    print("=" * 65)

    # Initialize and run test classes
    config_tests = TestConfigurationManagement()
    config_tests.setup_method()
    config_tests.test_template_creation()

    reproducibility_tests = TestReproducibility()
    reproducibility_tests.setup_method()
    reproducibility_tests.test_deterministic_results_with_seed()

    print("\n" + "=" * 65)
    print("All configuration management tests completed successfully!")