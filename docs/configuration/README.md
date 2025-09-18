# Configuration Management Documentation

This document describes the configuration system for the quantitative trading system, including structure, validation, and environment-specific settings.

## Configuration Architecture

The configuration system uses a hierarchical approach with support for multiple environments and inheritance patterns.

### Configuration Types

- **Base Configuration**: Core system settings
- **Environment Configurations**: Development, Testing, Production
- **Library Configurations**: Strategy-specific settings
- **User Configurations**: Personalized overrides

### Supported Formats

- **TOML** (default): Human-readable, excellent for configuration
- **YAML**: Widely used, good for complex structures
- **JSON**: Machine-readable, programmatic access

## Configuration Structure

### Base Configuration (`config/base.toml`)

```toml
# System configuration
[system]
name = "quant-trading-system"
version = "1.0.0"
environment = "development"
debug = false
log_level = "INFO"

# Trading configuration
[trading]
initial_capital = 1000000.0
currency = "USD"
timezone = "UTC"
max_position_size = 0.05  # 5% of portfolio
min_position_size = 0.01  # 1% of portfolio

# Data configuration
[data]
default_provider = "yfinance"
cache_enabled = true
cache_duration = 86400  # 24 hours in seconds
max_lookback_days = 3650  # 10 years

[data.providers.yfinance]
api_key = null
rate_limit = 100  # requests per minute
timeout = 30  # seconds
```

### Environment Configurations

#### Development (`config/development.toml`)

```toml
[system]
environment = "development"
debug = true
log_level = "DEBUG"

[trading]
initial_capital = 100000.0  # Smaller capital for development

[data]
cache_enabled = true
cache_duration = 604800  # 7 days for development

[backtesting]
start_date = "2023-01-01"
end_date = "2023-12-31"
benchmark = "SPY"

[risk]
max_drawdown = 0.20  # More lenient for development
var_confidence = 0.90
```

#### Testing (`config/testing.toml`)

```toml
[system]
environment = "testing"
debug = false
log_level = "INFO"

[trading]
initial_capital = 1000000.0

[data]
cache_enabled = true
cache_duration = 2592000  # 30 days for testing

[testing]
coverage_target = 80.0
performance_threshold = 0.001  # 1ms per operation
statistical_significance = 0.05

[risk]
max_drawdown = 0.15
var_confidence = 0.95
```

#### Production (`config/production.toml`)

```toml
[system]
environment = "production"
debug = false
log_level = "WARNING"

[trading]
initial_capital = 10000000.0  # Larger capital for production

[data]
cache_enabled = true
cache_duration = 86400  # 24 hours for production

[security]
api_key_encrypted = true
audit_trail = true
access_logging = true

[risk]
max_drawdown = 0.10  # Strict risk limits for production
var_confidence = 0.99
position_limits = { single_name = 0.05, sector = 0.20, total = 1.0 }

[monitoring]
enabled = true
health_check_interval = 60
alert_thresholds = { drawdown = 0.08, volatility = 0.25 }
```

### Library Configuration Template

```toml
# Momentum strategy configuration
[strategy.momentum]
name = "Dual Moving Average Crossover"
type = "momentum"
enabled = true

[strategy.momentum.parameters]
lookback_short = 20
lookback_long = 50
entry_threshold = 0.02
exit_threshold = 0.01

[strategy.momentum.risk]
max_position_size = 0.05
stop_loss = 0.03
take_profit = 0.06

[strategy.momentum.symbols]
universe = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
exclude = ["VIX", "UVXY"]

[strategy.momentum.backtest]
start_date = "2020-01-01"
end_date = "2023-12-31"
benchmark = "SPY"
commission = 0.001
slippage = 0.0005
```

## Configuration Validation

### Schema Definition

The configuration system includes comprehensive validation rules:

```python
# Example validation rules
validation_rules = {
    "trading.initial_capital": {
        "type": "float",
        "min": 0,
        "required": True
    },
    "risk.max_drawdown": {
        "type": "float",
        "min": 0,
        "max": 1,
        "required": True
    },
    "data.cache_duration": {
        "type": "int",
        "min": 0,
        "max": 31536000  # 1 year
    }
}
```

### Configuration Classes

#### ConfigurationSystem

```python
from data.src.models.configuration_system import ConfigurationSystem

# Initialize configuration system
config_system = ConfigurationSystem(config_type="TOML")

# Create base configuration
config_system.create_base_config()

# Create environment configurations
config_system.create_environment_configs()

# Load and validate configuration
config = config_system.load_config("config/production.toml")
validation_result = config_system.validate_config(config)
```

#### Configuration Operations

```python
# Get configuration value
initial_capital = config_system.get_value(
    "config/production.toml",
    "trading.initial_capital",
    default=1000000.0
)

# Set configuration value
success = config_system.set_value(
    "config/production.toml",
    "trading.initial_capital",
    2000000.0
)

# Synchronize with repository
repo = RepositoryStructure(root_path="/path/to/repo")
config_system.sync_with_repository(repo)
```

## Environment Inheritance

### Inheritance Pattern

Configurations support inheritance with environment-specific overrides:

```toml
# Base configuration sets defaults
[trading]
initial_capital = 1000000.0
max_position_size = 0.05

# Development inherits and overrides
[trading]
initial_capital = 100000.0  # Override for development
# max_position_size inherited from base

# Production inherits and overrides
[trading]
initial_capital = 10000000.0  # Override for production
max_position_size = 0.03    # Override for production
```

### Environment Resolution

The configuration system resolves values using this priority order:
1. Environment-specific configuration
2. Base configuration
3. Default values
4. Schema defaults

## Configuration Management CLI

### Commands

```bash
# Initialize configuration system
quant-config init --type TOML

# Validate configuration
quant-config validate --config config/production.toml

# Get configuration value
quant-config get --key trading.initial_capital

# Set configuration value
quant-config set --key trading.initial_capital --value 2000000

# List all configurations
quant-config list --structure

# Synchronize with repository
quant-config sync --repo /path/to/repo

# Detect configuration drift
quant-config drift --reference base.toml --current current.toml

# Export configuration
quant-config export --environment production --output prod-config.toml
```

## Configuration Security

### Sensitive Data Handling

```toml
# Use environment variables for sensitive data
[api]
key = "${API_KEY}"  # Will be resolved from environment
secret = "${API_SECRET}"  # Will be resolved from environment

[database]
host = "${DB_HOST}"
port = "${DB_PORT}"
username = "${DB_USER}"
password = "${DB_PASSWORD}"
```

### Encryption Support

The configuration system supports encryption for sensitive values:

```python
# Encrypt sensitive configuration
config_system.encrypt_value("api_secret", "my_secret_key")

# Decrypt configuration
secret_value = config_system.decrypt_value("api_secret", "my_secret_key")
```

## Configuration Templates

### Strategy Templates

```toml
# Mean reversion strategy template
[strategy.mean_reversion]
name = "Mean Reversion"
type = "mean_reversion"
enabled = true

[strategy.mean_reversion.parameters]
lookback_period = 20
entry_z_score = 2.0
exit_z_score = 0.5
bb_period = 20
bb_std = 2.0

[strategy.mean_reversion.risk]
max_position_size = 0.05
stop_loss = 0.02
take_profit = 0.04

[strategy.mean_reversion.filters]
min_volume = 1000000
min_price = 10.0
max_price = 1000.0
```

### Risk Management Templates

```toml
# Conservative risk profile
[risk.conservative]
max_drawdown = 0.08
var_confidence = 0.99
position_limits = { single_name = 0.03, sector = 0.15, total = 1.0 }
leverage_limit = 1.0

# Moderate risk profile
[risk.moderate]
max_drawdown = 0.12
var_confidence = 0.95
position_limits = { single_name = 0.05, sector = 0.20, total = 1.0 }
leverage_limit = 1.5

# Aggressive risk profile
[risk.aggressive]
max_drawdown = 0.20
var_confidence = 0.90
position_limits = { single_name = 0.08, sector = 0.30, total = 1.0 }
leverage_limit = 2.0
```

## Configuration Monitoring

### Drift Detection

```python
# Monitor configuration drift
reference_config = config_system.load_config("reference.toml")
current_config = config_system.load_config("current.toml")

drift_analysis = config_system.detect_drift(reference_config, current_config)

if drift_analysis["has_drift"]:
    print(f"Configuration drift detected: {drift_analysis['drift_score']}")
    print(f"Changed keys: {drift_analysis['changed_keys']}")
```

### Audit Logging

```python
# Enable configuration audit logging
config_system.enable_audit_logging("config_audit.log")

# All configuration changes will be logged
config_system.set_value("config.toml", "trading.initial_capital", 1500000)
# Log entry: 2024-01-15T10:30:00 - SET trading.initial_capital = 1500000.0
```

## Best Practices

### Configuration Management

1. **Version control configurations** (except sensitive data)
2. **Use environment variables** for secrets and API keys
3. **Validate configurations** before deployment
4. **Document all configuration options**
5. **Use configuration inheritance** to reduce duplication

### Security Practices

1. **Never commit sensitive data** to version control
2. **Use encryption** for stored secrets
3. **Implement access controls** for configuration files
4. **Audit configuration changes** in production
5. **Regular security reviews** of configuration system

### Performance Considerations

1. **Cache configuration values** for frequently accessed settings
2. **Lazy loading** for large configuration sections
3. **Configuration hot-reloading** for non-critical updates
4. **Memory optimization** for configuration storage
5. **Efficient validation** to minimize startup time

## Troubleshooting

### Common Issues

1. **Configuration validation failures**: Check schema and required fields
2. **Environment variable resolution**: Verify variable names and values
3. **File permission errors**: Ensure proper access rights
4. **Configuration drift**: Regular synchronization with repository
5. **Performance issues**: Optimize validation and caching

### Debug Mode

Enable detailed logging for configuration issues:

```bash
# Enable debug logging
export CONFIG_DEBUG=true
quant-config validate --config config.toml --verbose
```

## Configuration API Reference

### ConfigurationSystem Class

#### Methods

- `load_config(file_path: str) -> dict`: Load configuration from file
- `validate_config(config: dict) -> dict`: Validate configuration against schema
- `get_value(file_path: str, key_path: str, default=None) -> any`: Get configuration value
- `set_value(file_path: str, key_path: str, value: any) -> bool`: Set configuration value
- `create_base_config() -> bool`: Create base configuration file
- `create_environment_configs() -> bool`: Create environment configuration files
- `sync_with_repository(repo: RepositoryStructure) -> bool`: Sync with repository
- `detect_drift(reference: dict, current: dict) -> dict`: Detect configuration drift
- `export_configuration(output_path: str, environment: str = None) -> bool`: Export configuration

#### Attributes

- `config_type`: Configuration format (TOML, YAML, JSON)
- `base_config_path`: Path to base configuration file
- `environment_configs`: Dictionary of environment configuration paths
- `schema_definition`: Configuration schema definition
- `validation_rules`: Configuration validation rules
- `audit_logging`: Enable audit logging flag

### Configuration Examples

For complete configuration examples, see the `config/` directory and the configuration templates in `docs/configuration/templates/`.