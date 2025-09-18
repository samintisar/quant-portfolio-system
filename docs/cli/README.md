# CLI Documentation for Quantitative Trading System

This document describes the command-line interfaces available for the quantitative trading system, providing comprehensive tools for environment management, dependency validation, testing, and configuration management.

## Available CLI Tools

### 1. Environment Management CLI (`quant-env`)

**Location**: `cli/quant-env.py`
**Purpose**: Manage Python virtual environments for quantitative trading

#### Commands

##### Environment Creation
```bash
# Create a new environment with Python 3.11
quant-env create my-trading-env --python 3.11

# Create environment with core dependencies
quant-env create my-trading-env --python 3.11 --with-deps

# Create environment in specific path
quant-env create my-trading-env --python 3.11 --path /path/to/envs
```

##### Environment Management
```bash
# List all environments
quant-env list

# Get environment information
quant-env info my-trading-env

# Validate environment integrity
quant-env validate my-trading-env

# Export environment configuration
quant-env export my-trading-env --output my-env-config.json

# Remove environment (requires --force)
quant-env remove my-trading-env --force
```

##### Activation/Deactivation
```bash
# Activate environment (note: requires shell integration)
quant-env activate my-trading-env

# Deactivate current environment
quant-env deactivate
```

#### Output Formats

All commands support both human-readable and JSON output:

```bash
# Human-readable (default)
quant-env list

# JSON format
quant-env list --json

# Verbose output
quant-env list --verbose

# Dry-run mode
quant-env create test-env --dry-run
```

#### Error Handling

The CLI provides comprehensive error reporting with appropriate exit codes:
- `0`: Success
- `1`: Invalid input
- `2`: Operation failed
- `3`: Permission denied
- `4`: Timeout

### 2. Dependency Management CLI (`quant-dep`)

**Location**: `cli/quant-dep.py`
**Purpose**: Validate and manage package dependencies in trading environments

#### Commands

##### Dependency Validation
```bash
# Validate all dependencies in an environment
quant-dep validate --env my-trading-env

# Validate specific package
quant-dep validate --env my-trading-env --package numpy

# Check for security vulnerabilities
quant-dep validate --env my-trading-env --package pandas
```

##### Dependency Installation
```bash
# Install a package
quant-dep install --env my-trading-env --package numpy --version 2.1.0

# Install latest version
quant-dep install --env my-trading-env --package scipy

# Uninstall a package
quant-dep uninstall --env my-trading-env --package old-package
```

##### Dependency Information
```bash
# List all dependencies
quant-dep list --env my-trading-env

# Get package information
quant-dep info --env my-trading-env --package numpy

# Upgrade a package
quant-dep upgrade --env my-trading-env --package numpy --version 2.2.0
```

##### Conflict Detection
```bash
# Check for dependency conflicts
quant-dep check-conflicts --env my-trading-env

# Export dependency manifest
quant-dep export --env my-trading-env --output dependencies.json
```

#### Version Pinning and Security

The dependency management system includes:
- **Version pinning**: Exact version specifications for reproducibility
- **Security validation**: Checks against known vulnerabilities
- **Compatibility verification**: Python version compatibility checks
- **Conflict resolution**: Automatic detection of dependency conflicts

### 3. Testing Framework CLI (`quant-test`)

**Location**: `cli/quant-test.py`
**Purpose**: Execute tests, statistical validation, and performance benchmarks

#### Commands

##### Test Execution
```bash
# Run all tests
quant-test run

# Run specific test types
quant-test run --type unit
quant-test run --type integration
quant-test run --type contract

# Use specific testing framework
quant-test run --type unit --framework pytest
```

##### Library Validation
```bash
# Validate a trading library
quant-test validate --library momentum-strategy

# Run statistical tests
quant-test statistical --library momentum-strategy --test normality

# Run backtests
quant-test backtest --library momentum-strategy --start 2023-01-01 --end 2023-12-31
```

##### Coverage and Reporting
```bash
# Calculate test coverage
quant-test coverage --library momentum-strategy

# Generate test reports
quant-test report --library momentum-strategy --output test-report.json

# Run performance benchmarks
quant-test benchmark --test portfolio-optimization --iterations 1000
```

#### Statistical Validation

The testing framework includes comprehensive statistical validation:
- **Normality tests**: Shapiro-Wilk, Kolmogorov-Smirnov
- **Stationarity tests**: Augmented Dickey-Fuller
- **Autocorrelation tests**: Ljung-Box, Durbin-Watson
- **Backtesting validation**: Walk-forward analysis, out-of-sample testing

### 4. Configuration Management CLI (`quant-config`)

**Location**: `cli/quant-config.py`
**Purpose**: Manage system and library configurations

#### Commands

##### Configuration Initialization
```bash
# Initialize configuration system
quant-config init --type TOML

# Initialize with repository integration
quant-config init --type TOML --repo /path/to/repo
```

##### Configuration Management
```bash
# Validate configuration
quant-config validate --config config.toml
quant-config validate --environment development

# Get configuration values
quant-config get --key trading.initial_capital
quant-config get --environment development --key risk.max_position_size

# Set configuration values
quant-config set --key trading.initial_capital --value 1000000
quant-config set --environment testing --key backtest.start_date --value "2023-01-01"
```

##### Configuration Analysis
```bash
# List configurations
quant-config list
quant-config list --environment production
quant-config list --structure

# Synchronize with repository
quant-config sync --repo /path/to/repo

# Detect configuration drift
quant-config drift --reference base-config.toml --current current-config.toml

# Export configurations
quant-config export --environment production --output prod-config.toml
```

##### Library-Specific Configuration
```bash
# Get library configuration
quant-config library --library momentum-strategy --action get

# Update library configuration
quant-config library --library momentum-strategy --action update --updates "lookback_period=20,risk_factor=0.02"
```

## Configuration Files

### Environment Configuration

```toml
[environment]
name = "quant-trading-env"
python_version = "3.11"
base_path = "/path/to/envs"

[dependencies]
numpy = "2.1.0"
pandas = "2.2.1"
scipy = "1.13.0"
```

### Trading Configuration

```toml
[trading]
initial_capital = 1000000
max_position_size = 0.05
risk_free_rate = 0.02

[backtesting]
start_date = "2023-01-01"
end_date = "2023-12-31"
benchmark = "SPY"

[risk_management]
max_drawdown = 0.15
var_confidence = 0.95
position_limits = { single_name = 0.05, sector = 0.20 }
```

## Integration Examples

### Complete Workflow Example

```bash
# 1. Create environment
quant-env create my-trading-system --python 3.11 --with-deps

# 2. Initialize configuration
quant-config init --type TOML

# 3. Configure trading parameters
quant-config set --key trading.initial_capital --value 1000000
quant-config set --key risk.max_drawdown --value 0.15

# 4. Install additional dependencies
quant-dep install --env my-trading-system --package yfinance --version 0.2.28
quant-dep install --env my-trading-system --package vectorbt --version 1.2.0

# 5. Run validation tests
quant-test validate --library momentum-strategy
quant-test run --type integration

# 6. Generate reports
quant-test coverage --library momentum-strategy
quant-config export --environment production --output final-config.toml
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Trading System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup environment
        run: |
          python cli/quant-env.py create ci-env --python 3.11 --with-deps
          python cli/quant-dep.py install --env ci-env --package pytest

      - name: Validate configuration
        run: python cli/quant-config.py validate --environment testing

      - name: Run tests
        run: python cli/quant-test.py run --type all --framework pytest

      - name: Generate coverage report
        run: python cli/quant-test.py coverage --library momentum-strategy
```

## Best Practices

### Environment Management

1. **Always use virtual environments** for trading system development
2. **Pin dependency versions** for reproducible research
3. **Regularly validate environments** for integrity
4. **Export environment configurations** for team sharing

### Configuration Management

1. **Use environment-specific configurations** (development, testing, production)
2. **Validate configurations** before deployment
3. **Monitor configuration drift** in production
4. **Version control configuration files** with appropriate security measures

### Testing

1. **Run contract tests** before implementation
2. **Perform statistical validation** on all trading strategies
3. **Maintain high test coverage** (>80% target)
4. **Regular performance benchmarking** to identify optimization opportunities

## Performance Considerations

All CLI tools are optimized for performance with the following targets:
- **Environment validation**: < 5 seconds
- **Dependency validation**: < 2 seconds per package
- **Test execution**: < 1ms per data point
- **Configuration operations**: < 100ms

For large datasets or complex operations, consider using:
- `--dry-run` flag for validation without execution
- `--json` output for programmatic processing
- Batch operations for multiple items

## Security Considerations

- **Never commit sensitive configuration** to version control
- **Use environment variables** for secrets and API keys
- **Regularly validate dependencies** for security vulnerabilities
- **Restrict CLI permissions** based on user roles
- **Audit configuration changes** in production environments

## Troubleshooting

### Common Issues

1. **Permission denied errors**: Ensure proper file permissions and virtual environment activation
2. **Dependency conflicts**: Use `quant-dep check-conflicts` to identify issues
3. **Configuration validation failures**: Check syntax and required fields
4. **Test failures**: Review error messages and check environment setup

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
quant-env create test-env --verbose --dry-run
quant-test run --type unit --verbose
quant-config validate --config config.toml --verbose
```

### Support

For additional support:
- Check the log files in `logs/` directory
- Review the project documentation in `docs/`
- Open an issue in the project repository