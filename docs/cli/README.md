# CLI Reference Guide

This comprehensive guide covers all command-line interfaces available in the Quantitative Trading System.

## 🚀 Overview

The system provides CLI interfaces for all major modules:
- **Data Preprocessing**: Clean, validate, and normalize financial data
- **Feature Generation**: Create financial features and indicators
- **Quality Assessment**: Evaluate data quality and generate reports
- **Rules Management**: Manage validation and processing rules
- **Portfolio Operations**: Optimize portfolios and analyze performance

## 📋 Data Preprocessing CLI

### Preprocess Command

**Purpose**: Clean, validate, and normalize financial data using configured pipelines

**Usage**:
```bash
python -m data.src.cli.preprocess \
    --pipeline-id my_pipeline \
    --input-path data/raw/ \
    --output-path data/processed/ \
    [--config-path config/pipeline_config.py] \
    [--quality-threshold 0.95] \
    [--log-level INFO] \
    [--save-quality-report] \
    [--dry-run]
```

**Parameters**:
- `--pipeline-id`: Unique identifier for the pipeline configuration
- `--input-path`: Directory containing raw data files
- `--output-path`: Directory to save processed data
- `--config-path`: Path to pipeline configuration file (optional)
- `--quality-threshold`: Minimum quality score threshold (0.0-1.0)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--save-quality-report`: Save detailed quality assessment report
- `--dry-run`: Validate configuration without processing data

**Examples**:
```bash
# Basic preprocessing
python -m data.src.cli.preprocess \
    --pipeline-id equity_pipeline \
    --input-path data/raw/equity/ \
    --output-path data/processed/equity/

# High-quality processing with report
python -m data.src.cli.preprocess \
    --pipeline-id high_quality_pipeline \
    --input-path data/raw/ \
    --output-path data/processed/ \
    --quality-threshold 0.98 \
    --save-quality-report \
    --log-level DEBUG
```

### Quality Report Command

**Purpose**: Generate detailed data quality assessment reports

**Usage**:
```bash
python -m data.src.cli.quality_report \
    --dataset-id my_dataset \
    --input-path data/processed/ \
    --output-path reports/ \
    [--format html] \
    [--include-charts] \
    [--detailed-metrics]
```

**Parameters**:
- `--dataset-id`: Unique identifier for the dataset
- `--input-path`: Directory containing processed data files
- `--output-path`: Directory to save quality reports
- `--format`: Report format (html, json, csv)
- `--include-charts`: Include visual charts in the report
- `--detailed-metrics`: Include detailed statistical metrics

**Examples**:
```bash
# Generate HTML report with charts
python -m data.src.cli.quality_report \
    --dataset-id equity_data_2023 \
    --input-path data/processed/equity/ \
    --output-path reports/ \
    --format html \
    --include-charts

# Generate detailed JSON report
python -m data.src.cli.quality_report \
    --dataset-id portfolio_data \
    --input-path data/processed/ \
    --output-path reports/ \
    --format json \
    --detailed-metrics
```

### Batch Process Command

**Purpose**: Process multiple datasets in batch mode

**Usage**:
```bash
python -m data.src.cli.batch_process \
    --config-file config/batch_config.yaml \
    [--parallel-workers 4] \
    [--max-retries 3] \
    [--timeout 3600]
```

**Parameters**:
- `--config-file`: YAML configuration file for batch processing
- `--parallel-workers`: Number of parallel processing workers
- `--max-retries`: Maximum number of retry attempts for failed jobs
- `--timeout`: Timeout in seconds for each processing job

**Example Configuration**:
```yaml
datasets:
  - name: "equity_data"
    input_path: "data/raw/equity/"
    output_path: "data/processed/equity/"
    pipeline_id: "equity_pipeline"
    quality_threshold: 0.95

  - name: "fixed_income_data"
    input_path: "data/raw/fixed_income/"
    output_path: "data/processed/fixed_income/"
    pipeline_id: "fixed_income_pipeline"
    quality_threshold: 0.90

global_settings:
  log_level: "INFO"
  save_quality_reports: true
  backup_on_failure: true
```

## 🎯 Feature Generation CLI

### Generate Features Command

**Purpose**: Generate financial features and indicators from processed data

**Usage**:
```bash
python -m data.src.cli.feature_generator \
    --input-data data/processed/ \
    --output-path features/ \
    [--config-file config/features.yaml] \
    [--feature-types returns,volatility,momentum] \
    [--time-windows 5,21,63] \
    [--symbols AAPL,GOOGL,MSFT]
```

**Parameters**:
- `--input-data`: Path to processed data files
- `--output-path`: Directory to save generated features
- `--config-file`: YAML configuration file for feature generation
- `--feature-types`: Comma-separated list of feature types
- `--time-windows`: Comma-separated list of time windows
- `--symbols`: Comma-separated list of symbols to process

**Available Feature Types**:
- `returns`: Price returns for different periods
- `volatility`: Volatility measures and indicators
- `momentum`: Momentum and trend indicators
- `volume`: Volume-based indicators
- `technical`: Technical analysis indicators
- `statistical`: Statistical features and measures

**Examples**:
```bash
# Generate basic features for specific symbols
python -m data.src.cli.feature_generator \
    --input-data data/processed/ \
    --output-path features/ \
    --feature-types returns,volatility \
    --time-windows 5,21 \
    --symbols AAPL,GOOGL,MSFT

# Generate all features using configuration file
python -m data.src.cli.feature_generator \
    --input-data data/processed/ \
    --output-path features/ \
    --config-file config/features.yaml
```

### Feature Validation Command

**Purpose**: Validate generated features for statistical consistency

**Usage**:
```bash
python -m data.src.cli.validate_features \
    --input-path features/ \
    --output-path reports/ \
    [--significance-level 0.05] \
    [--correlation-threshold 0.95] \
    [--generate-report]
```

**Parameters**:
- `--input-path`: Directory containing feature files
- `--output-path`: Directory to save validation reports
- `--significance-level`: Statistical significance level for tests
- `--correlation-threshold`: Threshold for high correlation warnings
- `--generate-report`: Generate detailed validation report

## 📊 Quality Assessment CLI

### Data Quality Check Command

**Purpose**: Check data quality against predefined thresholds

**Usage**:
```bash
python -m data.src.cli.quality_check \
    --dataset-id my_dataset \
    --input-path data/processed/ \
    [--thresholds-file config/quality_thresholds.yaml] \
    [--fail-on-error] \
    [--export-metrics]
```

**Parameters**:
- `--dataset-id`: Unique identifier for the dataset
- `--input-path`: Directory containing data files
- `--thresholds-file`: YAML file with quality thresholds
- `--fail-on-error`: Exit with error code if quality thresholds not met
- `--export-metrics`: Export quality metrics to file

**Quality Metrics Checked**:
- **Completeness**: Percentage of non-missing values
- **Accuracy**: Deviation from expected values
- **Consistency**: Temporal consistency checks
- **Timeliness**: Data freshness and latency
- **Validity**: Data type and format validation

### Monitor Quality Command

**Purpose**: Continuously monitor data quality and generate alerts

**Usage**:
```bash
python -m data.src.cli.quality_monitor \
    --datasets config/datasets.yaml \
    --check-interval 300 \
    [--alert-email admin@example.com] \
    [--webhook-url https://hooks.example.com/quality] \
    [--log-file logs/quality_monitor.log]
```

**Parameters**:
- `--datasets`: YAML file listing datasets to monitor
- `--check-interval`: Interval in seconds between checks
- `--alert-email`: Email address for quality alerts
- `--webhook-url`: Webhook URL for real-time notifications
- `--log-file`: Path to monitoring log file

## 📋 Rules Management CLI

### Rules Management Commands

**Purpose**: Manage data validation and processing rules

**Usage**:
```bash
# List all rules
python -m data.src.cli.feature_commands rules list

# Add a new rule
python -m data.src.cli.feature_commands rules add \
    --name "price_validation" \
    --type "validation" \
    --conditions '{"field": "close", "operator": "greater_than", "value": 0}' \
    --actions '[{"type": "flag", "severity": "error"}]'

# Update an existing rule
python -m data.src.cli.feature_commands rules update \
    --rule-id "price_validation" \
    --conditions '{"field": "close", "operator": "greater_than", "value": 0.01}'

# Delete a rule
python -m data.src.cli.feature_commands rules delete --rule-id "price_validation"

# Validate rules syntax
python -m data.src.cli.feature_commands rules validate \
    --rules-file config/validation_rules.yaml
```

**Rule Types**:
- `validation`: Data validation rules
- `transformation`: Data transformation rules
- `filtering`: Data filtering rules
- `aggregation`: Data aggregation rules
- `notification`: Notification and alerting rules

**Rule Components**:
- **Conditions**: Logical conditions for rule triggering
- **Actions**: Actions to take when conditions are met
- **Severity**: Severity level (info, warning, error, critical)
- **Priority**: Rule execution priority

## 🏦 Portfolio Operations CLI

### Portfolio Optimization Command

**Purpose**: Optimize portfolio allocation using various optimization methods

**Usage**:
```bash
python -m portfolio.src.cli.optimize \
    --returns-data data/returns.csv \
    --config-file config/optimization.yaml \
    --output-path results/ \
    [--method mean_variance] \
    [--objective sharpe_ratio] \
    [--constraints-file config/constraints.yaml]
```

**Parameters**:
- `--returns-data`: Path to returns data file
- `--config-file`: Optimization configuration file
- `--output-path`: Directory to save optimization results
- `--method`: Optimization method (mean_variance, black_litterman, risk_parity)
- `--objective`: Optimization objective (sharpe_ratio, min_variance, max_return)
- `--constraints-file`: Portfolio constraints configuration

**Optimization Methods**:
- `mean_variance`: Classical mean-variance optimization
- `black_litterman`: Black-Litterman model with views
- `risk_parity`: Risk parity allocation
- `hierarchical_risk_parity`: Hierarchical risk parity
- `minimum_variance`: Minimum variance portfolio

### Performance Analysis Command

**Purpose**: Analyze portfolio performance and generate reports

**Usage**:
```bash
python -m portfolio.src.cli.analyze_performance \
    --portfolio-file results/portfolio.json \
    --benchmark-data data/benchmark.csv \
    --output-path reports/ \
    [--analysis-period 252] \
    [--confidence-level 0.95] \
    [--generate-charts]
```

**Parameters**:
- `--portfolio-file`: Path to portfolio allocation file
- `--benchmark-data`: Path to benchmark returns data
- `--output-path`: Directory to save analysis reports
- `--analysis-period`: Analysis period in trading days
- `--confidence-level`: Confidence level for risk metrics
- `--generate-charts`: Generate performance charts

**Performance Metrics**:
- **Return Metrics**: Total return, annual return, cumulative return
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio
- **Drawdown Metrics**: Maximum drawdown, drawdown duration
- **Risk-Adjusted Metrics**: Information ratio, Treynor ratio
- **Distribution Metrics**: VaR, CVaR, expected shortfall

### Backtesting Command

**Purpose**: Backtest trading strategies on historical data

**Usage**:
```bash
python -m strategies.src.cli.backtest \
    --strategy-file strategies/momentum_strategy.py \
    --data-path data/processed/ \
    --config-file config/backtest.yaml \
    --output-path results/ \
    [--initial-capital 1000000] \
    [--transaction-costs 0.001] \
    [--generate-report]
```

**Parameters**:
- `--strategy-file`: Path to strategy implementation file
- `--data-path`: Path to historical data files
- `--config-file`: Backtest configuration file
- `--output-path`: Directory to save backtest results
- `--initial-capital`: Initial capital for backtest
- `--transaction-costs`: Transaction costs as percentage
- `--generate-report`: Generate detailed backtest report

## ⚙️ Configuration CLI

### Configuration Management Commands

**Purpose**: Manage system configurations

**Usage**:
```bash
# Create default configuration
python -m config.cli create_default \
    --config-type pipeline \
    --output-path config/default_pipeline.yaml

# Validate configuration
python -m config.cli validate \
    --config-file config/pipeline.yaml \
    --config-type pipeline

# Update configuration
python -m config.cli update \
    --config-file config/pipeline.yaml \
    --updates '{"quality_thresholds": {"completeness": 0.98}}'

# Compare configurations
python -m config.cli compare \
    --config-file1 config/pipeline_v1.yaml \
    --config-file2 config/pipeline_v2.yaml \
    --output-path reports/config_comparison.html
```

## 📈 Monitoring CLI

### System Monitoring Command

**Purpose**: Monitor system performance and resource usage

**Usage**:
```bash
python -m monitoring.cli.system_monitor \
    --interval 60 \
    --duration 3600 \
    [--metrics cpu,memory,disk,network] \
    [--output-file metrics/system_metrics.csv] \
    [--alert-thresholds config/alert_thresholds.yaml]
```

**Parameters**:
- `--interval`: Monitoring interval in seconds
- `--duration`: Total monitoring duration in seconds
- `--metrics`: Comma-separated list of metrics to monitor
- `--output-file`: Path to save monitoring data
- `--alert-thresholds`: Configuration for alert thresholds

### Performance Benchmark Command

**Purpose**: Run performance benchmarks for system components

**Usage**:
```bash
python -m monitoring.cli.benchmark \
    --components preprocessing,feature_generation,optimization \
    --data-size large \
    --iterations 5 \
    --output-path benchmarks/ \
    [--generate-report]
```

**Parameters**:
- `--components`: Comma-separated list of components to benchmark
- `--data-size`: Data size category (small, medium, large, xlarge)
- `--iterations`: Number of benchmark iterations
- `--output-path`: Directory to save benchmark results
- `--generate-report`: Generate detailed benchmark report

## 🚨 Error Handling and Exit Codes

### Exit Codes

- **0**: Success - Command executed successfully
- **1**: General Error - Generic error occurred
- **2**: Configuration Error - Invalid configuration provided
- **3**: Data Error - Data quality or format issues
- **4**: Permission Error - Insufficient permissions
- **5**: Resource Error - Insufficient system resources
- **6**: Timeout Error - Operation timed out
- **7**: Validation Error - Input validation failed
- **8**: Network Error - Network connectivity issues

### Error Handling

All CLI commands include comprehensive error handling:

```bash
# Example error output
$ python -m data.src.cli.preprocess --pipeline-id invalid_pipeline
ERROR: Pipeline configuration not found: invalid_pipeline
Available pipelines:
- equity_pipeline
- fixed_income_pipeline
- crypto_pipeline

# Debug mode with verbose output
$ python -m data.src.cli.preprocess --pipeline-id equity_pipeline --log-level DEBUG
DEBUG: Loading configuration from config/pipeline_config.py
DEBUG: Validating input directory: data/raw/
DEBUG: Initializing preprocessing pipeline
INFO: Starting data preprocessing...
```

## 🔄 Advanced Usage

### Pipeline Chaining

**Purpose**: Chain multiple CLI commands for complex workflows

**Usage**:
```bash
#!/bin/bash
# Example: Complete data processing pipeline

# Step 1: Preprocess data
python -m data.src.cli.preprocess \
    --pipeline-id equity_pipeline \
    --input-path data/raw/ \
    --output-path data/processed/ \
    --quality-threshold 0.95

# Step 2: Generate features
python -m data.src.cli.feature_generator \
    --input-data data/processed/ \
    --output-path features/ \
    --feature-types returns,volatility,momentum

# Step 3: Optimize portfolio
python -m portfolio.src.cli.optimize \
    --returns-data features/returns.csv \
    --config-file config/optimization.yaml \
    --output-path results/

echo "Pipeline completed successfully"
```

### Parallel Processing

**Purpose**: Run multiple CLI commands in parallel

**Usage**:
```bash
#!/bin/bash
# Example: Parallel processing of multiple datasets

datasets=("equity" "fixed_income" "crypto")

for dataset in "${datasets[@]}"; do
    python -m data.src.cli.preprocess \
        --pipeline-id "${dataset}_pipeline" \
        --input-path "data/raw/${dataset}/" \
        --output-path "data/processed/${dataset}/" &
done

# Wait for all background processes to complete
wait
echo "All datasets processed in parallel"
```

## 📝 Best Practices

### Configuration Management

1. **Use Configuration Files**: Store configurations in YAML files for reproducibility
2. **Version Control**: Track configuration changes in version control
3. **Environment Separation**: Use different configurations for dev/test/prod
4. **Validation**: Always validate configurations before use

### Error Handling

1. **Check Exit Codes**: Always check command exit codes in scripts
2. **Logging**: Use appropriate log levels for debugging
3. **Graceful Degradation**: Handle errors gracefully when possible
4. **Retry Logic**: Implement retry logic for transient errors

### Performance Optimization

1. **Batch Processing**: Process data in batches for large datasets
2. **Parallel Processing**: Use parallel processing when supported
3. **Memory Management**: Monitor memory usage for large operations
4. **Resource Monitoring**: Use system monitoring tools

### Security

1. **Sensitive Data**: Never store sensitive data in configuration files
2. **Environment Variables**: Use environment variables for secrets
3. **Access Control**: Implement proper file permissions
4. **Audit Logging**: Enable audit logging for critical operations

## 🆘 Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# If configuration file is not found
ERROR: Configuration file not found: config/missing.yaml

# Solution: Check file path and create default configuration
python -m config.cli create_default --config-type pipeline
```

#### Data Quality Issues
```bash
# If data quality thresholds are not met
ERROR: Quality threshold not met: completeness=0.92 < 0.95

# Solution: Check quality report and adjust thresholds or clean data
python -m data.src.cli.quality_report --dataset-id problem_dataset
```

#### Memory Issues
```bash
# If running out of memory with large datasets
ERROR: Memory allocation failed

# Solution: Process data in smaller chunks or increase memory allocation
python -m data.src.cli.preprocess --batch-size 10000
```

#### Permission Issues
```bash
# If lacking write permissions
ERROR: Permission denied: data/processed/

# Solution: Check file permissions or run with appropriate privileges
chmod 755 data/processed/
```

### Debug Mode

Use debug mode for detailed troubleshooting:

```bash
# Enable debug logging
python -m data.src.cli.preprocess \
    --pipeline-id my_pipeline \
    --log-level DEBUG \
    --verbose
```

### Getting Help

For additional help and support:
- **Documentation**: Check the full [documentation](../README.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-username/quant-portfolio-system/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/your-username/quant-portfolio-system/discussions)

---

*CLI Reference Guide v1.0.0 | Updated: 2024-01-15*

## Configuration Files

### Preprocessing Pipeline Configuration

```json
{
  "cleaning": {
    "outlier_method": "iqr",
    "outlier_threshold": 1.5,
    "missing_strategy": "forward_fill",
    "time_gap_handling": "interpolate"
  },
  "validation": {
    "check_bounds": true,
    "check_duplicates": true,
    "check_financial_logic": true,
    "statistical_validation": true
  },
  "normalization": {
    "method": "z_score",
    "preserve_relationships": true,
    "handle_financial_edge_cases": true
  },
  "quality": {
    "completeness_threshold": 0.95,
    "accuracy_threshold": 0.99,
    "consistency_threshold": 0.98
  }
}
```

### Data Quality Configuration

```json
{
  "metrics": ["completeness", "accuracy", "consistency", "timeliness"],
  "thresholds": {
    "completeness": 0.95,
    "accuracy": 0.99,
    "consistency": 0.98,
    "timeliness": 0.90
  },
  "validation": {
    "statistical_tests": true,
    "outlier_detection": true,
    "trend_analysis": true,
    "correlation_analysis": true
  }
}
```

## Integration Examples

### Complete Data Processing Workflow

```bash
# 1. Setup environment and install dependencies
python scripts/setup_data_environment.py
pip install -r docs/requirements.txt

# 2. Ingest historical data
python scripts/initialize_historical_data.py --symbols AAPL,GOOGL,MSFT --start-date 2023-01-01

# 3. Preprocess data with quality reporting
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report --config config/pipeline_config.json

# 4. Generate comprehensive quality report
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json --comprehensive --statistical

# 5. Run automated data refresh
python scripts/automated_data_refresh.py --config config/refresh_config.json

# 6. Validate preprocessing results
python scripts/run_preprocessing.py --validate --output validation_report.json
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Data Processing Pipeline

on: [push, pull_request]

jobs:
  data_processing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup environment
        run: |
          python scripts/setup_data_environment.py
          pip install -r docs/requirements.txt

      - name: Run preprocessing
        run: |
          python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report

      - name: Generate quality report
        run: |
          python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json

      - name: Validate results
        run: |
          python scripts/run_preprocessing.py --validate

      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: processing-results
          path: |
            data/processed/
            quality_report.json
            validation_report.json
```

## Best Practices

### Data Preprocessing

1. **Always validate data quality** before quantitative analysis
2. **Use appropriate normalization** methods for different data types
3. **Monitor memory usage** when processing large datasets
4. **Preserve statistical relationships** during normalization
5. **Handle financial edge cases** (zero prices, negative volumes, extreme moves)

### Quality Management

1. **Set appropriate thresholds** for quality metrics based on use case
2. **Monitor quality trends** over time
3. **Investigate quality degradation** immediately
4. **Document quality issues** and resolution steps
5. **Automate quality reporting** for regular monitoring

### Performance Optimization

1. **Use appropriate file formats** (Parquet recommended for large datasets)
2. **Batch process** when possible to reduce overhead
3. **Monitor memory usage** and optimize for large datasets
4. **Cache intermediate results** for repeated processing
5. **Use parallel processing** for independent operations

## Performance Considerations

All CLI tools are optimized for performance with the following targets:
- **Preprocessing Speed**: 10 million data points in <30 seconds
- **Memory Efficiency**: <4GB memory usage for large datasets
- **Real-time Processing**: Sub-second processing for 1K data batches
- **Quality Assessment**: <5 seconds for standard quality reports

For large datasets or complex operations, consider using:
- `--format parquet` for better performance with large files
- `--batch-size` option to control memory usage
- `--parallel` flag for multi-core processing
- `--dry-run` flag for validation without execution

## Security Considerations

- **Never commit sensitive data** to version control
- **Use environment variables** for API keys and secrets
- **Validate data sources** before processing
- **Monitor data access patterns** for unusual activity
- **Implement proper access controls** for data storage

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or use streaming processing
2. **Permission denied**: Check file permissions and storage paths
3. **Missing dependencies**: Install required packages from requirements.txt
4. **Invalid data format**: Validate input data before processing
5. **Configuration errors**: Check JSON syntax and configuration structure

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable verbose output
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --verbose

# Dry run to test configuration
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --dry-run --verbose

# Debug quality assessment
python -m data.src.cli.quality_report --data data/processed/ --verbose --debug
```

### Support

For additional support:
- Check the processing logs in the output directory
- Review the project documentation in `docs/`
- Examine the configuration examples in `config/`
- Open an issue in the project repository with detailed error information