# CLI Documentation for Quantitative Trading System

This document describes the command-line interfaces available for the quantitative trading system, providing comprehensive tools for data preprocessing, quality reporting, and pipeline management.

## Available CLI Tools

### 1. Data Preprocessing CLI (`preprocess`)

**Location**: `data/src/cli/preprocess.py`
**Purpose**: Command-line interface for data preprocessing with quality reporting

#### Commands

##### Basic Data Preprocessing
```bash
# Preprocess data with default settings
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/

# Preprocess with specific configuration
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --config config/pipeline_config.json

# Preprocess with quality reporting
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report

# Preprocess with custom cleaning parameters
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --cleaning-method iqr --cleaning-threshold 1.5
```

##### Advanced Options
```bash
# Preprocess with specific normalization method
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --normalization z_score

# Preprocess with missing value handling strategy
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --missing-strategy forward_fill

# Preprocess with validation enabled
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --validate

# Preprocess with verbose output
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --verbose
```

##### File Format Support
```bash
# Process CSV files
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --format csv

# Process Parquet files
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --format parquet

# Process multiple formats
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --format auto
```

#### Output Formats

The preprocessing CLI supports multiple output formats:

```bash
# Generate comprehensive report
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report --report-format json

# Generate HTML report
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report --report-format html

# Save processing statistics
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --stats --stats-file processing_stats.json
```

### 2. Data Quality Reporting CLI (`quality_report`)

**Location**: `data/src/cli/quality_report.py`
**Purpose**: Automated data quality assessment and reporting

#### Commands

##### Generate Quality Reports
```bash
# Generate basic quality report
python -m data.src.cli.quality_report --data data/processed/

# Generate report with specific metrics
python -m data.src.cli.quality_report --data data/processed/ --metrics completeness,accuracy,consistency

# Generate report with custom thresholds
python -m data.src.cli.quality_report --data data/processed/ --threshold-completeness 0.95 --threshold-accuracy 0.99

# Generate comprehensive report
python -m data.src.cli.quality_report --data data/processed/ --comprehensive
```

##### Report Output Options
```bash
# Save report to JSON file
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json

# Save report to HTML file
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.html --format html

# Generate CSV summary
python -m data.src.cli.quality_report --data data/processed/ --output quality_summary.csv --format csv

# Generate multiple formats
python -m data.src.cli.quality_report --data data/processed/ --output quality_report --format all
```

##### Advanced Analysis
```bash
# Include statistical analysis
python -m data.src.cli.quality_report --data data/processed/ --statistical

# Include outlier detection
python -m data.src.cli.quality_report --data data/processed/ --outliers

# Include trend analysis
python -m data.src.cli.quality_report --data data/processed/ --trends

# Include correlation analysis
python -m data.src.cli.quality_report --data data/processed/ --correlation
```

#### Quality Metrics

The quality reporting system includes comprehensive metrics:
- **Completeness**: Percentage of non-null values
- **Accuracy**: Validation against business rules
- **Consistency**: Cross-field validation
- **Timeliness**: Data freshness and currency
- **Uniqueness**: Duplicate detection
- **Validity**: Format and type validation
- **Statistical Properties**: Distribution analysis, outlier detection

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