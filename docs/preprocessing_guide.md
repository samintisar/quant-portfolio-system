# Data Preprocessing Guide

This guide provides comprehensive documentation for the data preprocessing system in the quantitative trading system, including data cleaning, validation, normalization, and quality control.

## Overview

The preprocessing system provides:
- **Data Cleaning**: Missing value handling, outlier detection, time gap management
- **Data Validation**: Financial logic validation, bounds checking, statistical validation
- **Data Normalization**: Z-score, Min-Max, Robust scaling with financial specialization
- **Quality Control**: Automated quality assessment and reporting
- **Performance Optimization**: 10M data points in <30 seconds, <4GB memory usage

## Quick Start

### 1. Basic Preprocessing

```python
from data.src.preprocessing import DataPreprocessor
from data.src.lib import cleaning, validation, normalization

# Create preprocessor
preprocessor = DataPreprocessor()

# Load your data (example with pandas DataFrame)
import pandas as pd
data = pd.read_csv('data/raw/market_data.csv')

# Clean data
cleaned_data = cleaning.remove_outliers(data, method='iqr', threshold=1.5)
cleaned_data = cleaning.handle_missing_values(cleaned_data, strategy='forward_fill')

# Validate data
validation_report = validation.validate_financial_data(cleaned_data)
if not validation_report['is_valid']:
    print(f"Validation failed: {validation_report['errors']}")

# Normalize data
normalized_data = normalization.z_score_normalization(cleaned_data)
```

### 2. Complete Pipeline

```python
# Run complete preprocessing pipeline
pipeline_result = preprocessor.run_preprocessing_pipeline(
    data,
    cleaning_config={
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'missing_strategy': 'forward_fill'
    },
    validation_config={
        'check_bounds': True,
        'check_duplicates': True,
        'check_financial_logic': True
    },
    normalization_config={
        'method': 'z_score',
        'preserve_relationships': True,
        'handle_financial_edge_cases': True
    }
)

# Access results
processed_data = pipeline_result['data']
quality_report = pipeline_result['quality_report']
processing_stats = pipeline_result['statistics']
```

### 3. CLI Usage

```bash
# Preprocess data with quality reporting
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report

# Generate data quality report
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json

# Run with custom configuration
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --config config/pipeline_config.json
```

## Core Components

### 1. Data Cleaning (`data/src/lib/cleaning.py`)

#### Missing Value Handling
```python
from data.src.lib.cleaning import handle_missing_values

# Forward fill missing values
data_filled = handle_missing_values(data, strategy='forward_fill')

# Backward fill missing values
data_filled = handle_missing_values(data, strategy='backward_fill')

# Interpolate missing values
data_filled = handle_missing_values(data, strategy='interpolate')

# Drop rows with missing values
data_dropped = handle_missing_values(data, strategy='drop')

# Custom filling
data_filled = handle_missing_values(data, strategy='custom', fill_value=0)
```

#### Outlier Detection and Removal
```python
from data.src.lib.cleaning import remove_outliers, detect_outliers

# IQR method
data_clean = remove_outliers(data, method='iqr', threshold=1.5)

# Z-score method
data_clean = remove_outliers(data, method='z_score', threshold=3)

# Isolation Forest
data_clean = remove_outliers(data, method='isolation_forest', contamination=0.1)

# Detect outliers without removing
outliers = detect_outliers(data, method='iqr', threshold=1.5)
```

#### Time Gap Handling
```python
from data.src.lib.cleaning import handle_time_gaps

# Interpolate time gaps
data_filled = handle_time_gaps(data, method='interpolate')

# Forward fill time gaps
data_filled = handle_time_gaps(data, method='forward_fill')

# Drop time gaps
data_dropped = handle_time_gaps(data, method='drop')
```

### 2. Data Validation (`data/src/lib/validation.py`)

#### Financial Data Validation
```python
from data.src.lib.validation import validate_financial_data

# Comprehensive validation
validation_report = validate_financial_data(data)

# Check specific aspects
price_validation = validation.validate_price_data(data)
volume_validation = validation.validate_volume_data(data)
returns_validation = validation.validate_returns_data(data)
```

#### Statistical Validation
```python
from data.src.lib.validation import validate_statistical_properties

# Check statistical properties
stats_report = validation.validate_statistical_properties(data)

# Normality test
normality_test = validation.test_normality(data['returns'])

# Stationarity test
stationarity_test = validation.test_stationarity(data['price'])
```

#### Bounds Checking
```python
from data.src.lib.validation import validate_bounds

# Validate price bounds
price_bounds = validation.validate_bounds(
    data['price'],
    min_value=0,
    max_value=1000000
)

# Validate volume bounds
volume_bounds = validation.validate_bounds(
    data['volume'],
    min_value=0,
    max_value=1000000000
)
```

### 3. Data Normalization (`data/src/lib/normalization.py`)

#### Z-Score Normalization
```python
from data.src.lib.normalization import z_score_normalization

# Standard z-score normalization
normalized_data = z_score_normalization(data)

# With custom parameters
normalized_data = z_score_normalization(
    data,
    preserve_relationships=True,
    handle_financial_edge_cases=True
)
```

#### Min-Max Normalization
```python
from data.src.lib.normalization import min_max_normalization

# Min-max scaling to [0, 1]
normalized_data = min_max_normalization(data)

# Custom range
normalized_data = min_max_normalization(data, feature_range=(-1, 1))
```

#### Robust Scaling
```python
from data.src.lib.normalization import robust_scaling

# Robust scaling using median and IQR
scaled_data = robust_scaling(data)

# With custom parameters
scaled_data = robust_scaling(
    data,
    center=True,
    scale=True,
    quantile_range=(25.0, 75.0)
)
```

## Configuration

### Pipeline Configuration

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

### Loading Configuration

```python
from data.src.config.pipeline_config import PreprocessingConfig

# Load from JSON file
config = PreprocessingConfig.load('config/pipeline_config.json')

# Create programmatically
config = PreprocessingConfig(
    cleaning_config={
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'missing_strategy': 'forward_fill'
    },
    validation_config={
        'check_bounds': True,
        'check_duplicates': True,
        'check_financial_logic': True
    },
    normalization_config={
        'method': 'z_score',
        'preserve_relationships': True
    }
)
```

## Quality Assessment

### Quality Metrics

The system provides comprehensive quality metrics:

- **Completeness**: Percentage of non-null values
- **Accuracy**: Validation against business rules
- **Consistency**: Cross-field validation
- **Timeliness**: Data freshness and currency
- **Uniqueness**: Duplicate detection
- **Validity**: Format and type validation
- **Statistical Properties**: Distribution analysis, outlier detection

### Quality Reporting

```python
from data.src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Generate quality report
quality_report = preprocessor.assess_data_quality(data)

# Access quality metrics
completeness_score = quality_report['completeness']
accuracy_score = quality_report['accuracy']
consistency_score = quality_report['consistency']

# Get detailed issues
issues = quality_report['issues']
recommendations = quality_report['recommendations']
```

## Performance Optimization

### Batch Processing

```python
# Process large datasets in batches
batch_results = preprocessor.process_in_batches(
    data,
    batch_size=100000,
    max_memory_usage=4000000000  # 4GB
)
```

### Memory Management

```python
# Monitor memory usage
memory_usage = preprocessor.get_memory_usage()

# Optimize for memory
optimized_data = preprocessor.optimize_memory_usage(data)

# Use streaming for very large datasets
for chunk in preprocessor.stream_preprocessing(large_file):
    processed_chunk = preprocessor.process_chunk(chunk)
```

## Financial Data Specifics

### Handling Financial Edge Cases

```python
from data.src.lib.normalization import z_score_normalization

# Handle zero prices, negative volumes, extreme moves
normalized_data = z_score_normalization(
    data,
    handle_financial_edge_cases=True,
    zero_price_handling='log_transform',
    negative_volume_handling='absolute_value',
    extreme_move_handling='winsorization'
)
```

### Market-Specific Validation

```python
from data.src.lib.validation import validate_market_data

# Equity-specific validation
equity_validation = validate_market_data(
    data,
    asset_type='equity',
    min_price=0.01,
    max_price=10000,
    min_volume=0,
    max_volume=1000000000
)

# FX-specific validation
fx_validation = validate_market_data(
    data,
    asset_type='fx',
    min_rate=0,
    max_rate=1000,
    significant_digits=4
)
```

## Testing and Validation

### Unit Tests

```bash
# Run preprocessing library tests
pytest tests/unit/test_cleaning.py
pytest tests/unit/test_validation.py
pytest tests/unit/test_normalization_unit.py
```

### Performance Tests

```bash
# Run performance benchmarks
pytest tests/performance/test_preprocessing_performance.py

# Memory usage tests
pytest tests/performance/test_memory_usage.py
```

### Statistical Tests

```bash
# Run statistical validation tests
pytest tests/statistical/test_preprocessing_statistical.py
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use streaming processing
2. **Slow Performance**: Use appropriate file formats (Parquet) and batch processing
3. **Validation Failures**: Check data format and content against validation rules
4. **Normalization Issues**: Ensure data is cleaned before normalization
5. **Quality Issues**: Review preprocessing pipeline configuration

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
preprocessor = DataPreprocessor(verbose=True)

# Dry run to test configuration
dry_run_result = preprocessor.dry_run_pipeline(data, config)
```

### Performance Monitoring

```python
# Monitor processing performance
performance_stats = preprocessor.get_performance_stats()

# Check memory usage
memory_stats = preprocessor.get_memory_stats()

# Monitor quality trends
quality_trends = preprocessor.get_quality_trends()
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

## API Reference

### Classes

#### `DataPreprocessor`
- `run_preprocessing_pipeline(data, cleaning_config, validation_config, normalization_config)`
- `assess_data_quality(data)`
- `process_in_batches(data, batch_size, max_memory_usage)`
- `optimize_memory_usage(data)`

#### `CleaningLib`
- `handle_missing_values(data, strategy)`
- `remove_outliers(data, method, threshold)`
- `detect_outliers(data, method, threshold)`
- `handle_time_gaps(data, method)`

#### `ValidationLib`
- `validate_financial_data(data)`
- `validate_bounds(data, min_value, max_value)`
- `validate_statistical_properties(data)`
- `test_normality(data)`
- `test_stationarity(data)`

#### `NormalizationLib`
- `z_score_normalization(data, preserve_relationships, handle_financial_edge_cases)`
- `min_max_normalization(data, feature_range)`
- `robust_scaling(data, center, scale, quantile_range)`

## Contributing

1. Follow the existing code structure
2. Add appropriate tests
3. Update documentation
4. Use type hints
5. Follow PEP 8 guidelines

## License

This project is part of the quantitative trading system and follows the same license terms.