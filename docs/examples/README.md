# Preprocessing System Examples

This directory contains comprehensive examples and documentation for using the data preprocessing system.

## Files in This Directory

### 1. `preprocessing_configuration_examples.md`
- **Purpose**: Comprehensive guide to configuring the preprocessing system
- **Content**: JSON configuration examples for different use cases
- **Key Topics**:
  - Basic configuration structure
  - Daily equity data processing
  - High-frequency data handling
  - Cryptocurrency data processing
  - Multi-asset portfolio processing
  - Backtesting data preparation
  - Real-time processing
  - Machine learning preparation
  - Performance optimization
  - Error handling and fallbacks

### 2. `preprocessing_usage_examples.py`
- **Purpose**: Practical Python code examples
- **Content**: Working examples of common preprocessing tasks
- **Examples Included**:
  - Basic stock data preprocessing
  - Custom configuration usage
  - Multi-asset processing
  - Real-time processing simulation
  - Quality reporting
  - Performance benchmarking

### 3. `README.md` (This file)
- **Purpose**: Overview and usage guide for examples

## Getting Started

### Prerequisites
- Python 3.11+
- Required packages: `pip install -r ../../docs/requirements.txt`
- Basic familiarity with pandas and financial data

### Running the Examples

1. **Configuration Examples**:
   ```bash
   # View configuration examples
   cat preprocessing_configuration_examples.md
   ```

2. **Usage Examples**:
   ```bash
   # Run the usage examples
   python preprocessing_usage_examples.py
   ```

## Quick Start

### Basic Usage

```python
import pandas as pd
from data.src.lib.cleaning import DataCleaner
from data.src.lib.validation import DataValidator
from data.src.lib.normalization import DataNormalizer

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize preprocessing components
cleaner = DataCleaner()
validator = DataValidator()
normalizer = DataNormalizer()

# Process data
cleaned = cleaner.handle_missing_values(df, method='forward_fill')
cleaned, _ = cleaner.detect_outliers(cleaned, method='iqr', action='clip')
normalized, _ = normalizer.normalize_zscore(cleaned)

# Validate quality
validation_results = validator.run_comprehensive_validation(normalized)
quality_score = validator.get_data_quality_score(validation_results)

print(f"Data quality score: {quality_score:.3f}")
```

### Configuration-Based Usage

```python
import json

# Load configuration
with open('your_config.json', 'r') as f:
    config = json.load(f)

# Apply preprocessing pipeline
processed = apply_preprocessing_pipeline(df, config)
```

## Common Use Cases

### 1. Daily Stock Data
Use the "Daily Equity Data" configuration from `preprocessing_configuration_examples.md`

### 2. High-Frequency Trading Data
Use the "High-Frequency Data" configuration with shorter time windows

### 3. Multi-Asset Portfolios
See Example 3 in `preprocessing_usage_examples.py`

### 4. Real-Time Processing
See Example 4 for real-time batch processing simulation

### 5. Backtesting Preparation
Use the "Backtesting Data Preparation" configuration

## Configuration Templates

Copy and modify these templates from the configuration guide:

- **Basic Processing**: Minimal configuration for clean data
- **Advanced Processing**: Full configuration with all options
- **Performance Optimized**: For large datasets
- **Real-Time**: For streaming data applications

## Performance Guidelines

### Memory Usage
- Target: <4GB for 10M data points
- Monitor: Use `psutil` for memory tracking
- Optimize: Use chunk processing for large datasets

### Processing Speed
- Target: <30 seconds for 10M data points
- Monitor: Time individual operations
- Optimize: Use parallel processing where appropriate

### Data Quality
- Target: >0.9 quality score for most applications
- Monitor: Track completeness, consistency, accuracy
- Improve: Adjust configuration based on validation results

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk size
   - Use memory optimization settings
   - Process data in smaller batches

2. **Poor Quality Scores**
   - Review data issues in validation report
   - Adjust quality thresholds
   - Check data source quality

3. **Slow Processing**
   - Enable parallel processing
   - Optimize data types
   - Use appropriate chunk sizes

### Getting Help

1. Check the main documentation in `../../docs/`
2. Review error messages and logs
3. Test with sample datasets
4. Monitor system resources during processing

## Best Practices

1. **Always validate** data quality after preprocessing
2. **Start simple** and add complexity as needed
3. **Monitor performance** for large datasets
4. **Handle edge cases** (empty data, all NaN, constant values)
5. **Document your** preprocessing pipelines
6. **Version control** your configurations
7. **Test thoroughly** with realistic data

## Next Steps

After reviewing these examples:

1. **Adapt configurations** for your specific data
2. **Integrate with** your trading strategies
3. **Set up monitoring** for production use
4. **Customize preprocessing** rules as needed

For more detailed information, see:
- Main documentation: `../../docs/`
- Test cases: `../../tests/`
- Source code: `../../data/src/`