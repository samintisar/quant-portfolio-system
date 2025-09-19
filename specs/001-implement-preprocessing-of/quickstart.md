# Data Preprocessing System Quickstart

## Overview

The Data Preprocessing System provides comprehensive tools for cleaning, validating, and normalizing financial time series data. This quickstart guide will help you get up and running with basic preprocessing operations.

## Prerequisites

- Python 3.11+
- Required packages (install with `pip install -r docs/requirements.txt`)
- Basic familiarity with pandas and financial time series data

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd quant-portfolio-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r docs/requirements.txt
   ```

3. **Set up the preprocessing module**:
   ```bash
   cd data/src
   pip install -e .
   ```

## Basic Usage

### 1. Command Line Interface

The system provides a CLI for preprocessing data:

```bash
# Process a CSV file with default configuration
python -m data.cli.preprocess --input data.csv --output processed_data.csv

# Process with custom configuration
python -m data.cli.preprocess \
    --input data.csv \
    --output processed_data.csv \
    --config config/preprocessing_config.json

# Generate quality report
python -m data.cli.quality-report --input processed_data.csv --output quality_report.json
```

### 2. Python API

Basic preprocessing in Python:

```python
import pandas as pd
from data.preprocessing import DataPreprocessor

# Load your data
df = pd.read_csv('raw_data.csv')

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Process data with default settings
processed_df = preprocessor.process(df)

# Process with custom configuration
config = {
    'missing_value_handling': {
        'method': 'forward_fill',
        'threshold': 0.1
    },
    'outlier_detection': {
        'method': 'zscore',
        'threshold': 3.0,
        'action': 'clip'
    },
    'normalization': {
        'method': 'zscore',
        'preserve_stats': True
    }
}

processed_df = preprocessor.process(df, config)

# Get quality metrics
quality_metrics = preprocessor.get_quality_metrics()
print(f"Data quality score: {quality_metrics['overall_score']}")
```

### 3. Configuration Files

Create a configuration file (`preprocessing_config.json`):

```json
{
  "pipeline_id": "equity_daily_v1",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.1
  },
  "outlier_detection": {
    "method": "zscore",
    "threshold": 3.0,
    "action": "clip"
  },
  "normalization": {
    "method": "zscore",
    "preserve_stats": true
  },
  "quality_thresholds": {
    "completeness": 0.95,
    "consistency": 0.90,
    "accuracy": 0.95
  }
}
```

## Data Format

### Input Data Requirements

Your input data should include these columns (at minimum):

- `symbol`: Stock symbol (e.g., "AAPL")
- `timestamp`: Date/time column
- `close`: Closing price (required)

Optional columns:
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `volume`: Trading volume

Example CSV format:
```csv
symbol,timestamp,open,high,low,close,volume
AAPL,2024-01-01,150.25,152.50,149.75,151.30,1000000
MSFT,2024-01-01,370.50,375.25,369.80,372.15,500000
```

### Output Data Format

The preprocessor adds these columns:
- `*_normalized`: Normalized versions of price/volume columns
- `returns`: Calculated returns
- `volatility`: Rolling volatility
- `quality_flags`: Data quality indicators
- `outlier_flags`: Outlier detection results

## Preprocessing Methods

### Missing Value Handling

- **forward_fill**: Propagate last valid observation forward
- **interpolation**: Linear interpolation between valid points
- **mean**: Fill with column mean
- **median**: Fill with column median
- **drop**: Remove rows with missing values

### Outlier Detection

- **zscore**: Standard deviations from mean
- **iqr**: Interquartile range method
- **percentile**: Percentile-based detection
- **custom**: Custom threshold definition

### Normalization

- **zscore**: Standardization (mean=0, std=1)
- **minmax**: Scale to [0,1] range
- **robust**: Scale using median and IQR
- **percentile**: Percentile transformation

## Quality Assessment

The system provides comprehensive quality metrics:

- **Completeness**: Percentage of non-null values
- **Consistency**: Cross-field validation
- **Accuracy**: Domain validation and outlier detection
- **Overall Score**: Weighted combination of all metrics

Quality reports are generated in JSON format:
```json
{
  "dataset_id": "processed_20240101",
  "overall_score": 0.97,
  "completeness": 0.98,
  "consistency": 0.95,
  "accuracy": 0.99,
  "issues_found": [
    "Missing data points: 2%",
    "Outliers detected: 5"
  ]
}
```

## Examples

### Example 1: Basic Stock Data Processing

```python
import pandas as pd
from data.preprocessing import DataPreprocessor

# Load sample data
df = pd.read_csv('sample_stocks.csv')

# Process with default settings
preprocessor = DataPreprocessor()
processed_df = preprocessor.process(df)

# Check quality
quality = preprocessor.get_quality_metrics()
print(f"Quality score: {quality['overall_score']:.2f}")

# Save results
processed_df.to_csv('processed_stocks.csv', index=False)
```

### Example 2: Custom Configuration

```python
import pandas as pd
from data.preprocessing import DataPreprocessor

# Load data with some quality issues
df = pd.read_csv('noisy_data.csv')

# Custom configuration for high-frequency data
config = {
    'missing_value_handling': {
        'method': 'interpolation',
        'threshold': 0.05
    },
    'outlier_detection': {
        'method': 'iqr',
        'threshold': 1.5,
        'action': 'flag'
    },
    'normalization': {
        'method': 'robust',
        'preserve_stats': False
    }
}

preprocessor = DataPreprocessor()
processed_df = preprocessor.process(df, config)

# Generate detailed report
preprocessor.generate_quality_report('quality_report.html')
```

### Example 3: Batch Processing

```python
import os
from data.preprocessing import DataPreprocessor

# Process multiple files
input_dir = 'raw_data/'
output_dir = 'processed_data/'

preprocessor = DataPreprocessor()

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'processed_{filename}')

        df = pd.read_csv(input_path)
        processed_df = preprocessor.process(df)
        processed_df.to_csv(output_path, index=False)

        print(f"Processed {filename}, quality: {preprocessor.get_quality_metrics()['overall_score']:.2f}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/data/ -v

# Run specific test categories
python -m pytest tests/data/test_preprocessing.py -v
python -m pytest tests/data/test_quality_metrics.py -v

# Run with coverage
python -m pytest tests/data/ --cov=data.preprocessing --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Missing required columns**:
   - Ensure your data has `symbol`, `timestamp`, and `close` columns
   - Check column names match expected format

2. **Invalid timestamp format**:
   - Use standard datetime formats (ISO 8601 preferred)
   - Consider using `pd.to_datetime()` for conversion

3. **Memory issues with large datasets**:
   - Use chunk processing for large files
   - Consider data type optimization (float32 vs float64)

4. **Quality threshold failures**:
   - Review quality thresholds in configuration
   - Check data for systematic issues

### Getting Help

- Check the documentation in `docs/`
- Review test cases in `tests/data/`
- Submit issues with sample data for reproduction

## Next Steps

After completing this quickstart:

1. Explore advanced configuration options
2. Integrate with your trading strategies
3. Set up automated data pipelines
4. Customize preprocessing rules for your specific needs

For more detailed information, see the full documentation in the `docs/` directory.