# CLI Contract: Financial Features Generation

## Command Interface

### Primary Command
```bash
quant-features generate [OPTIONS]
```

### Commands

#### 1. Generate Features
```bash
quant-features generate \
  --input-data <path/to/data.csv> \
  --config <path/to/config.json> \
  --output <path/to/output.json> \
  [--format <json|csv>] \
  [--validate-only] \
  [--verbose]
```

**Options**:
- `--input-data, -i`: Path to input CSV/JSON file containing price data (required)
- `--config, -c`: Path to JSON configuration file (optional, uses defaults)
- `--output, -o`: Path to output file (optional, defaults to stdout)
- `--format, -f`: Output format - json or csv (default: json)
- `--validate-only, -v`: Validate input data without generating features
- `--verbose`: Enable detailed logging
- `--help`: Show help message
- `--version`: Show version information

#### 2. Validate Data
```bash
quant-features validate \
  --input-data <path/to/data.csv> \
  [--output <path/to/report.json>]
```

**Options**:
- `--input-data, -i`: Path to input data file (required)
- `--output, -o`: Path to validation report (optional)
- `--verbose`: Enable detailed logging

#### 3. Show Default Config
```bash
quant-features config show [--output <path/to/config.json>]
```

**Options**:
- `--output, -o`: Save config to file (optional)

#### 4. Generate Sample Config
```bash
quant-features config generate [--output <path/to/config.json>]
```

## Input Data Format

### CSV Format
```csv
symbol,date,open,high,low,close,volume,adjusted_close
AAPL,2023-01-03,130.28,133.46,129.89,132.05,112112000,132.05
SPY,2023-01-03,383.89,385.90,382.15,384.30,84369000,384.30
```

### JSON Format
```json
{
  "instruments": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "type": "STOCK",
      "exchange": "NASDAQ",
      "currency": "USD"
    }
  ],
  "price_data": [
    {
      "symbol": "AAPL",
      "date": "2023-01-03",
      "open": 130.28,
      "high": 133.46,
      "low": 129.89,
      "close": 132.05,
      "volume": 112112000,
      "adjusted_close": 132.05
    }
  ]
}
```

## Configuration Format

### JSON Configuration
```json
{
  "return_methods": ["ARITHMETIC", "LOGARITHMIC"],
  "volatility_windows": [20, 252],
  "momentum_types": ["SIMPLE_MOMENTUM", "RSI", "ROC"],
  "lookback_periods": [10, 20, 50],
  "missing_data_strategy": "FORWARD_FILL",
  "outlier_detection": {
    "method": "Z_SCORE",
    "threshold": 3.0
  },
  "validation": {
    "min_data_points": 20,
    "max_missing_percentage": 10.0,
    "return_bounds": [-10.0, 10.0],
    "volatility_bounds": [0.0, 5.0]
  }
}
```

## Output Formats

### JSON Output
```json
{
  "feature_sets": [
    {
      "instrument": {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "type": "STOCK"
      },
      "returns": [
        {
          "date": "2023-01-04",
          "return_type": "ARITHMETIC",
          "return_value": 0.0268,
          "period": 1,
          "is_valid": true
        }
      ],
      "volatility": [
        {
          "date": "2023-01-31",
          "window_size": 20,
          "volatility_type": "STANDARD_DEVIATION",
          "volatility_value": 0.0234,
          "mean_return": 0.0012,
          "sample_size": 20
        }
      ],
      "momentum": [
        {
          "date": "2023-01-31",
          "indicator_type": "RSI",
          "indicator_value": 65.43,
          "lookback_period": 14,
          "signal_strength": 0.31,
          "interpretation": "Slightly overbought"
        }
      ],
      "data_quality_score": 0.98
    }
  ],
  "summary": {
    "instruments_processed": 2,
    "total_data_points": 504,
    "features_calculated": 1512,
    "processing_time_ms": 1250,
    "memory_usage_mb": 45.2
  },
  "quality_report": {
    "overall_score": 0.96,
    "completeness_score": 0.98,
    "outliers_detected": 3,
    "gaps_found": 0,
    "recommendations": [
      "Data quality is excellent",
      "All features calculated successfully"
    ]
  },
  "metadata": {
    "calculation_timestamp": "2023-01-31T15:30:00Z",
    "parameters_hash": "abc123...",
    "version": "1.0.0",
    "configuration_used": { ... }
  }
}
```

### CSV Output
Multiple CSV files are generated:
- `returns_<symbol>.csv`: Return data
- `volatility_<symbol>.csv`: Volatility data
- `momentum_<symbol>.csv`: Momentum data
- `summary.csv`: Processing summary
- `quality_report.csv`: Data quality report

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | SUCCESS | Operation completed successfully |
| 1 | ERROR | General error occurred |
| 2 | INVALID_INPUT | Input data validation failed |
| 3 | CONFIG_ERROR | Configuration file error |
| 4 | FILE_ERROR | File I/O error |
| 5 | NUMERICAL_ERROR | Numerical calculation error |
| 6 | INSUFFICIENT_DATA | Not enough data for calculations |
| 7 | TIMEOUT | Operation timed out |
| 8 | MEMORY_ERROR | Insufficient memory |
| 9 | VALIDATION_FAILED | Data quality below threshold |

## Error Messages

### Standard Error Format
```json
{
  "error": {
    "code": "INSUFFICIENT_DATA",
    "message": "Insufficient historical data for volatility calculation",
    "details": {
      "instrument": "AAPL",
      "required_points": 252,
      "available_points": 150,
      "calculation": "252-day volatility"
    }
  }
}
```

### Common Error Codes
- `INVALID_FILE_FORMAT`: Input file format not recognized
- `MISSING_REQUIRED_FIELDS`: Required fields missing in input data
- `INVALID_DATE_FORMAT`: Date format not supported
- `NEGATIVE_PRICES`: Price values cannot be negative
- `MISSING_DATA_THRESHOLD`: Too much missing data
- `OUTLIER_THRESHOLD`: Too many outliers detected
- `NUMERICAL_INSTABILITY`: Calculation failed due to numerical issues
- `CONFIGURATION_ERROR`: Invalid configuration parameters

## Performance Monitoring

### Logging Levels
- `ERROR`: Critical errors
- `WARN`: Warnings and non-critical issues
- `INFO`: General information
- `DEBUG`: Detailed debugging information

### Performance Metrics
All commands output performance metrics:
- Processing time (milliseconds)
- Memory usage (MB)
- Data points processed
- Features generated
- Quality scores

## Integration Examples

### Basic Usage
```bash
# Generate features with default config
quant-features generate -i data/portfolio.csv -o results.json

# Validate data first
quant-features validate -i data/portfolio.csv

# Use custom configuration
quant-features generate -i data.csv -c config/custom.json -o results.json

# CSV output format
quant-features generate -i data.csv -f csv -o results/
```

### Pipeline Integration
```bash
# Unix pipeline
cat data.csv | quant-features generate -i - | jq '.summary'

# Error handling
quant-features generate -i data.csv || echo "Generation failed: $?"

# Performance monitoring
quant-features generate -i data.csv --verbose 2>&1 | grep "Processing time"
```

This CLI contract ensures reproducible research and aligns with constitutional requirements for CLI interfaces and reproducibility.