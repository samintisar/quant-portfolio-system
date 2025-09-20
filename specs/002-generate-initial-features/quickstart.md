# Quickstart: Financial Features Generation

## Prerequisites

- Python 3.11+
- Project dependencies installed
- Historical price data for financial instruments

## Installation

1. **Install dependencies**:
```bash
pip install -r docs/requirements.txt
```

2. **Verify installation**:
```bash
quant-features --version
```

## Getting Started in 5 Minutes

### 1. Prepare Your Data

Create a CSV file with your historical price data:

```csv
# portfolio.csv
symbol,date,open,high,low,close,volume
AAPL,2023-01-03,130.28,133.46,129.89,132.05,112112000
MSFT,2023-01-03,233.47,237.25,232.32,236.88,54257000
GOOGL,2023-01-03,89.05,90.45,88.55,89.13,32579000
SPY,2023-01-03,383.89,385.90,382.15,384.30,84369000
```

### 2. Generate Features (Basic)

```bash
# Basic feature generation
quant-features generate -i portfolio.csv -o features.json

# View results
cat features.json | jq '.summary'
```

### 3. Validate Your Data

```bash
# Check data quality before processing
quant-features validate -i portfolio.csv
```

### 4. Quick Results

```bash
# Generate and see summary
quant-features generate -i portfolio.csv | jq '.summary'
```

## Basic Configuration

### Create a Configuration File

```json
// config.json
{
  "return_methods": ["ARITHMETIC", "LOGARITHMIC"],
  "volatility_windows": [20, 60],
  "momentum_types": ["SIMPLE_MOMENTUM", "RSI"],
  "lookback_periods": [10, 20],
  "missing_data_strategy": "FORWARD_FILL"
}
```

### Use Custom Configuration

```bash
quant-features generate -i portfolio.csv -c config.json -o features.json
```

## Common Use Cases

### 1. Portfolio Analysis
```bash
# Analyze a portfolio of stocks
quant-features generate -i portfolio.csv | jq '.feature_sets[].returns[0]'
```

### 2. Risk Assessment
```bash
# Focus on volatility measures
quant-features generate -i portfolio.csv | jq '.feature_sets[].volatility'
```

### 3. Momentum Strategy
```bash
# Get momentum signals
quant-features generate -i portfolio.csv | jq '.feature_sets[].momentum'
```

### 4. Data Quality Check
```bash
# Validate data before processing
quant-features validate -i portfolio.csv
```

## Output Interpretation

### Understanding Feature Sets

Each feature set contains:

```json
{
  "instrument": {"symbol": "AAPL", "type": "STOCK"},
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
```

### Key Metrics

- **Returns**: Price changes over specified periods
- **Volatility**: Price fluctuation measures (standard deviation)
- **Momentum**: Trend strength indicators
- **Quality Score**: Data integrity assessment (0.0-1.0)

## Sample Workflow

### Complete Analysis Pipeline

```bash
# 1. Validate data
quant-features validate -i portfolio.csv

# 2. Generate features with full configuration
quant-features generate \
  -i portfolio.csv \
  -c config.json \
  -o results.json \
  --verbose

# 3. Extract specific insights
cat results.json | jq '.quality_report.recommendations'

# 4. Export to CSV for further analysis
quant-features generate -i portfolio.csv -f csv -o analysis/
```

## Troubleshooting

### Common Issues

**Insufficient Data Error**:
```bash
# Error: Not enough data for 252-day window
# Solution: Use smaller window sizes
echo '{"volatility_windows": [20, 60]}' > config.json
quant-features generate -i portfolio.csv -c config.json
```

**Missing Data Issues**:
```bash
# Check data quality first
quant-features validate -i portfolio.csv --verbose
```

**Performance Issues**:
```bash
# Monitor performance
quant-features generate -i portfolio.csv --verbose
```

### Getting Help

```bash
# Show help
quant-features --help

# Show default configuration
quant-features config show

# Generate sample configuration
quant-features config generate > config.json
```

## Next Steps

1. **Explore Features**: Dive deeper into specific feature types
2. **Custom Configuration**: Tailor settings to your needs
3. **Integration**: Use in automated trading systems
4. **Backtesting**: Apply features to historical analysis
5. **Monitoring**: Set up regular feature generation

## API Integration

For programmatic access, use the REST API:

```bash
# Start API server (if available)
python -m features.api

# Generate features via API
curl -X POST http://localhost:8000/features/generate \
  -H "Content-Type: application/json" \
  -d @portfolio.json
```

This quickstart provides the fastest path to generating financial features. For advanced usage and configuration options, refer to the full documentation.