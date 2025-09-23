# Data Pipeline Documentation

## Overview

The enhanced data pipeline provides comprehensive financial data processing capabilities, including data ingestion from Yahoo Finance, cleaning, validation, normalization, and quality reporting. This pipeline is designed for quantitative portfolio optimization and analysis.

## Features

### 1. Yahoo Finance Data Ingestion
- **Historical Data**: Fetch OHLCV data for any stock symbol
- **Multiple Symbols**: Batch processing of multiple symbols
- **Flexible Time Periods**: Support for various time ranges (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- **Real-time Data**: Access to current market data
- **Symbol Information**: Basic company and market data

### 2. Data Cleaning
- **Duplicate Removal**: Automatic removal of duplicate records
- **Missing Value Handling**: Forward-fill and backward-fill strategies
- **Data Sorting**: Chronological ordering by date
- **Outlier Detection**: Identification of extreme values

### 3. Data Validation
- **Required Columns**: Ensures essential data columns are present
- **Data Gaps**: Detects missing trading days
- **Extreme Returns**: Identifies unusual price movements
- **Zero/Negative Prices**: Validates price data integrity
- **Quality Scoring**: Overall data quality assessment

### 4. Data Normalization
- **Min-Max Normalization**: Scale values to [0, 1] range
- **Z-Score Normalization**: Standardize to mean=0, std=1
- **Returns Normalization**: Convert to cumulative returns
- **Selective Application**: Apply normalization only to relevant columns

### 5. Quality Reporting
- **Comprehensive Metrics**: Data completeness, volatility, density
- **Recommendations**: Automated suggestions for data improvement
- **Timestamp Tracking**: Audit trail for data processing
- **Detailed Statistics**: Extensive data characterization

## Usage

### Basic Usage

```python
from portfolio.data.yahoo_service import YahooFinanceService

# Initialize the service
service = YahooFinanceService()

# Fetch and process data for a single symbol
result = service.fetch_and_process_data(
    symbols=['AAPL'],
    period='1y',
    normalize_method='minmax'
)

# Access processed data
if result['AAPL']['success']:
    data = result['AAPL']['data']
    quality_report = result['AAPL']['quality_report']
    print(f"Processed {len(data)} data points")
```

### Advanced Usage

```python
# Process multiple symbols with different normalization
symbols = ['AAPL', 'MSFT', 'GOOGL']
results = service.fetch_and_process_data(
    symbols=symbols,
    period='2y',
    normalize_method='zscore'
)

# Analyze results
for symbol, result in results.items():
    if result['success']:
        data = result['data']
        validation = result['validation']
        quality = result['quality_report']

        print(f"{symbol}:")
        print(f"  Validation: {'✓' if validation['is_valid'] else '✗'}")
        print(f"  Completeness: {quality['quality_metrics']['completeness']:.1%}")
        print(f"  Volatility: {quality['quality_metrics']['volatility_annualized']:.2%}")
```

### Individual Operations

```python
# Step-by-step processing
symbol = 'AAPL'

# 1. Fetch raw data
raw_data = service.fetch_historical_data(symbol, period='1y')

# 2. Clean data
cleaned_data = service.clean_data(raw_data)

# 3. Validate data
is_valid, validation_report = service.validate_data(cleaned_data, symbol)

if is_valid:
    # 4. Normalize data
    normalized_data = service.normalize_data(cleaned_data, 'minmax')

    # 5. Generate quality report
    quality_report = service.generate_quality_report(normalized_data, symbol)

    print(f"Data quality: {quality_report['quality_metrics']['completeness']:.1%}")
```

## API Reference

### YahooFinanceService

#### Core Methods

- `fetch_historical_data(symbol, period)`: Fetch historical price data
- `fetch_multiple_symbols(symbols, period)`: Fetch data for multiple symbols
- `fetch_price_data(symbols, period)`: Fetch adjusted close prices
- `get_symbol_info(symbol)`: Get basic symbol information

#### Data Processing Methods

- `clean_data(data)`: Clean and preprocess data
- `validate_data(data, symbol)`: Validate data quality
- `normalize_data(data, method)`: Normalize data using specified method
- `generate_quality_report(data, symbol)`: Generate quality report

#### Pipeline Methods

- `fetch_and_process_data(symbols, period, normalize_method)`: Complete data pipeline

### Parameters

#### Time Periods
- `1d`: 1 day
- `5d`: 5 days
- `1mo`: 1 month
- `3mo`: 3 months
- `6mo`: 6 months
- `1y`: 1 year
- `2y`: 2 years
- `5y`: 5 years
- `10y`: 10 years
- `ytd`: Year to date
- `max`: Maximum available

#### Normalization Methods
- `minmax`: Scale to [0, 1] range
- `zscore`: Standardize to mean=0, std=1
- `returns`: Convert to cumulative returns
- `None`: No normalization

## Data Quality Metrics

### Completeness
- Calculated as the ratio of non-missing values to total values
- Expressed as a percentage (0-100%)
- Higher values indicate better data quality

### Volatility
- Annualized standard deviation of returns
- Calculated as daily_std * sqrt(252)
- Useful for risk assessment

### Data Density
- Ratio of actual data points to expected data points
- Accounts for market closures and holidays
- Values typically around 0.7 (70%) due to weekends

### Price Range
- Minimum and maximum adjusted close prices
- Useful for understanding price movements

## Error Handling

The pipeline includes comprehensive error handling:

- **Network Issues**: Graceful handling of connectivity problems
- **Invalid Symbols**: Automatic filtering of unavailable symbols
- **Data Quality**: Validation failures prevent processing of poor-quality data
- **Missing Data**: Robust handling of gaps and missing values

## Performance Considerations

- **Memory Usage**: Data is processed efficiently to minimize memory footprint
- **Batch Processing**: Multiple symbols processed in parallel when possible
- **Offline Storage**: Data saved locally to minimize API calls and enable offline access
- **Rate Limiting**: Respect Yahoo Finance API limits

## Offline Data Storage

The enhanced pipeline includes offline data storage capabilities:

### Features:
- **Automatic Saving**: Data is automatically saved to local files when fetched
- **Offline Loading**: Subsequent requests load from local files without API calls
- **Fallback Support**: Graceful fallback to online API when offline data unavailable
- **Data Management**: Utilities for listing, validating, and managing offline data

### Benefits:
- **Reliability**: Works without internet connection
- **Performance**: Faster data loading from local storage
- **Cost-effective**: Reduces API usage and potential rate limiting
- **Reproducible**: Same data available for consistent testing and analysis

### Usage:
```python
# Service with offline data enabled (default)
service = YahooFinanceService()

# First call: Fetches from Yahoo Finance and saves locally
data = service.fetch_historical_data('AAPL', '1y')

# Subsequent calls: Load from local files
data = service.fetch_historical_data('AAPL', '1y')

# Force online refresh
data = service.fetch_historical_data('AAPL', '1y', force_online=True)
```

### Data Management:
```bash
# List available offline data
python scripts/manage_data.py list

# Validate offline data
python scripts/manage_data.py validate

# Clear specific data type
python scripts/manage_data.py clear --type raw

# Refresh specific symbols
python scripts/manage_data.py refresh --symbols AAPL,MSFT
```

## Testing

Comprehensive test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Real Data Testing**: Validation with actual market data
- **Error Scenarios**: Testing of various failure modes

Run tests:
```bash
python -m pytest tests/unit/test_data_pipeline.py
python -m pytest tests/integration/test_data_pipeline_integration.py
```

## Examples

See `examples/data_pipeline_demo.py` for a comprehensive demonstration of the data pipeline capabilities.

## Dependencies

- `pandas>=1.5.0`: Data manipulation and analysis
- `numpy>=1.21.0`: Numerical computing
- `yfinance>=0.2.0`: Yahoo Finance API integration
- `pytest>=7.0.0`: Testing framework

## Best Practices

1. **Use Appropriate Time Periods**: Choose periods that match your analysis needs
2. **Validate Data**: Always check validation results before analysis
3. **Monitor Quality Reports**: Review recommendations for data improvement
4. **Handle Missing Data**: Account for potential data gaps in analysis
5. **Consider Normalization**: Choose appropriate normalization for your use case

## Future Enhancements

- Additional data sources (Alpha Vantage, Quandl, etc.)
- Real-time data streaming
- Advanced outlier detection
- Automated data correction
- Multi-timeframe analysis
- Enhanced caching strategies