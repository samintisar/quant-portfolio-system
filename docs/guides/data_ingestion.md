# Data Ingestion and Storage Guide

This guide explains how to use the data ingestion and storage system for the quantitative trading system.

## Overview

The system provides:
- **Historical data ingestion** from Yahoo Finance
- **Multi-asset class support** (equities, ETFs, FX, bonds, commodities)
- **Persistent storage** with multiple format options
- **Data validation** and quality control
- **Efficient retrieval** and management

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python scripts/setup_data_environment.py

# Install dependencies
pip install -r docs/requirements.txt
```

### 2. Basic Usage

```python
from data.src.feeds import create_default_ingestion_system, AssetClass
from datetime import datetime, timedelta

# Create ingestion system
ingestion = create_default_ingestion_system()

# Define date range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch data
results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)

# Save to storage
ingestion.save_results_to_storage(results)
```

### 3. Run Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Complete demonstration
python scripts/demo_data_ingestion_and_storage.py
```

## Project Structure

```
quant-portfolio-system/
├── data/                          # Data source code
│   ├── src/
│   │   ├── api/                   # REST API interfaces
│   │   ├── cli/                   # Command-line interfaces
│   │   │   ├── preprocess.py      # Data preprocessing CLI
│   │   │   └── quality_report.py  # Quality reporting CLI
│   │   ├── config/                # Configuration management
│   │   │   └── pipeline_config.py # Preprocessing configuration
│   │   ├── feeds/                 # Data ingestion modules
│   │   │   ├── __init__.py
│   │   │   ├── yahoo_finance_ingestion.py
│   │   │   └── data_ingestion_interface.py
│   │   ├── lib/                   # Core preprocessing libraries
│   │   │   ├── cleaning.py        # Data cleaning utilities
│   │   │   ├── validation.py      # Data validation utilities
│   │   │   └── normalization.py   # Data normalization utilities
│   │   ├── models/                # Data models and entities
│   │   ├── services/              # Data processing services
│   │   ├── storage/               # Data storage modules
│   │   │   ├── __init__.py
│   │   │   └── market_data_storage.py
│   │   └── preprocessing.py       # Main preprocessing orchestration
│   ├── storage/                   # Actual data storage location
│   │   ├── raw/                   # Raw market data
│   │   │   ├── equity/           # Stock data
│   │   │   ├── etf/              # ETF data
│   │   │   ├── fx/               # FX data
│   │   │   └── bond/             # Bond data
│   │   ├── processed/             # Processed data
│   │   └── metadata/             # Storage metadata
│   └── tests/                     # Data module tests
├── scripts/                      # Utility scripts
│   ├── setup_data_environment.py
│   ├── data_management.py
│   ├── automated_data_refresh.py
│   ├── initialize_historical_data.py
│   └── run_preprocessing.py
├── examples/                     # Usage examples
│   └── basic_usage.py
├── docs/                         # Documentation
│   ├── requirements.txt           # Single source of truth for dependencies
│   ├── data_ingestion_guide.md    # Data system documentation
│   └── cli/                       # CLI documentation
├── tests/                         # Comprehensive test suite
│   ├── unit/                      # Unit tests for libraries
│   ├── statistical/               # Statistical validation tests
│   ├── performance/               # Performance and memory tests
│   ├── integration/               # Integration tests
│   ├── data/                      # Data-specific tests
│   └── contract/                  # Contract tests
├── config/                       # Configuration files
├── output/                        # Analysis outputs
└── README.md
```

## Core Components

### 1. Data Ingestion (`data/src/feeds/`)

#### YahooFinanceIngestion
- Fetches data from Yahoo Finance API
- Supports equities, ETFs, FX, bonds, commodities
- Includes rate limiting and error handling
- Built-in data validation

#### UnifiedDataIngestion
- High-level interface for data operations
- Support for multiple data sources
- Batch processing capabilities
- DataFrame export functionality

### 2. Data Storage (`data/src/storage/`)

#### MarketDataStorage
- Persistent storage with multiple formats
- Versioning and metadata tracking
- Efficient retrieval with filtering
- Storage management utilities

#### Supported Formats
- **Parquet** (recommended): Efficient columnar storage
- **CSV**: Human-readable text format
- **HDF5**: Hierarchical data format
- **Feather**: Fast binary format
- **Pickle**: Python object serialization

### 3. Data Preprocessing (`data/src/lib/` and `data/src/preprocessing.py`)

#### Core Libraries
- **cleaning.py**: Missing value handling, outlier detection, time gap management
- **validation.py**: Data integrity checks, financial logic validation, statistical validation
- **normalization.py**: Z-score, Min-Max, Robust scaling with financial specialization

#### Main Preprocessor
- **preprocessing.py**: Orchestration module for complete preprocessing pipelines
- **Performance Optimized**: 10M data points in <30 seconds, <4GB memory usage
- **Quality Integration**: Automated quality assessment and reporting

#### Key Features
- **Multiple Cleaning Methods**: IQR, Z-score, isolation forest for outlier detection
- **Flexible Missing Value Handling**: Forward fill, backward fill, interpolation, drop
- **Financial-Aware Validation**: Price bounds, volume validation, logic checks
- **Normalization Options**: Z-score, Min-Max, Robust scaling with edge case handling
- **Real-time Processing**: Sub-second processing for 1K data batches

## Usage Examples

### Fetching Single Asset

```python
from data.src.feeds import create_default_ingestion_system, AssetClass
from datetime import datetime, timedelta

ingestion = create_default_ingestion_system()
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch Apple stock data
results = ingestion.fetch_equities(['AAPL'], start_date, end_date)

# Access the data
if results['AAPL'].success:
    aapl_data = results['AAPL'].data
    print(f"Latest price: ${aapl_data['close'].iloc[-1]:.2f}")
```

### Batch Processing

```python
# Multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
results = ingestion.fetch_equities(symbols, start_date, end_date)

# Multiple asset classes
equity_results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)
etf_results = ingestion.fetch_etfs(['SPY', 'QQQ'], start_date, end_date)
fx_results = ingestion.fetch_fx_pairs(['EURUSD', 'GBPUSD'], start_date, end_date)
```

### Storage Operations

```python
from data.src.storage import create_default_storage

# Save results to storage
save_results = ingestion.save_results_to_storage(results)

# Retrieve data later
storage = create_default_storage()
aapl_data = storage.load_data('AAPL', AssetClass.EQUITY)

# Check what's available
symbols = storage.get_available_symbols()
storage_info = storage.get_storage_info()
```

### DataFrame Export

```python
# Export to single DataFrame
combined_df = ingestion.export_results_to_dataframe(results, combine=True)

# Export to individual DataFrames
individual_dfs = ingestion.export_results_to_dataframe(results, combine=False)
```

### Data Preprocessing

```python
from data.src.preprocessing import PreprocessingOrchestrator
from data.src.lib import cleaning, validation, normalization

# Create preprocessing orchestrator
orchestrator = PreprocessingOrchestrator()

# Clean data (handle missing values, outliers)
cleaned_data = cleaning.remove_outliers(data, method='iqr', threshold=1.5)
cleaned_data = cleaning.handle_missing_values(cleaned_data, strategy='forward_fill')

# Validate data integrity
validation_report = validation.validate_financial_data(cleaned_data)
if not validation_report['is_valid']:
    print(f"Data validation failed: {validation_report['errors']}")

# Normalize data for analysis
normalized_data = normalization.z_score_normalization(cleaned_data)

# Complete preprocessing pipeline
pipeline_result = orchestrator.run_preprocessing_pipeline(
    data,
    cleaning_config={'outlier_method': 'iqr', 'missing_strategy': 'forward_fill'},
    validation_config={'check_bounds': True, 'check_duplicates': True},
    normalization_config={'method': 'z_score', 'preserve_relationships': True}
)
```

### CLI Preprocessing

```bash
# Preprocess data with quality reporting
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report

# Generate data quality report
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json

# Run preprocessing with custom configuration
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --config config/pipeline_config.json
```

## Data Format

### Raw Data Structure
Each stored dataset contains:
- **OHLCV** data (Open, High, Low, Close, Volume)
- **Returns** (simple and log returns)
- **Dollar volume** (price × volume)
- **Symbol** identifier
- **Datetime** index

### Example Data
```
            open    high     low   close  volume      return  log_return  dollar_volume  symbol
date
2024-09-18  175.2  176.8   174.5   176.3  500000    0.012345    0.012282      88150000    AAPL
2024-09-19  176.5  177.2   175.8   176.9  480000    0.003403    0.003397      84912000    AAPL
```

## Configuration

### Storage Configuration
```python
from data.src.storage import MarketDataStorage, StorageFormat, CompressionType

# Custom storage configuration
storage = MarketDataStorage(
    base_path="custom/storage/path",
    default_format=StorageFormat.PARQUET,
    compression=CompressionType.GZIP,
    enable_versioning=True
)
```

### Ingestion Configuration
```python
from data.src.feeds import YahooFinanceIngestion

# Custom ingestion configuration
ingestion = YahooFinanceIngestion(
    max_workers=10,      # Concurrent requests
    rate_limit=0.05      # 50ms between requests
)
```

## Best Practices

### 1. Data Management
- Use **Parquet** format for best performance
- Enable **versioning** for historical tracking
- **Clean up** old versions periodically
- **Monitor** storage usage

### 2. Performance
- Use **batch processing** for multiple symbols
- **Limit date ranges** to what you need
- **Cache** frequently accessed data
- **Avoid** redundant requests

### 3. Preprocessing
- **Always validate data quality** before quantitative analysis
- **Use appropriate normalization** methods for different data types
- **Monitor memory usage** when processing large datasets
- **Preserve statistical relationships** during normalization
- **Handle financial edge cases** (zero prices, negative volumes, extreme moves)

### 4. Quality Management
- **Set appropriate thresholds** for quality metrics based on use case
- **Monitor quality trends** over time
- **Investigate quality degradation** immediately
- **Document quality issues** and resolution steps
- **Automate quality reporting** for regular monitoring

### 5. Error Handling
- **Validate** data before use
- **Check** success status of operations
- **Log** errors for debugging
- **Retry** failed operations

## Performance Targets

### System Performance
- **Data Ingestion**: Concurrent processing with rate limiting
- **Preprocessing Speed**: 10 million data points in <30 seconds
- **Memory Efficiency**: <4GB memory usage for large datasets
- **Real-time Processing**: Sub-second processing for 1K data batches
- **Scalability**: Linear scaling with dataset size

### Quality Targets
- **Data Completeness**: >95% non-null values
- **Data Accuracy**: >99% validation pass rate
- **Processing Consistency**: <1% variance across runs
- **Quality Assessment**: <5 seconds for standard reports

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path includes project root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r docs/requirements.txt
   ```

3. **Permission Issues**
   ```bash
   # Ensure write permissions for storage directory
   chmod -R 755 data/storage/
   ```

4. **Network Issues**
   - Check internet connection
   - Verify Yahoo Finance API accessibility
   - Consider using a proxy if needed

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will provide detailed logs for troubleshooting
```

## API Reference

### Classes

#### `UnifiedDataIngestion`
- `fetch_equities(symbols, start_date, end_date)`
- `fetch_etfs(symbols, start_date, end_date)`
- `fetch_fx_pairs(symbols, start_date, end_date)`
- `save_results_to_storage(results)`
- `export_results_to_dataframe(results)`

#### `MarketDataStorage`
- `save_ingestion_result(result)`
- `load_data(symbol, asset_class, start_date, end_date)`
- `get_available_symbols()`
- `get_storage_info()`

### Enums

#### `AssetClass`
- `EQUITY`, `ETF`, `FX`, `BOND`, `COMMODITY`, `INDEX`

#### `StorageFormat`
- `CSV`, `PARQUET`, `HDF5`, `PICKLE`, `FEATHER`

## Contributing

1. Follow the existing code structure
2. Add appropriate tests
3. Update documentation
4. Use type hints
5. Follow PEP 8 guidelines

## License

This project is part of the quantitative trading system and follows the same license terms.