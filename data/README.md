# Data Module

This module contains comprehensive data management components for the quantitative trading system, including ingestion, preprocessing, storage, and quality control.

## Structure

- **api/**: REST API interfaces for data services
- **cli/**: Command-line interfaces for data processing
- **config/**: Configuration management for pipelines and processing
- **feeds/**: Data source connectors and market data ingestion
- **lib/**: Core preprocessing libraries (cleaning, validation, normalization)
- **models/**: Data models and entities
- **services/**: Data processing services
- **storage/**: Data persistence and caching mechanisms
- **preprocessing.py**: Main preprocessing orchestration module

## Key Components

### Data Feeds (`feeds/`)
- **Yahoo Finance API Integration**: Historical market data ingestion
- **Multi-asset Support**: Equities, ETFs, FX, Bonds, Commodities
- **Batch Processing**: Concurrent data fetching with rate limiting

### Preprocessing Libraries (`lib/`)
- **cleaning.py**: Missing value handling, outlier detection, time gap management
- **validation.py**: Data integrity checks, financial logic validation, statistical validation
- **normalization.py**: Z-score, Min-Max, Robust scaling with financial specialization

### CLI Tools (`cli/`)
- **preprocess.py**: Command-line data preprocessing with quality reporting
- **quality_report.py**: Automated data quality assessment and reporting

### Configuration (`config/`)
- **pipeline_config.py**: Comprehensive preprocessing pipeline configuration
- **configs/**: Environment-specific configuration files

## Usage

### Data Ingestion
```python
from data.src.feeds import create_default_ingestion_system, AssetClass
from datetime import datetime, timedelta

# Create ingestion system
ingestion = create_default_ingestion_system()
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch equity data
results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)
```

### Data Preprocessing
```python
from data.src.preprocessing import PreprocessingOrchestrator
from data.src.lib import cleaning, validation, normalization

# Initialize preprocessing orchestrator
orchestrator = PreprocessingOrchestrator()

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

### CLI Usage
```bash
# Preprocess data with quality reporting
python -m data.src.cli.preprocess --input data/raw/ --output data/processed/ --report

# Generate data quality report
python -m data.src.cli.quality_report --data data/processed/ --output quality_report.json
```

## Performance Targets

- **Processing Speed**: 10 million data points in under 30 seconds
- **Memory Efficiency**: Less than 4GB memory usage for large datasets
- **Real-time Processing**: Sub-second processing for 1K data batches
- **Scalability**: Linear scaling with dataset size

## Testing

Comprehensive test suite located in `tests/`:

```bash
# Unit tests for preprocessing libraries
pytest tests/unit/

# Performance and memory tests
pytest tests/performance/

# Statistical validation tests
pytest tests/statistical/

# Integration tests
pytest tests/integration/

# Contract tests
pytest tests/contract/
```

## Configuration

Preprocessing pipelines are configured via `data/src/config/pipeline_config.py`:

```python
from data.src.config.pipeline_config import PreprocessingConfig

# Load configuration
config = PreprocessingConfig.load('config/pipeline_config.json')

# Customize preprocessing steps
config.cleaning.outlier_method = 'iqr'
config.cleaning.missing_strategy = 'forward_fill'
config.normalization.method = 'z_score'
```