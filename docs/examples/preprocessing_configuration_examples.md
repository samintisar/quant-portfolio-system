# Preprocessing Configuration Examples

This document provides comprehensive examples of preprocessing system configurations for different use cases and data types.

## Basic Configuration Structure

All preprocessing configurations follow this JSON structure:

```json
{
  "pipeline_id": "unique_identifier",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.1,
    "window_size": 5
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

## Example 1: Daily Equity Data

For standard daily stock price data with typical issues:

```json
{
  "pipeline_id": "equity_daily_v1",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.05
  },
  "outlier_detection": {
    "method": "iqr",
    "threshold": 1.5,
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

**Python Usage:**
```python
from lib.cleaning import DataCleaner
from lib.validation import DataValidator
from lib.normalization import DataNormalizer
import json

# Load configuration
with open('equity_daily_config.json', 'r') as f:
    config = json.load(f)

# Initialize components
cleaner = DataCleaner()
validator = DataValidator()
normalizer = DataNormalizer()

# Process data
processed = cleaner.handle_missing_values(
    df,
    method=config['missing_value_handling']['method'],
    threshold=config['missing_value_handling']['threshold']
)

processed, _ = cleaner.detect_outliers(
    processed,
    method=config['outlier_detection']['method'],
    threshold=config['outlier_detection']['threshold'],
    action=config['outlier_detection']['action']
)

processed, _ = normalizer.normalize_zscore(processed)
```

## Example 2: High-Frequency Data

For intraday or high-frequency trading data:

```json
{
  "pipeline_id": "hf_trading_v1",
  "missing_value_handling": {
    "method": "interpolation",
    "threshold": 0.02,
    "window_size": 3
  },
  "outlier_detection": {
    "method": "custom",
    "threshold": 5.0,
    "action": "flag"
  },
  "normalization": {
    "method": "robust",
    "preserve_stats": false
  },
  "quality_thresholds": {
    "completeness": 0.98,
    "consistency": 0.95,
    "accuracy": 0.99
  },
  "time_gap_handling": {
    "expected_frequency": "1m",
    "fill_method": "interpolate",
    "max_gap_minutes": 5
  }
}
```

## Example 3: Cryptocurrency Data

For volatile cryptocurrency price data:

```json
{
  "pipeline_id": "crypto_v1",
  "missing_value_handling": {
    "method": "interpolation",
    "threshold": 0.03
  },
  "outlier_detection": {
    "method": "percentile",
    "threshold": 2.0,
    "action": "clip"
  },
  "normalization": {
    "method": "robust",
    "preserve_stats": true
  },
  "quality_thresholds": {
    "completeness": 0.92,
    "consistency": 0.88,
    "accuracy": 0.90
  },
  "volatility_adjustment": {
    "enabled": true,
    "window": 20,
    "min_volatility": 0.01
  }
}
```

## Example 4: Multi-Asset Portfolio

For handling multiple asset classes in a portfolio:

```json
{
  "pipeline_id": "multi_asset_portfolio_v1",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.08
  },
  "outlier_detection": {
    "method": "zscore",
    "threshold": 3.5,
    "action": "flag"
  },
  "normalization": {
    "method": "minmax",
    "feature_range": [0, 1],
    "preserve_stats": true
  },
  "quality_thresholds": {
    "completeness": 0.90,
    "consistency": 0.85,
    "accuracy": 0.90
  },
  "asset_specific_rules": {
    "equities": {
      "price_columns": ["open", "high", "low", "close"],
      "volume_columns": ["volume"],
      "outlier_multiplier": 1.5
    },
    "bonds": {
      "price_columns": ["price", "yield"],
      "outlier_multiplier": 2.0
    },
    "commodities": {
      "price_columns": ["open", "high", "low", "close"],
      "volume_columns": ["volume", "open_interest"],
      "outlier_multiplier": 3.0
    }
  }
}
```

## Example 5: Backtesting Data Preparation

For preparing data specifically for backtesting:

```json
{
  "pipeline_id": "backtesting_v1",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.05
  },
  "outlier_detection": {
    "method": "iqr",
    "threshold": 2.0,
    "action": "clip"
  },
  "normalization": {
    "method": "returns_normalization",
    "preserve_stats": true
  },
  "quality_thresholds": {
    "completeness": 0.95,
    "consistency": 0.90,
    "accuracy": 0.95
  },
  "backtesting_specific": {
    "calculate_returns": true,
    "calculate_volatility": true,
    "volatility_window": 20,
    "preserve_timestamps": true,
    "min_data_points": 252
  }
}
```

## Example 6: Real-time Processing

For real-time data streams:

```json
{
  "pipeline_id": "realtime_v1",
  "missing_value_handling": {
    "method": "forward_fill",
    "threshold": 0.01
  },
  "outlier_detection": {
    "method": "custom",
    "threshold": 4.0,
    "action": "flag"
  },
  "normalization": {
    "method": "volatility_normalization",
    "window": 20
  },
  "quality_thresholds": {
    "completeness": 0.99,
    "consistency": 0.95,
    "accuracy": 0.98
  },
  "realtime_constraints": {
    "max_processing_time_ms": 100,
    "batch_size": 1000,
    "memory_limit_mb": 512,
    "fallback_mode": "pass_through"
  }
}
```

## Example 7: Machine Learning Preparation

For preparing data for ML models:

```json
{
  "pipeline_id": "ml_preparation_v1",
  "missing_value_handling": {
    "method": "median",
    "threshold": 0.10
  },
  "outlier_detection": {
    "method": "iqr",
    "threshold": 1.5,
    "action": "remove"
  },
  "normalization": {
    "method": "robust",
    "preserve_stats": false
  },
  "quality_thresholds": {
    "completeness": 0.90,
    "consistency": 0.95,
    "accuracy": 0.95
  },
  "ml_specific": {
    "feature_engineering": true,
    "remove_highly_correlated": true,
    "correlation_threshold": 0.95,
    "variance_threshold": 0.01,
    "handle_multicollinearity": true
  }
}
```

## Configuration Parameter Reference

### Missing Value Handling Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `forward_fill` | Propagate last valid value forward | Time series data |
| `interpolation` | Linear interpolation between points | High-frequency data |
| `mean` | Fill with column mean | General numerical data |
| `median` | Fill with column median | Data with outliers |
| `drop` | Remove rows with missing values | When missing data is minimal |

### Outlier Detection Methods

| Method | Description | Sensitivity |
|--------|-------------|------------|
| `zscore` | Standard deviations from mean | Medium |
| `iqr` | Interquartile range method | Low |
| `percentile` | Percentile-based detection | Customizable |
| `custom` | Financial-specific logic | High for financial data |

### Outlier Actions

| Action | Description | Use Case |
|--------|-------------|----------|
| `clip` | Clip to acceptable bounds | Preserves data points |
| `remove` | Remove outlier rows | When outliers are errors |
| `flag` | Add flag columns | Analysis of outliers |

### Normalization Methods

| Method | Range | Robustness | Best For |
|--------|-------|------------|----------|
| `zscore` | Variable | Medium | General use |
| `minmax` | [0,1] or custom | Low | Bounded features |
| `robust` | Variable | High | Data with outliers |
| `percentile` | Variable | Medium | Non-normal distributions |

## Advanced Configuration Examples

### Conditional Processing

```json
{
  "pipeline_id": "conditional_v1",
  "conditional_rules": [
    {
      "condition": "data_type == 'equity'",
      "config": {
        "outlier_detection": {
          "method": "iqr",
          "threshold": 1.5
        }
      }
    },
    {
      "condition": "volatility > 0.3",
      "config": {
        "normalization": {
          "method": "robust"
        }
      }
    }
  ]
}
```

### Multi-Stage Processing

```json
{
  "pipeline_id": "multi_stage_v1",
  "stages": [
    {
      "name": "initial_cleaning",
      "operations": [
        {"type": "remove_duplicates"},
        {"type": "handle_missing", "method": "forward_fill"}
      ]
    },
    {
      "name": "outlier_handling",
      "operations": [
        {"type": "detect_outliers", "method": "zscore", "action": "flag"}
      ]
    },
    {
      "name": "final_preparation",
      "operations": [
        {"type": "normalize", "method": "zscore"},
        {"type": "validate_quality"}
      ]
    }
  ]
}
```

## Performance Optimization

### Large Dataset Configuration

```json
{
  "pipeline_id": "large_dataset_v1",
  "performance": {
    "chunk_size": 100000,
    "parallel_processing": true,
    "max_workers": 4,
    "memory_optimization": {
      "dtype_optimization": true,
      "chunk_processing": true,
      "garbage_collection": true
    }
  },
  "monitoring": {
    "track_performance": true,
    "log_progress": true,
    "alert_thresholds": {
      "memory_gb": 8.0,
      "time_minutes": 30
    }
  }
}
```

## Error Handling and Fallbacks

```json
{
  "pipeline_id": "robust_v1",
  "error_handling": {
    "fallback_strategies": [
      {
        "error_type": "memory_error",
        "action": "reduce_chunk_size",
        "retry_count": 3
      },
      {
        "error_type": "validation_error",
        "action": "use_relaxed_rules",
        "relaxed_config": "relaxed_config.json"
      }
    ],
    "logging": {
      "level": "INFO",
      "save_errors": true,
      "error_log_file": "preprocessing_errors.log"
    }
  }
}
```

## Testing Your Configuration

Always test your configuration with sample data:

```python
def test_preprocessing_config(config, sample_data):
    """Test preprocessing configuration."""
    try:
        # Apply preprocessing
        processed = apply_preprocessing(sample_data, config)

        # Check quality
        quality_score = calculate_quality_score(processed)

        # Validate results
        assert quality_score > config['quality_thresholds']['completeness']

        # Check data integrity
        assert not processed.isnull().all().any()

        print(f"Configuration test passed. Quality score: {quality_score:.3f}")
        return True

    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False
```

## Best Practices

1. **Start Simple**: Begin with basic configuration and gradually add complexity
2. **Validate Quality**: Always check data quality after preprocessing
3. **Monitor Performance**: Track processing time and memory usage
4. **Handle Edge Cases**: Plan for empty data, all-NaN, and constant values
5. **Document Configurations**: Keep clear documentation of each configuration
6. **Version Control**: Track configuration changes like code
7. **Test Thoroughly**: Validate with realistic test data
8. **Monitor in Production**: Continuously monitor preprocessing quality

For more information, see the main documentation in the `docs/` directory.