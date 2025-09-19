# Data Model: Data Preprocessing System

## Entity Overview

### 1. RawDataStream
**Purpose**: Represents ingested market data from various sources before preprocessing

**Fields**:
- `symbol`: str - Trading symbol/ticker (e.g., "AAPL", "MSFT")
- `timestamp`: datetime - Data point timestamp
- `open`: float - Opening price
- `high`: float - Highest price
- `low`: float - Lowest price
- `close`: float - Closing price
- `volume`: int - Trading volume
- `data_source`: str - Source of data (e.g., "yahoo_finance", "bloomberg")
- `frequency`: str - Data frequency (e.g., "1d", "1h", "1m")
- `quality_score`: float - Initial data quality assessment (0-1)

**Validation Rules**:
- symbol: Must be valid ticker format
- timestamp: Must be chronological, no duplicates
- prices: Must be >= 0, high >= low >= open/close
- volume: Must be >= 0
- frequency: Must match actual data frequency

### 2. ProcessedData
**Purpose**: Cleaned and normalized output ready for analysis

**Fields**:
- `symbol`: str - Trading symbol/ticker
- `timestamp`: datetime - Data point timestamp
- `open_normalized`: float - Normalized opening price
- `high_normalized`: float - Normalized highest price
- `low_normalized`: float - Normalized lowest price
- `close_normalized`: float - Normalized closing price
- `volume_normalized`: float - Normalized trading volume
- `returns`: float - Calculated returns
- `volatility`: float - Rolling volatility
- `preprocessing_version`: str - Version of preprocessing pipeline used
- `quality_flags`: dict - Data quality indicators
- `outlier_flags`: dict - Outlier detection results

**Validation Rules**:
- normalized values: Must be in expected range based on method
- returns: Must be finite, reasonable bounds (-50% to +50% daily)
- volatility: Must be >= 0
- quality_flags: Must contain completeness, consistency, accuracy scores

### 3. PreprocessingRules
**Purpose**: Configurable settings for handling data quality issues

**Fields**:
- `rule_id`: str - Unique identifier for the rule
- `rule_type`: str - Type of rule (missing_value, outlier, normalization, validation)
- `asset_class`: str - Applicable asset class (equity, fx, bond, commodity)
- `parameters`: dict - Rule-specific parameters
- `priority`: int - Execution priority (1-10)
- `active`: bool - Whether rule is active
- `created_at`: datetime - Rule creation timestamp
- `updated_at`: datetime - Last update timestamp

**Validation Rules**:
- rule_type: Must be one of predefined types
- asset_class: Must be valid asset class
- parameters: Must match schema for rule type
- priority: Must be 1-10

### 4. QualityMetrics
**Purpose**: Statistics measuring data completeness, consistency, and validity

**Fields**:
- `metric_id`: str - Unique metric identifier
- `dataset_id`: str - Associated dataset identifier
- `metric_type`: str - Type of metric (completeness, consistency, accuracy, timeliness)
- `value`: float - Metric value
- `threshold`: float - Threshold for alerting
- `status`: str - Status (pass, warn, fail)
- `timestamp`: datetime - When metric was calculated
- `metadata`: dict - Additional metric information

**Validation Rules**:
- metric_type: Must be one of predefined types
- value: Must be 0-1 for ratio metrics
- status: Must be one of pass/warn/fail
- threshold: Must be reasonable based on metric type

### 5. ProcessingLog
**Purpose**: Record of all transformations applied to the data

**Fields**:
- `log_id`: str - Unique log identifier
- `dataset_id`: str - Associated dataset identifier
- `operation`: str - Operation performed
- `input_shape`: tuple - Shape of input data
- `output_shape`: tuple - Shape of output data
- `parameters_used`: dict - Parameters used in operation
- `execution_time`: float - Time taken for operation
- `rows_affected`: int - Number of rows affected
- `timestamp`: datetime - When operation was performed
- `success`: bool - Whether operation succeeded
- `error_message`: str - Error message if failed

**Validation Rules**:
- operation: Must be valid preprocessing operation
- execution_time: Must be >= 0
- rows_affected: Must be >= 0
- success: Must be boolean

## Entity Relationships

```
RawDataStream → ProcessedData
    (1:N) - One raw stream can produce multiple processed versions

PreprocessingRules → ProcessedData
    (1:N) - Rules are applied to create processed data

ProcessedData → QualityMetrics
    (1:N) - Each processed dataset has quality metrics

ProcessedData → ProcessingLog
    (1:N) - Each processing operation generates log entries
```

## State Transitions

### Data Processing States
1. **Raw**: Data as received from source
2. **Validated**: Initial quality checks completed
3. **Cleaned**: Missing values handled, outliers processed
4. **Normalized**: Data scaled and transformed
5. **Final**: Ready for analysis, with quality metrics attached

### Rule Management States
1. **Draft**: Rule being defined
2. **Active**: Rule in use
3. **Inactive**: Rule disabled but preserved
4. **Archived**: Rule no longer needed
5. **Deleted**: Rule removed from system

## Data Quality Dimensions

### Completeness
- Percentage of non-null values
- Gap analysis for time series
- Coverage across required fields

### Consistency
- Cross-field validation (e.g., high >= low)
- Temporal consistency checks
- Cross-series correlation validation

### Accuracy
- Domain validation (e.g., price ranges)
- Statistical outlier detection
- Cross-source verification when available

### Timeliness
- Data latency measurement
- Update frequency compliance
- Real-time processing performance

## Configuration Schema

### Preprocessing Pipeline Configuration
```json
{
  "pipeline_id": "string",
  "description": "string",
  "asset_classes": ["string"],
  "rules": [
    {
      "rule_type": "string",
      "parameters": {},
      "priority": "integer"
    }
  ],
  "quality_thresholds": {
    "completeness": "float",
    "consistency": "float",
    "accuracy": "float"
  },
  "output_format": "string"
}
```

### Rule Parameter Schemas

#### Missing Value Handling
```json
{
  "method": "forward_fill|interpolation|mean|median|drop",
  "threshold": "float",
  "window_size": "integer"
}
```

#### Outlier Detection
```json
{
  "method": "zscore|iqr|percentile|custom",
  "threshold": "float",
  "action": "clip|remove|flag"
}
```

#### Normalization
```json
{
  "method": "zscore|minmax|robust|percentile",
  "range": "array",
  "preserve_stats": "boolean"
}
```

## Performance Considerations

### Memory Usage
- Chunk processing for large datasets
- Efficient data types (float32 vs float64)
- Lazy loading when possible

### Processing Speed
- Vectorized operations using pandas/numpy
- Parallel processing for independent operations
- Caching of intermediate results

### Scalability
- Support for incremental processing
- Streaming capabilities for real-time data
- Distributed processing options for very large datasets