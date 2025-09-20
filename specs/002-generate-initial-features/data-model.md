# Data Model: Initial Financial Features

## Entity Relationships

### Core Entities

#### FinancialInstrument
Represents tradable financial assets with metadata and price history.

**Fields**:
- `symbol` (string, required): Unique identifier (e.g., "AAPL", "SPY")
- `name` (string, optional): Full name of the instrument
- `type` (enum, required): Instrument type - STOCK, ETF, INDEX, FUTURES
- `exchange` (string, optional): Trading exchange (e.g., "NASDAQ", "NYSE")
- `currency` (string, optional): Trading currency (default: "USD")
- `sector` (string, optional): Economic sector classification
- `start_date` (date, required): First available data date
- `end_date` (date, required): Last available data date

**Validation Rules**:
- Symbol must be unique across all instruments
- Symbol must match exchange format (alphanumeric, no spaces)
- Type must be one of predefined enum values
- Start_date <= end_date

#### PriceData
Time series of price information for each instrument.

**Fields**:
- `instrument_id` (foreign key to FinancialInstrument)
- `date` (date, required): Trading date
- `open` (decimal, required): Opening price
- `high` (decimal, required): Highest price
- `low` (decimal, required): Lowest price
- `close` (decimal, required): Closing price
- `volume` (integer, optional): Trading volume
- `adjusted_close` (decimal, optional): Dividend/split adjusted close

**Validation Rules**:
- Low <= High, Open <= High, Close <= High
- Low >= Open, Low >= Close (standard price relationships)
- Volume >= 0
- All price fields > 0
- Dates must be trading days (no weekends/holidays)
- Monotonically increasing dates

#### FeatureSet
Container for all calculated features for a specific instrument and time period.

**Fields**:
- `instrument_id` (foreign key to FinancialInstrument)
- `calculation_date` (datetime, required): When features were calculated
- `parameters_hash` (string, required): Hash of calculation parameters
- `data_quality_score` (decimal, required): Quality assessment (0.0-1.0)
- `validation_flags` (json, optional): Validation results and warnings

#### ReturnSeries
Calculated returns for each instrument and calculation method.

**Fields**:
- `feature_set_id` (foreign key to FeatureSet)
- `date` (date, required): Return calculation date
- `return_type` (enum, required): ARITHMETIC, LOGARITHMIC, PERCENTAGE
- `return_value` (decimal, required): Calculated return
- `period` (integer, required): Number of days for return calculation
- `is_valid` (boolean, required): Statistical validity flag

**Validation Rules**:
- Return_value must be within reasonable bounds [-10, +10]
- Period > 0 and <= 252 (1 year maximum)
- Return_type must be predefined enum value
- Date must have corresponding price data

#### VolatilityMeasure
Rolling volatility calculations.

**Fields**:
- `feature_set_id` (foreign key to FeatureSet)
- `date` (date, required): Volatility calculation date
- `window_size` (integer, required): Rolling window size in days
- `volatility_type` (enum, required): STANDARD_DEVIATION, ANNUALIZED
- `volatility_value` (decimal, required): Calculated volatility
- `mean_return` (decimal, required): Mean return over window
- `sample_size` (integer, required): Number of data points used

**Validation Rules**:
- Volatility_value >= 0
- Window_size >= 5 and <= 252
- Sample_size >= window_size * 0.8 (allowing some missing data)
- Volatility_value < 10 (1000% annual volatility maximum)

#### MomentumIndicator
Various momentum indicators for trend analysis.

**Fields**:
- `feature_set_id` (foreign key to FeatureSet)
- `date` (date, required): Indicator calculation date
- `indicator_type` (enum, required): SIMPLE_MOMENTUM, RSI, ROC
- `indicator_value` (decimal, required): Calculated indicator value
- `lookback_period` (integer, required): Number of periods for calculation
- `signal_strength` (decimal, optional): Normalized signal strength (-1 to +1)
- `interpretation` (string, optional): Brief interpretation of signal

**Validation Rules**:
- Indicator_value within type-specific bounds
- Lookback_period >= 1 and <= 252
- Signal_strength between -1 and +1 if present
- Valid date with corresponding price data

## Entity Relationships

```
FinancialInstrument (1) <---> (N) PriceData
FinancialInstrument (1) <---> (N) FeatureSet
FeatureSet (1) <---> (N) ReturnSeries
FeatureSet (1) <---> (N) VolatilityMeasure
FeatureSet (1) <---> (N) MomentumIndicator
```

## State Transitions

### Data Processing Pipeline
1. **Raw Data**: Price data loaded from source
2. **Validated**: Data quality checks performed
3. **Features**: Financial features calculated
4. **Validated**: Statistical validation completed
5. **Ready**: Available for analysis and modeling

### Quality States
- `RAW`: Unprocessed data from source
- `CLEANED`: Missing data handled, outliers addressed
- `VALIDATED`: Statistical integrity verified
- `FEATURED`: All features calculated
- `READY`: Available for quantitative analysis

## Configuration Model

### FeatureParameters
Configuration for feature calculations.

**Fields**:
- `return_methods` (array): Return calculation methods to use
- `volatility_windows` (array): Window sizes for volatility
- `momentum_types` (array): Momentum indicators to calculate
- `lookback_periods` (array): Periods for momentum calculations
- `missing_data_strategy` (string): How to handle missing data
- `outlier_detection` (object): Outlier detection parameters
- `validation_rules` (object): Custom validation thresholds

### DataQualityMetrics
Quality assessment results.

**Fields**:
- `completeness_score` (decimal): Proportion of non-missing data
- `outlier_count` (integer): Number of outliers detected
- `gaps_detected` (integer): Number of data gaps
- `statistical_anomalies` (array): Detected statistical issues
- `recommendation` (string): Data quality recommendation

## API Design Implications

### Input Requirements
- Instruments array with symbols and metadata
- Price data time series for each instrument
- Configuration parameters for calculations
- Data quality validation rules

### Output Structure
- Feature sets grouped by instrument
- All calculated returns, volatility, and momentum indicators
- Quality scores and validation flags
- Metadata for reproducibility

### Error Handling
- Missing instrument data
- Insufficient history for calculations
- Numerical instability issues
- Data quality violations

This data model supports the constitutional requirements for library-first architecture, CLI interfaces, and comprehensive testing while maintaining financial soundness and statistical rigor.