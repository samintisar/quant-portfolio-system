# Data Model: Core Forecasting Models for Returns & Volatility

## Entity Relationships

```
Asset (1) ---< Forecast >--- (N) TimeSeriesPoint
Asset (1) ---< VolatilityForecast >--- (N) TimeSeriesPoint
Asset (1) ---< RegimeLabel >--- (N) TimeSeriesPoint
Asset (1) ---< SignalValidation >--- (N) Forecast
MarketRegime (1) ---< RegimeLabel >--- (N) Asset
EconomicScenario (1) ---< ScenarioImpact >--- (N) Asset
```

## Core Entities

### Asset
Represents a financial instrument being forecasted.

**Fields**:
- `asset_id`: string (unique identifier, e.g., "AAPL.US")
- `symbol`: string (ticker symbol)
- `name`: string (full name)
- `asset_class`: enum (Equity, Bond, Commodity, Currency, ETF)
- `sector`: string (GICS sector classification)
- `country`: string (primary market country)
- `currency`: string (trading currency)
- `min_data_points`: integer (minimum historical points required)
- `created_at`: datetime
- `updated_at`: datetime

**Validation Rules**:
- asset_id must be unique and follow format "SYMBOL.COUNTRY"
- symbol must be valid exchange ticker
- asset_class must be one of predefined enum values
- min_data_points >= 252 (1 year of daily data)

### Forecast
Generated return forecast for an asset.

**Fields**:
- `forecast_id`: string (unique identifier)
- `asset_id`: string (foreign key to Asset)
- `model_type`: string (ARIMA, GARCH, HMM, Ensemble)
- `forecast_horizon`: integer (number of periods ahead)
- `created_at`: datetime
- `parameters`: JSON (model parameters used)
- `forecast_values`: array of floats (point forecasts)
- `confidence_intervals`: JSON (lower/upper bounds by confidence level)
- `metrics`: JSON (model performance metrics)

**Validation Rules**:
- forecast_horizon between 1 and 252 (max 1 year ahead)
- model_type must be implemented model
- confidence_intervals must include 95% level
- metrics must include RMSE, MAE, and direction accuracy

### VolatilityForecast
Generated volatility forecast for an asset.

**Fields**:
- `volatility_id`: string (unique identifier)
- `asset_id`: string (foreign key to Asset)
- `model_type`: string (GARCH, EGARCH, GJR-GARCH, Stochastic)
- `forecast_horizon`: integer (number of periods ahead)
- `created_at`: datetime
- `parameters`: JSON (GARCH parameters, distribution type)
- `volatility_forecasts`: array of floats (conditional variance)
- `long_run_variance`: float (unconditional variance estimate)
- `persistence`: float (GARCH persistence parameter)

**Validation Rules**:
- volatility_forecasts must be positive
- long_run_variance > 0
- persistence between 0 and 1 (stationarity)
- model_type must support volatility modeling

### MarketRegime
Classified market state with transition characteristics.

**Fields**:
- `regime_id`: string (unique identifier)
- `regime_name`: string (Bull, Bear, High_Volatility)
- `regime_description`: string (human-readable description)
- `expected_duration`: float (average periods in regime)
- `volatility_level`: enum (Low, Medium, High)
- `return_characteristics`: JSON (typical return distribution)
- `transition_probabilities`: JSON (matrix of regime transition)
- `created_at`: datetime
- `updated_at`: datetime

**Validation Rules**:
- regime_name must be unique and descriptive
- expected_duration > 0
- transition_probabilities must sum to 1.0 by row
- return_characteristics must include mean and volatility

### RegimeLabel
Asset-specific regime classification over time.

**Fields**:
- `label_id`: string (unique identifier)
- `asset_id`: string (foreign key to Asset)
- `regime_id`: string (foreign key to MarketRegime)
- `date`: date (classification date)
- `probability`: float (probability of being in regime)
- `confidence`: float (model confidence in classification)
- `features_used`: JSON (features contributing to classification)
- `created_at`: datetime

**Validation Rules**:
- probability between 0 and 1
- confidence between 0 and 1
- date must be within historical data range
- features_used must include key indicators

### EconomicScenario
Modeled economic condition with probabilistic impacts.

**Fields**:
- `scenario_id`: string (unique identifier)
- `scenario_name`: string (Recession, Growth, Stagflation, etc.)
- `scenario_description`: string (detailed description)
- `probability`: float (base probability of scenario)
- `economic_indicators`: JSON (key indicator values for scenario)
- `duration_estimate`: float (expected duration in periods)
- `created_at`: datetime
- `updated_at`: datetime

**Validation Rules**:
- scenario_name must be unique and meaningful
- probability between 0 and 1
- economic_indicators must include GDP, inflation, rates
- duration_estimate > 0

### ScenarioImpact
Impact of economic scenario on asset returns.

**Fields**:
- `impact_id`: string (unique identifier)
- `scenario_id`: string (foreign key to EconomicScenario)
- `asset_id`: string (foreign key to Asset)
- `expected_return`: float (mean return under scenario)
- `return_volatility`: float (volatility under scenario)
- `correlation_change`: float (correlation change vs baseline)
- `probability_adjustment`: float (asset-specific probability adjustment)
- `confidence_interval`: JSON (return distribution bounds)
- `created_at`: datetime

**Validation Rules**:
- expected_return must be reasonable (-50% to +100%)
- return_volatility > 0
- correlation_change between -1 and 1
- probability_adjustment between -1 and 1

### SignalValidation
Quality assessment of generated signals.

**Fields**:
- `validation_id`: string (unique identifier)
- `forecast_id`: string (foreign key to Forecast)
- `validation_type`: string (Statistical, Economic, Backtest)
- `validation_metrics`: JSON (specific metrics by type)
- `passed_checks`: boolean (overall validation result)
- `warnings`: array of strings (validation warnings)
- `errors`: array of strings (validation errors)
- `created_at`: datetime

**Validation Rules**:
- validation_type must be implemented validation method
- validation_metrics must include significance tests
- warnings and errors must be actionable
- passed_checks must be true for signal deployment

## State Transitions

### Forecast Generation
```
Historical Data → Model Training → Forecast Generation → Validation
     ↓              ↓                ↓                   ↓
Data Quality   Parameter Opt      Point Estimates    Statistical Tests
Requirements   Selection         Confidence Intervals Economic Sense Check
```

### Regime Detection
```
Market Data → Feature Extraction → HMM Training → Regime Classification → Validation
     ↓            ↓                ↓              ↓                   ↓
Multiple Asset Technical Indicators  EM Algorithm Viterbi Decoding   Persistence Check
```

### Scenario Modeling
```
Economic Data → Network Learning → Probability Estimation → Impact Assessment → Integration
     ↓            ↓               ↓                   ↓                ↓
Indicators   Structure Learning  Parameter Fits   Asset-specific    Combined Forecast
             from Data         Monte Carlo        Impacts        Generation
```

## Data Quality Constraints

### Input Data Requirements
- Minimum 252 daily observations (1 year)
- Maximum 5% missing values
- No gaps longer than 5 consecutive days
- Prices must be positive and non-zero
- Volumes must be non-negative

### Model Output Constraints
- Return forecasts must be reasonable (-50% to +100%)
- Volatility forecasts must be positive
- Regime probabilities must sum to 1.0
- Scenario impacts must be economically plausible
- Confidence intervals must contain point estimates

### Performance Requirements
- Model training time < 5 minutes per asset
- Forecast generation < 1 second per asset
- Memory usage < 4GB for 500 assets
- Statistical significance p < 0.05
- Out-of-sample performance >= benchmark

---
*Data Model v1.0 | Created: 2025-09-20*