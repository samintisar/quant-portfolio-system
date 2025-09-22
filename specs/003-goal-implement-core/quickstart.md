# Quickstart Guide: Core Forecasting Models

## Overview

This quickstart guide demonstrates how to use the core forecasting models to generate return and volatility predictions, detect market regimes, and model economic scenarios.

## Prerequisites

- Python 3.11+
- Required packages installed (see requirements.txt)
- Historical market data available
- API access key (if using API endpoints)

## Installation

1. **Install dependencies**:
```bash
pip install -r docs/requirements.txt
```

2. **Verify installation**:
```bash
python -c "import statsmodels, arch, hmmlearn, pgmpy; print('All dependencies available')"
```

## Quickstart Examples

### 1. Generate Return Forecasts

#### Using CLI (Recommended for reproducibility)
```bash
# Generate ARIMA forecast for Apple stock
python -m forecasting.cli.return_forecast \
    --assets AAPL.US \
    --model-type ARIMA \
    --parameters '{"p": 2, "d": 1, "q": 1}' \
    --horizon 30 \
    --confidence-levels 0.95 \
    --output results/apple_forecast.json

# Auto-fit ARIMA parameters
python -m forecasting.cli.return_forecast \
    --assets AAPL.US MSFT.US GOOGL.US \
    --auto-fit \
    --horizon 21 \
    --output results/multi_forecast.json
```

#### Using Python API
```python
from forecasting.models.arima_model import ARIMAForecaster
from forecasting.config import ForecastConfig

# Initialize forecaster
config = ForecastConfig(
    assets=["AAPL.US"],
    model_type="ARIMA",
    parameters={"p": 2, "d": 1, "q": 1},
    forecast_horizon=30,
    confidence_levels=[0.90, 0.95]
)

forecaster = ARIMAForecaster(config)
result = forecaster.forecast()

print(f"Forecast: {result.forecast_values}")
print(f"95% CI: {result.confidence_intervals[0.95]}")
```

#### Using REST API
```bash
curl -X POST "http://localhost:8000/api/v1/forecasts/returns" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "assets": ["AAPL.US"],
    "model_config": {
      "model_type": "ARIMA",
      "parameters": {"p": 2, "d": 1, "q": 1}
    },
    "forecast_horizon": 30,
    "confidence_levels": [0.95]
  }'
```

### 2. Generate Volatility Forecasts

#### CLI Usage
```bash
# GARCH volatility forecast
python -m forecasting.cli.volatility_forecast \
    --assets AAPL.US \
    --model-type GARCH \
    --distribution normal \
    --horizon 21 \
    --output results/apple_volatility.json

# EGARCH with asymmetric effects
python -m forecasting.cli.volatility_forecast \
    --assets SPY.US \
    --model-type EGARCH \
    --distribution student_t \
    --horizon 30 \
    --output results/spy_volatility.json
```

#### Python API
```python
from forecasting.models.garch_model import GARCHForecaster

config = ForecastConfig(
    assets=["AAPL.US"],
    model_type="GARCH",
    forecast_horizon=21
)

vol_forecaster = GARCHForecaster(config)
vol_result = vol_forecaster.forecast()

print(f"Volatility forecast: {vol_result.volatility_forecasts}")
print(f"Long-run variance: {vol_result.long_run_variance}")
print(f"Persistence: {vol_result.persistence}")
```

### 3. Detect Market Regimes

#### CLI Usage
```bash
# Detect 3-regime model for multiple assets
python -m forecasting.cli.regime_detection \
    --assets AAPL.US MSFT.US SPY.US \
    --n-regimes 3 \
    --features returns volatility volume \
    --lookback 252 \
    --output results/market_regimes.json

# Analyze regime transitions
python -m forecasting.cli.regime_analysis \
    --regime-file results/market_regimes.json \
    --output results/regime_analysis.json
```

#### Python API
```python
from forecasting.models.hmm_model import HMMRegimeDetector

config = ForecastConfig(
    assets=["AAPL.US", "MSFT.US", "SPY.US"],
    n_regimes=3,
    features=["returns", "volatility", "volume"],
    lookback_period=252
)

detector = HMMRegimeDetector(config)
regime_result = detector.detect_regimes()

print(f"Regime labels: {regime_result.regime_labels}")
print(f"Transition probabilities: {regime_result.transition_probabilities}")
print(f"Expected durations: {regime_result.expected_durations}")
```

### 4. Model Economic Scenarios

#### CLI Usage
```bash
# Model recession scenario impacts
python -m forecasting.cli.scenario_modeling \
    --assets AAPL.US MSFT.US \
    --scenarios recession growth \
    --indicators gdp inflation unemployment \
    --output results/scenario_impacts.json

# Custom scenario definition
python -m forecasting.cli.scenario_modeling \
    --assets SPY.US \
    --scenario-config '{"recession": {"gdp": -0.02, "inflation": 0.01, "unemployment": 0.08}}' \
    --output results/custom_scenario.json
```

#### Python API
```python
from forecasting.models.scenario_model import ScenarioModeler

scenario_config = {
    "recession": {
        "gdp": -0.02,
        "inflation": 0.01,
        "unemployment": 0.08
    },
    "growth": {
        "gdp": 0.03,
        "inflation": 0.02,
        "unemployment": 0.04
    }
}

modeler = ScenarioModeler(assets=["AAPL.US", "MSFT.US"])
scenario_result = modeler.model_scenarios(scenario_config)

print(f"Scenario impacts: {scenario_result.scenario_impacts}")
print(f"Probability adjustments: {scenario_result.probability_adjustments}")
```

### 5. Validate Signals

#### CLI Usage
```bash
# Statistical validation of forecasts
python -m forecasting.cli.signal_validation \
    --forecast-ids forecast_123 forecast_456 \
    --validation-types Statistical Economic \
    --benchmark buy_and_hold \
    --output results/validation_report.json

# Comprehensive backtest validation
python -m forecasting.cli.signal_validation \
    --forecast-file results/apple_forecast.json \
    --validation-types Statistical Economic Backtest \
    --transaction-costs 0.001 \
    --output results/comprehensive_validation.json
```

#### Python API
```python
from forecasting.validation.signal_validator import SignalValidator

validator = SignalValidator()
validation_result = validator.validate_forecasts(
    forecast_ids=["forecast_123", "forecast_456"],
    validation_types=["Statistical", "Economic", "Backtest"],
    benchmark_id="buy_and_hold"
)

print(f"Validation passed: {validation_result.passed_checks}")
print(f"Quality score: {validation_result.quality_score}")
print(f"Warnings: {validation_result.warnings}")
```

## Configuration Files

### Model Configuration
Create `config/forecasting_config.json`:
```json
{
  "default_model": {
    "return_forecasting": {
      "model_type": "ARIMA",
      "auto_fit": true,
      "max_p": 5,
      "max_d": 2,
      "max_q": 5
    },
    "volatility_forecasting": {
      "model_type": "GARCH",
      "distribution": "normal",
      "volatility_target": 0.15
    },
    "regime_detection": {
      "n_regimes": 3,
      "features": ["returns", "volatility"],
      "lookback_period": 252
    }
  },
  "validation": {
    "significance_level": 0.05,
    "min_forecast_horizon": 5,
    "max_forecast_horizon": 252,
    "benchmark": "buy_and_hold"
  }
}
```

### Asset Configuration
Create `config/assets.json`:
```json
{
  "universe": [
    {
      "asset_id": "AAPL.US",
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "asset_class": "Equity",
      "sector": "Technology",
      "country": "US",
      "currency": "USD"
    },
    {
      "asset_id": "SPY.US",
      "symbol": "SPY",
      "name": "SPDR S&P 500 ETF",
      "asset_class": "ETF",
      "sector": "Broad Market",
      "country": "US",
      "currency": "USD"
    }
  ]
}
```

## Data Preparation

### Required Data Format
```
data/market_data/
├── daily/
│   ├── AAPL.US.csv
│   ├── MSFT.US.csv
│   └── SPY.US.csv
└── indicators/
    ├── gdp.csv
    ├── inflation.csv
    └── unemployment.csv
```

### Sample Data Structure (AAPL.US.csv)
```csv
date,open,high,low,close,volume
2023-01-03,130.28,133.41,129.89,132.05,112117500
2023-01-04,132.81,134.74,132.65,134.76,108105800
2023-01-05,134.73,135.20,131.84,132.03,129365000
```

## Running the Full Pipeline

### Complete Workflow
```bash
# 1. Prepare data
python scripts/prepare_data.py --input data/raw/ --output data/processed/

# 2. Generate all forecasts
python scripts/generate_forecasts.py --config config/forecasting_config.json

# 3. Detect regimes
python scripts/detect_regimes.py --assets universe.txt --output results/regimes/

# 4. Model scenarios
python scripts/model_scenarios.py --scenarios config/scenarios.json

# 5. Validate signals
python scripts/validate_signals.py --results-dir results/ --output report/

# 6. Generate comprehensive report
python scripts/generate_report.py --results-dir results/ --output final_report.pdf
```

## Monitoring and Troubleshooting

### Common Issues
1. **Insufficient Data**: Ensure minimum 252 observations per asset
2. **Model Convergence**: Check for numerical stability in GARCH models
3. **Regime Stability**: Monitor regime transition probabilities
4. **Memory Usage**: Use chunked processing for large datasets

### Performance Monitoring
```bash
# Monitor memory usage
python scripts/performance_monitor.py --process forecasting

# Check model convergence
python scripts/model_diagnostics.py --forecast-file results/forecast.json

# Validate data quality
python scripts/data_quality_check.py --input data/processed/
```

## Next Steps

1. **Explore Advanced Models**: Experiment with ensemble methods and machine learning
2. **Customize Regimes**: Adjust regime detection parameters for specific assets
3. **Backtest Strategies**: Implement trading strategies based on forecasts
4. **Real-time Integration**: Set up automated forecasting pipelines
5. **Risk Management**: Implement position sizing based on volatility forecasts

## Getting Help

- Documentation: `docs/forecasting/`
- API Reference: `docs/api/`
- Troubleshooting: `docs/troubleshooting.md`
- Examples: `examples/forecasting/`

---
*Quickstart v1.0 | Created: 2025-09-20*