# Market Regime Detection Module

## Overview

This module implements Hidden Markov Model (HMM) based market regime detection to enable adaptive portfolio optimization strategies. The system automatically identifies three distinct market states and switches optimization methods accordingly.

## Regime Classification

The detector identifies three market regimes:

### 1. Bull Market 🐂
- **Characteristics**: High returns, low volatility
- **Strategy**: Black-Litterman optimization
- **Rationale**: Growth-tilted approach with lower risk aversion to capture upside
- **Settings**: `risk_aversion=2.0`, `tau=0.05`

### 2. Bear Market 🐻  
- **Characteristics**: Negative returns, high volatility
- **Strategy**: CVaR (Conditional Value-at-Risk) optimization
- **Rationale**: Tail risk minimization for downside protection
- **Settings**: `alpha=0.05` (95% confidence level)

### 3. Sideways Market ↔️
- **Characteristics**: Low returns, low volatility
- **Strategy**: Risk Parity
- **Rationale**: Balanced risk allocation across assets
- **Implementation**: Inverse volatility weighting

## Features Used for Detection

The HMM model uses three engineered features from benchmark (SPY) returns:

1. **Rolling Returns** (60-day window)
   - Captures medium-term momentum
   - Primary signal for bull vs bear classification

2. **Realized Volatility** (20-day window)
   - Annualized standard deviation
   - Distinguishes high-vol from low-vol regimes

3. **VIX Proxy** (volatility-of-volatility)
   - Rolling std of realized volatility
   - Captures market stress and uncertainty

## Usage

### Basic Example

```python
from portfolio.regime.detector import RegimeDetector
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.optimizer.optimizer import SimplePortfolioOptimizer

# Load benchmark data
service = YahooFinanceService()
spy_data = service.fetch_historical_data('SPY', period='10y')
spy_returns = spy_data['Adj Close'].pct_change().dropna()

# Initialize and fit detector
detector = RegimeDetector(
    n_states=3,
    return_window=60,
    volatility_window=20,
    random_state=42
)

# Detect regimes
regimes = detector.fit_predict(spy_returns)
print(regimes.value_counts())

# Analyze regime statistics
stats = detector.get_regime_statistics(regimes, spy_returns)
print(stats)
```

### Regime-Adaptive Optimization

```python
from portfolio.backtesting.walk_forward import WalkForwardBacktester

# Run adaptive backtest
backtester = WalkForwardBacktester(
    train_period="3y",
    test_period="3mo",
    transaction_cost=0.00075
)

result = backtester.run_regime_adaptive_backtest(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start_date="2015-01-01",
    end_date="2025-01-01",
    spy_returns=spy_returns,
    detector=detector,
    weight_cap=0.20
)

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
```

### Notebook Integration

The regime detection system is demonstrated in Cell 12 of `portfolio_optimization_lab.ipynb`:

```python
# Load detector
from portfolio.regime.detector import RegimeDetector

# Fetch SPY data
spy_data = service.fetch_historical_data('SPY', period='10y')
spy_returns = spy_data['Adj Close'].pct_change().dropna()

# Fit and detect
detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
regimes = detector.fit_predict(spy_returns)

# Run adaptive backtest
result = backtester.run_regime_adaptive_backtest(
    symbols=symbols,
    spy_returns=spy_returns,
    detector=detector,
    weight_cap=0.20
)
```

## Performance Analysis

### Expected Benefits

1. **Risk Management**: CVaR protection during bear markets reduces tail losses
2. **Return Enhancement**: Growth tilt during bull markets captures upside
3. **Stability**: Risk parity during sideways markets maintains diversification

### Evaluation Metrics

- **Regime Detection Accuracy**: Validated against known market events
- **Sharpe Ratio Improvement**: Compare adaptive vs fixed strategies
- **Drawdown Reduction**: Particularly during bear regimes
- **Transition Smoothness**: Analyze regime switching frequency

## Visualization

The module provides built-in visualization tools:

```python
import matplotlib.pyplot as plt

# Create regime timeline with equity curve
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot regimes
colors = {'Bull': 'lightgreen', 'Bear': 'lightcoral', 'Sideways': 'lightyellow'}
for regime, color in colors.items():
    mask = regimes == regime
    axes[0].fill_between(
        range(len(regimes)),
        0, 1,
        where=(regimes == regime).values,
        color=color,
        alpha=0.5,
        label=regime
    )

# Plot equity curve
equity = result.equity_curve
axes[1].plot(equity.values, linewidth=2)
plt.show()
```

## Technical Details

### HMM Configuration

- **Model Type**: Gaussian HMM with full covariance
- **Number of States**: 3 (Bull/Bear/Sideways)
- **Training Iterations**: 100
- **Feature Scaling**: StandardScaler for normalization

### Regime Labeling Logic

States are labeled based on mean rolling returns:
- **Highest mean** → Bull
- **Lowest mean** → Bear  
- **Middle mean** → Sideways

### Transition Matrix

The detector computes regime transition probabilities:

```python
transitions = detector.get_regime_transition_matrix(regimes)
print(transitions)
```

Example output:
```
            Bull    Bear    Sideways
Bull        0.92    0.03    0.05
Bear        0.04    0.88    0.08
Sideways    0.06    0.07    0.87
```

High diagonal values indicate regime persistence.

## Files

- `detector.py`: Main RegimeDetector class
- `__init__.py`: Module exports
- `README.md`: This documentation
- `../examples/regime_adaptive_demo.py`: Standalone demonstration script

## Dependencies

```
hmmlearn>=0.3.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
```

## Future Enhancements

Potential improvements for advanced users:

1. **Multi-asset regime detection**: Use correlation matrix changes
2. **Macro overlays**: Incorporate economic indicators (rates, inflation)
3. **Regime forecasting**: Predict next regime transition
4. **Custom view integration**: User-defined views for Black-Litterman
5. **Dynamic parameter tuning**: Adjust windows based on regime

## References

- Nystrup, P., et al. (2020). "Multi-period portfolio selection with drawdown control"
- Kritzman, M., et al. (2012). "Regime Shifts: Implications for Dynamic Strategies"
- Ang, A., & Bekaert, G. (2002). "Regime switches in interest rates"

## Testing

Run the demonstration script:

```bash
python examples/regime_adaptive_demo.py
```

Or use pytest:

```bash
pytest tests/unit/test_regime_detector.py
```

## Support

For questions or issues, please see the main repository documentation.
