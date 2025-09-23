# Portfolio Optimization Quickstart

This quickstart guide will help you get up and running with the portfolio optimization system in under 10 minutes.

## Prerequisites

- Python 3.11+ installed
- Basic knowledge of Python and pandas
- Understanding of portfolio theory concepts

## Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd quant-portfolio-system
```

2. **Install dependencies**:
```bash
pip install -r docs/requirements.txt
```

3. **Verify installation**:
```bash
python -c "import pandas, numpy, cvxpy; print('Dependencies installed successfully')"
```

## Basic Usage

### 1. Simple Portfolio Optimization

Create a file `basic_optimization.py`:

```python
from portfolio.optimizer import PortfolioOptimizer
from portfolio.constraints import PortfolioConstraints

# Define assets for your portfolio
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Set basic constraints
constraints = PortfolioConstraints(
    max_position_size=0.20,  # 20% max per asset
    max_sector_concentration=0.40,  # 40% max per sector
    max_drawdown=0.15  # 15% max drawdown
)

# Create optimizer and run optimization
optimizer = PortfolioOptimizer()
result = optimizer.optimize_mean_variance(assets, constraints)

# Display results
print(f"Optimization successful: {result.success}")
print(f"Optimal weights: {result.optimal_weights}")
print(f"Expected Sharpe ratio: {result.performance.sharpe_ratio:.3f}")
```

Run it:
```bash
python basic_optimization.py
```

### 2. Black-Litterman with Market Views

Create `black_litterman_example.py`:

```python
from portfolio.optimizer import PortfolioOptimizer
from portfolio.constraints import PortfolioConstraints
from portfolio.views import MarketView

# Define assets and constraints
assets = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'V']
constraints = PortfolioConstraints()

# Define market views (your expectations)
views = [
    MarketView('AAPL', expected_return=0.15, confidence=0.7),
    MarketView('GOOGL', expected_return=0.12, confidence=0.5),
    MarketView('MSFT', expected_return=0.10, confidence=0.6)
]

# Run Black-Litterman optimization
optimizer = PortfolioOptimizer()
result = optimizer.optimize_black_litterman(assets, constraints, views)

print("Black-Litterman Results:")
for asset, weight in result.optimal_weights.items():
    print(f"{asset}: {weight:.2%}")
```

### 3. Risk-Managed CVaR Optimization

Create `cvar_optimization.py`:

```python
from portfolio.optimizer import PortfolioOptimizer
from portfolio.constraints import PortfolioConstraints

# Define assets with risk focus
assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'AGG']

# Set conservative constraints
constraints = PortfolioConstraints(
    max_position_size=0.25,
    max_drawdown=0.10,  # More conservative
    max_volatility=0.15
)

# Run CVaR optimization
optimizer = PortfolioOptimizer()
result = optimizer.optimize_cvar(assets, constraints)

print(f"CVaR Optimization Results:")
print(f"Expected annual return: {result.performance.annual_return:.2%}")
print(f"Expected volatility: {result.performance.annual_volatility:.2%}")
print(f"Maximum drawdown: {result.performance.max_drawdown:.2%}")
```

## Running Tests

Verify everything works by running the test suite:

```bash
# Run all tests
pytest tests/

# Run specific optimization tests
pytest tests/unit/test_optimizer.py -v

# Run performance tests
pytest tests/performance/ -v
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file:
```env
# Default data period for historical data
DEFAULT_DATA_PERIOD=5y

# Risk-free rate for calculations
RISK_FREE_RATE=0.02

# Maximum number of assets in portfolio
MAX_ASSETS=50
```

### Sample Portfolio Configuration

Create `portfolio_config.json`:
```json
{
  "assets": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
  "constraints": {
    "max_position_size": 0.20,
    "max_sector_concentration": 0.40,
    "max_drawdown": 0.15,
    "min_return": 0.08
  },
  "optimization_method": "mean_variance",
  "benchmark": "SPY"
}
```

Use it in your code:
```python
import json
from portfolio.optimizer import PortfolioOptimizer

with open('portfolio_config.json', 'r') as f:
    config = json.load(f)

optimizer = PortfolioOptimizer()
result = optimizer.optimize_from_config(config)
```

## Expected Output

When you run the basic optimization example, you should see output like:

```
Optimization successful: True
Optimal weights: {'AAPL': 0.15, 'GOOGL': 0.18, 'MSFT': 0.22, 'AMZN': 0.25, 'TSLA': 0.20}
Expected Sharpe ratio: 1.234
Expected annual return: 14.5%
Expected volatility: 11.8%
Maximum drawdown: 12.3%
```

## Common Issues

**Data Download Issues**:
- Ensure internet connection for Yahoo Finance API
- Check asset symbols are valid
- Try reducing data period (e.g., from "5y" to "2y")

**Optimization Fails**:
- Check constraints are not too restrictive
- Ensure sufficient assets (minimum 5 recommended)
- Verify asset data quality (no missing values)

**Performance Issues**:
- Reduce number of assets (<50 recommended)
- Use shorter data periods
- Check memory usage (system requirements: 4GB+ RAM)

## Next Steps

1. **Experiment**: Try different assets and constraints
2. **Backtest**: Use historical data to test performance
3. **Visualize**: Generate performance charts
4. **Compare**: Test different optimization methods

For detailed API documentation, see the `/contracts` directory.

This quickstart should have you running portfolio optimizations in minutes!