# Portfolio Module

This module contains portfolio optimization and risk management components.

## Structure

- **execution/**: Trade execution and order management
- **optimization/**: Portfolio optimization algorithms and utilities
- **risk/**: Risk management and monitoring tools

## Key Features

- Mean-variance optimization
- Risk constraints and position limits
- Performance attribution
- Risk metrics calculation (VaR, CVaR, max drawdown)
- Portfolio rebalancing strategies

## Usage

```python
from portfolio.src.optimization import PortfolioOptimizer
from portfolio.src.risk import RiskManager

# Optimize portfolio
optimizer = PortfolioOptimizer()
weights = optimizer.optimize(returns_data)

# Manage risk
risk_manager = RiskManager()
risk_metrics = risk_manager.calculate_portfolio_risk(weights, returns)
```

## Testing

Integration tests are located in `tests/integration/` and can be run with:

```bash
pytest tests/integration/
```