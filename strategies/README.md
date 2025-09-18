# Strategies Module

This module contains trading strategy implementations and backtesting tools.

## Structure

- **momentum/**: Momentum-based trading strategies
- **meanrev/**: Mean reversion strategies
- **arbitrage/**: Statistical arbitrage strategies

## Strategy Types

### Momentum Strategies
- Moving average crossover
- Relative strength index (RSI)
- Rate of change (ROC)
- Trend following

### Mean Reversion Strategies
- Bollinger Bands
- Pairs trading
- Statistical arbitrage
- Range trading

### Arbitrage Strategies
- Statistical arbitrage
- Pairs trading
- Market neutral strategies

## Usage

```python
from strategies.src.momentum import MomentumStrategy
from strategies.src.meanrev import MeanReversionStrategy

# Create momentum strategy
momentum = MomentumStrategy(lookback_period=20, entry_threshold=0.02)
signals = momentum.generate_signals(price_data)

# Create mean reversion strategy
mean_rev = MeanReversionStrategy(bb_period=20, bb_std=2.0)
signals = mean_rev.generate_signals(price_data)
```

## Backtesting

Strategies can be backtested using the testing framework:

```python
from tests.unit.test_mathematical_functions import MathematicalFunctions

# Calculate strategy performance
sharpe = MathematicalFunctions.calculate_sharpe_ratio(strategy_returns)
max_dd = MathematicalFunctions.calculate_max_drawdown(strategy_returns)
```