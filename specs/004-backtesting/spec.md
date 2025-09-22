# Spec: 004 - Backtesting Engine

## Overview
Implement a simple backtesting engine that validates portfolio strategies against historical data with basic benchmark comparison.

## Requirements
- Walk-forward backtesting
- Rolling window validation
- Basic benchmark comparison (equal-weight, market-cap)
- Performance attribution
- Risk metrics calculation

## Implementation Plan
1. Simple rolling window approach
2. Basic portfolio rebalancing logic
3. Benchmark portfolio creation
4. Performance metrics calculation
5. Simple visualization of results

## Success Criteria
- Backtests 3-5 years of data in < 60 seconds
- Clear performance vs benchmarks
- Shows value of ML enhancement
- Provides risk metrics

## Anti-Overengineering Rules
- No complex transaction cost modeling
- No slippage simulation
- No market impact modeling
- No advanced benchmark strategies
- No complex performance attribution

## Files to Create
- `backtesting/engine.py` - Backtesting core logic
- `backtesting/benchmarks.py` - Benchmark portfolios
- `backtesting/metrics.py` - Performance metrics
- `tests/test_backtesting.py` - Backtesting tests

## Dependencies
- pandas
- numpy
- vectorbt
- matplotlib