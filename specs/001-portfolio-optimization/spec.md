# Spec: 001 - Portfolio Optimization Core Functionality

## Overview
Implement core portfolio optimization methods with clean, simple approaches that demonstrate understanding of quantitative finance concepts without overengineering.

## Requirements
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Optimization with ML views
- CVaR Optimization for tail risk
- Basic risk constraints (position limits, sector limits)
- Performance metrics calculation (Sharpe, Sortino, Max Drawdown)

## Implementation Plan
1. Use existing optimization libraries (CVXPY, Riskfolio-Lib)
2. Focus on 15-20 large cap US stocks
3. Simple constraint handling
4. Clear performance metrics
5. Basic benchmark comparison

## Success Criteria
- Sharpe ratio > 1.5
- Max drawdown < 15%
- Optimization completes in < 5 seconds
- Code is readable and well-documented

## Anti-Overengineering Rules
- No complex ensemble methods
- No advanced regime detection
- No real-time optimization
- Focus on daily data only
- Simple validation approach

## Files to Create
- `portfolio/optimization.py` - Core optimization methods
- `portfolio/metrics.py` - Performance calculation
- `portfolio/constraints.py` - Constraint handling
- `tests/test_optimization.py` - Basic optimization tests

## Dependencies
- cvxpy
- riskfolio-lib
- pandas
- numpy