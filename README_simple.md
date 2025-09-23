# Simple Portfolio Optimization System

A clean, simplified portfolio optimization system that removes overengineering and focuses on core functionality.

## Overview

This system provides basic portfolio optimization capabilities without the complex abstractions, factory patterns, and overengineered architecture found in many "resume-driven" projects. It's designed to be simple, maintainable, and focused on solving the actual problem.

## Key Features

- **Simple portfolio optimization** using mean-variance approach
- **Essential performance metrics** (Sharpe ratio, return, volatility, drawdown)
- **Basic API endpoints** for optimization and analysis
- **Clean configuration** using YAML files
- **Standard Python logging** without custom frameworks
- **Focused test suite** covering core functionality

## Files Structure

- `portfolio_simple.py` - Core portfolio optimization functionality
- `performance_simple.py` - Performance metrics calculation
- `config_simple.yaml` - Configuration file
- `config_loader.py` - Configuration loading utilities
- `api_simple.py` - Simple FastAPI endpoints
- `tests_simple.py` - Focused test suite
- `example_simple.py` - Usage examples

## Quick Start

### Basic Usage

```python
from portfolio_simple import SimplePortfolioOptimizer

# Initialize optimizer
optimizer = SimplePortfolioOptimizer()

# Optimize portfolio
result = optimizer.optimize_portfolio(['SPY', 'AAPL', 'GOOGL'])

# View results
print("Optimal weights:")
for asset, weight in result['optimization']['weights'].items():
    print(f"  {asset}: {weight:.2%}")
```

### Performance Analysis

```python
from performance_simple import SimplePerformanceCalculator

# Calculate performance metrics
calc = SimplePerformanceCalculator()
metrics = calc.calculate_metrics(portfolio_returns)

print(f"Annual Return: {metrics['annual_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### API Usage

Start the API server:

```bash
python api_simple.py
```

Optimize a portfolio:

```bash
curl -X POST "http://localhost:8000/optimize" \
     -H "Content-Type: application/json" \
     -d '{"symbols": ["SPY", "AAPL", "GOOGL"]}'
```

## Configuration

Edit `config_simple.yaml` to adjust system parameters:

```yaml
portfolio:
  risk_free_rate: 0.02
  max_position_size: 0.05
  trading_days_per_year: 252
  default_period: "5y"
```

## Running Tests

```bash
python tests_simple.py
```

## What Was Removed

The original system had significant overengineering that has been removed:

### Removed Complexity
- **Abstract base classes** and factory patterns
- **20+ Pydantic models** for simple requests
- **Complex constraint validation systems**
- **Overengineered logging framework**
- **50+ performance metrics** when 8-10 suffice
- **Complex API middleware** and decorators
- **Multiple optimization methods** with similar functionality
- **Enterprise-grade error handling**

### Simplified To
- **Single optimizer class** with core methods
- **Basic request/response models**
- **Simple validation** where needed
- **Standard Python logging**
- **Essential metrics** only
- **Clean API endpoints**
- **Mean-variance optimization** as primary method
- **Standard exception handling**

## Core Functionality

### Portfolio Optimization
- Mean-variance optimization
- Efficient frontier calculation
- Basic constraint handling
- Performance metrics calculation

### Performance Metrics
- Total return
- Annualized return/volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Beta/alpha (with benchmark)

### API Endpoints
- `POST /optimize` - Optimize portfolio
- `POST /analyze` - Analyze existing portfolio
- `GET /assets/{symbol}` - Get asset data
- `GET /health` - Health check

## Dependencies

- pandas
- numpy
- yfinance
- fastapi (for API)
- uvicorn (for API server)
- pyyaml (for configuration)

## Running the Example

```bash
python example_simple.py
```

This will demonstrate:
1. Basic portfolio optimization
2. Efficient frontier calculation
3. Custom portfolio analysis

## Design Philosophy

This system follows the principle that **simple solutions are better than complex ones**. It prioritizes:

- **Clarity** over cleverness
- **Functionality** over framework
- **Maintainability** over complexity
- **Real-world usage** over theoretical perfection

The result is a system that's easier to understand, modify, and extend while still providing all the essential portfolio optimization functionality.