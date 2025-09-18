# Claude Code Guidelines: Quantitative Trading System

This file contains the runtime development guidelines for the Claude Code AI assistant when working on this quantitative trading and portfolio optimization project.

## Project Overview
You are working on a quantitative trading and portfolio optimization system that researches, tests, and implements systematic investment strategies. The system combines:
- **Data ingestion** from financial markets (Yahoo Finance API)
- **Predictive modeling** using statistical and ML techniques
- **Portfolio optimization** under risk and regulatory constraints
- **Backtesting** for performance validation
- **Risk management** with comprehensive monitoring

## Core Technologies
- **Language**: Python 3.11+
- **Data**: Pandas, NumPy, Yahoo Finance API
- **ML/Stats**: Scikit-learn, PyTorch, PyMC, statsmodels
- **Optimization**: CVXPY, Riskfolio-Lib
- **Backtesting**: Vectorbt
- **Visualization**: Matplotlib, Plotly, Streamlit

## Mathematical Focus Areas
1. **Time Series Analysis**: ARIMA, GARCH models for returns/volatility
2. **Portfolio Theory**: Mean-variance optimization, Black-Litterman
3. **Risk Management**: VaR, CVaR, Monte Carlo simulations
4. **Statistical Learning**: Factor models, regime detection
5. **Constraint Optimization**: Portfolio construction with regulatory limits

## Development Principles
- **Statistical Rigor**: All models must be statistically validated
- **Reproducibility**: Seed control and version-controlled configurations
- **Financial Soundness**: Validate against established benchmarks
- **Risk-First**: Risk constraints are non-negotiable
- **Test-Driven**: Mathematical correctness verified through testing

## Performance Targets
- **Sharpe Ratio**: > 1.5 for optimized portfolios
- **Max Drawdown**: < 15% under normal conditions
- **Benchmark Outperformance**: > 200 bps annually vs S&P 500
- **Concentration Limits**: < 5% single name, < 20% sector

## Code Conventions
- Use type hints for all financial data structures
- Document mathematical formulations in docstrings
- Include statistical significance tests for model outputs
- Implement logging for all portfolio decisions
- Follow PEP 8 with finance-specific naming (e.g., `sharpe_ratio`, `max_drawdown`)

## When Working on Features
1. **Always validate** mathematical correctness first
2. **Implement backtesting** for any trading strategy
3. **Calculate risk metrics** for all portfolio changes
4. **Benchmark against** simple buy-and-hold strategies
5. **Document assumptions** clearly in code comments

## Constitutional Compliance
This project follows the Quantitative Trading System Constitution (v1.0.0):
- Library-first architecture with mathematical focus
- CLI interfaces for reproducible research
- Test-first development with statistical validation
- Rigorous data quality and validation
- Comprehensive risk management and observability
- Strict versioning for model governance
- Simplicity with financial soundness

## Recent Changes
<!-- Auto-updated by scripts - keep last 3 entries -->
- 2025-01-17: Initial constitution and templates created
- 2025-01-17: Project structure established for quantitative trading
- 2025-01-17: Mathematical validation frameworks defined

---
*Based on Quantitative Trading Constitution v1.0.0 - See `.specify/memory/constitution.md`*
*Updated: 2025-01-17 | Lines: 65*