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
- **Data Preprocessing**: Custom libraries for cleaning, validation, normalization

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
- **Data Processing**: 10M data points in <30 seconds, memory usage <4GB

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
- 2025-09-18: Complete data preprocessing system with cleaning, validation, and normalization
- 2025-09-18: Comprehensive unit and performance tests for preprocessing libraries
- 2025-09-18: Performance benchmarks: 10M data points in <30 seconds, <4GB memory usage

## Repository Organization
**MAINTAIN CLEAN PROJECT STRUCTURE - NO RANDOM FILES IN ROOT!**

### Directory Structure:
```
quant-portfolio-system/
├── data/                          # Data handling ONLY
│   ├── src/                       # Source code modules
│   │   ├── feeds/                 # Data ingestion
│   │   ├── lib/                   # Preprocessing libraries (cleaning, validation, normalization)
│   │   ├── models/                # Data models and entities
│   │   ├── services/              # Data processing services
│   │   ├── config/                # Configuration management
│   │   ├── cli/                   # Command-line interfaces
│   │   └── storage/               # Data storage
│   └── storage/                   # Actual data files (created at runtime)
├── scripts/                       # Utility and demo scripts
├── examples/                      # Usage examples
├── docs/                          # Documentation ONLY
├── portfolio/                     # Portfolio optimization
├── strategies/                    # Trading strategies
├── tests/                         # Unit tests
│   ├── unit/                      # Unit tests for libraries
│   ├── statistical/               # Statistical validation tests
│   ├── performance/               # Performance and memory tests
│   └── integration/               # Integration tests
├── config/                        # Configuration files
└── output/                        # Analysis outputs
```

### Strict Rules:
1. **NEVER** create random files in project root
2. **ALWAYS** organize files in proper directories
3. **KEEP** docs/requirements.txt updated as SINGLE source of truth
4. **NO** individual pip install commands - update requirements.txt first
5. **USE** proper directory structure for all new files
6. **NEVER** create separate "enhanced", "expanded", "improved", or similar variant files - update existing files instead

## Dependency Management
1. **ALWAYS** update `docs/requirements.txt` before installing packages
2. **USE** `pip install -r docs/requirements.txt` for all installations
3. **NEVER** use individual `pip install` commands
4. **KEEP** requirements.txt organized by functional areas
5. **INCLUDE** all required dependencies (even "optional" ones that are actually needed)

## Data Preprocessing System

### Core Libraries
- **Cleaning Library** (`data/src/lib/cleaning.py`): Missing value handling, outlier detection, time gap management
- **Validation Library** (`data/src/lib/validation.py`): Data integrity checks, financial logic validation
- **Normalization Library** (`data/src/lib/normalization.py`): Z-score, Min-Max, Robust scaling with financial specialization

### Performance Requirements
- **Processing Speed**: 10 million data points in under 30 seconds
- **Memory Efficiency**: Less than 4GB memory usage for large datasets
- **Real-time Processing**: Sub-second processing for 1K data batches
- **Scalability**: Linear scaling with dataset size

### Usage Guidelines
1. **Always validate data quality** before quantitative analysis
2. **Use appropriate normalization** methods for different data types
3. **Monitor memory usage** when processing large datasets
4. **Preserve statistical relationships** during normalization
5. **Handle financial edge cases** (zero prices, negative volumes, extreme moves)

### Testing
- **Unit Tests**: Comprehensive mathematical validation in `tests/unit/`
- **Performance Tests**: Benchmarking and memory validation in `tests/performance/`
- **Statistical Tests**: Significance testing for preprocessing impact

### Configuration
- Preprocessing pipelines configured via `data/src/config/pipeline_config.py`
- Quality thresholds and validation rules customizable per dataset
- CLI interfaces for batch processing and quality reporting

---
*Based on Quantitative Trading Constitution v1.0.0 - See `.specify/memory/constitution.md`*
*Updated: 2025-09-18 | Lines: 120*