# Claude Code Guidelines: Quantitative Trading System

This file contains the runtime development guidelines for the Claude Code AI assistant when working on this quantitative trading and portfolio optimization project.

## Project Overview
You are working on a **resume-focused portfolio optimization system** that demonstrates core quantitative finance concepts without overengineering. The system prioritizes:
- **Simplicity**: Clean, readable code that showcases understanding
- **Core functionality**: Essential portfolio optimization techniques only
- **Demonstrative value**: Clear examples of quantitative finance concepts
- **Maintainability**: Easy to understand and explain in interviews

## Core Technologies (Minimal Set)
- **Language**: Python 3.11+
- **Data**: Pandas, NumPy, Yahoo Finance API
- **Optimization**: Basic CVXPY for mean-variance optimization
- **Performance**: Simple metrics calculation (Sharpe, max drawdown)
- **API**: FastAPI for basic endpoints
- **Testing**: pytest for unit and integration tests

## Mathematical Focus Areas (Resume-Ready)
1. **Portfolio Theory**: Mean-variance optimization (core concept for interviews)
2. **Risk Metrics**: Sharpe ratio, max drawdown, volatility calculation
3. **Performance Analysis**: Basic return calculation and benchmarking
4. **Data Processing**: Clean financial data handling with Pandas

## Development Principles (Simplicity First)
- **KISS (Keep It Simple, Stupid)**: No overengineering for resume projects
- **Readable > Clever**: Code should be easy to explain in interviews
- **Core functionality only**: Implement essential features, not edge cases
- **Clean imports**: Minimal dependencies, clear project structure
- **Working demo > Perfect system**: Focus on demonstrable value

## Performance Targets (Reasonable for Resume Projects)
- **Functional API**: Endpoints that work for common use cases
- **Clean code structure**: Easy to navigate and understand
- **Basic test coverage**: Unit tests for core calculations
- **Documentation**: Clear README and inline comments
- **Demonstrable results**: Can show working optimization examples

## Code Conventions (Resume-Friendly)
- **Simple file structure**: Flat hierarchy when possible
- **Clear function names**: Self-explanatory without complex abstractions
- **Minimal dependencies**: Only essential libraries
- **Type hints**: Basic type annotations for clarity
- **Error handling**: Simple try-catch blocks, not complex error systems

## When Working on Features
1. **Start simple**: Implement basic version first
2. **Add complexity only if needed**: For resume projects, basic is often enough
3. **Test core functionality**: Ensure optimization works with sample data
4. **Clean code matters**: More important than perfect algorithms
5. **Document decisions**: Explain why choices were made for interviews

## Resume Project Guidelines (Most Important)
- **This is a resume project, not an enterprise system**: Keep it simple and demonstrable
- **NO overengineering**: Complex abstractions, excessive logging, or enterprise patterns
- **Focus on interview topics**: Mean-variance optimization, basic risk metrics, clean data handling
- **Working demo is the goal**: Something you can show and explain confidently
- **Clean code > complex features**: Recruiters care more about code quality than sophisticated algorithms

## Recent Changes
- Simplified architecture: Removed overengineered components, focused on core functionality
- Clean file structure: Consolidated portfolio modules, removed duplicate files
- Basic API: Simple FastAPI endpoints for optimization and analysis
- Essential tests: Unit and integration tests for core calculations

## Repository Organization (Simplified for Resume Projects)
**KEEP IT SIMPLE - Flat structure preferred**

### Directory Structure:
```
quant-portfolio-system/
├── portfolio/                     # Core portfolio optimization
│   ├── optimizer.py               # Main optimization logic
│   ├── performance.py             # Performance calculations
│   ├── data/                      # Data handling (Yahoo Finance)
│   └── api/                       # Simple FastAPI endpoints
├── tests/                         # Tests (essential only)
│   ├── unit/                      # Core calculation tests
│   ├── integration/               # API and workflow tests
│   └── performance/               # Basic performance tests
├── examples/                      # Usage examples for demo
├── scripts/                       # Simple utility scripts
├── config/                        # Configuration files
├── docs/                          # Documentation
└── requirements.txt               # Single source of truth for dependencies
```

### Critical Rules for Resume Projects:
1. **NEVER** create files with "enhanced", "advanced", or "enterprise" in the name
2. **ALWAYS** prefer simple, flat structure over complex hierarchies
3. **KEEP** requirements.txt updated and minimal
4. **NO** unnecessary abstractions or design patterns
5. **USE** existing files instead of creating new variants
6. **FOCUS** on code that demonstrates understanding, not complexity

## Dependency Management (Simple for Resume Projects)
1. **ALWAYS** update `requirements.txt` before installing packages
2. **USE** `pip install -r requirements.txt` for all installations
3. **NEVER** use individual `pip install` commands
4. **KEEP** dependencies minimal: Only essential libraries for the project
5. **PIN** versions for reproducibility in resume demos

## Data Handling (Keep it Simple)
- **Basic cleaning**: Handle missing values, remove outliers
- **Simple validation**: Check data integrity before analysis
- **Essential normalization**: Only when necessary for optimization
- **Yahoo Finance API**: Primary data source for demo purposes

### Performance Requirements (Resume Project Realistic)
- **Functional with sample data**: Works with common stock symbols
- **Reasonable speed**: Optimizations complete in seconds, not minutes
- **Memory efficient**: Doesn't crash with typical datasets
- **Demonstrable**: Can run examples in interviews without issues

### Usage Guidelines (Practical for Resume)
1. **Focus on working examples**: Ensure demo cases work reliably
2. **Handle common edge cases**: Missing data, invalid symbols
3. **Keep data processing minimal**: Essential steps only
4. **Document data sources**: Clear explanation of data provenance
5. **Test with real symbols**: Use common stocks (AAPL, GOOGL, etc.)

### Testing (Essential Only)
- **Unit tests**: Core calculations (optimization, performance metrics)
- **Integration tests**: API endpoints work end-to-end
- **Example tests**: Ensure sample code runs correctly
- **No complex test frameworks**: Keep it simple with pytest

### Configuration (Simple Setup)
- **Single config file**: Easy to understand and modify
- **Environment variables**: For API keys and settings
- **Default values**: Work out-of-the-box for common cases
- **Documentation**: Clear setup instructions

---
*Updated for Resume Project Focus - Simpler is Better*
*Updated: 2025-09-22 | Lines: 95*
