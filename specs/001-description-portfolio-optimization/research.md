# Research Portfolio Optimization Implementation

## Research Decisions

### Output Format (FR-007)
**Decision**: JSON format with CSV export option
**Rationale**: JSON is programmatic and easy to work with in Python, CSV for Excel compatibility
**Alternatives considered**: Excel only (too complex), CSV only (limited structure)

### Time Periods (FR-008)
**Decision**: Daily data with weekly/monthly aggregation options
**Rationale**: Daily provides sufficient granularity for portfolio optimization, aggregation for different analysis needs
**Alternatives considered**: Weekly only (loses detail), Intraday (overkill for this scope)

### Optimization Methods
**Decision**: Focus on 3 core methods for resume project:
1. Mean-Variance Optimization (Modern Portfolio Theory)
2. Black-Litterman Model (with simple market views)
3. CVaR Optimization (Conditional Value at Risk)

**Rationale**: These 3 methods demonstrate quantitative finance knowledge without overengineering
**Alternatives considered**: More complex methods would overcomplicate the resume project

### Risk Constraints
**Decision**: Simple, measurable constraints:
- Maximum position size (5% per asset)
- Maximum sector concentration (20% per sector)
- Maximum drawdown limit (15%)
- Basic volatility constraint

**Rationale**: Demonstrates risk management understanding without complex calculations
**Alternatives considered**: Complex VaR calculations, regime detection (overengineering)

### Performance Metrics
**Decision**: Core metrics that matter for interviews:
- Sharpe Ratio (primary)
- Maximum Drawdown
- Annual Returns
- Benchmark Comparison (S&P 500)
- Basic volatility metrics

**Rationale**: These are the key metrics interviewers look for
**Alternatives considered**: Complex metrics (overengineering for resume project)

### Data Sources
**Decision**: Yahoo Finance API via yfinance library
**Rationale**: Free, reliable, and sufficient for demonstration
**Alternatives considered**: Paid APIs (unnecessary cost), Web scraping (unreliable)

### Libraries
**Decision**: Use proven, resume-friendly libraries:
- Pandas/NumPy (essential)
- CVXPY (optimization)
- Riskfolio-Lib (specialized for portfolio optimization)
- yfinance (data)
- Matplotlib/Plotly (visualization)
- pytest (testing)

**Rationale**: These libraries are well-known in the industry and look good on resume
**Alternatives considered**: Building custom implementations (overengineering)

## Simplicity Commitment

This project intentionally avoids:
- Complex ensemble methods
- Advanced machine learning
- Real-time data processing
- Complex deployment pipelines
- Over-engineered testing frameworks

Focus is on clean, working code that demonstrates quantitative finance fundamentals.

## Research Complete

All NEEDS CLARIFICATION items from the specification have been resolved with simple, practical decisions suitable for a resume project.