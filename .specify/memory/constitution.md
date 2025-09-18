# Quantitative Trading & Portfolio Optimization System Constitution

## Core Principles

### I. Library-First Architecture
Every quantitative trading component must be implemented as a standalone library. Libraries must be self-contained, independently testable, and thoroughly documented. Each library must serve a specific purpose in the trading workflow - no organizational-only libraries allowed.

### II. CLI Interface
Every trading library exposes functionality via CLI. Text I/O protocol: stdin/args → stdout, errors → stderr. Support both JSON (for machine consumption) and human-readable formats (for analysis). Commands must be idempotent and support dry-run mode for safety.

### III. Test-First (NON-NEGOTIABLE)
TDD mandatory for all trading algorithms: Tests written → User approved → Tests fail → Then implement. Red-Green-Refactor cycle strictly enforced. All trading strategies must have backtest validation before live deployment.

### IV. Integration Testing
Focus areas requiring integration tests: Trading model contract tests, Data feed contract changes, Portfolio optimization integrations, Risk management module interactions, Market regime detection system.

### V. Observability & Monitoring
All trading operations must emit structured logs with timestamps, strategy parameters, and market conditions. Real-time monitoring of portfolio metrics, risk limits, and system health. Performance tracking with latency benchmarks for order execution.

### VI. Versioning & Risk Management
MAJOR.MINOR.BUILD format with strict risk governance. Breaking changes require full regression testing and risk committee approval. All strategy versions must maintain consistent risk profiles and performance characteristics.

### VII. Simplicity in Trading Systems
Start with simple, well-understood trading models before adding complexity. No future-proofing for hypothetical market conditions. Each additional feature must demonstrate clear alpha generation or risk reduction.

## Trading System Requirements

### Data Quality & Processing
- All market data must be validated, normalized, and timestamped
- Handle missing data, outliers, and market anomalies appropriately
- Maintain data lineage and audit trails for compliance

### Risk Management
- Position limits enforced at all times
- Real-time VaR and CVaR calculations
- Automatic position liquidation if risk thresholds breached
- Stress testing against historical and synthetic market scenarios

### Performance Standards
- Backtesting execution speed: <1ms per data point
- Live trading latency: <10ms for order execution
- Maximum drawdown protection: 15% from peak
- Sharpe ratio target: >1.0 after costs

## Development Workflow

### Strategy Development Process
1. Research phase with historical analysis
2. Hypothesis formulation with statistical validation
3. Contract-first implementation with failing tests
4. Strategy implementation and backtesting
5. Paper trading validation
6. Live deployment with risk limits

### Code Review Requirements
- All trading algorithms require review by quantitative analysts
- Risk models must be validated by risk management team
- Performance characteristics must be documented and approved
- Compliance checks for regulatory requirements

### Quality Gates
- Maximum 3 libraries per strategy (data, strategy, execution)
- No wrapping of trading framework features - use directly
- All strategies must include comprehensive error handling
- Automated testing must cover 95%+ of code paths

## Governance

The Constitution supersedes all other development practices. Amendments require documentation, quantitative validation, and migration plan.

All trading strategy development must verify compliance with risk limits and performance targets. Any complexity beyond simple trading models must be justified with backtest results and risk analysis. Use CLAUDE.md for runtime development guidance specific to quantitative trading.

**Version**: 2.1.1 | **Ratified**: 2025-07-16 | **Last Amended**: 2025-07-16