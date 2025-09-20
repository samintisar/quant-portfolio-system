# Phase 0: Research Findings

## Research Tasks Completed

### Momentum Indicators Research
**Decision**: Implement 3 core momentum types
- **Simple Momentum**: Price change over N periods (close[t] / close[t-N] - 1)
- **Relative Strength Index (RSI)**: 14-period RSI for overbought/oversold signals
- **Rate of Change (ROC)**: Percentage change over N periods

**Rationale**: These cover the most widely used momentum indicators in quantitative finance, providing complementary signals (trend strength, mean reversion, velocity).

**Alternatives considered**:
- MACD (too complex for initial features)
- Stochastic Oscillator (similar to RSI)
- Williams %R (less commonly used)

### Return Calculation Methods Research
**Decision**: Support 3 return types
- **Arithmetic Returns**: (price[t] - price[t-1]) / price[t-1]
- **Logarithmic Returns**: ln(price[t] / price[t-1])
- **Percentage Returns**: (price[t] / price[t-1] - 1) * 100

**Rationale**: Log returns are preferred for mathematical modeling, arithmetic for intuitive understanding, percentage for reporting.

**Alternatives considered**:
- Money-weighted returns (too complex for basic features)
- Time-weighted returns (similar to arithmetic)

### Data Volume and Scale Research
**Decision**: Target 1000 instruments at daily frequency
- Processing: 1000 instruments × 252 trading days × 10 years = ~2.5M data points
- Performance: <15 seconds for full calculation, <2GB memory usage

**Rationale**: Represents realistic institutional portfolio scale while being computationally feasible.

**Alternatives considered**:
- 5000+ instruments (would require distributed processing)
- Intraday frequency (beyond initial scope)

### Rolling Window Size Research
**Decision**: Default 20-day (1 month) and 252-day (1 year) windows
- Short-term volatility: 20 days
- Long-term volatility: 252 days
- Configurable via JSON parameters

**Rationale**: Standard financial industry timeframes for volatility calculations.

**Alternatives considered**:
- Fixed 30-day (not trading calendar aware)
- Exponential weighting (more complex, phase 2 feature)

### Missing Data Handling Research
**Decision**: Multi-strategy approach
- Forward fill for short gaps (<3 days)
- Linear interpolation for medium gaps (3-10 days)
- Drop instruments with >10% missing data
- Flag outliers using 3-sigma rule

**Rationale**: Balances data preservation with statistical integrity.

**Alternatives considered**:
- Drop all missing data (too aggressive)
- Always use interpolation (can introduce bias)

### Statistical Validation Research
**Decision**: Implement 4 validation layers
- **Numerical stability**: No NaN/Inf values
- **Bounds checking**: Returns within [-100%, +1000%]
- **Time series integrity**: Monotonic timestamps
- **Cross-instrument consistency**: Same date coverage

**Rationale**: Comprehensive validation prevents garbage-in-garbage-out scenarios.

**Alternatives considered**:
- Basic NaN checking only (insufficient)
- Full statistical hypothesis testing (overkill for initial features)

## Technology Stack Decisions

### Core Libraries Confirmed
- **Pandas**: DataFrame operations and time series handling
- **NumPy**: Numerical computations and array operations
- **Yahoo Finance API**: Data source for testing
- **Scikit-learn**: Statistical validation utilities

### Performance Optimization Strategy
- Vectorized operations using Pandas/NumPy
- Memory-efficient chunked processing for large datasets
- Parallel processing across instruments where beneficial
- Lazy evaluation for feature calculations

### Configuration Management
- JSON parameter files for reproducible runs
- CLI argument parsing with defaults
- Environment variables for API keys
- Version-controlled template configurations

## Mathematical Formulations

### Returns
**Arithmetic**: r_t = (P_t - P_{t-1}) / P_{t-1}
**Logarithmic**: r_t = ln(P_t / P_{t-1})
**Percentage**: r_t = ((P_t / P_{t-1}) - 1) * 100

### Volatility
**Rolling Standard Deviation**: σ_t = √(Σ(r_{t-i} - μ_r)² / (N-1))
**Annualized**: σ_annual = σ_daily × √252

### Momentum Indicators
**Simple Momentum**: M_t = (P_t / P_{t-N}) - 1
**RSI**: RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
**ROC**: ROC_t = ((P_t - P_{t-N}) / P_{t-N}) × 100

## Research Summary

All NEEDS CLARIFICATION items have been resolved through research. The implementation will focus on:
- 3 core momentum indicators (Simple, RSI, ROC)
- 3 return calculation methods (Arithmetic, Log, Percentage)
- 1000 instrument scale with daily frequency
- 20/252-day rolling windows
- Multi-strategy missing data handling
- Comprehensive statistical validation

The approach balances simplicity with financial soundness, following the constitutional principle of "starting simple" while providing sufficient capability for quantitative research.

**Status**: Research complete, all unknowns resolved
**Next**: Proceed to Phase 1 - Design & Contracts