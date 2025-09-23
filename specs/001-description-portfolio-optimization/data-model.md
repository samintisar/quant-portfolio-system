# Data Model: Portfolio Optimization

## Core Entities

### Asset
**Description**: Individual financial instrument in the portfolio
**Fields**:
- `symbol` (string): Ticker symbol (e.g., "AAPL", "GOOGL")
- `name` (string): Full company name
- `sector` (string): Industry sector classification
- `prices` (DataFrame): Historical price data with datetime index
- `returns` (Series): Daily/weekly/monthly returns
- `volatility` (float): Annualized volatility
- `market_cap` (float): Market capitalization (for weighting)

**Validation Rules**:
- Symbol must be valid Yahoo Finance ticker
- Minimum 1 year of historical data required
- No missing values in price data (handled in preprocessing)

### Portfolio
**Description**: Collection of assets with optimized weights
**Fields**:
- `name` (string): Portfolio identifier
- `assets` (List[Asset]): Assets in the portfolio
- `weights` (Dict[str, float]): Asset symbol -> weight allocation
- `optimization_method` (string): "mean_variance", "black_litterman", "cvar"
- `constraints` (PortfolioConstraints): Applied risk constraints
- `performance` (PortfolioPerformance): Calculated performance metrics
- `created_date` (datetime): When portfolio was optimized
- `data_period` (Dict): Start/end dates of historical data used

**Validation Rules**:
- Weights must sum to 1.0 (100%)
- No negative weights (long-only portfolio)
- All constraints must be satisfied

### PortfolioConstraints
**Description**: Risk and allocation constraints for optimization
**Fields**:
- `max_position_size` (float): Maximum weight per asset (default: 0.05)
- `max_sector_concentration` (float): Maximum sector weight (default: 0.20)
- `max_drawdown` (float): Maximum acceptable drawdown (default: 0.15)
- `min_return` (float): Minimum required return (default: 0.0)
- `max_volatility` (float): Maximum acceptable volatility (default: 0.25)
- `risk_free_rate` (float): Risk-free rate for calculations (default: 0.02)

**Validation Rules**:
- All values between 0 and 1 (as percentages)
- Constraints must be mathematically feasible

### PortfolioPerformance
**Description**: Calculated performance metrics for a portfolio
**Fields**:
- `sharpe_ratio` (float): Risk-adjusted return measure
- `max_drawdown` (float): Maximum loss from peak
- `annual_return` (float): Annualized return
- `annual_volatility` (float): Annualized volatility
- `benchmark_return` (float): Benchmark (S&P 500) return
- `alpha` (float): Excess return over benchmark
- `beta` (float): Portfolio sensitivity to benchmark
- `information_ratio` (float): Risk-adjusted excess return

**Validation Rules**:
- All metrics calculated from actual portfolio returns
- Benchmark comparison uses same time period

### MarketView
**Description**: Investor view for Black-Litterman model
**Fields**:
- `asset_symbol` (string): Asset the view applies to
- `expected_return` (float): Investor's expected return
- `confidence` (float): Confidence level (0-1)
- `view_type` (string): "absolute" or "relative"
- `benchmark_asset` (string): For relative views (optional)

**Validation Rules**:
- Confidence must be between 0 and 1
- Expected return must be reasonable (e.g., -50% to +100% annually)

### OptimizationResult
**Description**: Complete optimization output
**Fields**:
- `success` (bool): Whether optimization succeeded
- `optimal_weights` (Dict[str, float]): Final asset weights
- `objective_value` (float): Optimization objective value
- `iterations` (int): Number of optimization iterations
- `execution_time` (float): Time taken to optimize
- `warning_messages` (List[str]): Any optimization warnings
- `error_messages` (List[str]): Any optimization errors

**Validation Rules**:
- Weights must satisfy all constraints if successful
- Error messages must explain failures clearly

## Data Relationships

```
Portfolio contains [1..*] Assets
Portfolio has 1 PortfolioConstraints
Portfolio has 1 PortfolioPerformance
Portfolio uses [0..*] MarketViews (for Black-Litterman)
Portfolio produces 1 OptimizationResult
```

## State Transitions

### Portfolio Optimization Process
1. **Data Input** → **Validation**: Check asset data quality
2. **Validation** → **Optimization**: Run selected optimization method
3. **Optimization** → **Performance Calculation**: Compute metrics
4. **Performance Calculation** → **Output**: Generate results

### Error States
- **Insufficient Data**: Less than 1 year of historical data
- **Infeasible Constraints**: Constraints that cannot be satisfied
- **Optimization Failure**: Numerical issues in optimization
- **Invalid Views**: Contradictory market views in Black-Litterman

## Data Formats

### Input Data
- Asset prices: CSV with columns (Date, Open, High, Low, Close, Volume)
- Portfolio config: JSON with assets and constraints
- Market views: JSON list of view objects

### Output Data
- Portfolio allocation: JSON with weights and metrics
- Performance report: JSON with all calculated metrics
- Backtest results: JSON with time series of portfolio value
- Visualization data: JSON for plotting

This data model provides a clean, simple structure for implementing portfolio optimization while maintaining the flexibility needed for different optimization methods and constraints.