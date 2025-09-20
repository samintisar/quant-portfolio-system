# Portfolio Optimization Guide

This comprehensive guide covers portfolio optimization techniques, implementation, and best practices for the Quantitative Trading System.

## 🎯 What is Portfolio Optimization?

Portfolio optimization is the process of selecting the best portfolio (asset distribution) out of the set of all portfolios being considered, according to some objective. The objective typically maximizes factors such as expected return, and minimizes costs like financial risk.

### Key Concepts

- **Expected Return**: The anticipated return on investment
- **Risk**: Typically measured as volatility (standard deviation of returns)
- **Diversification**: Spreading investments to reduce risk
- **Constraints**: Investment restrictions and requirements
- **Efficient Frontier**: Set of optimal portfolios offering highest expected return for given risk

## 🧮 Mathematical Foundations

### Markowitz Mean-Variance Optimization

The classical approach to portfolio optimization:

$$\max_{w} \mu^T w - \frac{\lambda}{2} w^T \Sigma w$$

Subject to:
$$\sum_{i=1}^{n} w_i = 1$$
$$w_i \geq 0 \quad \text{(no short selling)}$$

Where:
- $w$ = portfolio weights vector
- $\mu$ = expected returns vector
- $\Sigma$ = covariance matrix
- $\lambda$ = risk aversion parameter

### Sharpe Ratio Maximization

$$\max_{w} \frac{\mu^T w - r_f}{\sqrt{w^T \Sigma w}}$$

Where $r_f$ is the risk-free rate.

### Risk Parity

$$\min_{w} \sum_{i=1}^{n} \left(w_i \sigma_i - \frac{1}{n}\right)^2$$

Where $\sigma_i$ is the volatility of asset $i$.

## 🔧 Implementation

### 1. Basic Portfolio Optimization

```python
from portfolio.src.optimization import PortfolioOptimizer
from portfolio.src.optimization import OptimizationConfig
import pandas as pd
import numpy as np

# Sample returns data
returns = pd.DataFrame({
    'AAPL': [0.01, 0.02, -0.01, 0.03, 0.01],
    'GOOGL': [0.02, 0.01, 0.02, -0.01, 0.02],
    'MSFT': [0.015, 0.015, 0.005, 0.02, 0.01]
}, index=pd.date_range('2023-01-01', periods=5))

# Initialize optimizer
optimizer = PortfolioOptimizer()

# Configure optimization
config = OptimizationConfig(
    objective="sharpe_ratio",
    risk_free_rate=0.02,
    constraints={
        "min_weight": 0.0,      # Allow zero weight
        "max_weight": 0.5,      # Maximum 50% per position
        "min_positions": 2,      # Minimum 2 positions
        "max_positions": 5       # Maximum 5 positions
    }
)

# Optimize portfolio
optimal_portfolio = optimizer.optimize_portfolio(
    returns=returns,
    config=config
)

print("Optimal Portfolio Weights:")
for symbol, weight in optimal_portfolio.weights.items():
    print(f"{symbol}: {weight:.2%}")
```

### 2. Advanced Constraints

```python
# Advanced constraint configuration
advanced_config = OptimizationConfig(
    objective="sharpe_ratio",
    risk_free_rate=0.02,
    constraints={
        # Basic constraints
        "min_weight": 0.05,      # Minimum 5% per position
        "max_weight": 0.25,     # Maximum 25% per position

        # Sector constraints
        "sector_limits": {
            "Technology": 0.4,   # Max 40% in tech
            "Healthcare": 0.3,   # Max 30% in healthcare
            "Finance": 0.3       # Max 30% in finance
        },

        # Risk constraints
        "max_volatility": 0.2,   # Max 20% portfolio volatility
        "max_drawdown": 0.15,   # Max 15% maximum drawdown

        # Position constraints
        "min_positions": 3,
        "max_positions": 8,

        # Turnover constraints
        "max_turnover": 0.3,     # Max 30% annual turnover

        # Liquidity constraints
        "min_liquidity_rank": 5  # Only invest in top 5 most liquid assets
    },
    risk_model="factor",        # Use factor risk model
    transaction_costs=0.001    # 0.1% transaction costs
)

# Optimize with advanced constraints
optimal_portfolio = optimizer.optimize_portfolio(
    returns=returns,
    config=advanced_config
)
```

### 3. Black-Litterman Model

```python
from portfolio.src.models.black_litterman import BlackLittermanModel

# Black-Litterman implementation
bl_model = BlackLittermanModel(
    equilibrium_returns=returns.mean(),
    covariance_matrix=returns.cov(),
    risk_aversion=3.0,
    views=[
        {
            "asset": "AAPL",
            "view": 0.15,        # 15% expected return
            "confidence": 0.7   # 70% confidence
        },
        {
            "asset": "GOOGL",
            "view": 0.12,        # 12% expected return
            "confidence": 0.6   # 60% confidence
        }
    ]
)

# Get Black-Litterman returns
bl_returns = bl_model.bl_returns()

# Optimize with BL returns
bl_portfolio = optimizer.optimize_portfolio(
    returns=bl_returns,
    config=config
)
```

### 4. Risk Parity Portfolio

```python
from portfolio.src.optimization import RiskParityOptimizer

# Risk parity implementation
risk_parity_optimizer = RiskParityOptimizer()

# Configure risk parity
risk_parity_config = OptimizationConfig(
    objective="risk_contribution",
    constraints={
        "min_weight": 0.0,
        "max_weight": 0.4,
        "target_risk_contribution": {
            "AAPL": 0.25,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "AMZN": 0.25
        }
    }
)

# Optimize risk parity portfolio
risk_parity_portfolio = risk_parity_optimizer.optimize_portfolio(
    returns=returns,
    config=risk_parity_config
)
```

## 📊 Performance Analysis

### 1. Performance Metrics

```python
from portfolio.src.analysis import PerformanceAnalyzer

# Initialize performance analyzer
analyzer = PerformanceAnalyzer()

# Calculate performance metrics
performance = analyzer.calculate_performance(
    returns=returns,
    weights=optimal_portfolio.weights,
    benchmark_returns=returns.mean(axis=1)  # Equal weight benchmark
)

print("Portfolio Performance Metrics:")
print(f"Annual Return: {performance['annual_return']:.2%}")
print(f"Annual Volatility: {performance['annual_volatility']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {performance['sortino_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
print(f"Calmar Ratio: {performance['calmar_ratio']:.2f}")
print(f"Information Ratio: {performance['information_ratio']:.2f}")
print(f"Tracking Error: {performance['tracking_error']:.2%}")
```

### 2. Risk Analysis

```python
from portfolio.src.risk import RiskAnalyzer

# Initialize risk analyzer
risk_analyzer = RiskAnalyzer()

# Calculate risk metrics
risk_metrics = risk_analyzer.calculate_risk_metrics(
    returns=returns,
    weights=optimal_portfolio.weights
)

print("Risk Analysis:")
print(f"Portfolio Volatility: {risk_metrics['volatility']:.2%}")
print(f"Beta: {risk_metrics['beta']:.2f}")
print(f"Alpha: {risk_metrics['alpha']:.2%}")
print(f"VaR (95%): {risk_metrics['var_95']:.2%}")
print(f"CVaR (95%): {risk_metrics['cvar_95']:.2%}")
print(f"Expected Shortfall: {risk_metrics['expected_shortfall']:.2%}")
```

### 3. Attribution Analysis

```python
from portfolio.src.analysis import AttributionAnalyzer

# Initialize attribution analyzer
attribution_analyzer = AttributionAnalyzer()

# Calculate performance attribution
attribution = attribution_analyzer.calculate_attribution(
    returns=returns,
    weights=optimal_portfolio.weights,
    factor_returns=returns  # Simplified factor returns
)

print("Performance Attribution:")
print(f"Asset Selection: {attribution['asset_selection']:.2%}")
print(f"Factor Allocation: {attribution['factor_allocation']:.2%}")
print(f"Interaction Effect: {attribution['interaction']:.2%}")
print(f"Total Active Return: {attribution['active_return']:.2%}")
```

## 🎛️ Advanced Optimization Techniques

### 1. Multi-Objective Optimization

```python
from portfolio.src.optimization import MultiObjectiveOptimizer

# Multi-objective optimization
multi_optimizer = MultiObjectiveOptimizer()

# Configure multiple objectives
multi_config = OptimizationConfig(
    objective="multi_objective",
    objectives=[
        {"name": "return", "weight": 0.4},
        {"name": "sharpe_ratio", "weight": 0.3},
        {"name": "risk_adjusted_return", "weight": 0.3}
    ],
    constraints={
        "min_weight": 0.0,
        "max_weight": 0.3,
        "max_volatility": 0.18
    }
)

# Optimize with multiple objectives
multi_obj_portfolio = multi_optimizer.optimize_portfolio(
    returns=returns,
    config=multi_config
)
```

### 2. Robust Optimization

```python
from portfolio.src.optimization import RobustOptimizer

# Robust optimization considering parameter uncertainty
robust_optimizer = RobustOptimizer()

# Configure robust optimization
robust_config = OptimizationConfig(
    objective="robust_sharpe_ratio",
    uncertainty_set="ellipsoidal",  # Type of uncertainty set
    uncertainty_level=0.1,        # Uncertainty level (10%)
    constraints={
        "min_weight": 0.05,
        "max_weight": 0.25,
        "worst_case_return": 0.05   # Minimum worst-case return
    }
)

# Optimize robust portfolio
robust_portfolio = robust_optimizer.optimize_portfolio(
    returns=returns,
    config=robust_config
)
```

### 3. Transaction Cost Optimization

```python
from portfolio.src.optimization import TransactionCostOptimizer

# Optimization with transaction costs
tc_optimizer = TransactionCostOptimizer()

# Configure transaction cost optimization
tc_config = OptimizationConfig(
    objective="net_return_after_costs",
    transaction_costs={
        "fixed_cost": 0.001,     # 0.1% fixed cost
        "proportional_cost": 0.0005,  # 0.05% proportional cost
        "market_impact": 0.001,  # 0.1% market impact
        "bid_ask_spread": 0.0002 # 0.02% bid-ask spread
    },
    constraints={
        "min_weight": 0.05,
        "max_weight": 0.25,
        "max_turnover": 0.2       # Max 20% turnover
    }
)

# Optimize with transaction costs
tc_portfolio = tc_optimizer.optimize_portfolio(
    returns=returns,
    config=tc_config,
    current_weights=current_weights  # Current portfolio weights
)
```

## 📈 Backtesting and Validation

### 1. Walk-Forward Optimization

```python
from portfolio.src.backtesting import WalkForwardOptimizer

# Walk-forward optimization
wf_optimizer = WalkForwardOptimizer()

# Configure walk-forward
wf_config = {
    "training_period": 252,      # 1 year training
    "testing_period": 63,       # 3 months testing
    "rebalancing_frequency": 63, # Quarterly rebalancing
    "optimization_config": config
}

# Run walk-forward optimization
wf_results = wf_optimizer.walk_forward_optimization(
    returns=returns,
    config=wf_config
)

print("Walk-Forward Results:")
print(f"Average Annual Return: {wf_results['avg_annual_return']:.2%}")
print(f"Average Volatility: {wf_results['avg_volatility']:.2%}")
print(f"Average Sharpe Ratio: {wf_results['avg_sharpe_ratio']:.2f}")
```

### 2. Out-of-Sample Testing

```python
from portfolio.src.validation import OutOfSampleValidator

# Out-of-sample validation
validator = OutOfSampleValidator()

# Split data into in-sample and out-of-sample
split_date = returns.index[int(len(returns) * 0.7)]
in_sample_returns = returns[returns.index <= split_date]
out_sample_returns = returns[returns.index > split_date]

# Optimize on in-sample data
in_sample_portfolio = optimizer.optimize_portfolio(
    returns=in_sample_returns,
    config=config
)

# Validate on out-of-sample data
validation_results = validator.validate_portfolio(
    portfolio=in_sample_portfolio,
    returns=out_sample_returns
)

print("Out-of-Sample Validation:")
print(f"Out-of-Sample Return: {validation_results['return']:.2%}")
print(f"Out-of-Sample Volatility: {validation_results['volatility']:.2%}")
print(f"Out-of-Sample Sharpe Ratio: {validation_results['sharpe_ratio']:.2f}")
```

### 3. Stress Testing

```python
from portfolio.src.stress_testing import StressTester

# Stress testing
stress_tester = StressTester()

# Define stress scenarios
scenarios = {
    "market_crash": {"shock": -0.3, "duration": 21},      # -30% over 21 days
    "volatility_spike": {"volatility_multiplier": 2.0, "duration": 10},
    "liquidity_crisis": {"liquidity_shock": -0.5, "duration": 5}
}

# Run stress tests
stress_results = stress_tester.run_stress_tests(
    portfolio=optimal_portfolio,
    returns=returns,
    scenarios=scenarios
)

print("Stress Test Results:")
for scenario, result in stress_results.items():
    print(f"{scenario}:")
    print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"  Final Return: {result['final_return']:.2%}")
    print(f"  Recovery Time: {result['recovery_time']} days")
```

## 🎯 Best Practices

### 1. Data Quality

```python
from data.src.lib.validation import DataValidator
from data.src.lib.cleaning import DataCleaner

# Validate input data
validator = DataValidator()
cleaner = DataCleaner()

# Check data quality
quality_report = validator.validate_return_data(returns)

# Clean data if necessary
if quality_report['overall_score'] < 0.9:
    returns = cleaner.clean_returns(returns)
```

### 2. Robustness Testing

```python
from portfolio.src.validation import RobustnessValidator

# Test optimization robustness
robustness_validator = RobustnessValidator()

# Test sensitivity to input parameters
sensitivity_results = robustness_validator.test_sensitivity(
    optimizer=optimizer,
    returns=returns,
    config=config,
    parameters=['risk_free_rate', 'max_weight'],
    variation=0.1  # 10% variation
)

print("Sensitivity Analysis:")
for param, sensitivity in sensitivity_results.items():
    print(f"{param}: {sensitivity:.3f}")
```

### 3. Performance Monitoring

```python
from portfolio.src.monitoring import PortfolioMonitor

# Set up portfolio monitoring
monitor = PortfolioMonitor()

# Monitor portfolio performance
monitor.add_portfolio(
    name="Optimal Portfolio",
    weights=optimal_portfolio.weights,
    returns=returns
)

# Generate monitoring report
monitoring_report = monitor.generate_report(
    period="daily",
    metrics=["return", "volatility", "sharpe_ratio", "drawdown"]
)

print("Portfolio Monitoring Report:")
print(monitoring_report)
```

## 🚀 Real-World Example

### Complete Portfolio Optimization Workflow

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Data Preparation
def prepare_portfolio_data(symbols, start_date, end_date):
    """Prepare returns data for portfolio optimization"""

    # Download data (simplified example)
    from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion

    ingestion = YahooFinanceIngestion()
    raw_data = ingestion.download_data(symbols, start_date, end_date)

    # Calculate daily returns
    returns = raw_data['close'].pct_change().dropna()

    return returns

# 2. Optimization Configuration
def create_optimization_config(risk_profile="moderate"):
    """Create optimization configuration based on risk profile"""

    configs = {
        "conservative": {
            "objective": "min_risk",
            "constraints": {
                "min_weight": 0.02,
                "max_weight": 0.20,
                "max_volatility": 0.12,
                "min_positions": 5,
                "max_positions": 10
            }
        },
        "moderate": {
            "objective": "sharpe_ratio",
            "constraints": {
                "min_weight": 0.02,
                "max_weight": 0.25,
                "max_volatility": 0.18,
                "min_positions": 4,
                "max_positions": 8
            }
        },
        "aggressive": {
            "objective": "max_return",
            "constraints": {
                "min_weight": 0.01,
                "max_weight": 0.30,
                "max_volatility": 0.25,
                "min_positions": 3,
                "max_positions": 6
            }
        }
    }

    return OptimizationConfig(**configs[risk_profile])

# 3. Complete Optimization Workflow
def optimize_portfolio_workflow(symbols, start_date, end_date, risk_profile="moderate"):
    """Complete portfolio optimization workflow"""

    print(f"Starting portfolio optimization for {len(symbols)} symbols...")

    # Step 1: Prepare data
    print("Step 1: Preparing data...")
    returns = prepare_portfolio_data(symbols, start_date, end_date)

    # Step 2: Validate data
    print("Step 2: Validating data...")
    validator = DataValidator()
    quality_report = validator.validate_return_data(returns)

    if quality_report['overall_score'] < 0.8:
        print(f"Warning: Data quality score {quality_report['overall_score']:.2f} is low")

    # Step 3: Configure optimization
    print("Step 3: Configuring optimization...")
    config = create_optimization_config(risk_profile)

    # Step 4: Optimize portfolio
    print("Step 4: Optimizing portfolio...")
    optimizer = PortfolioOptimizer()
    optimal_portfolio = optimizer.optimize_portfolio(returns, config)

    # Step 5: Analyze results
    print("Step 5: Analyzing results...")
    analyzer = PerformanceAnalyzer()
    performance = analyzer.calculate_performance(
        returns=returns,
        weights=optimal_portfolio.weights
    )

    # Step 6: Generate report
    print("Step 6: Generating report...")
    report = generate_optimization_report(
        portfolio=optimal_portfolio,
        performance=performance,
        config=config
    )

    return optimal_portfolio, performance, report

# 4. Report Generation
def generate_optimization_report(portfolio, performance, config):
    """Generate comprehensive optimization report"""

    report = f"""
Portfolio Optimization Report
=============================

Optimization Configuration:
- Objective: {config.objective}
- Risk Profile: {config.constraints.get('risk_profile', 'moderate')}
- Number of Assets: {len(portfolio.weights)}
- Minimum Position: {config.constraints.get('min_weight', 0):.1%}
- Maximum Position: {config.constraints.get('max_weight', 1):.1%}

Optimal Portfolio Weights:
"""

    for symbol, weight in sorted(portfolio.weights.items(), key=lambda x: x[1], reverse=True):
        report += f"- {symbol}: {weight:.2%}\n"

    report += f"""
Performance Metrics:
- Annual Return: {performance['annual_return']:.2%}
- Annual Volatility: {performance['annual_volatility']:.2%}
- Sharpe Ratio: {performance['sharpe_ratio']:.2f}
- Sortino Ratio: {performance['sortino_ratio']:.2f}
- Max Drawdown: {performance['max_drawdown']:.2%}
- Calmar Ratio: {performance['calmar_ratio']:.2f}

Risk Metrics:
- Beta: {performance.get('beta', 'N/A')}
- Alpha: {performance.get('alpha', 'N/A')}
- Information Ratio: {performance.get('information_ratio', 'N/A')}
- Tracking Error: {performance.get('tracking_error', 'N/A')}

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return report

# Example Usage
if __name__ == "__main__":
    # Define portfolio universe
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'PYPL'
    ]

    # Define time period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years

    # Run optimization
    portfolio, performance, report = optimize_portfolio_workflow(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        risk_profile="moderate"
    )

    # Print report
    print(report)
```

## 🆘 Troubleshooting

### Common Issues

#### Optimization Fails to Converge
```python
# Try different solvers
config.solver = 'SCS'  # or 'ECOS', 'OSQP'

# Relax constraints
config.constraints['max_weight'] = 0.4
config.constraints['min_weight'] = 0.01

# Check data quality
if returns.isnull().any().any():
    returns = returns.fillna(returns.mean())
```

#### Unrealistic Portfolio Weights
```python
# Add realistic constraints
config.constraints.update({
    'max_weight': 0.2,      # Maximum 20% per position
    'min_weight': 0.02,     # Minimum 2% per position
    'max_positions': 10,    # Maximum 10 positions
    'min_positions': 5     # Minimum 5 positions
})
```

#### Poor Out-of-Sample Performance
```python
# Use robust optimization
config.objective = 'robust_sharpe_ratio'
config.uncertainty_level = 0.1

# Add transaction costs
config.transaction_costs = 0.001
config.constraints['max_turnover'] = 0.2

# Use walk-forward optimization
wf_optimizer = WalkForwardOptimizer()
wf_results = wf_optimizer.walk_forward_optimization(returns, config)
```

## 📚 Additional Resources

### Documentation
- [API Reference](../api/README.md)
- [Performance Analysis](../reference/performance_metrics.md)
- [Risk Management](../guides/risk_management.md)
- [Mathematical Formulations](../reference/mathematics.md)

### Research Papers
- Markowitz, H. (1952). "Portfolio Selection"
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
- Michaud, R. (1998). "Efficient Asset Management"

### Tools and Libraries
- [CVXPY](https://www.cvxpy.org/) - Convex optimization
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) - Portfolio optimization
- [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/) - Portfolio optimization with constraints

---

*Last Updated: 2024-01-15*