# Quick Start Guide: Quantitative Trading System

## Prerequisites

### System Requirements
- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Internet connection for data APIs

### Required Python Packages
```bash
pip install pandas numpy scipy scikit-learn
pip install yfinance quandl fredapi
pip install backtrader zipline
pip install cvxopt pypfopt
pip install matplotlib plotly dash
pip install pytest black flake8
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export YAHOO_API_KEY="your_yahoo_api_key"
export QUANDL_API_KEY="your_quandl_api_key"
export FRED_API_KEY="your_fred_api_key"
```

## Project Structure

```
quant-portfolio-system/
├── src/
│   ├── data_pipeline/         # Data ingestion and processing
│   ├── strategies/            # Trading strategy implementations
│   ├── risk_management/       # Risk calculation and monitoring
│   ├── backtesting/          # Backtesting framework
│   └── portfolio_optimization/ # Portfolio optimization algorithms
├── config/                   # Configuration files
├── data/                     # Data storage
├── logs/                     # System logs
├── tests/                    # Test files
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

## Getting Started

### 1. Data Ingestion
```bash
# Download historical data for Apple
data-ingest --source yahoo --symbol AAPL --start 2023-01-01 --end 2023-12-31

# Download multiple symbols
data-ingest --source yahoo --symbol "AAPL,GOOGL,MSFT" --start 2023-01-01 --end 2023-12-31

# Validate data quality
data-validate --input data/AAPL_2023.csv --checks completeness,accuracy,timeliness
```

### 2. Strategy Configuration
Create a strategy configuration file `config/momentum_strategy.yaml`:
```yaml
strategy:
  id: "momentum_ma_crossover"
  name: "Momentum Moving Average Crossover"
  type: "MOMENTUM"
  version: "1.0.0"

parameters:
  fast_ma: 12
  slow_ma: 26
  rsi_period: 14
  signal_threshold: 0.7

risk_management:
  max_position_size: 0.05
  stop_loss_percent: 0.02
  take_profit_percent: 0.03
  max_portfolio_risk: 0.10

data_requirements:
  min_history_days: 252
  required_fields: ["open", "high", "low", "close", "volume"]
  data_sources: ["yahoo"]
```

### 3. Backtesting
```bash
# Run backtest
strategy-backtest --config config/momentum_strategy.yaml --data data/AAPL_2023.csv --start 2023-01-01 --end 2023-12-31

# Optimize parameters
strategy-optimize --config config/momentum_strategy.yaml --data data/AAPL_2023.csv --objective sharpe_ratio
```

### 4. Risk Management
```bash
# Calculate portfolio risk
risk-calculate --portfolio data/portfolio.json --method var --confidence 0.95

# Validate risk limits
risk-validate --portfolio data/portfolio.json --limits config/risk_limits.yaml

# Run stress testing
risk-stress-test --portfolio data/portfolio.json --scenarios config/stress_scenarios.yaml
```

## Common Workflows

### Daily Data Update
```bash
# Update data for all symbols
scripts/update_data.sh

# Validate data quality
scripts/validate_data.sh

# Generate daily risk report
risk-report --portfolio data/portfolio.json --type daily --period today
```

### Strategy Development
```bash
# Create new strategy
scripts/create_strategy.sh --name new_strategy --type MOMENTUM

# Test strategy logic
pytest tests/strategies/test_new_strategy.py

# Validate configuration
strategy-validate --config config/new_strategy.yaml
```

### Portfolio Management
```bash
# Check portfolio performance
scripts/portfolio_performance.sh

# Generate trading signals
strategy-signal --config config/strategy.yaml --data data/latest.csv

# Monitor risk limits
scripts/risk_monitoring.sh
```

## Configuration Files

### Data Sources Configuration (`config/data_sources.yaml`)
```yaml
data_sources:
  yahoo:
    api_key: "${YAHOO_API_KEY}"
    rate_limit: 100
    timeout: 30
  quandl:
    api_key: "${QUANDL_API_KEY}"
    rate_limit: 50
    timeout: 60
  fred:
    api_key: "${FRED_API_KEY}"
    rate_limit: 120
    timeout: 45
```

### Risk Limits Configuration (`config/risk_limits.yaml`)
```yaml
risk_limits:
  portfolio:
    max_var_percent: 0.025
    max_drawdown_percent: 0.10
    max_leverage_ratio: 2.0
  position:
    max_position_size_percent: 0.05
    max_single_asset_percent: 0.10
    max_sector_exposure_percent: 0.20
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_pipeline.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Integration Tests
```bash
# Test data pipeline integration
pytest tests/integration/test_data_pipeline.py

# Test strategy integration
pytest tests/integration/test_strategies.py

# Test risk management integration
pytest tests/integration/test_risk_management.py
```

## Monitoring and Logging

### System Monitoring
```bash
# Start monitoring dashboard
python scripts/monitoring_dashboard.py

# Check system health
scripts/health_check.sh

# View logs
tail -f logs/system.log
```

### Performance Metrics
- Data ingestion rate
- Backtest execution time
- Risk calculation speed
- Memory usage
- API response times

## Troubleshooting

### Common Issues

1. **Data Ingestion Errors**
   ```bash
   # Check API credentials
   echo $YAHOO_API_KEY
   echo $QUANDL_API_KEY
   echo $FRED_API_KEY

   # Test API connectivity
   python scripts/test_api_connectivity.py
   ```

2. **Backtest Failures**
   ```bash
   # Validate data format
   data-validate --input data/test.csv --checks format

   # Check strategy configuration
   strategy-validate --config config/strategy.yaml
   ```

3. **Risk Calculation Errors**
   ```bash
   # Check portfolio data format
   python scripts/validate_portfolio.py

   # Verify risk limits configuration
   python scripts/validate_risk_limits.py
   ```

### Performance Issues
1. **Slow Data Processing**
   - Check data file sizes
   - Monitor memory usage
   - Optimize data storage format

2. **Backtest Performance**
   - Reduce backtest period for testing
   - Optimize strategy algorithms
   - Use faster data formats (Parquet)

3. **Risk Calculation Speed**
   - Reduce Monte Carlo scenarios
   - Optimize correlation matrix calculation
   - Use faster computational methods

## Next Steps

1. **Explore Strategies**
   - Review existing strategy implementations
   - Create custom strategies
   - Test different market conditions

2. **Enhance Risk Management**
   - Customize risk limits
   - Add new stress test scenarios
   - Implement advanced risk models

3. **Scale System**
   - Add more data sources
   - Implement distributed processing
   - Set up automated trading

4. **Production Deployment**
   - Set up monitoring and alerting
   - Implement backup and recovery
   - Configure production environments

## Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **API Reference**: `docs/api/` directory
- **Configuration**: `config/` directory
- **Community**: GitHub issues and discussions

## Support

For technical support:
1. Check the troubleshooting section
2. Review the documentation
3. Search existing issues
4. Create new issue with detailed information