# Quantitative Trading System

A systematic investment strategy research and implementation platform combining predictive modeling, portfolio optimization, and risk management.

## 🚀 Quick Start

### Setup Environment
```bash
# Setup the data environment
python scripts/setup_data_environment.py

# Install dependencies
pip install -r docs/requirements.txt

# Run basic usage example
python examples/basic_usage.py
```

### Basic Data Ingestion
```python
from data.src.feeds import create_default_ingestion_system, AssetClass
from datetime import datetime, timedelta

ingestion = create_default_ingestion_system()
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch equity data
results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)

# Save to storage
ingestion.save_results_to_storage(results)
```

## 📊 Features

### Data Ingestion & Storage
- **Historical Data**: Yahoo Finance API integration
- **Multi-Asset Classes**: Equities, ETFs, FX, Bonds, Commodities
- **Persistent Storage**: Multiple formats (Parquet, CSV, HDF5)
- **Data Validation**: Comprehensive quality checks
- **Batch Processing**: Concurrent data fetching

### Portfolio Optimization
- **Mean-Variance Optimization**: Modern portfolio theory
- **Risk Constraints**: Regulatory and position limits
- **Black-Litterman**: Advanced portfolio construction
- **Factor Models**: Statistical risk modeling

### Risk Management
- **VaR/CVaR**: Value at Risk calculations
- **Monte Carlo**: Scenario analysis
- **Stress Testing**: Extreme event simulation
- **Real-time Monitoring**: Portfolio risk tracking

### Backtesting & Analysis
- **Vectorbt**: High-performance backtesting
- **Performance Metrics**: Sharpe, Sortino, Max Drawdown
- **Benchmark Analysis**: vs S&P 500 and other indices
- **Strategy Validation**: Statistical significance testing

## 📁 Project Structure

```
quant-portfolio-system/
├── data/                          # Data ingestion & storage
│   ├── src/
│   │   ├── feeds/                 # Data ingestion modules
│   │   └── storage/               # Data storage modules
│   └── storage/                   # Persistent data storage
├── portfolio/                     # Portfolio optimization
├── strategies/                    # Trading strategies
├── scripts/                       # Utility scripts
│   ├── setup_data_environment.py
│   └── demo_data_ingestion_and_storage.py
├── examples/                      # Usage examples
│   └── basic_usage.py
├── docs/                          # Documentation
│   ├── requirements.txt
│   └── data_ingestion_guide.md
├── tests/                         # Unit tests
└── output/                        # Analysis outputs
```

## 🛠️ Usage Examples

### Data Ingestion
```python
# Fetch multiple asset classes
equity_results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)
etf_results = ingestion.fetch_etfs(['SPY', 'QQQ'], start_date, end_date)
fx_results = ingestion.fetch_fx_pairs(['EURUSD', 'GBPUSD'], start_date, end_date)

# Export to DataFrame
df = ingestion.export_results_to_dataframe(results)
```

### Data Storage & Retrieval
```python
from data.src.storage import create_default_storage

# Save results
ingestion.save_results_to_storage(results)

# Retrieve data later
storage = create_default_storage()
aapl_data = storage.load_data('AAPL', AssetClass.EQUITY)
```

### Available Scripts
- `scripts/setup_data_environment.py` - Environment setup
- `scripts/demo_data_ingestion_and_storage.py` - Complete demonstration
- `examples/basic_usage.py` - Basic usage examples

## 📖 Documentation

- [Data Ingestion Guide](docs/data_ingestion_guide.md) - Complete data system documentation
- [CLAUDE.md](CLAUDE.md) - Development guidelines and constitution
- [README.md](README.md) - Project overview and quick start

## 🔧 Dependencies

### Core Libraries
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `yfinance>=0.2.28` - Yahoo Finance API
- `scipy>=1.11.0` - Statistical functions

### Optional Libraries
- `pyarrow>=12.0.0` - Parquet format support
- `matplotlib>=3.7.0` - Plotting
- `plotly>=5.0.0` - Interactive visualization

## 🎯 Performance Targets

- **Sharpe Ratio**: > 1.5 for optimized portfolios
- **Max Drawdown**: < 15% under normal conditions
- **Benchmark Outperformance**: > 200 bps annually vs S&P 500
- **Data Quality**: > 95% completeness rate

## 📈 Getting Help

1. Check the [Data Ingestion Guide](docs/data_ingestion_guide.md)
2. Run the setup script: `python scripts/setup_data_environment.py`
3. Try the examples in `examples/` directory
4. Review the complete demo: `python scripts/demo_data_ingestion_and_storage.py`

## 📋 Next Steps

1. **Setup**: Run the environment setup script
2. **Ingest Data**: Fetch historical market data
3. **Build Models**: Implement predictive models
4. **Optimize Portfolios**: Apply portfolio theory
5. **Backtest**: Validate strategies
6. **Deploy**: Monitor and trade

---

Built with ❤️ for quantitative trading research and systematic investing.

