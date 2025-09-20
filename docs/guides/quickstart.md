# Quick Start Guide

This guide will help you get up and running with the Quantitative Trading System in minutes.

## 🚀 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.11+**: Download from [python.org](https://python.org)
- **Git**: For version control
- **Text Editor**: VS Code, PyCharm, or similar
- **Terminal**: Command line interface

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/quant-portfolio-system.git
cd quant-portfolio-system
```

### 2. Install Dependencies
```bash
pip install -r docs/requirements.txt
```

### 3. Verify Installation
```bash
python -c "import pandas as pd; import numpy as np; print('Dependencies installed successfully!')"
```

## ⚙️ Configuration

### 1. Create Your First Pipeline
Create a simple configuration file `config/my_first_pipeline.py`:

```python
from data.src.config.pipeline_config import PipelineConfigManager

# Initialize configuration manager
config_manager = PipelineConfigManager()

# Create basic equity pipeline
config = config_manager.create_default_config(
    pipeline_id="my_first_pipeline",
    description="My first quantitative trading pipeline",
    asset_classes=["equity"],
    rules=[
        {
            "type": "validation",
            "conditions": [{"field": "close", "operator": "greater_than", "value": 0}],
            "actions": [{"type": "flag", "severity": "warning"}]
        }
    ],
    quality_thresholds={
        "completeness": 0.95,
        "accuracy": 0.90
    }
)

# Save configuration
config_manager.save_config(config)
```

### 2. Set Up Data Sources
Create `config/data_sources.py`:

```python
YAHOO_FINANCE_SYMBOLS = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "JNJ", "V"
]
```

## 🔄 Data Processing

### 1. Ingest Data
```python
from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion

# Initialize data ingestion
ingestion = YahooFinanceIngestion()

# Download data for your symbols
data = ingestion.download_data(
    symbols=YAHOO_FINANCE_SYMBOLS,
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Downloaded data for {len(data)} symbols")
```

### 2. Preprocess Data
```python
from data.src.preprocessing import PreprocessingOrchestrator

# Initialize preprocessing orchestrator
orchestrator = PreprocessingOrchestrator(config_manager)

# Process the data
results = orchestrator.preprocess_data(
    data=data,
    pipeline_id="my_first_pipeline"
)

print(f"Processing completed with quality score: {results['quality_score']}")
```

## 📊 Feature Generation

### 1. Generate Basic Features
```python
from services.feature_service import FeatureGenerator
from services.feature_service import FeatureGenerationConfig

# Initialize feature generator
feature_generator = FeatureGenerator()

# Configure feature generation
feature_config = FeatureGenerationConfig(
    return_periods=[1, 5, 21],  # Daily, weekly, monthly returns
    volatility_windows=[5, 21],  # Weekly and monthly volatility
    momentum_periods=[5, 14]     # Short and medium momentum
)

# Generate features
features = feature_generator.generate_features(
    price_data=results['processed_data'],
    custom_config=feature_config
)

print(f"Generated {len(features.feature_names)} features")
```

## 🎯 Portfolio Optimization

### 1. Prepare Returns Data
```python
import pandas as pd

# Extract returns from features
returns = pd.DataFrame({
    symbol: features.get_feature(f'returns_1', symbol)
    for symbol in YAHOO_FINANCE_SYMBOLS
    if features.has_feature(f'returns_1', symbol)
})

# Drop missing values
returns = returns.dropna()

print(f"Returns data shape: {returns.shape}")
```

### 2. Optimize Portfolio
```python
from portfolio.src.optimization import PortfolioOptimizer
from portfolio.src.optimization import OptimizationConfig

# Initialize optimizer
optimizer = PortfolioOptimizer()

# Configure optimization
opt_config = OptimizationConfig(
    objective="sharpe_ratio",
    risk_free_rate=0.02,
    constraints={
        "min_weight": 0.01,      # Minimum 1% per position
        "max_weight": 0.20,      # Maximum 20% per position
        "max_sector_exposure": 0.30  # Maximum 30% sector exposure
    }
)

# Optimize portfolio
optimal_portfolio = optimizer.optimize_portfolio(
    returns=returns,
    config=opt_config
)

print("Optimal Portfolio Weights:")
for symbol, weight in optimal_portfolio.weights.items():
    print(f"{symbol}: {weight:.2%}")
```

## 📈 Performance Analysis

### 1. Calculate Portfolio Performance
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

print(f"Portfolio Performance:")
print(f"Annual Return: {performance['annual_return']:.2%}")
print(f"Annual Volatility: {performance['annual_volatility']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

### 2. Visualize Results
```python
import matplotlib.pyplot as plt

# Plot cumulative returns
cumulative_returns = (1 + returns).cumprod()
portfolio_returns = (returns * optimal_portfolio.weights).sum(axis=1)
portfolio_cumulative = (1 + portfolio_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns.index, portfolio_cumulative, label='Optimal Portfolio')
plt.plot(cumulative_returns.index, cumulative_returns.mean(axis=1), label='Equal Weight')
plt.title('Portfolio Performance Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
```

## 🔧 Running via CLI

### 1. Process Data Using CLI
```bash
# Process data using your pipeline
python -m data.src.cli.preprocess \
    --pipeline-id my_first_pipeline \
    --input-path data/raw/ \
    --output-path data/processed/
```

### 2. Generate Quality Report
```bash
# Generate quality report
python -m data.src.cli.quality_report \
    --dataset-id my_first_dataset \
    --input-path data/processed/ \
    --output-path reports/
```

## 📝 Next Steps

### What You've Accomplished
✅ Installed the system
✅ Created your first data pipeline
✅ Ingested financial data
✅ Preprocessed and cleaned data
✅ Generated financial features
✅ Optimized a portfolio
✅ Analyzed performance

### Where to Go Next

1. **Advanced Features**: Explore more sophisticated features and models
2. **Risk Management**: Implement comprehensive risk management strategies
3. **Backtesting**: Test your strategies on historical data
4. **Real-time Trading**: Set up live trading capabilities
5. **Machine Learning**: Incorporate ML models for prediction

### Recommended Reading

- [Data Preprocessing Guide](preprocessing.md) - Advanced data cleaning techniques
- [Feature Generation Guide](feature_generation.md) - More feature engineering options
- [Portfolio Optimization Guide](portfolio_optimization.md) - Advanced optimization techniques
- [Risk Management Guide](risk_management.md) - Comprehensive risk management

## 🆘 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If you encounter permission issues
pip install --user -r docs/requirements.txt

# If you need a specific Python version
python3.11 -m pip install -r docs/requirements.txt
```

#### Data Download Issues
```python
# If Yahoo Finance data download fails
import yfinance as yf
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
print(data.head())
```

#### Memory Issues
```python
# For large datasets, process in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Getting Help

- **Documentation**: Check the full [documentation](../README.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-username/quant-portfolio-system/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/your-username/quant-portfolio-system/discussions)

## 🎉 Congratulations!

You've successfully set up and run your first quantitative trading pipeline! You're now ready to explore more advanced features and build sophisticated trading strategies.

**Next Steps:**
1. Experiment with different asset classes
2. Try advanced optimization techniques
3. Implement risk management strategies
4. Set up automated trading

Happy trading! 🚀