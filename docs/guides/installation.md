# Installation Guide

This comprehensive guide covers installing and setting up the Quantitative Trading System.

## 🖥️ System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Internet**: Required for data downloads

### Recommended Requirements
- **Operating System**: Windows 11+, macOS 12+, or Linux (Ubuntu 22.04+)
- **Python**: 3.11 or higher
- **RAM**: 16GB or more
- **Storage**: 50GB+ SSD recommended
- **CPU**: Multi-core processor
- **GPU**: Optional for machine learning features

## 📦 Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/quant-portfolio-system.git
cd quant-portfolio-system
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Install base dependencies
pip install --upgrade pip
pip install -r docs/requirements.txt
```

#### 4. Verify Installation
```bash
# Run verification script
python scripts/verify_installation.py
```

### Method 2: Docker Installation

#### 1. Install Docker
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- [Docker for Linux](https://docs.docker.com/engine/install/)

#### 2. Build Docker Image
```bash
docker build -t quant-portfolio-system .
```

#### 3. Run Container
```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data quant-portfolio-system
```

### Method 3: Development Installation

#### 1. Clone Repository
```bash
git clone https://github.com/your-username/quant-portfolio-system.git
cd quant-portfolio-system
```

#### 2. Create Development Environment
```bash
# Create development virtual environment
python -m venv venv-dev

# Activate environment
# Windows:
venv-dev\Scripts\activate
# macOS/Linux:
source venv-dev/bin/activate
```

#### 3. Install Development Dependencies
```bash
# Install development dependencies
pip install --upgrade pip
pip install -r docs/requirements.txt
pip install -r requirements-dev.txt  # If exists
```

#### 4. Install in Development Mode
```bash
pip install -e .
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (if using premium services)
YAHOO_FINANCE_API_KEY=your_yahoo_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
QUANDL_API_KEY=your_quandl_key

# Database Configuration
DATABASE_URL=sqlite:///data/quant_portfolio.db

# API Configuration
API_HOST=localhost
API_PORT=8000
API_SECRET_KEY=your_secret_key

# Data Storage
DATA_PATH=./data
LOG_PATH=./logs
CACHE_PATH=./cache

# Performance Settings
MAX_WORKERS=4
MEMORY_LIMIT_GB=8
```

### Database Setup

#### SQLite (Default)
```bash
# SQLite database will be created automatically
# No additional setup required
```

#### PostgreSQL (Optional)
```bash
# Install PostgreSQL
# Ubuntu:
sudo apt-get install postgresql postgresql-contrib
# macOS:
brew install postgresql
# Windows: Download from postgresql.org

# Create database
createdb quant_portfolio

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://username:password@localhost/quant_portfolio
```

## 🔧 Verification

### 1. Run Basic Tests
```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v
```

### 2. Test Data Processing
```bash
# Test data ingestion
python scripts/test_data_ingestion.py

# Test preprocessing
python scripts/test_preprocessing.py
```

### 3. Test API Endpoints
```bash
# Start API server
python data/src/api/preprocessing_api.py

# Test health endpoint
curl http://localhost:8000/health
```

## 📁 Project Structure

After installation, your project structure should look like this:

```
quant-portfolio-system/
├── data/                          # Data handling modules
│   ├── src/                       # Source code
│   │   ├── api/                   # API endpoints
│   │   ├── cli/                   # Command-line interfaces
│   │   ├── config/                # Configuration management
│   │   ├── feeds/                 # Data ingestion
│   │   ├── lib/                   # Core libraries
│   │   ├── models/                # Data models
│   │   ├── services/              # Processing services
│   │   └── storage/               # Data storage
│   └── storage/                   # Data files (created at runtime)
├── portfolio/                     # Portfolio optimization
├── strategies/                    # Trading strategies
├── tests/                         # Test suites
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── examples/                      # Usage examples
├── config/                        # Configuration files
├── output/                        # Analysis outputs
├── requirements.txt              # Dependencies
├── CLAUDE.md                      # Development guidelines
└── README.md                      # Project overview
```

## 🚀 Getting Started

### 1. Create Your First Pipeline
```python
from data.src.config.pipeline_config import PipelineConfigManager

config_manager = PipelineConfigManager()
config = config_manager.create_default_config(
    pipeline_id="demo_pipeline",
    description="Demo pipeline for testing",
    asset_classes=["equity"],
    rules=[],
    quality_thresholds={"completeness": 0.9}
)
```

### 2. Download Sample Data
```python
from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion

ingestion = YahooFinanceIngestion()
data = ingestion.download_data(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### 3. Process Data
```python
from data.src.preprocessing import PreprocessingOrchestrator

orchestrator = PreprocessingOrchestrator(config_manager)
results = orchestrator.preprocess_data(data, "demo_pipeline")
```

## 🛠️ Troubleshooting

### Common Installation Issues

#### Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.11 -m pip install -r docs/requirements.txt
```

#### Permission Issues
```bash
# Install for current user only
pip install --user -r docs/requirements.txt

# Use sudo (Linux/macOS)
sudo pip install -r docs/requirements.txt
```

#### Network Issues
```bash
# Use pip with timeout
pip install --timeout=60 -r docs/requirements.txt

# Use different index URL
pip install -i https://pypi.org/simple/ -r docs/requirements.txt
```

#### Memory Issues
```bash
# Install with no cache
pip install --no-cache-dir -r docs/requirements.txt

# Install packages one by one
while read package; do
    pip install "$package"
done < docs/requirements.txt
```

### Testing Issues

#### Test Failures
```bash
# Run tests with verbose output
python -m pytest tests/unit/ -v --tb=short

# Run specific test
python -m pytest tests/unit/test_preprocessing.py::test_clean_data -v

# Run tests with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🔧 Development Setup

### 1. Install Development Tools
```bash
# Install development dependencies
pip install black pytest pytest-cov mypy pre-commit

# Install pre-commit hooks
pre-commit install
```

### 2. Code Formatting
```bash
# Format code
black .

# Check formatting
black --check .

# Type checking
mypy data/src/
```

### 3. Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run performance tests
python -m pytest tests/performance/
```

## 🔄 Upgrading

### Upgrade Dependencies
```bash
# Upgrade all packages
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U

# Upgrade specific package
pip install --upgrade package_name
```

### Upgrade System Version
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r docs/requirements.txt

# Run migration scripts if any
python scripts/migrate.py
```

## 📚 Additional Resources

### Documentation
- [Full Documentation](../README.md)
- [API Reference](../api/README.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)

### Community
- [GitHub Issues](https://github.com/your-username/quant-portfolio-system/issues)
- [GitHub Discussions](https://github.com/your-username/quant-portfolio-system/discussions)
- [Discord Server](https://discord.gg/quant-portfolio-system)

### Support
- [Installation FAQ](../faq/installation_faq.md)
- [Contact Support](mailto:support@quant-portfolio-system.com)

---

*Last Updated: 2024-01-15*