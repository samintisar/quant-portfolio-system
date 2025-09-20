# Testing Guide

This comprehensive guide covers testing strategies, frameworks, and best practices for the Quantitative Trading System.

## 🧪 Testing Philosophy

### Testing Principles

1. **Test-Driven Development (TDD)**: Write tests before code
2. **Comprehensive Coverage**: Test all critical paths and edge cases
3. **Statistical Validation**: Validate mathematical correctness
4. **Performance Testing**: Ensure system meets performance targets
5. **Integration Testing**: Verify component interactions
6. **Mocking and Isolation**: Test components in isolation

### Testing Pyramid

```
                    ╱╲
                   ╱  ╲
      Unit Tests   ╱    ╲  Integration Tests
                 ╱      ╲
                ╱________╲   E2E Tests
               ╱          ╲
              ╱            ╲
             ╱              ╲
```

- **Unit Tests (70%)**: Fast, isolated component tests
- **Integration Tests (20%)**: Component interaction tests
- **End-to-End Tests (10%)**: Full system workflow tests

## 🏗️ Testing Framework

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_data_processing.py
│   ├── test_feature_generation.py
│   ├── test_portfolio_optimization.py
│   └── test_risk_management.py
├── integration/             # Integration tests
│   ├── test_data_pipeline.py
│   ├── test_api_integration.py
│   └── test_portfolio_workflow.py
├── performance/             # Performance tests
│   ├── test_benchmarks.py
│   ├── test_memory_usage.py
│   └── test_concurrent_processing.py
├── statistical/             # Statistical validation tests
│   ├── test_mathematical_correctness.py
│   ├── test_statistical_significance.py
│   └── test_backtesting_validation.py
└── contract/                # Contract tests
    ├── test_api_contract.py
    └── test_data_contract.py
```

### Test Configuration

```python
# conftest.py - Pytest configuration
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def sample_data():
    """Sample financial data for testing"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'dates': dates,
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    })

@pytest.fixture(scope="session")
def sample_returns():
    """Sample returns data for testing"""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    returns_data = {}
    for symbol in symbols:
        returns_data[symbol] = np.random.normal(0.001, 0.02, 252)

    return pd.DataFrame(returns_data, index=dates)

@pytest.fixture
def mock_optimizer():
    """Mock portfolio optimizer for testing"""
    class MockOptimizer:
        def optimize_portfolio(self, returns, config):
            n_assets = returns.shape[1]
            equal_weights = np.ones(n_assets) / n_assets
            return type('Portfolio', (), {
                'weights': dict(zip(returns.columns, equal_weights))
            })()

    return MockOptimizer()

# Custom markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "statistical: marks tests as statistical validation tests"
    )
```

## 🧪 Unit Testing

### Data Processing Tests

```python
# tests/unit/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
from data.src.lib.cleaning import DataCleaner
from data.src.lib.validation import DataValidator
from data.src.lib.normalization import DataNormalizer

class TestDataCleaning:
    """Test data cleaning functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.cleaner = DataCleaner()
        self.validator = DataValidator()

    def test_handle_missing_values_interpolation(self, sample_data):
        """Test missing value handling with interpolation"""
        # Introduce missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[10:15, 'close'] = np.nan

        # Clean data
        cleaned_data = self.cleaner.handle_missing_values(
            data_with_missing,
            method='interpolation'
        )

        # Verify no missing values remain
        assert not cleaned_data['close'].isna().any()

        # Verify interpolation is reasonable
        original_values = sample_data.loc[8:17, 'close']
        cleaned_values = cleaned_data.loc[8:17, 'close']

        # Values should be within reasonable range
        assert np.all(cleaned_values >= original_values.min() * 0.8)
        assert np.all(cleaned_values <= original_values.max() * 1.2)

    def test_handle_outliers_iqr_method(self, sample_data):
        """Test outlier detection using IQR method"""
        # Introduce outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.loc[50, 'close'] = 1000  # Extreme outlier

        # Detect outliers
        outliers = self.cleaner.detect_outliers(
            data_with_outliers,
            method='iqr'
        )

        # Verify outlier detection
        assert 50 in outliers['close']

        # Handle outliers
        cleaned_data = self.cleaner.handle_outliers(
            data_with_outliers,
            method='iqr'
        )

        # Verify outliers are handled
        assert cleaned_data.loc[50, 'close'] < 1000

    def test_validate_price_data(self, sample_data):
        """Test price data validation"""
        # Validate data
        validation_result = self.validator.validate_price_data(sample_data)

        # Verify validation results
        assert validation_result['is_valid']
        assert len(validation_result['validation_issues']) == 0

        # Test invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'close'] = -100  # Negative price

        invalid_result = self.validator.validate_price_data(invalid_data)
        assert not invalid_result['is_valid']
        assert len(invalid_result['validation_issues']) > 0

    def test_normalize_data_zscore(self, sample_data):
        """Test Z-score normalization"""
        # Normalize data
        normalized_data = self.cleaner.normalize_data(
            sample_data,
            method='zscore'
        )

        # Verify normalization
        numeric_columns = sample_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # Mean should be approximately 0
            assert abs(normalized_data[col].mean()) < 1e-10
            # Standard deviation should be 1
            assert abs(normalized_data[col].std() - 1.0) < 1e-10

    def test_detect_time_gaps(self, sample_data):
        """Test time gap detection"""
        # Introduce time gap
        data_with_gap = sample_data.copy()
        data_with_gap = data_with_gap.drop(data_with_gap.index[20:25])

        # Detect gaps
        gaps = self.cleaner.detect_time_gaps(data_with_gap)

        # Verify gap detection
        assert len(gaps) > 0
        assert gaps[0]['start_date'] < gaps[0]['end_date']

class TestDataValidation:
    """Test data validation functionality"""

    def test_validate_ohlc_relationships(self, sample_data):
        """Test OHLC relationship validation"""
        # Validate OHLC relationships
        result = self.validator.validate_ohlc_relationships(sample_data)

        # Verify validation
        assert result['is_valid']

        # Test invalid OHLC data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'low'] - 1  # High < Low

        invalid_result = self.validator.validate_ohlc_relationships(invalid_data)
        assert not invalid_result['is_valid']
        assert len(invalid_result['validation_issues']) > 0

    def test_validate_data_quality(self, sample_data):
        """Test comprehensive data quality validation"""
        # Validate data quality
        quality_report = self.validator.validate_data_quality(sample_data)

        # Verify quality metrics
        assert 'completeness' in quality_report
        assert 'accuracy' in quality_report
        assert 'consistency' in quality_report
        assert 'overall_score' in quality_report

        # Score should be between 0 and 1
        assert 0 <= quality_report['overall_score'] <= 1

    def test_validate_volume_data(self, sample_data):
        """Test volume data validation"""
        # Validate volume data
        result = self.validator.validate_volume_data(sample_data)

        # Verify validation
        assert result['is_valid']

        # Test invalid volume data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'volume'] = -1000  # Negative volume

        invalid_result = self.validator.validate_volume_data(invalid_data)
        assert not invalid_result['is_valid']
```

### Feature Generation Tests

```python
# tests/unit/test_feature_generation.py
import pytest
import pandas as pd
import numpy as np
from services.feature_service import FeatureGenerator
from services.feature_service import FeatureGenerationConfig
from models.price_data import PriceData, Frequency, PriceType
from models.financial_instrument import FinancialInstrument, InstrumentType

class TestFeatureGeneration:
    """Test feature generation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.feature_generator = FeatureGenerator()

    def create_test_price_data(self):
        """Create test price data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(102, 5, 100),
            'low': np.random.normal(98, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)

        instrument = FinancialInstrument(
            symbol="TEST",
            name="Test Instrument",
            instrument_type=InstrumentType.EQUITY
        )

        return PriceData(
            prices=prices,
            instrument=instrument,
            frequency=Frequency.DAILY,
            price_type=PriceType.OHLC
        )

    def test_generate_returns(self):
        """Test returns calculation"""
        price_data = self.create_test_price_data()

        # Generate returns
        returns = self.feature_generator.calculate_returns(
            price_data.prices['close'],
            periods=[1, 5, 21]
        )

        # Verify returns calculation
        assert 'returns_1' in returns
        assert 'returns_5' in returns
        assert 'returns_21' in returns

        # Check for reasonable values
        for col in returns.columns:
            assert not returns[col].isna().all()
            assert abs(returns[col].mean()) < 0.1  # Reasonable mean return

    def test_generate_volatility(self):
        """Test volatility calculation"""
        price_data = self.create_test_price_data()

        # Generate volatility
        volatility = self.feature_generator.calculate_volatility(
            price_data.prices['close'],
            windows=[5, 21, 63]
        )

        # Verify volatility calculation
        assert 'volatility_5' in volatility
        assert 'volatility_21' in volatility
        assert 'volatility_63' in volatility

        # Volatility should be positive
        for col in volatility.columns:
            assert (volatility[col] >= 0).all()

    def test_generate_momentum(self):
        """Test momentum calculation"""
        price_data = self.create_test_price_data()

        # Generate momentum
        momentum = self.feature_generator.calculate_momentum(
            price_data.prices['close'],
            periods=[5, 14, 21]
        )

        # Verify momentum calculation
        assert 'momentum_5' in momentum
        assert 'momentum_14' in momentum
        assert 'momentum_21' in momentum

    def test_generate_all_features(self):
        """Test complete feature generation"""
        price_data = self.create_test_price_data()

        # Configure feature generation
        config = FeatureGenerationConfig(
            return_periods=[1, 5, 21],
            volatility_windows=[5, 21],
            momentum_periods=[5, 14]
        )

        # Generate all features
        feature_set = self.feature_generator.generate_features(
            price_data=price_data,
            custom_config=config
        )

        # Verify feature generation
        assert len(feature_set.feature_names) > 0
        assert feature_set.quality_score.value > 0.5
        assert len(feature_set.get_feature('returns_1')) > 0

    def test_feature_validation(self):
        """Test feature validation"""
        price_data = self.create_test_price_data()

        # Generate features
        feature_set = self.feature_generator.generate_features(price_data=price_data)

        # Validate features
        validation_result = self.feature_generator.validate_features(feature_set)

        # Verify validation
        assert 'is_valid' in validation_result
        assert 'validation_issues' in validation_result
        assert 'quality_metrics' in validation_result

    def test_feature_transformation(self):
        """Test feature transformation"""
        price_data = self.create_test_price_data()

        # Generate features
        feature_set = self.feature_generator.generate_features(price_data=price_data)

        # Apply transformations
        transformed_features = self.feature_generator.transform_features(
            feature_set,
            transformations=['normalize', 'standardize']
        )

        # Verify transformations
        assert len(transformed_features.feature_names) > 0
        # Check if normalization worked (values around 0)
        feature_data = transformed_features.to_dict()
        for feature_name, values in feature_data.items():
            if values:
                assert abs(np.mean(values)) < 1.0
```

### Portfolio Optimization Tests

```python
# tests/unit/test_portfolio_optimization.py
import pytest
import pandas as pd
import numpy as np
from portfolio.src.optimization import PortfolioOptimizer
from portfolio.src.optimization import OptimizationConfig
from portfolio.src.analysis import PerformanceAnalyzer

class TestPortfolioOptimization:
    """Test portfolio optimization functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = PortfolioOptimizer()
        self.analyzer = PerformanceAnalyzer()

    def create_test_returns(self):
        """Create test returns data"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

        # Create correlated returns
        np.random.seed(42)
        mean_returns = np.array([0.001, 0.0015, 0.0012, 0.0008, 0.001])
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.00015, 0.0001, 0.00012],
            [0.0002, 0.0005, 0.00018, 0.00012, 0.00015],
            [0.00015, 0.00018, 0.00035, 0.0001, 0.00013],
            [0.0001, 0.00012, 0.0001, 0.0003, 0.0001],
            [0.00012, 0.00015, 0.00013, 0.0001, 0.00025]
        ])

        returns = np.random.multivariate_normal(mean_returns, cov_matrix, 252)
        return pd.DataFrame(returns, index=dates, columns=symbols)

    def test_mean_variance_optimization(self, sample_returns):
        """Test mean-variance optimization"""
        # Configure optimization
        config = OptimizationConfig(
            objective="sharpe_ratio",
            risk_free_rate=0.02,
            constraints={
                "min_weight": 0.0,
                "max_weight": 1.0,
                "sum_weights": 1.0
            }
        )

        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        # Verify optimization results
        assert portfolio is not None
        assert len(portfolio.weights) == sample_returns.shape[1]

        # Check weight constraints
        total_weight = sum(portfolio.weights.values())
        assert abs(total_weight - 1.0) < 1e-6  # Sum to 1

        for weight in portfolio.weights.values():
            assert 0 <= weight <= 1  # Weights between 0 and 1

    def test_minimum_variance_optimization(self, sample_returns):
        """Test minimum variance optimization"""
        # Configure minimum variance optimization
        config = OptimizationConfig(
            objective="min_variance",
            constraints={
                "min_weight": 0.05,
                "max_weight": 0.4,
                "sum_weights": 1.0
            }
        )

        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        # Verify results
        assert portfolio is not None
        for weight in portfolio.weights.values():
            assert 0.05 <= weight <= 0.4  # Respect constraints

    def test_maximum_return_optimization(self, sample_returns):
        """Test maximum return optimization"""
        # Configure maximum return optimization
        config = OptimizationConfig(
            objective="max_return",
            constraints={
                "min_weight": 0.0,
                "max_weight": 0.3,
                "max_volatility": 0.15
            }
        )

        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        # Verify results
        assert portfolio is not None
        for weight in portfolio.weights.values():
            assert weight <= 0.3  # Max weight constraint

    def test_risk_parity_optimization(self, sample_returns):
        """Test risk parity optimization"""
        # Configure risk parity optimization
        config = OptimizationConfig(
            objective="risk_parity",
            constraints={
                "min_weight": 0.1,
                "max_weight": 0.4,
                "sum_weights": 1.0
            }
        )

        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        # Verify results
        assert portfolio is not None
        assert len(portfolio.weights) == sample_returns.shape[1]

    def test_constraint_handling(self, sample_returns):
        """Test constraint handling"""
        # Test with multiple constraints
        config = OptimizationConfig(
            objective="sharpe_ratio",
            constraints={
                "min_weight": 0.05,
                "max_weight": 0.25,
                "min_positions": 3,
                "max_positions": 5,
                "max_volatility": 0.2,
                "sector_limits": {
                    "Technology": 0.4,
                    "Healthcare": 0.3
                }
            }
        )

        # Optimize portfolio
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        # Verify constraints are respected
        active_positions = [w for w in portfolio.weights.values() if w > 0.01]
        assert 3 <= len(active_positions) <= 5

        for weight in portfolio.weights.values():
            assert 0.05 <= weight <= 0.25

    def test_performance_calculation(self, sample_returns):
        """Test performance calculation"""
        # Create simple equal-weight portfolio
        weights = {symbol: 1.0/len(sample_returns.columns)
                  for symbol in sample_returns.columns}

        # Calculate performance
        performance = self.analyzer.calculate_performance(
            returns=sample_returns,
            weights=weights
        )

        # Verify performance metrics
        assert 'annual_return' in performance
        assert 'annual_volatility' in performance
        assert 'sharpe_ratio' in performance
        assert 'max_drawdown' in performance

        # Check reasonable values
        assert performance['annual_volatility'] > 0
        assert performance['max_drawdown'] <= 0

    def test_efficient_frontier(self, sample_returns):
        """Test efficient frontier calculation"""
        # Calculate efficient frontier
        efficient_portfolios = self.optimizer.calculate_efficient_frontier(
            returns=sample_returns,
            num_portfolios=20
        )

        # Verify efficient frontier
        assert len(efficient_portfolios) == 20

        # Check that portfolios are on efficient frontier
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]

        # Returns should generally increase with volatility
        correlation = np.corrcoef(returns, volatilities)[0, 1]
        assert correlation > 0.5  # Positive correlation

    def test_robustness_to_outliers(self, sample_returns):
        """Test optimization robustness to outliers"""
        # Add extreme outlier to one asset
        returns_with_outlier = sample_returns.copy()
        returns_with_outlier.iloc[0, 0] = 10.0  # 1000% return outlier

        # Optimize both original and outlier data
        config = OptimizationConfig(
            objective="sharpe_ratio",
            constraints={
                "min_weight": 0.0,
                "max_weight": 0.5
            }
        )

        portfolio_original = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config
        )

        portfolio_outlier = self.optimizer.optimize_portfolio(
            returns=returns_with_outlier,
            config=config
        )

        # Verify robustness (weights shouldn't change drastically)
        original_weights = list(portfolio_original.weights.values())
        outlier_weights = list(portfolio_outlier.weights.values())

        # Calculate weight differences
        weight_diffs = [abs(o - e) for o, e in zip(original_weights, outlier_weights)]
        max_diff = max(weight_diffs)

        # Max weight difference should be reasonable
        assert max_diff < 0.3  # Less than 30% weight difference

    def test_transaction_costs(self, sample_returns):
        """Test optimization with transaction costs"""
        # Configure with transaction costs
        config = OptimizationConfig(
            objective="sharpe_ratio",
            transaction_costs=0.001,  # 0.1% transaction cost
            constraints={
                "min_weight": 0.0,
                "max_weight": 0.3
            }
        )

        # Current weights (different from optimal)
        current_weights = {symbol: 0.2 for symbol in sample_returns.columns}

        # Optimize with transaction costs
        portfolio = self.optimizer.optimize_portfolio(
            returns=sample_returns,
            config=config,
            current_weights=current_weights
        )

        # Verify results
        assert portfolio is not None
        assert len(portfolio.weights) == sample_returns.shape[1]
```

## 🔗 Integration Testing

### API Integration Tests

```python
# tests/integration/test_api_integration.py
import pytest
import requests
import json
from datetime import datetime, timedelta

class TestAPIIntegration:
    """Test API integration"""

    @pytest.fixture(scope="module")
    def api_base_url(self):
        """API base URL"""
        return "http://localhost:8000"

    @pytest.fixture(scope="module")
    def api_headers(self):
        """API headers"""
        return {
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json"
        }

    def test_health_check(self, api_base_url):
        """Test API health check"""
        response = requests.get(f"{api_base_url}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_pipeline_creation(self, api_base_url, api_headers):
        """Test pipeline creation via API"""
        pipeline_data = {
            "pipeline_id": "test_pipeline",
            "description": "Test pipeline for integration",
            "asset_classes": ["equity"],
            "rules": [
                {
                    "type": "validation",
                    "conditions": [{"field": "close", "operator": "greater_than", "value": 0}],
                    "actions": [{"type": "flag", "severity": "warning"}]
                }
            ],
            "quality_thresholds": {"completeness": 0.9}
        }

        response = requests.post(
            f"{api_base_url}/pipelines",
            headers=api_headers,
            json=pipeline_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_id"] == "test_pipeline"

    def test_data_processing_workflow(self, api_base_url, api_headers):
        """Test complete data processing workflow"""
        # Create test data
        test_data = {
            "dates": ["2023-01-01", "2023-01-02"],
            "close": [100.5, 101.2],
            "volume": [1000000, 1200000]
        }

        # Process data
        process_response = requests.post(
            f"{api_base_url}/preprocessing/process",
            headers=api_headers,
            json={
                "pipeline_id": "test_pipeline",
                "input_data": test_data,
                "output_format": "json"
            }
        )

        assert process_response.status_code == 200
        process_result = process_response.json()
        assert process_result["success"]

        # Assess quality
        quality_response = requests.post(
            f"{api_base_url}/quality/assess",
            headers=api_headers,
            json={
                "dataset_id": "test_dataset",
                "data": test_data
            }
        )

        assert quality_response.status_code == 200
        quality_result = quality_response.json()
        assert "overall_score" in quality_result["data"]

    def test_error_handling(self, api_base_url, api_headers):
        """Test API error handling"""
        # Test invalid pipeline
        response = requests.post(
            f"{api_base_url}/preprocessing/process",
            headers=api_headers,
            json={
                "pipeline_id": "nonexistent_pipeline",
                "input_data": {"close": [100, 101]}
            }
        )

        assert response.status_code == 404
```

### End-to-End Workflow Tests

```python
# tests/integration/test_complete_workflow.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion
from data.src.preprocessing import PreprocessingOrchestrator
from data.src.config.pipeline_config import PipelineConfigManager
from services.feature_service import FeatureGenerator
from portfolio.src.optimization import PortfolioOptimizer
from portfolio.src.optimization import OptimizationConfig

class TestCompleteWorkflow:
    """Test complete end-to-end workflow"""

    @pytest.mark.slow
    def test_full_workflow_execution(self):
        """Test complete workflow from data ingestion to portfolio optimization"""
        # Step 1: Data Ingestion
        ingestion = YahooFinanceIngestion()

        # Use small sample for testing
        symbols = ['AAPL', 'GOOGL']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        try:
            raw_data = ingestion.download_data(symbols, start_date, end_date)
            assert len(raw_data) == 2
        except Exception as e:
            # Skip if network issues
            pytest.skip(f"Network issue: {e}")

        # Step 2: Data Preprocessing
        config_manager = PipelineConfigManager()
        orchestrator = PreprocessingOrchestrator(config_manager)

        pipeline_config = {
            'pipeline_id': 'test_workflow_pipeline',
            'description': 'Test workflow pipeline',
            'asset_classes': ['equity'],
            'rules': [
                {
                    'type': 'validation',
                    'conditions': [
                        {'field': 'close', 'operator': 'greater_than', 'value': 0}
                    ],
                    'actions': [{'type': 'flag', 'severity': 'warning'}]
                }
            ],
            'quality_thresholds': {'completeness': 0.8}
        }

        processed_data = {}
        for symbol, data in raw_data.items():
            result = orchestrator.preprocess_data(data, pipeline_config)
            processed_data[symbol] = result['processed_data']

        assert len(processed_data) == 2

        # Step 3: Feature Generation
        feature_generator = FeatureGenerator()
        all_features = {}

        for symbol, data in processed_data.items():
            features = feature_generator.generate_features(price_data=data)
            all_features[symbol] = features

        assert len(all_features) == 2

        # Step 4: Portfolio Optimization
        # Extract returns from features
        returns_data = {}
        for symbol, feature_set in all_features.items():
            if feature_set.has_feature('returns_1', symbol):
                returns_data[symbol] = feature_set.get_feature('returns_1', symbol)

        returns_df = pd.DataFrame(returns_data).dropna()
        assert len(returns_df) > 0

        # Optimize portfolio
        optimizer = PortfolioOptimizer()
        config = OptimizationConfig(
            objective='sharpe_ratio',
            risk_free_rate=0.02,
            constraints={
                'min_weight': 0.1,
                'max_weight': 0.8,
                'min_positions': 2
            }
        )

        portfolio = optimizer.optimize_portfolio(returns_df, config)
        assert portfolio is not None
        assert len(portfolio.weights) == 2

        # Verify constraints
        total_weight = sum(portfolio.weights.values())
        assert abs(total_weight - 1.0) < 1e-6

        for weight in portfolio.weights.values():
            assert 0.1 <= weight <= 0.8

    def test_workflow_error_handling(self):
        """Test workflow error handling"""
        # Test with invalid data
        invalid_data = {
            'close': [100, -50, 150],  # Negative price
            'volume': [1000000, 1200000, 900000]
        }

        config_manager = PipelineConfigManager()
        orchestrator = PreprocessingOrchestrator(config_manager)

        pipeline_config = {
            'pipeline_id': 'test_error_pipeline',
            'description': 'Test error handling pipeline',
            'asset_classes': ['equity'],
            'rules': [
                {
                    'type': 'validation',
                    'conditions': [
                        {'field': 'close', 'operator': 'greater_than', 'value': 0}
                    ],
                    'actions': [{'type': 'flag', 'severity': 'error'}]
                }
            ],
            'quality_thresholds': {'completeness': 0.9}
        }

        # Should handle error gracefully
        try:
            result = orchestrator.preprocess_data(invalid_data, pipeline_config)
            assert 'success' in result
            # May fail validation but should not crash
        except Exception as e:
            # Should handle known exceptions gracefully
            assert isinstance(e, (ValueError, KeyError))
```

## 📊 Performance Testing

### Benchmark Tests

```python
# tests/performance/test_benchmarks.py
import pytest
import time
import pandas as pd
import numpy as np
import psutil
import os
from datetime import datetime, timedelta

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing"""
        dates = pd.date_range('2020-01-01', periods=10000, freq='D')
        symbols = [f'STOCK_{i:03d}' for i in range(100)]

        # Create correlated returns
        np.random.seed(42)
        base_returns = np.random.normal(0.001, 0.02, (10000, 100))

        # Add some correlation structure
        for i in range(0, 100, 10):
            factor = np.random.normal(0, 0.01, 10000)
            for j in range(min(10, 100-i)):
                base_returns[:, i+j] += 0.3 * factor

        returns_df = pd.DataFrame(base_returns, index=dates, columns=symbols)
        return returns_df

    @pytest.mark.performance
    def test_data_processing_performance(self, large_dataset):
        """Test data processing performance"""
        from data.src.preprocessing import PreprocessingOrchestrator
        from data.src.config.pipeline_config import PipelineConfigManager

        config_manager = PipelineConfigManager()
        orchestrator = PreprocessingOrchestrator(config_manager)

        pipeline_config = {
            'pipeline_id': 'performance_test_pipeline',
            'description': 'Performance test pipeline',
            'asset_classes': ['equity'],
            'rules': [],
            'quality_thresholds': {'completeness': 0.9}
        }

        # Measure processing time
        start_time = time.time()

        # Process in chunks to simulate real usage
        chunk_size = 1000
        results = []

        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset.iloc[i:i+chunk_size]
            result = orchestrator.preprocess_data(
                {'close': chunk},
                pipeline_config
            )
            results.append(result)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance targets
        assert processing_time < 30.0  # Should process in under 30 seconds
        assert len(results) == 10  # Should process all chunks

        print(f"Data processing performance: {processing_time:.2f} seconds for 10M data points")

    @pytest.mark.performance
    def test_feature_generation_performance(self, large_dataset):
        """Test feature generation performance"""
        from services.feature_service import FeatureGenerator

        feature_generator = FeatureGenerator()

        # Create price data from returns
        price_data = large_dataset.cumsum() + 100

        # Measure feature generation time
        start_time = time.time()

        # Generate features for first 20 symbols
        for symbol in large_dataset.columns[:20]:
            from services.feature_service import FeatureGenerationConfig
            from models.price_data import PriceData, Frequency, PriceType
            from models.financial_instrument import FinancialInstrument, InstrumentType

            config = FeatureGenerationConfig(
                return_periods=[1, 5, 21],
                volatility_windows=[5, 21],
                momentum_periods=[5, 14]
            )

            prices_df = pd.DataFrame({
                'close': price_data[symbol]
            }, index=price_data.index)

            instrument = FinancialInstrument(
                symbol=symbol,
                name=f"Test {symbol}",
                instrument_type=InstrumentType.EQUITY
            )

            price_data_obj = PriceData(
                prices=prices_df,
                instrument=instrument,
                frequency=Frequency.DAILY,
                price_type=PriceType.CLOSE
            )

            features = feature_generator.generate_features(
                price_data=price_data_obj,
                custom_config=config
            )

        end_time = time.time()
        generation_time = end_time - start_time

        # Verify performance targets
        assert generation_time < 60.0  # Should generate features in under 60 seconds

        print(f"Feature generation performance: {generation_time:.2f} seconds for 20 symbols")

    @pytest.mark.performance
    def test_portfolio_optimization_performance(self, large_dataset):
        """Test portfolio optimization performance"""
        from portfolio.src.optimization import PortfolioOptimizer
        from portfolio.src.optimization import OptimizationConfig

        optimizer = PortfolioOptimizer()

        # Use subset for optimization testing
        subset_data = large_dataset.iloc[:252]  # 1 year of data
        subset_symbols = subset_data.columns[:20]  # 20 symbols

        returns_subset = subset_data[subset_symbols]

        config = OptimizationConfig(
            objective='sharpe_ratio',
            risk_free_rate=0.02,
            constraints={
                'min_weight': 0.02,
                'max_weight': 0.2,
                'min_positions': 5,
                'max_positions': 15
            }
        )

        # Measure optimization time
        start_time = time.time()

        portfolio = optimizer.optimize_portfolio(returns_subset, config)

        end_time = time.time()
        optimization_time = end_time - start_time

        # Verify performance targets
        assert optimization_time < 10.0  # Should optimize in under 10 seconds
        assert portfolio is not None
        assert len(portfolio.weights) == 20

        print(f"Portfolio optimization performance: {optimization_time:.2f} seconds")

    @pytest.mark.performance
    def test_memory_usage(self, large_dataset):
        """Test memory usage during processing"""
        process = psutil.Process(os.getpid())

        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process data
        from data.src.preprocessing import PreprocessingOrchestrator
        from data.src.config.pipeline_config import PipelineConfigManager

        config_manager = PipelineConfigManager()
        orchestrator = PreprocessingOrchestrator(config_manager)

        pipeline_config = {
            'pipeline_id': 'memory_test_pipeline',
            'description': 'Memory test pipeline',
            'asset_classes': ['equity'],
            'rules': [],
            'quality_thresholds': {'completeness': 0.9}
        }

        # Process data
        result = orchestrator.preprocess_data(
            {'close': large_dataset},
            pipeline_config
        )

        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Verify memory targets
        assert memory_increase < 4000  # Should use less than 4GB additional memory

        print(f"Memory usage: {memory_increase:.2f} MB increase")

    @pytest.mark.performance
    def test_concurrent_processing(self, large_dataset):
        """Test concurrent processing performance"""
        import concurrent.futures
        import threading

        from data.src.preprocessing import PreprocessingOrchestrator
        from data.src.config.pipeline_config import PipelineConfigManager

        config_manager = PipelineConfigManager()
        orchestrator = PreprocessingOrchestrator(config_manager)

        pipeline_config = {
            'pipeline_id': 'concurrent_test_pipeline',
            'description': 'Concurrent test pipeline',
            'asset_classes': ['equity'],
            'rules': [],
            'quality_thresholds': {'completeness': 0.9}
        }

        def process_chunk(chunk_data, chunk_id):
            """Process a chunk of data"""
            result = orchestrator.preprocess_data(
                {'close': chunk_data},
                pipeline_config
            )
            return chunk_id, result

        # Split data into chunks
        chunk_size = 1000
        chunks = []
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset.iloc[i:i+chunk_size]
            chunks.append((chunk, i // chunk_size))

        # Process concurrently
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk, chunk_id)
                      for chunk, chunk_id in chunks]

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        end_time = time.time()
        concurrent_time = end_time - start_time

        # Process sequentially for comparison
        start_time = time.time()

        sequential_results = []
        for chunk, chunk_id in chunks:
            result = process_chunk(chunk, chunk_id)
            sequential_results.append(result)

        end_time = time.time()
        sequential_time = end_time - start_time

        # Verify performance improvement
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5  # Should be at least 1.5x faster

        print(f"Concurrent processing speedup: {speedup:.2f}x")
```

### Statistical Validation Tests

```python
# tests/statistical/test_statistical_validation.py
import pytest
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class TestStatisticalValidation:
    """Statistical validation tests"""

    def create_test_returns(self, n_assets=5, n_periods=252):
        """Create test returns with known statistical properties"""
        np.random.seed(42)

        # Create mean returns and covariance matrix
        mean_returns = np.random.normal(0.001, 0.0005, n_assets)

        # Create positive definite covariance matrix
        A = np.random.normal(0, 0.02, (n_assets, n_assets))
        cov_matrix = A @ A.T + np.eye(n_assets) * 0.0001

        # Generate returns
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)

        dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
        symbols = [f'ASSET_{i:01d}' for i in range(n_assets)]

        return pd.DataFrame(returns, index=dates, columns=symbols)

    def test_returns_distribution_normality(self):
        """Test that generated returns follow expected distribution"""
        returns = self.create_test_returns()

        for symbol in returns.columns:
            symbol_returns = returns[symbol]

            # Test for normality using Shapiro-Wilk test
            _, p_value = stats.shapiro(symbol_returns)

            # At 5% significance level, we don't reject normality
            assert p_value > 0.05, f"{symbol} returns not normally distributed (p={p_value:.4f})"

    def test_portfolio_optimization_statistical_properties(self):
        """Test statistical properties of portfolio optimization"""
        returns = self.create_test_returns()

        from portfolio.src.optimization import PortfolioOptimizer
        from portfolio.src.optimization import OptimizationConfig

        optimizer = PortfolioOptimizer()

        config = OptimizationConfig(
            objective='sharpe_ratio',
            risk_free_rate=0.02,
            constraints={
                'min_weight': 0.0,
                'max_weight': 1.0
            }
        )

        # Run optimization multiple times with different seeds
        n_trials = 20
        portfolio_returns = []

        for i in range(n_trials):
            np.random.seed(i)
            trial_returns = returns + np.random.normal(0, 0.001, returns.shape)

            portfolio = optimizer.optimize_portfolio(trial_returns, config)
            portfolio_return = (trial_returns * pd.Series(portfolio.weights)).sum(axis=1)
            portfolio_returns.append(portfolio_return.mean())

        portfolio_returns = np.array(portfolio_returns)

        # Test that portfolio returns are reasonable
        assert np.all(portfolio_returns > -0.1)  # No extremely negative returns
        assert np.all(portfolio_returns < 0.1)   # No extremely positive returns

        # Test that optimization is stable
        return_std = np.std(portfolio_returns)
        assert return_std < 0.02  # Low variation across trials

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation accuracy"""
        returns = self.create_test_returns()

        from portfolio.src.risk import RiskAnalyzer

        analyzer = RiskAnalyzer()

        # Equal weight portfolio
        weights = {symbol: 1.0/len(returns.columns) for symbol in returns.columns}

        risk_metrics = analyzer.calculate_risk_metrics(returns, weights)

        # Verify risk metrics are reasonable
        assert risk_metrics['volatility'] > 0
        assert risk_metrics['volatility'] < 0.5  # Less than 50% volatility

        # VaR should be negative
        assert risk_metrics['var_95'] < 0
        assert risk_metrics['var_99'] < 0

        # CVaR should be more negative than VaR
        assert risk_metrics['cvar_95'] <= risk_metrics['var_95']
        assert risk_metrics['cvar_99'] <= risk_metrics['var_99']

    def test_efficient_frontier_properties(self):
        """Test efficient frontier mathematical properties"""
        returns = self.create_test_returns()

        from portfolio.src.optimization import PortfolioOptimizer

        optimizer = PortfolioOptimizer()
        efficient_portfolios = optimizer.calculate_efficient_frontier(returns, 20)

        # Extract returns and volatilities
        portfolio_returns = [p['return'] for p in efficient_portfolios]
        portfolio_volatilities = [p['volatility'] for p in efficient_portfolios]

        # Test that efficient frontier is non-decreasing in return-volatility space
        for i in range(1, len(portfolio_returns)):
            assert portfolio_returns[i] >= portfolio_returns[i-1]

            # If returns increase, volatility should not decrease
            if portfolio_returns[i] > portfolio_returns[i-1]:
                assert portfolio_volatilities[i] >= portfolio_volatilities[i-1]

    def test_backtesting_statistical_significance(self):
        """Test statistical significance of backtesting results"""
        returns = self.create_test_returns()

        from portfolio.src.backtesting import BacktestEngine

        engine = BacktestEngine()

        # Simple strategy: equal weight vs market cap weight
        strategy_returns = []
        benchmark_returns = []

        # Walk-forward analysis
        window_size = 63  # 3 months
        for i in range(window_size, len(returns), 21):  # Rebalance monthly
            train_data = returns.iloc[i-window_size:i]
            test_data = returns.iloc[i:min(i+21, len(returns))]

            # Equal weight strategy
            equal_weights = {symbol: 1.0/len(train_data.columns) for symbol in train_data.columns}
            strategy_return = (test_data * pd.Series(equal_weights)).sum(axis=1)
            strategy_returns.extend(strategy_return)

            # Benchmark (equal weight all assets)
            benchmark_return = test_data.mean(axis=1)
            benchmark_returns.extend(benchmark_return)

        # Test for significant outperformance
        strategy_returns = np.array(strategy_returns)
        benchmark_returns = np.array(benchmark_returns)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(strategy_returns, benchmark_returns)

        # Calculate information ratio
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

        print(f"Backtesting t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        print(f"Information Ratio: {information_ratio:.3f}")

        # Test that strategy is not significantly worse than benchmark
        assert p_value > 0.05 or information_ratio > 0

    def test_cross_validation_stability(self):
        """Test cross-validation stability"""
        returns = self.create_test_returns()

        from portfolio.src.optimization import PortfolioOptimizer
        from portfolio.src.optimization import OptimizationConfig

        optimizer = PortfolioOptimizer()

        config = OptimizationConfig(
            objective='sharpe_ratio',
            risk_free_rate=0.02,
            constraints={
                'min_weight': 0.05,
                'max_weight': 0.4
            }
        )

        # Time series cross-validation
        window_size = 126  # 6 months
        test_size = 63     # 3 months

        cv_results = []

        for i in range(0, len(returns) - window_size - test_size, test_size):
            train_data = returns.iloc[i:i+window_size]
            test_data = returns.iloc[i+window_size:i+window_size+test_size]

            # Optimize on training data
            portfolio = optimizer.optimize_portfolio(train_data, config)

            # Test on out-of-sample data
            test_return = (test_data * pd.Series(portfolio.weights)).sum(axis=1)

            cv_results.append({
                'train_period': (train_data.index[0], train_data.index[-1]),
                'test_period': (test_data.index[0], test_data.index[-1]),
                'test_return': test_return.mean(),
                'test_volatility': test_return.std(),
                'sharpe_ratio': test_return.mean() / test_return.std() * np.sqrt(252)
            })

        # Test stability across periods
        sharpe_ratios = [r['sharpe_ratio'] for r in cv_results]

        # Sharpe ratios should be reasonably stable
        sharpe_std = np.std(sharpe_ratios)
        assert sharpe_std < 2.0  # Standard deviation less than 2

        # Average Sharpe ratio should be reasonable
        avg_sharpe = np.mean(sharpe_ratios)
        assert -2.0 < avg_sharpe < 4.0  # Reasonable range

        print(f"Cross-validation Sharpe ratio: {avg_sharpe:.3f} ± {sharpe_std:.3f}")

    def test_parameter_sensitivity(self):
        """Test parameter sensitivity analysis"""
        returns = self.create_test_returns()

        from portfolio.src.optimization import PortfolioOptimizer

        optimizer = PortfolioOptimizer()

        # Test sensitivity to risk-free rate
        risk_free_rates = [0.0, 0.01, 0.02, 0.03, 0.04]
        portfolio_weights = []

        for rf_rate in risk_free_rates:
            config = {
                'objective': 'sharpe_ratio',
                'risk_free_rate': rf_rate,
                'constraints': {
                    'min_weight': 0.0,
                    'max_weight': 1.0
                }
            }

            portfolio = optimizer.optimize_portfolio(returns, config)
            portfolio_weights.append(list(portfolio.weights.values()))

        # Calculate weight correlations
        weight_correlations = []
        for i in range(len(portfolio_weights)):
            for j in range(i+1, len(portfolio_weights)):
                corr = np.corrcoef(portfolio_weights[i], portfolio_weights[j])[0, 1]
                weight_correlations.append(corr)

        # Test that weights are not too sensitive to risk-free rate
        avg_correlation = np.mean(weight_correlations)
        assert avg_correlation > 0.7  # High correlation between weights

        print(f"Average weight correlation: {avg_correlation:.3f}")
```

## 📈 Test Coverage and Reporting

### Coverage Configuration

```python
# .coveragerc - Coverage configuration
[run]
source = .
omit =
    */tests/*
    */venv/*
    */env/*
    setup.py
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

show_missing = true
precision = 2

[html]
directory = htmlcov
```

### Test Configuration

```ini
# pytest.ini - Pytest configuration
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --tb=short
    --strict-markers
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --junit-xml=test-results.xml

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    statistical: marks tests as statistical validation tests
    unit: marks tests as unit tests
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml - GitHub Actions workflow
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r docs/requirements.txt
        pip install pytest pytest-cov pytest-xdist

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --cov-report=xml

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v -m performance

    - name: Run statistical tests
      run: |
        pytest tests/statistical/ -v -m statistical

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

This comprehensive testing guide provides complete coverage of testing strategies for the quantitative trading system, including unit tests, integration tests, performance benchmarks, and statistical validation.

---

*Last Updated: 2024-01-15*