"""
Unit tests for walk-forward backtesting functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from portfolio.backtesting.walk_forward import (
    WalkForwardBacktester,
    BacktestConfig,
    BacktestResult,
    run_walk_forward_backtest
)


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.train_years == 1
        assert config.test_quarters == 1
        assert config.transaction_cost_bps == 7.5
        assert config.rebalance_frequency == 'quarterly'
        assert config.benchmark_symbol == 'SPY'
        assert config.include_equal_weight_baseline == True
        assert config.include_ml_overlay == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            train_years=2,
            test_quarters=2,
            transaction_cost_bps=10.0,
            rebalance_frequency='monthly'
        )
        assert config.train_years == 2
        assert config.test_quarters == 2
        assert config.transaction_cost_bps == 10.0
        assert config.rebalance_frequency == 'monthly'


class TestWalkForwardBacktester:
    """Test WalkForwardBacktester class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BacktestConfig()
        self.backtester = WalkForwardBacktester(self.config)
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']

    def create_sample_price_data(self, n_days=500):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays

        np.random.seed(42)
        price_data = {}
        for symbol in self.symbols:
            base_price = 100 + np.random.normal(0, 20)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            price_data[symbol] = prices

        return pd.DataFrame(price_data, index=dates)

    @patch('portfolio.backtesting.walk_forward.SimplePortfolioOptimizer')
    def test_initialization(self, mock_optimizer):
        """Test backtester initialization."""
        backtester = WalkForwardBacktester(self.config)
        assert backtester.config == self.config
        assert backtester.train_days == 252  # 1 year * 252 trading days
        assert backtester.test_days == 63    # 1 quarter * ~63 trading days

    def test_generate_walk_forward_windows(self):
        """Test walk-forward window generation."""
        prices = self.create_sample_price_data()
        windows = self.backtester._generate_walk_forward_windows(prices)

        # Check that windows are properly formatted
        assert len(windows) > 0
        for train_start, train_end, test_end in windows:
            assert isinstance(train_start, pd.Timestamp)
            assert isinstance(train_end, pd.Timestamp)
            assert isinstance(test_end, pd.Timestamp)
            assert train_start < train_end < test_end

        # Check that windows don't overlap in test periods
        for i in range(len(windows) - 1):
            assert windows[i][2] < windows[i + 1][0]

    def test_calculate_turnover(self):
        """Test turnover calculation."""
        old_weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        new_weights = {'AAPL': 0.5, 'GOOGL': 0.3, 'MSFT': 0.2}

        turnover = self.backtester._calculate_turnover(old_weights, new_weights)
        expected_turnover = (abs(0.5-0.4) + abs(0.3-0.3) + abs(0.2-0.3)) / 2
        assert abs(turnover - expected_turnover) < 1e-10

    def test_calculate_mean_variance_weights(self):
        """Test mean-variance weight calculation."""
        prices = self.create_sample_price_data()
        weights = self.backtester._calculate_mean_variance_weights(prices, self.symbols)

        assert isinstance(weights, dict)
        assert len(weights) == len(self.symbols)
        assert all(symbol in weights for symbol in self.symbols)

        # Weights should sum to 1 (approximately)
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-10

    def test_calculate_equal_weights(self):
        """Test ML overlay weight calculation with fallback."""
        prices = self.create_sample_price_data()
        weights = self.backtester._calculate_ml_overlay_weights(prices, self.symbols)

        # Should return weights that sum to 1
        assert isinstance(weights, dict)
        assert len(weights) <= len(self.symbols)
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-10

    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        metrics = self.backtester._calculate_comprehensive_metrics(returns)

        # Check essential metrics are present
        essential_metrics = [
            'total_return', 'annual_return', 'annual_volatility',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'win_rate', 'loss_rate'
        ]

        for metric in essential_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_generate_report(self):
        """Test report generation."""
        # Create mock results
        mock_result = BacktestResult(
            dates=[pd.Timestamp('2021-01-01')],
            returns=pd.Series([0.01, 0.02, -0.01]),
            weights_history=pd.DataFrame({'AAPL': [0.4], 'GOOGL': [0.3], 'MSFT': [0.3]}),
            metrics={'sharpe_ratio': 1.5, 'annual_return': 0.12, 'max_drawdown': -0.05},
            transaction_costs=0.001,
            turnover=0.25
        )

        results = {
            'optimized': mock_result,
            'equal_weight': mock_result,
            'comparison': {}
        }

        report = self.backtester.generate_report(results)

        # Check report contains expected sections
        assert "WALK-FORWARD BACKTESTING REPORT" in report
        assert "Configuration:" in report
        assert "OPTIMIZED STRATEGY" in report
        assert "Performance Metrics:" in report
        assert "Trading Metrics:" in report

    def test_run_backtest_insufficient_data(self):
        """Test backtest with insufficient data."""
        prices = pd.DataFrame()  # Empty data
        windows = self.backtester._generate_walk_forward_windows(prices)
        assert len(windows) == 0


class TestConvenienceFunction:
    """Test convenience functions."""

    def test_run_walk_forward_backtest_with_default_config(self):
        """Test convenience function with default configuration."""
        with patch('portfolio.backtesting.walk_forward.WalkForwardBacktester') as mock_class:
            mock_backtester = Mock()
            mock_class.return_value = mock_backtester
            mock_backtester.run_backtest.return_value = {'test': 'result'}

            result = run_walk_forward_backtest(['AAPL'], '2020-01-01', '2021-01-01')

            mock_class.assert_called_once()
            mock_backtester.run_backtest.assert_called_once_with(
                ['AAPL'], '2020-01-01', '2021-01-01'
            )
            assert result == {'test': 'result'}

    def test_run_walk_forward_backtest_with_custom_config(self):
        """Test convenience function with custom configuration."""
        config = BacktestConfig(train_years=2, transaction_cost_bps=10.0)

        with patch('portfolio.backtesting.walk_forward.WalkForwardBacktester') as mock_class:
            mock_backtester = Mock()
            mock_class.return_value = mock_backtester
            mock_backtester.run_backtest.return_value = {'test': 'result'}

            result = run_walk_forward_backtest(
                ['AAPL'], '2020-01-01', '2021-01-01', config
            )

            mock_class.assert_called_once_with(config)
            assert result == {'test': 'result'}


if __name__ == '__main__':
    pytest.main([__file__])