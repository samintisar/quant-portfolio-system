"""
Simple tests for the simplified portfolio optimization system.

Focuses on core functionality without overengineered test patterns.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from portfolio_simple import SimplePortfolioOptimizer
from performance_simple import SimplePerformanceCalculator
from config_loader import load_config, get_config, update_config


class TestSimplePortfolioOptimizer(unittest.TestCase):
    """Test the simplified portfolio optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = SimplePortfolioOptimizer()

    def test_init(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.risk_free_rate, 0.02)
        self.assertEqual(self.optimizer.trading_days_per_year, 252)

    def test_calculate_returns(self):
        """Test return calculation."""
        # Create sample price data
        prices = pd.Series([100, 101, 102, 101, 100])
        returns = self.optimizer.calculate_returns(prices.to_frame('TEST'))

        self.assertEqual(len(returns), 4)  # One less than prices
        self.assertTrue(all(returns >= -1))  # No returns less than -100%

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        # Create sample returns
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        metrics = self.optimizer.calculate_portfolio_metrics(returns)

        self.assertIn('total_return', metrics)
        self.assertIn('annual_return', metrics)
        self.assertIn('annual_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

    @patch('yfinance.download')
    def test_fetch_data(self, mock_download):
        """Test data fetching."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'SPY': [100, 101, 102],
            'AAPL': [150, 152, 151]
        })
        mock_download.return_value = mock_data

        data = self.optimizer.fetch_data(['SPY', 'AAPL'])

        self.assertEqual(data.shape, (3, 2))
        self.assertListEqual(list(data.columns), ['SPY', 'AAPL'])

    def test_mean_variance_optimize(self):
        """Test mean-variance optimization."""
        # Create sample returns
        returns = pd.DataFrame({
            'A': [0.01, 0.02, -0.01],
            'B': [0.02, -0.01, 0.01]
        })

        result = self.optimizer.mean_variance_optimize(returns)

        self.assertIn('weights', result)
        self.assertIn('expected_return', result)
        self.assertIn('expected_volatility', result)
        self.assertIn('sharpe_ratio', result)

        # Check weights sum to 1 (approximately)
        weights = result['weights']
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)


class TestSimplePerformanceCalculator(unittest.TestCase):
    """Test the simplified performance calculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.calc = SimplePerformanceCalculator()

    def test_init(self):
        """Test calculator initialization."""
        self.assertEqual(self.calc.risk_free_rate, 0.02)
        self.assertEqual(self.calc.trading_days_per_year, 252)

    def test_calculate_returns(self):
        """Test return calculation."""
        prices = pd.Series([100, 101, 102, 101, 100])
        returns = self.calc.calculate_returns(prices)

        self.assertEqual(len(returns), 4)
        self.assertAlmostEqual(returns.iloc[0], 0.01, places=5)

    def test_calculate_portfolio_returns(self):
        """Test portfolio return calculation."""
        price_data = pd.DataFrame({
            'A': [100, 101, 102],
            'B': [200, 202, 204]
        })
        weights = {'A': 0.6, 'B': 0.4}

        portfolio_returns = self.calc.calculate_portfolio_returns(price_data, weights)

        self.assertEqual(len(portfolio_returns), 2)
        self.assertTrue(all(portfolio_returns >= -1))

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        metrics = self.calc.calculate_metrics(returns)

        self.assertIn('total_return', metrics)
        self.assertIn('annual_return', metrics)
        self.assertIn('annual_volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)

    def test_calculate_metrics_with_benchmark(self):
        """Test metrics calculation with benchmark."""
        portfolio_returns = pd.Series([0.01, -0.02, 0.03])
        benchmark_returns = pd.Series([0.005, -0.01, 0.02])

        metrics = self.calc.calculate_metrics(portfolio_returns, benchmark_returns)

        self.assertIn('beta', metrics)
        self.assertIn('alpha', metrics)
        self.assertIn('information_ratio', metrics)

    def test_generate_report(self):
        """Test report generation."""
        metrics = {
            'total_return': 0.05,
            'annual_return': 0.10,
            'annual_volatility': 0.15,
            'sharpe_ratio': 0.5,
            'max_drawdown': -0.02,
            'win_rate': 0.6
        }

        report = self.calc.generate_report(metrics)
        self.assertIsInstance(report, str)
        self.assertIn('Portfolio Performance Report', report)


class TestConfigLoader(unittest.TestCase):
    """Test the configuration loader."""

    def test_get_default_config(self):
        """Test default configuration."""
        from config_loader import get_default_config
        config = get_default_config()

        self.assertIn('portfolio', config)
        self.assertIn('data', config)
        self.assertIn('api', config)

        # Check some default values
        self.assertEqual(config['portfolio']['risk_free_rate'], 0.02)
        self.assertEqual(config['portfolio']['trading_days_per_year'], 252)

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_content = """
portfolio:
  risk_free_rate: 0.03
  max_position_size: 0.1

data:
  min_data_points: 500
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            self.assertEqual(config['portfolio']['risk_free_rate'], 0.03)
            self.assertEqual(config['portfolio']['max_position_size'], 0.1)
            self.assertEqual(config['data']['min_data_points'], 500)
        finally:
            os.unlink(temp_file)

    def test_update_config(self):
        """Test updating configuration."""
        # Reset to default
        update_config({'portfolio': {'risk_free_rate': 0.02}})

        # Update configuration
        update_config({'portfolio': {'risk_free_rate': 0.04}})
        config = get_config()

        self.assertEqual(config['portfolio']['risk_free_rate'], 0.04)


class TestIntegration(unittest.TestCase):
    """Integration tests for the simplified system."""

    @patch('yfinance.download')
    def test_end_to_end_optimization(self, mock_download):
        """Test end-to-end optimization workflow."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'SPY': [100, 101, 102, 103, 104],
            'AAPL': [150, 152, 151, 153, 155]
        })
        mock_download.return_value = mock_data

        # Create optimizer and optimize
        optimizer = SimplePortfolioOptimizer()
        result = optimizer.optimize_portfolio(['SPY', 'AAPL'])

        # Check result structure
        self.assertIn('optimization', result)
        self.assertIn('performance', result)
        self.assertIn('assets', result)
        self.assertIn('timestamp', result)

        # Check optimization results
        opt_result = result['optimization']
        self.assertIn('weights', opt_result)
        self.assertIn('expected_return', opt_result)
        self.assertIn('expected_volatility', opt_result)
        self.assertIn('sharpe_ratio', opt_result)

        # Check performance results
        perf_result = result['performance']
        self.assertIn('total_return', perf_result)
        self.assertIn('annual_return', perf_result)
        self.assertIn('sharpe_ratio', perf_result)

    @patch('yfinance.download')
    def test_efficient_frontier_calculation(self, mock_download):
        """Test efficient frontier calculation."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'A': [100, 101, 102],
            'B': [200, 202, 204]
        })
        mock_download.return_value = mock_data

        optimizer = SimplePortfolioOptimizer()
        frontier = optimizer.get_efficient_frontier(['A', 'B'], n_points=5)

        self.assertEqual(len(frontier), 5)
        for point in frontier:
            self.assertIn('return', point)
            self.assertIn('volatility', point)
            self.assertIn('sharpe_ratio', point)
            self.assertIn('weights', point)


if __name__ == '__main__':
    unittest.main()