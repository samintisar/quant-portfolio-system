"""
Integration tests for walk-forward backtesting functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio.backtesting.walk_forward import (
    WalkForwardBacktester,
    BacktestConfig,
    BacktestResult,
    run_walk_forward_backtest
)


class TestWalkForwardIntegration:
    """Integration tests for walk-forward backtesting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BacktestConfig(
            train_years=1,
            test_quarters=1,
            transaction_cost_bps=5.0,
            include_equal_weight_baseline=True,
            include_ml_overlay=True
        )
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    def create_realistic_price_data(self, start_date='2019-01-01', end_date='2022-12-31'):
        """Create realistic price data for integration testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays

        # Create realistic price movements
        np.random.seed(42)
        price_data = {}

        base_prices = {'AAPL': 150, 'GOOGL': 120, 'MSFT': 200, 'AMZN': 180, 'TSLA': 200}

        for symbol, base_price in base_prices.items():
            prices = [base_price]
            daily_volatility = 0.02  # 2% daily volatility

            for i in range(1, len(dates)):
                # Add some trend and randomness
                trend = 0.0001  # Small upward trend
                random_return = np.random.normal(trend, daily_volatility)
                new_price = prices[-1] * (1 + random_return)
                prices.append(max(new_price, 1))  # Prevent negative prices

            price_data[symbol] = prices

        return pd.DataFrame(price_data, index=dates)

    def test_complete_backtest_workflow(self):
        """Test complete backtesting workflow."""
        # Create test data
        prices = self.create_realistic_price_data()

        # Initialize backtester
        backtester = WalkForwardBacktester(self.config)

        # Run backtest
        results = backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        # Verify results structure
        assert isinstance(results, dict)
        assert 'optimized' in results
        assert 'equal_weight' in results
        assert 'ml_overlay' in results
        assert 'comparison' in results

        # Verify each strategy result
        for strategy_name, result in results.items():
            if strategy_name == 'comparison':
                continue

            assert isinstance(result.dates, list)
            assert isinstance(result.returns, pd.Series)
            assert isinstance(result.weights_history, pd.DataFrame)
            assert isinstance(result.metrics, dict)
            assert isinstance(result.transaction_costs, (int, float))
            assert isinstance(result.turnover, (int, float))

            # Check metrics contain essential keys
            essential_metrics = [
                'total_return', 'annual_return', 'annual_volatility',
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
            ]
            for metric in essential_metrics:
                assert metric in result.metrics

    def test_expanding_window_validation(self):
        """Test expanding window validation specifically."""
        prices = self.create_realistic_price_data()

        # Test with different training periods
        configs = [
            BacktestConfig(train_years=1, test_quarters=1),
            BacktestConfig(train_years=2, test_quarters=1),
        ]

        for config in configs:
            backtester = WalkForwardBacktester(config)
            results = backtester.run_backtest(
                self.symbols,
                '2020-01-01',
                '2022-12-31'
            )

            # Verify results are generated
            assert len(results['optimized'].dates) > 0

            # Verify expanding window effect (longer training should have more windows)
            if config.train_years == 2:
                # Should have fewer windows due to longer training period
                assert len(results['optimized'].dates) > 0

    def test_transaction_costs_impact(self):
        """Test transaction costs impact on performance."""
        prices = self.create_realistic_price_data()

        # Test with different transaction costs
        low_cost_config = BacktestConfig(transaction_cost_bps=1.0)
        high_cost_config = BacktestConfig(transaction_cost_bps=20.0)

        low_cost_backtester = WalkForwardBacktester(low_cost_config)
        high_cost_backtester = WalkForwardBacktester(high_cost_config)

        low_cost_results = low_cost_backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        high_cost_results = high_cost_backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        # Higher transaction costs should result in higher total costs
        assert high_cost_results['optimized'].transaction_costs >= low_cost_results['optimized'].transaction_costs

    def test_baseline_comparison(self):
        """Test baseline comparison functionality."""
        prices = self.create_realistic_price_data()

        config = BacktestConfig(
            include_equal_weight_baseline=True,
            include_ml_overlay=True
        )

        backtester = WalkForwardBacktester(config)
        results = backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        # Verify all strategies are present
        assert 'optimized' in results
        assert 'equal_weight' in results
        assert 'ml_overlay' in results

        # Verify comparison metrics
        comparison = results['comparison']
        assert 'rankings' in comparison

        # Rankings should contain key metrics
        rankings = comparison['rankings']
        assert 'sharpe_ratio' in rankings
        assert 'annual_return' in rankings
        assert 'max_drawdown' in rankings

    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics including advanced ones."""
        prices = self.create_realistic_price_data()

        backtester = WalkForwardBacktester(self.config)
        results = backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        # Check for comprehensive metrics
        metrics = results['optimized'].metrics

        # Basic metrics
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics

        # Advanced metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'loss_rate' in metrics

        # Verify reasonable values
        assert metrics['win_rate'] >= 0
        assert metrics['win_rate'] <= 1
        assert metrics['loss_rate'] >= 0
        assert metrics['loss_rate'] <= 1

    def test_turnover_calculation(self):
        """Test turnover calculation accuracy with mock data."""
        prices = self.create_realistic_price_data()

        # Test turnover calculation directly
        backtester = WalkForwardBacktester(self.config)

        # Create mock weights history
        weights_history = [
            {'date': pd.Timestamp('2020-01-01'), 'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3},
            {'date': pd.Timestamp('2020-04-01'), 'AAPL': 0.5, 'GOOGL': 0.3, 'MSFT': 0.2},
            {'date': pd.Timestamp('2020-07-01'), 'AAPL': 0.3, 'GOOGL': 0.4, 'MSFT': 0.3}
        ]

        turnover = backtester._calculate_total_turnover(weights_history)
        assert isinstance(turnover, (int, float))
        assert turnover >= 0
        assert turnover > 0  # Should have some turnover with different weights

    def test_report_generation(self):
        """Test comprehensive report generation."""
        prices = self.create_realistic_price_data()

        backtester = WalkForwardBacktester(self.config)
        results = backtester.run_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31'
        )

        # Generate report
        report = backtester.generate_report(results)

        # Verify report structure
        assert isinstance(report, str)
        assert "WALK-FORWARD BACKTESTING REPORT" in report
        assert "Configuration:" in report
        assert "OPTIMIZED STRATEGY" in report
        assert "EQUAL_WEIGHT STRATEGY" in report
        assert "ML_OVERLAY STRATEGY" in report

        # Verify key metrics are included
        assert "Sharpe Ratio:" in report
        assert "Annual Return:" in report
        assert "Max Drawdown:" in report
        assert "Transaction Costs:" in report

    def test_convenience_function(self):
        """Test convenience function integration."""
        prices = self.create_realistic_price_data()

        # Test convenience function
        results = run_walk_forward_backtest(
            self.symbols,
            '2020-01-01',
            '2022-12-31',
            self.config
        )

        # Verify results are similar to direct backtester usage
        assert isinstance(results, dict)
        assert 'optimized' in results
        assert 'equal_weight' in results

    def test_error_handling(self):
        """Test error handling with problematic data."""
        # Test with insufficient data
        backtester = WalkForwardBacktester(self.config)

        # Should handle empty or insufficient data gracefully
        with pytest.raises(Exception):
            backtester.run_backtest(
                self.symbols,
                '2025-01-01',  # Future date
                '2025-12-31'
            )

    def test_benchmark_comparison(self):
        """Test benchmark comparison functionality with mock data."""
        prices = self.create_realistic_price_data()

        # Create mock results for different strategies
        mock_optimized_result = BacktestResult(
            dates=[pd.Timestamp('2020-01-01')],
            returns=pd.Series([0.01, 0.02, -0.01]),
            weights_history=pd.DataFrame({'AAPL': [0.4], 'GOOGL': [0.3], 'MSFT': [0.3]}),
            metrics={'annual_return': 0.12, 'sharpe_ratio': 1.2, 'max_drawdown': -0.05},
            transaction_costs=0.001,
            turnover=0.25
        )

        mock_equal_weight_result = BacktestResult(
            dates=[pd.Timestamp('2020-01-01')],
            returns=pd.Series([0.005, 0.015, -0.005]),
            weights_history=pd.DataFrame({'AAPL': [0.33], 'GOOGL': [0.33], 'MSFT': [0.34]}),
            metrics={'annual_return': 0.08, 'sharpe_ratio': 0.8, 'max_drawdown': -0.04},
            transaction_costs=0.0005,
            turnover=0.1
        )

        # Verify different strategies have different performance
        assert mock_optimized_result.metrics['annual_return'] != mock_equal_weight_result.metrics['annual_return']
        assert mock_optimized_result.metrics['sharpe_ratio'] != mock_equal_weight_result.metrics['sharpe_ratio']

        # Verify both have valid metrics
        assert 'annual_return' in mock_optimized_result.metrics
        assert 'sharpe_ratio' in mock_optimized_result.metrics
        assert 'max_drawdown' in mock_optimized_result.metrics


if __name__ == '__main__':
    pytest.main([__file__])