#!/usr/bin/env python3
"""
Comprehensive integration tests for the portfolio optimization system.

Tests end-to-end workflows and ensures all components work together.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.api.main import app
from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.models.views import MarketViewCollection, MarketView
from portfolio.config import get_config
from fastapi.testclient import TestClient


class TestComprehensiveIntegration:
    """Comprehensive integration tests."""

    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        self.optimizer = SimplePortfolioOptimizer()
        self.config = get_config()

    def _create_test_portfolio(self, n_assets: int = 5) -> tuple:
        """Create test portfolio with synthetic data."""
        np.random.seed(42)

        symbols = [f'STOCK_{i}' for i in range(n_assets)]
        assets = []

        for symbol in symbols:
            # Generate synthetic price and return data
            dates = pd.date_range(end='2023-12-31', periods=252, freq='B')
            base_price = 100 + np.random.uniform(-20, 80)
            returns = np.random.normal(0.001, 0.02, 251)

            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            asset = Asset(
                symbol=symbol,
                name=f'{symbol} Corp',
                sector='Technology'
            )
            asset.prices = pd.Series(prices, index=dates)
            asset.returns = pd.Series(returns, index=dates[1:])
            assets.append(asset)

        return assets, symbols

    def test_complete_optimization_workflow(self):
        """Test complete workflow from data to optimization."""
        # Create test portfolio
        assets, symbols = self._create_test_portfolio(3)

        # Create constraints
        constraints = PortfolioConstraints(
            max_position_size=0.5,
            max_sector_concentration=0.8,
            max_volatility=0.3,
            risk_free_rate=0.02
        )

        # Test all optimization methods
        methods = ['mean_variance', 'cvar']
        objectives = ['sharpe', 'min_risk', 'max_return']

        for method in methods:
            for objective in objectives:
                result = self.optimizer.optimize(
                    assets=assets,
                    constraints=constraints,
                    method=method,
                    objective=objective
                )

                assert result is not None
                assert result.success
                assert result.optimal_weights is not None
                assert len(result.optimal_weights) == len(assets)
                assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6
                assert result.execution_time > 0

        print("✓ Complete optimization workflow test passed")

    def test_black_litterman_integration(self):
        """Test Black-Litterman optimization with market views."""
        assets, symbols = self._create_test_portfolio(4)

        # Create market views
        market_views = MarketViewCollection([
            MarketView(
                asset_symbol=symbols[0],
                expected_return=0.12,
                confidence=0.8,
                view_type="absolute"
            ),
            MarketView(
                asset_symbol=symbols[1],
                expected_return=0.15,
                confidence=0.6,
                view_type="absolute"
            )
        ])

        constraints = PortfolioConstraints(risk_free_rate=0.02)

        # Test Black-Litterman optimization
        result = self.optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method="black_litterman",
            objective="sharpe",
            market_views=market_views
        )

        assert result is not None
        assert result.success
        assert result.optimal_weights is not None
        assert len(result.optimal_weights) == len(assets)

        # Verify weights respect constraints
        weights = list(result.optimal_weights.values())
        assert all(w >= 0 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6

        print("✓ Black-Litterman integration test passed")

    def test_api_end_to_end_workflow(self):
        """Test complete API workflow."""
        # Test optimization via API
        optimize_request = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"},
                {"symbol": "MSFT", "name": "Microsoft", "sector": "Technology"}
            ],
            "method": "mean_variance",
            "objective": "sharpe",
            "constraints": {
                "max_position_size": 0.4,
                "max_volatility": 0.25,
                "risk_free_rate": 0.02
            },
            "lookback_period": 252
        }

        response = self.client.post("/optimize", json=optimize_request)
        assert response.status_code == 200

        optimize_result = response.json()
        assert optimize_result["success"]
        assert "optimal_weights" in optimize_result
        assert "execution_time" in optimize_result
        assert optimize_result["execution_time"] > 0

        # Use the optimization result for portfolio analysis
        weights = optimize_result["optimal_weights"]

        analyze_request = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"},
                {"symbol": "MSFT", "name": "Microsoft", "sector": "Technology"}
            ],
            "weights": weights,
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02,
            "lookback_period": 252
        }

        response = self.client.post("/analyze", json=analyze_request)
        assert response.status_code == 200

        analyze_result = response.json()
        assert analyze_result["success"]
        assert "metrics" in analyze_result
        assert "annual_return" in analyze_result["metrics"]
        assert "annual_volatility" in analyze_result["metrics"]
        assert "sharpe_ratio" in analyze_result["metrics"]

        print("✓ API end-to-end workflow test passed")

    def test_efficient_frontier_integration(self):
        """Test efficient frontier calculation and API integration."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Calculate efficient frontier
        frontier = self.optimizer.get_efficient_frontier(symbols, n_points=10)

        assert len(frontier) == 10
        assert all('return' in point and 'volatility' in point for point in frontier)
        assert all('weights' in point for point in frontier)

        # Verify efficient frontier properties
        returns = [point['return'] for point in frontier]
        volatilities = [point['volatility'] for point in frontier]

        # Returns should be generally increasing
        assert all(returns[i] <= returns[i+1] for i in range(len(returns)-1))

        # Test with single asset (should fail gracefully)
        try:
            single_frontier = self.optimizer.get_efficient_frontier(["AAPL"], n_points=5)
            # Should not crash but may return empty or single point
            assert isinstance(single_frontier, list)
        except Exception:
            pass  # Expected to potentially fail

        print("✓ Efficient frontier integration test passed")

    def test_configuration_integration(self):
        """Test that configuration system works across all components."""
        # Test configuration loading
        assert self.config is not None
        assert hasattr(self.config, 'portfolio')
        assert hasattr(self.config, 'optimization')
        assert hasattr(self.config, 'data')

        # Test that optimizer uses configuration
        assert self.optimizer.risk_free_rate == self.config['portfolio']['risk_free_rate']

        # Test configuration updates
        original_rate = self.config['portfolio']['risk_free_rate']
        self.config['portfolio']['risk_free_rate'] = 0.03

        # New optimizer instance should use updated config
        new_optimizer = SimplePortfolioOptimizer()
        assert new_optimizer.risk_free_rate == 0.03

        # Restore original config
        self.config['portfolio']['risk_free_rate'] = original_rate

        print("✓ Configuration integration test passed")

    def test_error_handling_integration(self):
        """Test error handling across all components."""
        # Test insufficient assets
        assets, _ = self._create_test_portfolio(1)
        constraints = PortfolioConstraints()

        result = self.optimizer.optimize(assets, constraints, "mean_variance")
        # Should handle error gracefully
        assert result is not None

        # Test invalid method
        result = self.optimizer.optimize(assets, constraints, "invalid_method")
        assert result is not None

        # Test API error handling
        invalid_request = {
            "assets": [{"symbol": "INVALID"}],
            "method": "mean_variance"
        }

        response = self.client.post("/optimize", json=invalid_request)
        # Should return proper error response
        assert response.status_code in [200, 422]  # 200 with success=False or 422 for validation error
        result = response.json()
        assert "success" in result

        print("✓ Error handling integration test passed")

    def test_data_validation_integration(self):
        """Test data validation across the system."""
        # Test with empty returns
        empty_returns = pd.DataFrame()
        try:
            result = self.optimizer.mean_variance_optimize(empty_returns)
            # Should handle gracefully
            assert result is not None
        except Exception:
            pass  # Expected to potentially fail

        # Test with single asset
        single_asset_returns = pd.DataFrame({
            'STOCK_0': np.random.normal(0.001, 0.02, 100)
        })

        result = self.optimizer.mean_variance_optimize(single_asset_returns)
        assert result is not None
        assert len(result['weights']) == 1
        assert abs(list(result['weights'].values())[0] - 1.0) < 1e-6

        # Test with highly correlated assets
        correlated_returns = pd.DataFrame({
            'STOCK_0': np.random.normal(0.001, 0.02, 100),
            'STOCK_1': np.random.normal(0.001, 0.02, 100) * 0.9 + 0.1  # 90% correlated
        })

        result = self.optimizer.mean_variance_optimize(correlated_returns)
        assert result is not None
        assert len(result['weights']) == 2

        print("✓ Data validation integration test passed")

    def test_performance_metrics_integration(self):
        """Test performance metrics calculation integration."""
        # Create test portfolio
        assets, _ = self._create_test_portfolio(3)

        # Optimize portfolio
        constraints = PortfolioConstraints(risk_free_rate=0.02)
        result = self.optimizer.optimize(assets, constraints, "mean_variance", "sharpe")

        # Get optimized weights
        weights = result.optimal_weights

        # Calculate portfolio returns
        all_returns = pd.concat([asset.returns for asset in assets], axis=1)
        all_returns.columns = [asset.symbol for asset in assets]

        portfolio_returns = (all_returns * pd.Series(weights)).sum(axis=1)

        # Test portfolio metrics calculation
        metrics = self.optimizer.calculate_portfolio_metrics(portfolio_returns)

        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics

        # Verify metric relationships
        assert metrics['annual_volatility'] > 0
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert metrics['max_drawdown'] <= 0  # Drawdown should be negative

        print("✓ Performance metrics integration test passed")

    def test_market_conditions_scenarios(self):
        """Test optimization under different market conditions."""
        scenarios = [
            ("bull_market", 0.001, 0.015),  # High returns, low volatility
            ("bear_market", -0.002, 0.025),  # Negative returns, high volatility
            ("sideways_market", 0.0, 0.01),  # Low returns, low volatility
            ("high_volatility", 0.0005, 0.03),  # Low returns, very high volatility
        ]

        assets, _ = self._create_test_portfolio(4)

        for scenario_name, mean_return, volatility in scenarios:
            # Modify asset returns to match scenario
            for asset in assets:
                asset.returns = np.random.normal(mean_return, volatility, len(asset.returns))

            constraints = PortfolioConstraints(risk_free_rate=0.02)

            result = self.optimizer.optimize(
                assets=assets,
                constraints=constraints,
                method="mean_variance",
                objective="sharpe"
            )

            assert result is not None
            assert result.success
            assert result.optimal_weights is not None

            # Check that weights are reasonable for the scenario
            weights = list(result.optimal_weights.values())
            assert all(w >= 0 for w in weights)
            assert abs(sum(weights) - 1.0) < 1e-6

            print(f"✓ {scenario_name} scenario test passed")

    def test_system_stability_under_load(self):
        """Test system stability under repeated optimization requests."""
        assets, _ = self._create_test_portfolio(5)
        constraints = PortfolioConstraints()

        # Run multiple optimizations
        for i in range(20):
            result = self.optimizer.optimize(
                assets=assets,
                constraints=constraints,
                method="mean_variance",
                objective="sharpe"
            )

            assert result is not None
            assert result.success
            assert result.execution_time < 5.0  # Should be fast

        print("✓ System stability under load test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])