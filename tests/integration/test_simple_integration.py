#!/usr/bin/env python3
"""
Simple integration tests for portfolio optimization system.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.api.main import app
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.optimizer.optimizer import PortfolioOptimizer
from portfolio.config import get_config
from fastapi.testclient import TestClient


class TestSimpleIntegration:
    """Simple integration tests to verify basic functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
        self.optimizer = PortfolioOptimizer()

    def test_basic_optimization_workflow(self):
        """Test basic Mean-Variance optimization workflow."""
        # Create mock assets
        assets = []
        np.random.seed(42)

        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            asset = Asset(symbol=symbol, name=symbol, sector='Technology')
            asset.returns = returns
            assets.append(asset)

        # Create constraints
        constraints = PortfolioConstraints(
            max_position_size=0.50,
            max_sector_concentration=0.80,
            max_volatility=0.30,
            risk_free_rate=0.02
        )

        # Run optimization
        result = self.optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method="mean_variance",
            objective="sharpe"
        )

        # Validate results
        assert result.success
        assert result.optimal_weights is not None
        assert len(result.optimal_weights) == len(assets)
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01
        assert result.execution_time > 0

    def test_api_optimization_endpoint(self):
        """Test API optimization endpoint."""
        # Create optimization request
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "constraints": {
                "max_position_size": 0.5,
                "max_sector_concentration": 0.8,
                "max_volatility": 0.3,
                "risk_free_rate": 0.02
            },
            "method": "mean_variance",
            "objective": "sharpe",
            "lookback_period": 252
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"]
        assert "optimal_weights" in result
        assert "execution_time" in result
        assert result["execution_time"] > 0

    def test_black_litterman_optimization(self):
        """Test Black-Litterman optimization."""
        # Create mock assets
        assets = []
        np.random.seed(42)

        for symbol in ['AAPL', 'GOOGL']:
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            asset = Asset(symbol=symbol, name=symbol, sector='Technology')
            asset.returns = returns
            assets.append(asset)

        constraints = PortfolioConstraints(
            max_position_size=0.60,
            max_volatility=0.25,
            risk_free_rate=0.02
        )

        # Create market views
        from portfolio.models.views import MarketViewCollection, MarketView
        market_views = MarketViewCollection(views=[
            MarketView(asset_symbol="AAPL", expected_return=0.12, confidence=0.7)
        ])

        # Run Black-Litterman optimization
        result = self.optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method="black_litterman",
            objective="sharpe",
            market_views=market_views
        )

        assert result.success
        assert result.optimal_weights is not None

    def test_cvar_optimization(self):
        """Test CVaR optimization."""
        # Create mock assets
        assets = []
        np.random.seed(42)

        for symbol in ['AAPL', 'GOOGL', 'JPM']:
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            asset = Asset(symbol=symbol, name=symbol, sector='Mixed')
            asset.returns = returns
            assets.append(asset)

        constraints = PortfolioConstraints(
            max_position_size=0.40,
            max_volatility=0.25,
            risk_free_rate=0.02
        )

        # Run CVaR optimization
        result = self.optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method="cvar",
            objective="min_cvar"
        )

        assert result.success
        assert result.optimal_weights is not None

    def test_constraint_validation(self):
        """Test constraint validation."""
        # Test valid constraints
        valid_constraints = PortfolioConstraints(
            max_position_size=0.25,
            max_sector_concentration=0.40,
            max_volatility=0.25,
            risk_free_rate=0.02
        )

        # Should not raise an exception
        assert valid_constraints.max_position_size == 0.25

    def test_configuration_loading(self):
        """Test configuration loading."""
        config = get_config()

        # Check basic structure
        assert hasattr(config, 'optimization')
        assert hasattr(config, 'data')
        assert hasattr(config, 'performance')

        # Check optimization config
        opt_config = config.optimization
        assert hasattr(opt_config, 'risk_free_rate')
        assert opt_config.risk_free_rate >= 0

    def test_api_health_and_root(self):
        """Test API health and root endpoints."""
        # Test health endpoint
        health_response = self.client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"

        # Test root endpoint
        root_response = self.client.get("/")
        assert root_response.status_code == 200
        root_data = root_response.json()
        assert "message" in root_data
        assert "version" in root_data

    def test_optimizer_info(self):
        """Test optimizer info endpoint."""
        info = self.optimizer.get_optimizer_info()

        assert 'available_methods' in info
        assert 'default_method' in info
        assert isinstance(info['available_methods'], list)
        assert len(info['available_methods']) > 0

        # Check that essential methods are available
        methods = info['available_methods']
        assert 'mean_variance' in methods
        assert 'black_litterman' in methods
        assert 'cvar' in methods

    def test_multiple_objectives(self):
        """Test different optimization objectives."""
        # Create mock assets
        assets = []
        np.random.seed(42)

        for symbol in ['AAPL', 'GOOGL']:
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))
            asset = Asset(symbol=symbol, name=symbol, sector='Technology')
            asset.returns = returns
            assets.append(asset)

        constraints = PortfolioConstraints(
            max_position_size=0.60,
            max_volatility=0.25,
            risk_free_rate=0.02
        )

        # Test different objectives
        objectives = ['sharpe', 'min_variance', 'max_return']

        for objective in objectives:
            result = self.optimizer.optimize(
                assets=assets,
                constraints=constraints,
                method="mean_variance",
                objective=objective
            )

            assert result.success
            assert result.optimal_weights is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])