#!/usr/bin/env python3
"""
Simplified integration tests for portfolio optimization system.

Tests core functionality with minimal complexity.
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.api.main import app
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.optimizer.optimizer import PortfolioOptimizer
from portfolio.performance.calculator import PerformanceCalculator
from portfolio.performance.risk_metrics import RiskMetricsCalculator
from fastapi.testclient import TestClient


class TestPortfolioOptimizationIntegration:
    """Test core portfolio optimization functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.client = TestClient(app)
        self.optimizer = PortfolioOptimizer()
        self.performance_calc = PerformanceCalculator()
        self.risk_calc = RiskMetricsCalculator()

    def test_api_health_check(self):
        """Test API health endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data

    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "operational"

    def test_mean_variance_optimization_with_mock_data(self):
        """Test Mean-Variance optimization with mock data."""
        # Create mock assets with synthetic returns
        assets = []
        np.random.seed(42)  # For reproducible results

        for i, symbol in enumerate(['AAPL', 'GOOGL', 'MSFT', 'JPM', 'JNJ']):
            # Generate synthetic returns
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))

            asset = Asset(
                symbol=symbol,
                name=symbol,
                sector=['Tech', 'Tech', 'Tech', 'Financial', 'Healthcare'][i]
            )
            asset.returns = returns
            assets.append(asset)

        # Create basic constraints
        constraints = PortfolioConstraints(
            max_position_size=0.30,
            max_sector_concentration=0.60,
            max_volatility=0.25,
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
        assert result.success is True
        assert result.optimal_weights is not None
        assert len(result.optimal_weights) == len(assets)
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01
        assert result.execution_time > 0

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))

        # Calculate basic risk metrics
        var = self.risk_calc.calculate_var(returns, 0.95)
        cvar = self.risk_calc.calculate_cvar(returns, 0.95)
        volatility = self.risk_calc.calculate_volatility(returns)

        # Validate results
        assert var > 0
        assert cvar >= var  # CVaR should be >= VaR
        assert volatility > 0

        # Test all metrics function
        all_metrics = self.risk_calc.calculate_all_metrics(returns)
        assert 'var_95' in all_metrics
        assert 'cvar_95' in all_metrics
        assert 'volatility' in all_metrics

    def test_performance_calculation(self):
        """Test performance calculation."""
        # Create sample returns and weights
        np.random.seed(42)
        asset_returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.001, 0.025, 252),
            'MSFT': np.random.normal(0.0008, 0.018, 252)
        })

        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}

        # Calculate portfolio returns
        portfolio_returns = (asset_returns * pd.Series(weights)).sum(axis=1)

        # Calculate performance metrics
        performance = self.performance_calc.calculate_performance_metrics(
            portfolio_returns,
            risk_free_rate=0.02
        )

        # Validate results
        assert performance['annual_return'] is not None
        assert performance['annual_volatility'] is not None
        assert performance['sharpe_ratio'] is not None

    def test_cvar_optimization(self):
        """Test CVaR optimization."""
        # Create mock assets
        assets = []
        np.random.seed(42)

        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'JPM']:
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

        assert result.success is True
        assert result.optimal_weights is not None
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01

    def test_optimization_methods_info(self):
        """Test getting optimization methods info."""
        info = self.optimizer.get_optimizer_info()

        assert 'available_methods' in info
        assert 'default_method' in info
        assert isinstance(info['available_methods'], list)
        assert len(info['available_methods']) > 0

    def test_constraints_validation(self):
        """Test constraints validation."""
        # Test with empty assets list
        with pytest.raises(Exception):
            self.optimizer.optimize(
                assets=[],
                constraints=PortfolioConstraints(),
                method="mean_variance"
            )

    def test_correlation_matrix_calculation(self):
        """Test correlation matrix calculation."""
        # Create sample returns matrix
        np.random.seed(42)
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.025, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100)
        })

        corr_matrix = self.risk_calc.calculate_correlation_matrix(returns_matrix)

        # Validate correlation matrix
        assert len(corr_matrix) == 3
        for symbol in corr_matrix:
            assert corr_matrix[symbol][symbol] == 1.0  # Diagonal should be 1
            for other_symbol in corr_matrix[symbol]:
                assert -1 <= corr_matrix[symbol][other_symbol] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])