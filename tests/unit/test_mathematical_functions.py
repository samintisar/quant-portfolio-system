"""
Unit tests for mathematical functions used in quantitative trading.

Tests financial mathematics, statistical calculations, and portfolio optimization
functions with high precision requirements.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy import stats
import time


@dataclass
class TestDataPoint:
    """Test data structure for financial calculations."""
    value: float
    weight: float = 1.0


class MathematicalFunctions:
    """Core mathematical functions for quantitative analysis."""

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a series of returns."""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # Daily adjustment
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if not returns:
            return 0.0

        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return min(drawdowns) if len(drawdowns) > 0 else 0.0

    @staticmethod
    def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical method."""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        return -np.percentile(returns_array, (1 - confidence) * 100)

    @staticmethod
    def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, (1 - confidence) * 100)
        tail_returns = returns_array[returns_array <= var_threshold]
        return -np.mean(tail_returns) if len(tail_returns) > 0 else 0.0

    @staticmethod
    def calculate_portfolio_variance(weights: List[float], covariance_matrix: np.ndarray) -> float:
        """Calculate portfolio variance given weights and covariance matrix."""
        weights_array = np.array(weights)
        return float(weights_array.T @ covariance_matrix @ weights_array)

    @staticmethod
    def calculate_beta(returns: List[float], market_returns: List[float]) -> float:
        """Calculate beta against market returns."""
        if len(returns) != len(market_returns) or len(returns) < 2:
            return 1.0

        returns_array = np.array(returns)
        market_array = np.array(market_returns)
        covariance = np.cov(returns_array, market_array)[0, 1]
        market_variance = np.cov(market_array, market_array)[0, 0]  # Use consistent variance calculation

        return covariance / market_variance if market_variance > 0 else 1.0


class TestMathematicalFunctions:
    """Test suite for mathematical functions."""

    def setup_method(self):
        """Set up test data."""
        self.math_functions = MathematicalFunctions()

        # Sample returns data
        self.positive_returns = [0.01, 0.02, 0.015, 0.005, 0.025]
        self.mixed_returns = [0.01, -0.02, 0.015, -0.005, 0.025]
        self.negative_returns = [-0.01, -0.02, -0.015, -0.005, -0.025]
        self.empty_returns = []

        # Portfolio data
        self.weights = [0.3, 0.4, 0.3]
        self.covariance_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.06]
        ])

        # Market data
        self.market_returns = [0.008, 0.012, 0.010, 0.006, 0.015]

    def test_calculate_sharpe_ratio_positive(self):
        """Test Sharpe ratio calculation with positive returns."""
        result = self.math_functions.calculate_sharpe_ratio(self.positive_returns)
        assert isinstance(result, float)
        assert result > 0  # Should be positive with all positive returns

        # Test with different risk-free rates
        result_no_rf = self.math_functions.calculate_sharpe_ratio(self.positive_returns, 0.0)
        result_with_rf = self.math_functions.calculate_sharpe_ratio(self.positive_returns, 0.05)
        assert result_no_rf > result_with_rf  # Higher risk-free rate reduces Sharpe

    def test_calculate_sharpe_ratio_mixed(self):
        """Test Sharpe ratio calculation with mixed returns."""
        result = self.math_functions.calculate_sharpe_ratio(self.mixed_returns)
        assert isinstance(result, float)
        # Should handle volatility correctly

    def test_calculate_sharpe_ratio_edge_cases(self):
        """Test Sharpe ratio calculation edge cases."""
        # Empty returns
        result = self.math_functions.calculate_sharpe_ratio(self.empty_returns)
        assert result == 0.0

        # Single return
        result = self.math_functions.calculate_sharpe_ratio([0.01])
        assert result == 0.0

        # Zero volatility returns
        constant_returns = [0.01, 0.01, 0.01]
        result = self.math_functions.calculate_sharpe_ratio(constant_returns)
        assert isinstance(result, float)

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        result = self.math_functions.calculate_max_drawdown(self.mixed_returns)
        assert isinstance(result, float)
        assert result <= 0  # Drawdown should be negative or zero

        # Test with all positive returns
        result_positive = self.math_functions.calculate_max_drawdown(self.positive_returns)
        assert result_positive <= 0

        # Test with empty returns
        result_empty = self.math_functions.calculate_max_drawdown(self.empty_returns)
        assert result_empty == 0.0

    def test_calculate_var(self):
        """Test Value at Risk calculation."""
        result = self.math_functions.calculate_var(self.mixed_returns)
        assert isinstance(result, float)
        assert result >= 0  # VaR should be non-negative

        # Test different confidence levels
        var_95 = self.math_functions.calculate_var(self.mixed_returns, 0.95)
        var_99 = self.math_functions.calculate_var(self.mixed_returns, 0.99)
        assert var_99 >= var_95  # Higher confidence = higher VaR

        # Test with empty returns
        result_empty = self.math_functions.calculate_var(self.empty_returns)
        assert result_empty == 0.0

    def test_calculate_cvar(self):
        """Test Conditional Value at Risk calculation."""
        result = self.math_functions.calculate_cvar(self.mixed_returns)
        assert isinstance(result, float)
        assert result >= 0  # CVaR should be non-negative

        # CVaR should be >= VaR
        var_95 = self.math_functions.calculate_var(self.mixed_returns, 0.95)
        cvar_95 = self.math_functions.calculate_cvar(self.mixed_returns, 0.95)
        assert cvar_95 >= var_95

        # Test with empty returns
        result_empty = self.math_functions.calculate_cvar(self.empty_returns)
        assert result_empty == 0.0

    def test_calculate_portfolio_variance(self):
        """Test portfolio variance calculation."""
        result = self.math_functions.calculate_portfolio_variance(self.weights, self.covariance_matrix)
        assert isinstance(result, float)
        assert result > 0  # Variance should be positive

        # Test with equal weights
        equal_weights = [1/3, 1/3, 1/3]
        result_equal = self.math_functions.calculate_portfolio_variance(equal_weights, self.covariance_matrix)
        assert isinstance(result_equal, float)
        assert result_equal > 0

    def test_calculate_beta(self):
        """Test beta calculation."""
        result = self.math_functions.calculate_beta(self.mixed_returns, self.market_returns)
        assert isinstance(result, float)

        # Test with perfect correlation (asset vs itself should have beta of 1)
        same_returns = self.market_returns
        perfect_corr = self.math_functions.calculate_beta(same_returns, same_returns)
        assert abs(perfect_corr - 1.0) < 1e-10  # Should be exactly 1

        # Test with different lengths (should handle gracefully)
        result_diff_length = self.math_functions.calculate_beta([0.01, 0.02], [0.008])
        assert result_diff_length == 1.0  # Default value for invalid input

    def test_performance_requirements(self):
        """Test that functions meet performance requirements."""
        # Generate large dataset for performance testing
        large_returns = np.random.normal(0.001, 0.02, 10000).tolist()
        large_market = np.random.normal(0.0008, 0.015, 10000).tolist()

        # Test Sharpe ratio performance
        start_time = time.time()
        for _ in range(1000):
            self.math_functions.calculate_sharpe_ratio(large_returns[:100])
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        assert avg_time < 0.001, f"Sharpe ratio too slow: {avg_time:.6f}s"

        # Test portfolio variance performance
        start_time = time.time()
        for _ in range(1000):
            self.math_functions.calculate_portfolio_variance(self.weights, self.covariance_matrix)
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        assert avg_time < 0.001, f"Portfolio variance too slow: {avg_time:.6f}s"

    def test_numerical_precision(self):
        """Test numerical precision of calculations."""
        # Test with very small numbers
        small_returns = [1e-10, 2e-10, -1e-10]
        result = self.math_functions.calculate_sharpe_ratio(small_returns)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

        # Test with very large numbers
        large_returns = [1e6, 2e6, -1e6]
        result = self.math_functions.calculate_sharpe_ratio(large_returns)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_statistical_validity(self):
        """Test statistical validity of results."""
        # Generate normally distributed returns
        np.random.seed(42)
        normal_returns = np.random.normal(0.001, 0.02, 1000).tolist()

        # VaR at 95% should be approximately 1.645 std dev below mean
        var_95 = self.math_functions.calculate_var(normal_returns, 0.95)
        expected_var = -(np.mean(normal_returns) - 1.645 * np.std(normal_returns))
        assert abs(var_95 - expected_var) < 0.01, f"VaR not statistically valid: {var_95} vs {expected_var}"

        # Test that diversification reduces portfolio variance
        single_asset_var = self.math_functions.calculate_portfolio_variance([1.0, 0.0, 0.0], self.covariance_matrix)
        diversified_var = self.math_functions.calculate_portfolio_variance([0.33, 0.33, 0.34], self.covariance_matrix)
        assert diversified_var < single_asset_var, "Diversification should reduce variance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])