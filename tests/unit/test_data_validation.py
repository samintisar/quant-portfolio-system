"""
Unit tests for data validation in quantitative trading system.

Tests financial data validation, market data quality checks, and portfolio
constraint validation with strict requirements.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time


@dataclass
class ValidationResult:
    """Result of a data validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


class DataValidator:
    """Data validation utilities for quantitative trading."""

    @staticmethod
    def validate_price_data(prices: List[float], min_price: float = 0.01, max_price: float = 1000000) -> ValidationResult:
        """Validate price data for reasonableness."""
        errors = []
        warnings = []
        metrics = {}

        if not prices:
            errors.append("Price data is empty")
            return ValidationResult(False, errors, warnings, metrics)

        prices_array = np.array(prices)

        # Check for negative prices
        negative_count = np.sum(prices_array < 0)
        if negative_count > 0:
            errors.append(f"Found {negative_count} negative prices")

        # Check for zero prices
        zero_count = np.sum(prices_array == 0)
        if zero_count > 0:
            warnings.append(f"Found {zero_count} zero prices")

        # Check for unreasonably high prices
        high_count = np.sum(prices_array > max_price)
        if high_count > 0:
            warnings.append(f"Found {high_count} prices above {max_price}")

        # Check for unreasonably low prices
        low_count = np.sum(prices_array < min_price)
        if low_count > 0:
            warnings.append(f"Found {low_count} prices below {min_price}")

        # Calculate basic statistics
        metrics = {
            'count': len(prices),
            'mean': float(np.mean(prices_array)),
            'std': float(np.std(prices_array)),
            'min': float(np.min(prices_array)),
            'max': float(np.max(prices_array)),
            'negative_count': int(negative_count),
            'zero_count': int(zero_count),
            'high_count': int(high_count),
            'low_count': int(low_count)
        }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)

    @staticmethod
    def validate_returns_data(returns: List[float], max_return: float = 1.0) -> ValidationResult:
        """Validate returns data for outliers and anomalies."""
        errors = []
        warnings = []
        metrics = {}

        if not returns:
            errors.append("Returns data is empty")
            return ValidationResult(False, errors, warnings, metrics)

        returns_array = np.array(returns)

        # Check for extreme returns using vectorized operations
        extreme_mask = np.abs(returns_array) > max_return
        extreme_positive = np.sum(returns_array[extreme_mask] > 0)
        extreme_negative = np.sum(returns_array[extreme_mask] < 0)

        if extreme_positive > 0:
            warnings.append(f"Found {extreme_positive} returns > {max_return*100}%")

        if extreme_negative > 0:
            warnings.append(f"Found {extreme_negative} returns < {-max_return*100}%")

        # Check for missing and infinite values
        nan_mask = np.isnan(returns_array)
        inf_mask = np.isinf(returns_array)
        nan_count = np.sum(nan_mask)
        inf_count = np.sum(inf_mask)

        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values in returns")

        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values in returns")

        # Calculate basic statistics efficiently
        valid_returns = returns_array[~nan_mask & ~inf_mask]
        if len(valid_returns) > 0:
            metrics = {
                'count': len(returns),
                'mean': float(np.mean(valid_returns)),
                'std': float(np.std(valid_returns)),
                'min': float(np.min(valid_returns)),
                'max': float(np.max(valid_returns)),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'extreme_positive': int(extreme_positive),
                'extreme_negative': int(extreme_negative),
                'skewness': float(stats.skew(valid_returns)) if len(valid_returns) > 2 else 0.0,
                'kurtosis': float(stats.kurtosis(valid_returns)) if len(valid_returns) > 3 else 0.0
            }
        else:
            metrics = {
                'count': len(returns),
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'extreme_positive': int(extreme_positive),
                'extreme_negative': int(extreme_negative),
                'skewness': 0.0,
                'kurtosis': 0.0
            }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)

    @staticmethod
    def validate_portfolio_weights(weights: List[float], tolerance: float = 1e-6) -> ValidationResult:
        """Validate portfolio weights sum to 1 and are within bounds."""
        errors = []
        warnings = []
        metrics = {}

        if not weights:
            errors.append("Weights list is empty")
            return ValidationResult(False, errors, warnings, metrics)

        weights_array = np.array(weights)

        # Check for negative weights
        negative_weights = np.sum(weights_array < 0)
        if negative_weights > 0:
            errors.append(f"Found {negative_weights} negative weights")

        # Check for weights > 1
        large_weights = np.sum(weights_array > 1)
        if large_weights > 0:
            errors.append(f"Found {large_weights} weights > 1")

        # Check sum of weights
        weight_sum = np.sum(weights_array)
        if abs(weight_sum - 1.0) > tolerance:
            errors.append(f"Weights sum to {weight_sum:.6f}, not 1.0")

        # Check for concentration
        max_weight = np.max(weights_array)
        if max_weight > 0.5:
            warnings.append(f"High concentration: max weight = {max_weight:.3f}")

        # Calculate concentration metrics
        metrics = {
            'count': len(weights),
            'sum': float(weight_sum),
            'max_weight': float(max_weight),
            'min_weight': float(np.min(weights_array)),
            'negative_count': int(negative_weights),
            'large_count': int(large_weights),
            'herfindahl_index': float(np.sum(weights_array ** 2))
        }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)

    @staticmethod
    def validate_covariance_matrix(cov_matrix: np.ndarray, tolerance: float = 1e-8) -> ValidationResult:
        """Validate covariance matrix properties."""
        errors = []
        warnings = []
        metrics = {}

        if cov_matrix is None:
            errors.append("Covariance matrix is None")
            return ValidationResult(False, errors, warnings, metrics)

        # Check if matrix is square
        if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
            errors.append("Covariance matrix must be square")
            return ValidationResult(False, errors, warnings, metrics)

        n = cov_matrix.shape[0]

        # Check for negative variances (diagonal elements)
        variances = np.diag(cov_matrix)
        negative_variances = np.sum(variances < 0)
        if negative_variances > 0:
            errors.append(f"Found {negative_variances} negative variances")

        # Check for zero variances
        zero_variances = np.sum(variances == 0)
        if zero_variances > 0:
            warnings.append(f"Found {zero_variances} zero variances")

        # Check symmetry
        if not np.allclose(cov_matrix, cov_matrix.T, atol=tolerance):
            errors.append("Covariance matrix is not symmetric")

        # Check positive semi-definiteness
        try:
            eigenvalues = np.linalg.eigvals(cov_matrix)
            negative_eigenvalues = np.sum(eigenvalues < -tolerance)
            if negative_eigenvalues > 0:
                errors.append(f"Found {negative_eigenvalues} negative eigenvalues - not PSD")
        except np.linalg.LinAlgError:
            errors.append("Failed to compute eigenvalues")

        # Calculate condition number
        try:
            cond_number = np.linalg.cond(cov_matrix)
            if cond_number > 1e10:
                warnings.append(f"High condition number: {cond_number:.2e}")
        except np.linalg.LinAlgError:
            warnings.append("Could not compute condition number")

        metrics = {
            'shape': cov_matrix.shape,
            'min_variance': float(np.min(variances)),
            'max_variance': float(np.max(variances)),
            'mean_variance': float(np.mean(variances)),
            'negative_variances': int(negative_variances),
            'zero_variances': int(zero_variances),
            'condition_number': float(cond_number) if 'cond_number' in locals() else float('inf')
        }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)

    @staticmethod
    def validate_time_series_data(dates: List[datetime], values: List[float], freq: str = 'D') -> ValidationResult:
        """Validate time series data for consistency and gaps."""
        errors = []
        warnings = []
        metrics = {}

        if len(dates) != len(values):
            errors.append(f"Date and value lengths mismatch: {len(dates)} vs {len(values)}")
            return ValidationResult(False, errors, warnings, metrics)

        if not dates:
            errors.append("Time series is empty")
            return ValidationResult(False, errors, warnings, metrics)

        # Sort by date
        sorted_indices = np.argsort(dates)
        sorted_dates = [dates[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # Check for duplicates
        unique_dates = list(set(sorted_dates))
        if len(unique_dates) != len(sorted_dates):
            errors.append(f"Found {len(sorted_dates) - len(unique_dates)} duplicate dates")

        # Check for gaps
        if len(sorted_dates) > 1:
            time_diffs = [(sorted_dates[i] - sorted_dates[i-1]).days for i in range(1, len(sorted_dates))]
            expected_diff = 1 if freq == 'D' else (7 if freq == 'W' else 21)  # Weekly/Monthly

            gaps = sum(1 for diff in time_diffs if diff > expected_diff)
            if gaps > 0:
                warnings.append(f"Found {gaps} gaps in time series")

            # Check for inconsistent intervals
            unique_diffs = set(time_diffs)
            if len(unique_diffs) > 1:
                warnings.append(f"Inconsistent time intervals: {unique_diffs}")

        # Check for outliers in values
        values_array = np.array(sorted_values)
        q1, q3 = np.percentile(values_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = np.sum((values_array < lower_bound) | (values_array > upper_bound))
        if outliers > 0:
            warnings.append(f"Found {outliers} outliers using IQR method")

        metrics = {
            'count': len(sorted_dates),
            'start_date': sorted_dates[0].isoformat(),
            'end_date': sorted_dates[-1].isoformat(),
            'duplicate_dates': int(len(sorted_dates) - len(unique_dates)),
            'gaps': int(gaps) if 'gaps' in locals() else 0,
            'outliers': int(outliers),
            'date_range_days': (sorted_dates[-1] - sorted_dates[0]).days
        }

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, metrics)


class TestDataValidator:
    """Test suite for data validation functions."""

    def setup_method(self):
        """Set up test data."""
        self.validator = DataValidator()

        # Sample price data
        self.valid_prices = [100.0, 101.5, 102.3, 101.8, 103.2]
        self.invalid_prices = [100.0, -50.0, 0.0, 2000000.0]
        self.empty_prices = []

        # Sample returns data
        self.valid_returns = [0.01, -0.02, 0.015, -0.005, 0.025]
        self.invalid_returns = [0.01, 2.0, -1.5, np.nan, np.inf]
        self.empty_returns = []

        # Sample portfolio weights
        self.valid_weights = [0.3, 0.4, 0.3]
        self.invalid_weights = [0.5, 0.6, -0.1]
        self.sum_invalid_weights = [0.5, 0.5, 0.1]

        # Sample covariance matrix
        self.valid_cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.06]
        ])
        self.invalid_cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, -0.01, 0.03],  # Negative variance
            [0.01, 0.03, 0.06]
        ])

        # Sample time series data
        base_date = datetime(2023, 1, 1)
        self.valid_dates = [base_date + timedelta(days=i) for i in range(5)]
        self.valid_values = [100.0, 101.5, 102.3, 101.8, 103.2]
        self.gappy_dates = [base_date + timedelta(days=i) for i in [0, 1, 5, 6, 7]]
        self.duplicate_dates = [base_date + timedelta(days=i) for i in [0, 1, 1, 2, 3]]

    def test_validate_price_data_valid(self):
        """Test price validation with valid data."""
        result = self.validator.validate_price_data(self.valid_prices)
        assert result.is_valid
        assert len(result.errors) == 0
        assert 'count' in result.metrics
        assert result.metrics['count'] == len(self.valid_prices)

    def test_validate_price_data_invalid(self):
        """Test price validation with invalid data."""
        result = self.validator.validate_price_data(self.invalid_prices)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert 'negative_count' in result.metrics
        assert result.metrics['negative_count'] > 0

    def test_validate_price_data_empty(self):
        """Test price validation with empty data."""
        result = self.validator.validate_price_data(self.empty_prices)
        assert not result.is_valid
        assert "Price data is empty" in result.errors

    def test_validate_returns_data_valid(self):
        """Test returns validation with valid data."""
        result = self.validator.validate_returns_data(self.valid_returns)
        assert result.is_valid
        assert len(result.errors) == 0
        assert 'skewness' in result.metrics
        assert 'kurtosis' in result.metrics

    def test_validate_returns_data_invalid(self):
        """Test returns validation with invalid data."""
        result = self.validator.validate_returns_data(self.invalid_returns)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.metrics['nan_count'] > 0
        assert result.metrics['inf_count'] > 0

    def test_validate_portfolio_weights_valid(self):
        """Test portfolio weights validation with valid data."""
        result = self.validator.validate_portfolio_weights(self.valid_weights)
        assert result.is_valid
        assert len(result.errors) == 0
        assert abs(result.metrics['sum'] - 1.0) < 1e-6

    def test_validate_portfolio_weights_invalid(self):
        """Test portfolio weights validation with invalid data."""
        result = self.validator.validate_portfolio_weights(self.invalid_weights)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.metrics['negative_count'] > 0

    def test_validate_portfolio_weights_sum_error(self):
        """Test portfolio weights validation with sum not equal to 1."""
        result = self.validator.validate_portfolio_weights(self.sum_invalid_weights)
        assert not result.is_valid
        assert "Weights sum to" in result.errors[0]

    def test_validate_covariance_matrix_valid(self):
        """Test covariance matrix validation with valid data."""
        result = self.validator.validate_covariance_matrix(self.valid_cov_matrix)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.metrics['negative_variances'] == 0

    def test_validate_covariance_matrix_invalid(self):
        """Test covariance matrix validation with invalid data."""
        result = self.validator.validate_covariance_matrix(self.invalid_cov_matrix)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.metrics['negative_variances'] > 0

    def test_validate_time_series_data_valid(self):
        """Test time series validation with valid data."""
        result = self.validator.validate_time_series_data(self.valid_dates, self.valid_values)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.metrics['duplicate_dates'] == 0

    def test_validate_time_series_data_gaps(self):
        """Test time series validation with gaps."""
        result = self.validator.validate_time_series_data(self.gappy_dates, self.valid_values)
        assert result.is_valid  # Gaps are warnings, not errors
        assert len(result.warnings) > 0
        assert result.metrics['gaps'] > 0

    def test_validate_time_series_data_duplicates(self):
        """Test time series validation with duplicate dates."""
        result = self.validator.validate_time_series_data(self.duplicate_dates, self.valid_values)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.metrics['duplicate_dates'] > 0

    def test_performance_requirements(self):
        """Test that validation meets performance requirements."""
        # Generate large dataset for performance testing
        large_prices = np.random.uniform(50, 200, 10000).tolist()
        large_returns = np.random.normal(0.001, 0.02, 10000).tolist()
        large_weights = np.random.dirichlet(np.ones(100)).tolist()
        large_cov = np.random.uniform(0.01, 0.1, (100, 100))
        large_cov = (large_cov + large_cov.T) / 2  # Make symmetric

        # Test price validation performance
        start_time = time.time()
        for _ in range(100):
            self.validator.validate_price_data(large_prices[:1000])
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.001, f"Price validation too slow: {avg_time:.6f}s"

        # Test returns validation performance
        start_time = time.time()
        for _ in range(100):
            self.validator.validate_returns_data(large_returns[:1000])
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.002, f"Returns validation too slow: {avg_time:.6f}s"

        # Test covariance matrix validation performance
        start_time = time.time()
        for _ in range(50):
            self.validator.validate_covariance_matrix(large_cov[:20, :20])
        end_time = time.time()
        avg_time = (end_time - start_time) / 50
        assert avg_time < 0.001, f"Covariance validation too slow: {avg_time:.6f}s"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with single values
        single_price = [100.0]
        result = self.validator.validate_price_data(single_price)
        assert result.is_valid

        # Test with very small numbers
        tiny_prices = [1e-10, 2e-10, 3e-10]
        result = self.validator.validate_price_data(tiny_prices)
        assert len(result.warnings) > 0  # Should warn about low prices

        # Test with very large numbers
        huge_prices = [1e8, 2e8, 3e8]
        result = self.validator.validate_price_data(huge_prices)
        assert len(result.warnings) > 0  # Should warn about high prices

        # Test weights with very small tolerance
        almost_one_weights = [0.333333, 0.333333, 0.333334]
        result = self.validator.validate_portfolio_weights(almost_one_weights, tolerance=1e-5)
        assert result.is_valid

        # Test with singular covariance matrix
        singular_matrix = np.array([[1, 1], [1, 1]])
        result = self.validator.validate_covariance_matrix(singular_matrix)
        assert len(result.warnings) > 0  # Should warn about high condition number


# Import scipy stats for skewness/kurtosis
from scipy import stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"])