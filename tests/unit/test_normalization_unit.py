"""
Unit tests for data normalization methods.

Tests the mathematical correctness of normalization operations
including z-score, min-max, robust scaling, and financial-specific normalization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.normalization import DataNormalizer


class TestDataNormalizer:
    """Test DataNormalizer mathematical functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = DataNormalizer()

        # Create test data with known statistical properties
        np.random.seed(42)
        self.n_samples = 100

        # Generate test data
        base_prices = np.random.lognormal(4.0, 0.2, self.n_samples)
        base_volumes = np.random.lognormal(15.0, 0.5, self.n_samples)

        self.test_data = pd.DataFrame({
            'open': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'high': base_prices * np.random.uniform(1.00, 1.05, self.n_samples),
            'low': base_prices * np.random.uniform(0.95, 1.00, self.n_samples),
            'close': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'volume': base_volumes.astype(int)
        })

        # Simple series for testing
        self.test_series = pd.Series([1, 2, 3, 4, 5], name='test_values')

    def test_zscore_normalization_mathematical_correctness(self):
        """Test z-score normalization mathematical correctness."""
        # Test Series normalization
        normalized, params = self.normalizer.normalize_zscore(self.test_series)

        # Test mathematical properties
        expected_mean = 0.0
        expected_std = 1.0

        actual_mean = normalized.mean()
        actual_std = normalized.std()

        assert abs(actual_mean - expected_mean) < 1e-10
        assert abs(actual_std - expected_std) < 1e-10

        # Test parameter storage
        assert params['method'] == 'zscore'
        assert params['mean'] == self.test_series.mean()
        assert params['std'] == self.test_series.std()

        # Test denormalization
        denormalized = self.normalizer.denormalize_data(normalized, params)
        pd.testing.assert_series_equal(denormalized, self.test_series)

    def test_zscore_dataframe_normalization(self):
        """Test z-score normalization on DataFrame."""
        normalized, params = self.normalizer.normalize_zscore(self.test_data)

        # Test all numeric columns are normalized
        numeric_columns = self.test_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            col_mean = normalized[col].mean()
            col_std = normalized[col].std()

            assert abs(col_mean - 0.0) < 1e-10
            assert abs(col_std - 1.0) < 1e-10

        # Test parameter structure
        assert params['method'] == 'zscore'
        assert 'column_params' in params
        for col in numeric_columns:
            assert col in params['column_params']

    def test_minmax_normalization_mathematical_correctness(self):
        """Test min-max normalization mathematical correctness."""
        feature_range = (0, 1)

        # Test Series normalization
        normalized, params = self.normalizer.normalize_minmax(self.test_series, feature_range)

        # Test range bounds
        assert abs(normalized.min() - feature_range[0]) < 1e-10
        assert abs(normalized.max() - feature_range[1]) < 1e-10

        # Test parameter storage
        assert params['method'] == 'minmax'
        assert params['original_min'] == self.test_series.min()
        assert params['original_max'] == self.test_series.max()
        assert params['feature_range'] == feature_range

        # Test denormalization
        denormalized = self.normalizer.denormalize_data(normalized, params)
        pd.testing.assert_series_equal(denormalized, self.test_series)

    def test_custom_feature_range_normalization(self):
        """Test min-max normalization with custom feature range."""
        feature_range = (-1, 1)

        normalized, params = self.normalizer.normalize_minmax(self.test_series, feature_range)

        # Test custom range
        assert abs(normalized.min() - feature_range[0]) < 1e-10
        assert abs(normalized.max() - feature_range[1]) < 1e-10

        # Test denormalization with custom range
        denormalized = self.normalizer.denormalize_data(normalized, params)
        pd.testing.assert_series_equal(denormalized, self.test_series)

    def test_robust_normalization_mathematical_correctness(self):
        """Test robust normalization mathematical correctness."""
        # Test Series normalization
        normalized, params = self.normalizer.normalize_robust(self.test_series)

        # Test parameter storage
        assert params['method'] == 'robust'
        assert params['median'] == self.test_series.median()
        assert params['q1'] == self.test_series.quantile(0.25)
        assert params['q3'] == self.test_series.quantile(0.75)
        assert params['iqr'] == params['q3'] - params['q1']

        # Test denormalization
        denormalized = self.normalizer.denormalize_data(normalized, params)
        pd.testing.assert_series_equal(denormalized, self.test_series)

    def test_robust_scaling_with_outliers(self):
        """Test robust scaling with outliers."""
        # Create data with outliers
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 100])
        regular_data = pd.Series([1, 2, 3, 4, 5])

        # Normalize both
        normalized_outliers, _ = self.normalizer.normalize_robust(data_with_outliers)
        normalized_regular, _ = self.normalizer.normalize_robust(regular_data)

        # Robust scaling should be less affected by outliers
        outliers_std = normalized_outliers.std()
        regular_std = normalized_regular.std()

        # Outliers should not drastically change the scaling
        assert abs(outliers_std - regular_std) < 2.0

    def test_percentile_normalization(self):
        """Test percentile-based normalization."""
        percentiles = [10, 25, 50, 75, 90]

        normalized, params = self.normalizer.normalize_percentile(self.test_series, percentiles)

        # Test parameter storage
        assert params['method'] == 'percentile'
        assert params['percentiles'] == percentiles
        assert len(params['percentile_values']) == len(percentiles)

        # Test percentile values are correct
        expected_percentiles = [self.test_series.quantile(p/100) for p in percentiles]
        np.testing.assert_array_almost_equal(params['percentile_values'], expected_percentiles)

    def test_financial_data_normalization(self):
        """Test financial data normalization with price and volume."""
        normalized, params = self.normalizer.normalize_financial_data(self.test_data)

        # Test price columns are normalized
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in self.test_data.columns:
                # Should be approximately normalized
                col_mean = normalized[col].mean()
                col_std = normalized[col].std()
                assert abs(col_mean) < 1.0  # Should be close to zero
                assert 0.5 < col_std < 2.0  # Should be close to one

        # Test volume columns with log transformation
        volume_columns = ['volume']
        for col in volume_columns:
            if col in self.test_data.columns:
                # Check log transformation was applied
                col_params = params['column_transformations']['volume'][col]
                assert col_params['log_transformed'] == True

        # Test parameter structure
        assert params['method'] == 'zscore'  # Default method
        assert 'price_columns' in params
        assert 'volume_columns' in params
        assert 'column_transformations' in params

    def test_statistical_relationships_preservation(self):
        """Test that statistical relationships are preserved."""
        # Calculate original correlations
        original_corr = self.test_data[['open', 'high', 'low', 'close']].corr()

        # Normalize with preservation
        normalized, params = self.normalizer.normalize_financial_data(
            self.test_data, preserve_stats=True
        )

        # Calculate normalized correlations
        normalized_corr = normalized[['open', 'high', 'low', 'close']].corr()

        # Test correlation preservation (allowing some numerical error)
        for i in range(len(original_corr.columns)):
            for j in range(len(original_corr.columns)):
                if i != j:
                    original_val = original_corr.iloc[i, j]
                    normalized_val = normalized_corr.iloc[i, j]
                    assert abs(original_val - normalized_val) < 0.1  # Allow small differences

    def test_returns_normalization(self):
        """Test returns-based normalization."""
        normalized, params = self.normalizer.apply_returns_normalization(self.test_data)

        # Test returns calculation
        assert 'returns' in normalized.columns
        assert 'returns_normalized' in normalized.columns

        # Test normalized returns properties
        returns_mean = normalized['returns_normalized'].mean()
        returns_std = normalized['returns_normalized'].std()

        assert abs(returns_mean) < 1e-10  # Should be approximately zero
        assert abs(returns_std - 1.0) < 1e-10  # Should be approximately one

        # Test parameter storage
        assert params['method'] == 'returns_normalization'
        assert params['price_column'] == 'close'
        assert 'returns_params' in params

    def test_volatility_normalization(self):
        """Test volatility-based normalization."""
        window = 20

        normalized, params = self.normalizer.apply_volatility_normalization(
            self.test_data, window=window
        )

        # Test volatility normalization column exists
        vol_norm_col = 'close_vol_norm'
        assert vol_norm_col in normalized.columns

        # Test parameter storage
        assert params['method'] == 'volatility_normalization'
        assert params['window'] == window
        assert 'volatility_stats' in params

        # Test volatility statistics are reasonable
        vol_stats = params['volatility_stats']
        assert vol_stats['mean_vol'] > 0
        assert vol_stats['std_vol'] >= 0
        assert vol_stats['min_vol'] >= 0
        assert vol_stats['max_vol'] > 0

    def test_constant_series_handling(self):
        """Test handling of constant series (edge case)."""
        constant_series = pd.Series([5, 5, 5, 5, 5], name='constant')

        # Test z-score normalization
        normalized, params = self.normalizer.normalize_zscore(constant_series)
        # Should handle gracefully (replace inf with 0)
        assert not np.isinf(normalized).any()
        assert not normalized.isna().any()

        # Test min-max normalization
        normalized, params = self.normalizer.normalize_minmax(constant_series)
        # Should be middle of range
        expected_value = np.mean(params['feature_range'])
        assert all(abs(normalized - expected_value) < 1e-10)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5], name='with_nan')

        # Test normalization
        normalized, params = self.normalizer.normalize_zscore(series_with_nan)

        # NaN should be preserved or handled appropriately
        assert normalized.isna().sum() == series_with_nan.isna().sum()

    def test_infinite_value_handling(self):
        """Test handling of infinite values."""
        series_with_inf = pd.Series([1, 2, np.inf, 4, 5], name='with_inf')

        # Test normalization
        normalized, params = self.normalizer.normalize_zscore(series_with_inf)

        # Inf should be replaced with NaN
        assert np.isinf(normalized).sum() == 0
        assert normalized.isna().sum() > 0

    def test_normalization_summary(self):
        """Test normalization summary generation."""
        normalized, params = self.normalizer.normalize_zscore(self.test_data)

        summary = self.normalizer.get_normalization_summary(params)

        # Test summary structure
        assert summary['method'] == 'zscore'
        assert 'columns_transformed' in summary
        assert 'transformation_details' in summary

        # Test column details
        for col in summary['columns_transformed']:
            col_details = summary['transformation_details'][col]
            assert 'method' in col_details
            assert 'original_stats' in col_details

            # Test original stats
            stats = col_details['original_stats']
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats

    def test_different_normalization_methods(self):
        """Test different normalization methods on same data."""
        methods = ['zscore', 'minmax', 'robust']

        results = {}
        for method in methods:
            if method == 'zscore':
                normalized, params = self.normalizer.normalize_zscore(self.test_series)
            elif method == 'minmax':
                normalized, params = self.normalizer.normalize_minmax(self.test_series)
            elif method == 'robust':
                normalized, params = self.normalizer.normalize_robust(self.test_series)

            results[method] = {
                'normalized': normalized,
                'params': params
            }

        # Test each method produces different but valid results
        assert not results['zscore']['normalized'].equals(results['minmax']['normalized'])
        assert not results['zscore']['normalized'].equals(results['robust']['normalized'])

        # Test all can be denormalized to original
        for method, result in results.items():
            denormalized = self.normalizer.denormalize_data(
                result['normalized'], result['params']
            )
            pd.testing.assert_series_equal(denormalized, self.test_series)

    def test_column_selection_normalization(self):
        """Test normalization of specific columns."""
        columns_to_normalize = ['close', 'volume']

        normalized, params = self.normalizer.normalize_zscore(self.test_data, columns=columns_to_normalize)

        # Test only specified columns were normalized
        for col in columns_to_normalize:
            if col in self.test_data.columns:
                # Should be normalized (mean ≈ 0, std ≈ 1)
                assert abs(normalized[col].mean()) < 1e-10
                assert abs(normalized[col].std() - 1.0) < 1e-10

        # Test other columns were not modified
        for col in self.test_data.columns:
            if col not in columns_to_normalize:
                pd.testing.assert_series_equal(normalized[col], self.test_data[col])

    def test_reproducibility(self):
        """Test normalization reproducibility."""
        # Normalize same data twice
        normalized1, params1 = self.normalizer.normalize_zscore(self.test_series)
        normalized2, params2 = self.normalizer.normalize_zscore(self.test_series)

        # Results should be identical
        pd.testing.assert_series_equal(normalized1, normalized2)
        assert params1 == params2

    def test_numerical_precision(self):
        """Test numerical precision of normalization operations."""
        # Create data with extreme values
        extreme_series = pd.Series([1e-10, 1e-5, 1.0, 1e5, 1e10], name='extreme')

        # Test z-score normalization
        normalized, params = self.normalizer.normalize_zscore(extreme_series)

        # Should handle extreme values without overflow/underflow
        assert not np.isinf(normalized).any()
        assert not np.isnan(normalized).any()

        # Test denormalization precision
        denormalized = self.normalizer.denormalize_data(normalized, params)
        pd.testing.assert_series_equal(denormalized, extreme_series)