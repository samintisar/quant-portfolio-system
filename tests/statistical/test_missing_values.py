"""
Statistical validation tests for missing value imputation methods
Tests mathematical correctness and statistical properties of missing value handling
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import math


class TestMissingValueImputation:
    """Test suite for statistical validation of missing value imputation methods"""

    @pytest.fixture
    def test_datasets(self):
        """Generate test datasets with controlled missing value patterns"""
        np.random.seed(42)  # For reproducible results

        # Dataset 1: Completely random missing values
        data_random = np.random.normal(100, 15, 1000)
        mask_random = np.random.random(1000) < 0.1  # 10% missing
        data_random[mask_random] = np.nan

        # Dataset 2: Missing values correlated with value (MNAR - Missing Not At Random)
        data_mnar = np.random.normal(100, 15, 1000)
        mask_mnar = data_mnar > np.percentile(data_mnar, 80)  # Missing in high values
        data_mnar[mask_mnar] = np.nan

        # Dataset 3: Time series with consecutive missing values
        data_timeseries = np.sin(np.linspace(0, 4*np.pi, 1000)) * 50 + 100
        mask_timeseries = np.zeros(1000, dtype=bool)
        # Create blocks of missing values
        for i in range(0, 1000, 100):
            mask_timeseries[i:i+5] = True  # 5 consecutive missing values every 100
        data_timeseries[mask_timeseries] = np.nan

        return {
            'random_missing': pd.Series(data_random),
            'mnar_missing': pd.Series(data_mnar),
            'timeseries_missing': pd.Series(data_timeseries)
        }

    def test_mean_imputation_statistical_properties(self, test_datasets):
        """
        Test: Mean imputation preserves mean but affects variance
        Expected: Mean preserved exactly, variance reduced appropriately
        """
        dataset = test_datasets['random_missing']
        original_mean = dataset.mean()
        original_var = dataset.var()
        original_std = dataset.std()

        # Apply mean imputation
        imputed_mean = dataset.fillna(original_mean)

        # Statistical validation
        assert imputed_mean.mean() == pytest.approx(original_mean, rel=1e-10), \
            "Mean imputation should preserve original mean exactly"

        # Variance should be reduced
        imputed_var = imputed_mean.var()
        assert imputed_var < original_var, \
            "Mean imputation should reduce variance"

        # Calculate expected variance reduction
        n_missing = dataset.isna().sum()
        n_total = len(dataset)
        missing_ratio = n_missing / n_total

        # Expected variance after mean imputation
        # For mean imputation: variance = (n_observed/n_total) * original_var
        # where n_observed = n_total - n_missing
        n_observed = n_total - n_missing
        expected_var = original_var * (n_observed / n_total)
        assert imputed_var == pytest.approx(expected_var, rel=1e-10), \
            "Variance reduction should match theoretical expectation"

    def test_median_imputation_robustness(self, test_datasets):
        """
        Test: Median imputation is robust to outliers
        Expected: Median preserved, less affected by extreme values
        """
        # Create dataset with outliers
        data_with_outliers = np.concatenate([
            np.random.normal(100, 10, 950),
            np.random.normal(1000, 50, 50)  # Outliers
        ])

        # Add missing values
        mask = np.random.random(1000) < 0.1
        data_with_outliers[mask] = np.nan
        dataset = pd.Series(data_with_outliers)

        original_median = dataset.median()
        original_mean = dataset.mean()

        # Apply median imputation
        imputed_median = dataset.fillna(original_median)

        # Statistical validation
        assert imputed_median.median() == pytest.approx(original_median, rel=1e-10), \
            "Median imputation should preserve original median exactly"

        # Compare with mean imputation for robustness
        imputed_mean = dataset.fillna(original_mean)

        # Median imputation should be less affected by outliers
        median_std = imputed_median.std()
        mean_std = imputed_mean.std()

        # In presence of outliers, median imputation should result in lower std
        assert median_std <= mean_std, \
            "Median imputation should be more robust to outliers than mean imputation"

    def test_mode_imputation_categorical_data(self):
        """
        Test: Mode imputation for categorical data
        Expected: Mode preserved, maintains data distribution
        """
        # Create categorical dataset
        categories = ['A', 'B', 'C', 'D']
        probabilities = [0.4, 0.3, 0.2, 0.1]
        data = np.random.choice(categories, size=1000, p=probabilities)

        # Add missing values
        mask = np.random.random(1000) < 0.15
        data[mask] = np.nan
        dataset = pd.Series(data)

        original_mode = dataset.mode()[0]

        # Apply mode imputation
        imputed_mode = dataset.fillna(original_mode)

        # Statistical validation
        assert imputed_mode.mode()[0] == original_mode, \
            "Mode imputation should preserve original mode"

        # Check distribution preservation
        original_counts = dataset.value_counts(normalize=True)
        imputed_counts = imputed_mode.value_counts(normalize=True)

        # Mode frequency should increase
        assert imputed_counts[original_mode] > original_counts[original_mode], \
            "Mode frequency should increase after imputation"

        # Other category frequencies should decrease proportionally
        for category in categories:
            if category != original_mode:
                assert imputed_counts[category] <= original_counts[category], \
                    f"Frequency of category {category} should not increase"

    def test_linear_interpolation_time_series(self, test_datasets):
        """
        Test: Linear interpolation for time series data
        Expected: Preserves trends, smooth transitions
        """
        dataset = test_datasets['timeseries_missing']

        # Apply linear interpolation
        interpolated = dataset.interpolate(method='linear')

        # Check that missing values are filled
        assert interpolated.isna().sum() == 0, \
            "Linear interpolation should fill all missing values"

        # Check that interpolation preserves monotonicity where appropriate
        # In a sinusoidal pattern, interpolated values should follow the trend
        non_missing_indices = dataset.dropna().index
        for i in range(len(non_missing_indices) - 1):
            idx1, idx2 = non_missing_indices[i], non_missing_indices[i + 1]
            if idx2 - idx1 > 1:  # There are missing values between these points
                # Check that interpolated values are between the known points
                for j in range(idx1 + 1, idx2):
                    min_val = min(dataset[idx1], dataset[idx2])
                    max_val = max(dataset[idx1], dataset[idx2])
                    assert min_val <= interpolated[j] <= max_val, \
                        f"Interpolated value at index {j} should be between known values"

    def test_forward_fill_time_series(self, test_datasets):
        """
        Test: Forward fill (last observation carried forward) for time series
        Expected: Preserves last known value, appropriate for step functions
        """
        dataset = test_datasets['timeseries_missing']

        # Apply forward fill
        filled = dataset.fillna(method='ffill')

        # Check that missing values are filled (except leading missing)
        remaining_missing = filled.isna().sum()
        assert remaining_missing <= dataset.isna().sum(), \
            "Forward fill should reduce missing values"

        # Verify forward fill logic
        for i in range(1, len(filled)):
            if pd.isna(dataset[i]) and not pd.isna(filled[i]):
                # This value was filled, should equal the last non-missing value
                last_valid_idx = i - 1
                while last_valid_idx >= 0 and pd.isna(dataset[last_valid_idx]):
                    last_valid_idx -= 1

                if last_valid_idx >= 0:
                    assert filled[i] == dataset[last_valid_idx], \
                        f"Forward fill at index {i} should equal last valid value"

    def test_knn_imputation_local_structure(self):
        """
        Test: KNN imputation preserves local data structure
        Expected: Imputed values based on nearest neighbors, maintains correlations
        """
        # Create dataset with known correlation structure
        np.random.seed(42)
        n_samples = 500

        # Create correlated features
        x = np.random.normal(0, 1, n_samples)
        y = 2 * x + np.random.normal(0, 0.5, n_samples)  # Strong correlation with x
        z = -0.5 * x + np.random.normal(0, 0.3, n_samples)  # Negative correlation with x

        data = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # Add missing values to one column
        missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        data.loc[missing_indices, 'y'] = np.nan

        # Calculate original correlations
        original_corr = data.corr()

        # This test will fail until KNN imputation is implemented
        with pytest.raises(Exception) as exc_info:
            # KNN imputation would go here
            pass

        # The implementation should preserve correlations
        assert "knn" in str(exc_info.value).lower() or "not implemented" in str(exc_info.value).lower()

    def test_regression_imputation_predictive_relationship(self):
        """
        Test: Regression imputation uses predictive relationships
        Expected: Imputed values based on regression model, preserves relationships
        """
        # Create dataset with linear relationship
        np.random.seed(42)
        n_samples = 1000

        x = np.random.normal(50, 10, n_samples)
        y = 2.5 * x + 30 + np.random.normal(0, 5, n_samples)

        data = pd.DataFrame({'x': x, 'y': y})

        # Add missing values to y
        missing_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
        data.loc[missing_indices, 'y'] = np.nan

        # Calculate original regression parameters
        complete_data = data.dropna()
        original_slope = np.cov(complete_data['x'], complete_data['y'])[0, 1] / np.var(complete_data['x'])
        original_intercept = complete_data['y'].mean() - original_slope * complete_data['x'].mean()

        # This test will fail until regression imputation is implemented
        with pytest.raises(Exception) as exc_info:
            # Regression imputation would go here
            pass

        # The implementation should maintain the regression relationship
        assert "regression" in str(exc_info.value).lower() or "not implemented" in str(exc_info.value).lower()

    def test_multiple_imputation_uncertainty_quantification(self):
        """
        Test: Multiple imputation accounts for uncertainty
        Expected: Provides multiple imputed datasets with uncertainty estimates
        """
        # Create dataset with missing values
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        mask = np.random.random(1000) < 0.2
        data[mask] = np.nan
        dataset = pd.Series(data)

        # This test will fail until multiple imputation is implemented
        with pytest.raises(Exception) as exc_info:
            # Multiple imputation would go here
            pass

        # The implementation should provide uncertainty estimates
        assert "multiple" in str(exc_info.value).lower() or "uncertainty" in str(exc_info.value).lower()

    def test_imputation_impact_on_statistical_tests(self, test_datasets):
        """
        Test: Evaluate impact of different imputation methods on statistical tests
        Expected: Document how each method affects test results
        """
        dataset = test_datasets['random_missing']

        # Original statistical properties
        original_mean = dataset.mean()
        original_std = dataset.std()
        n_complete = dataset.dropna().shape[0]

        # Apply different imputation methods
        imputations = {
            'mean': dataset.fillna(original_mean),
            'median': dataset.fillna(dataset.median()),
            'zero': dataset.fillna(0)
        }

        # Test t-statistic for difference from hypothesized mean
        hypothesized_mean = 100

        results = {}
        for method, imputed_data in imputations.items():
            t_stat, p_value = stats.ttest_1samp(imputed_data.dropna(), hypothesized_mean)
            results[method] = {'t_stat': t_stat, 'p_value': p_value}

        # Validate that different methods produce different results
        t_stats = [results[method]['t_stat'] for method in results]
        assert len(set(round(t, 4) for t in t_stats)) > 1, \
            "Different imputation methods should produce different statistical test results"

    def test_imputation_bias_assessment(self, test_datasets):
        """
        Test: Assess bias introduced by different imputation methods
        Expected: Quantify bias for different missing data mechanisms
        """
        # Test different missing data mechanisms
        results = {}

        for mechanism, dataset in test_datasets.items():
            original_true_mean = dataset.dropna().mean()
            original_true_std = dataset.dropna().std()

            # Apply imputation
            imputed_mean = dataset.fillna(dataset.mean())
            imputed_median = dataset.fillna(dataset.median())

            # Calculate bias
            bias_mean = abs(imputed_mean.mean() - original_true_mean)
            bias_median = abs(imputed_median.mean() - original_true_mean)

            results[mechanism] = {
                'bias_mean': bias_mean,
                'bias_median': bias_median,
                'true_mean': original_true_mean,
                'true_std': original_true_std
            }

        # Validate that bias differs by missing data mechanism
        biases = [results[mechanism]['bias_mean'] for mechanism in results]
        assert len(set(round(b, 6) for b in biases)) > 1, \
            "Bias should differ by missing data mechanism"

    @pytest.mark.parametrize("missing_ratio", [0.05, 0.1, 0.2, 0.3])
    def test_imputation_performance_scaling(self, missing_ratio):
        """
        Test: Imputation performance scales with missing data ratio
        Expected: Processing time and accuracy change predictably with missing ratio
        """
        # Create dataset with controlled missing ratio
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        mask = np.random.random(1000) < missing_ratio
        data[mask] = np.nan
        dataset = pd.Series(data)

        original_mean = dataset.dropna().mean()

        # Test simple mean imputation
        import time
        start_time = time.time()
        imputed = dataset.fillna(original_mean)
        processing_time = time.time() - start_time

        # Validate that processing is fast
        assert processing_time < 0.1, \
            f"Mean imputation should be fast (< 0.1s for {missing_ratio*100}% missing)"

        # Validate that all missing values are filled
        assert imputed.isna().sum() == 0, \
            "All missing values should be filled"

    def test_imputation_edge_cases(self):
        """
        Test: Handle edge cases in missing value imputation
        Expected: Graceful handling of all-missing columns, single values, etc.
        """
        # Test all missing column
        all_missing = pd.Series([np.nan] * 100)

        with pytest.raises(ValueError) as exc_info:
            # Attempting to impute all-missing series should fail
            mean_val = all_missing.mean()
            # This should raise an error or handle gracefully
            pass

        # Test single value with missing
        single_missing = pd.Series([1.0, np.nan, 3.0])
        imputed_single = single_missing.fillna(single_missing.mean())

        assert imputed_single.isna().sum() == 0, \
            "Should handle small datasets with missing values"

        # Test constant value with missing
        constant_missing = pd.Series([5.0, 5.0, np.nan, 5.0])
        imputed_constant = constant_missing.fillna(constant_missing.mean())

        assert imputed_constant.iloc[2] == 5.0, \
            "Should correctly impute constant values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])