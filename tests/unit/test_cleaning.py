"""
Unit tests for data cleaning library mathematical functions.

Tests the statistical and mathematical correctness of cleaning operations
including missing value handling, outlier detection, and time gap processing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.cleaning import DataCleaner


class TestDataCleaner:
    """Test DataCleaner mathematical functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()

        # Create test data with known properties
        np.random.seed(42)  # For reproducible tests
        self.n_samples = 100

        # Generate test financial data
        dates = pd.date_range(start='2023-01-01', periods=self.n_samples, freq='D')

        # Base price series with known statistical properties
        base_prices = np.random.lognormal(4.0, 0.2, self.n_samples)

        # Create OHLCV data
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'high': base_prices * np.random.uniform(1.00, 1.05, self.n_samples),
            'low': base_prices * np.random.uniform(0.95, 1.00, self.n_samples),
            'close': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'volume': np.random.lognormal(15.0, 0.5, self.n_samples).astype(int)
        })

        # Ensure logical OHLC relationships
        self.test_data['high'] = np.maximum(
            self.test_data['high'],
            self.test_data[['open', 'close']].max(axis=1) * 1.001
        )
        self.test_data['low'] = np.minimum(
            self.test_data['low'],
            self.test_data[['open', 'close']].min(axis=1) * 0.999
        )

    def test_missing_value_detection_math(self):
        """Test mathematical correctness of missing value detection."""
        # Insert missing values at known positions
        test_df = self.test_data.copy()
        missing_indices = [10, 20, 30, 40, 50]
        for idx in missing_indices:
            test_df.loc[idx, 'close'] = np.nan

        # Test missing ratio calculation
        expected_missing_ratio = len(missing_indices) / (len(test_df) * len(test_df.columns))
        actual_missing_ratio = test_df.isnull().sum().sum() / (test_df.shape[0] * test_df.shape[1])

        assert abs(actual_missing_ratio - expected_missing_ratio) < 1e-10

        # Test column-specific missing ratio
        close_missing_ratio = test_df['close'].isnull().sum() / len(test_df)
        expected_close_ratio = len(missing_indices) / len(test_df)
        assert abs(close_missing_ratio - expected_close_ratio) < 1e-10

    def test_forward_fill_mathematical_properties(self):
        """Test mathematical properties of forward fill."""
        test_df = self.test_data.copy()

        # Create missing values
        test_df.loc[10:15, 'close'] = np.nan
        test_df.loc[20:25, 'close'] = np.nan

        # Apply forward fill
        filled = self.cleaner.handle_missing_values(test_df, method='forward_fill')

        # Test that forward fill preserves last known value
        last_good_value = test_df.loc[9, 'close']
        for i in range(10, 16):
            assert filled.loc[i, 'close'] == last_good_value

        # Test no missing values remain
        assert not filled['close'].isnull().any()

    def test_interpolation_mathematical_correctness(self):
        """Test mathematical correctness of interpolation."""
        test_df = self.test_data.copy()

        # Create consecutive missing values
        test_df.loc[20:25, 'close'] = np.nan

        # Apply interpolation
        interpolated = self.cleaner.handle_missing_values(test_df, method='interpolation')

        # Test linear interpolation properties
        start_val = test_df.loc[19, 'close']
        end_val = test_df.loc[26, 'close']

        # Interpolated values should be linear progression
        for i in range(20, 26):
            expected_val = start_val + (end_val - start_val) * (i - 19) / (26 - 19)
            assert abs(interpolated.loc[i, 'close'] - expected_val) < 1e-10

    def test_zscore_outlier_detection(self):
        """Test z-score outlier detection mathematical correctness."""
        # Create data with known outliers
        test_df = self.test_data.copy()

        # Add known outliers (5 standard deviations from mean)
        mean_close = test_df['close'].mean()
        std_close = test_df['close'].std()

        outlier_indices = [10, 30, 50]
        for idx in outlier_indices:
            test_df.loc[idx, 'close'] = mean_close + 5 * std_close

        # Detect outliers
        _, outlier_masks = self.cleaner.detect_outliers(
            test_df, method='zscore', threshold=3.0, action='flag'
        )

        # Test that our outliers are detected
        close_outliers = outlier_masks['close']
        for idx in outlier_indices:
            assert close_outliers[idx] == True

        # Test z-score calculation
        for idx in outlier_indices:
            z_score = abs((test_df.loc[idx, 'close'] - mean_close) / std_close)
            assert z_score > 3.0

    def test_iqr_outlier_detection(self):
        """Test IQR outlier detection mathematical correctness."""
        test_df = self.test_data.copy()

        # Add outliers beyond 1.5 * IQR
        Q1 = test_df['close'].quantile(0.25)
        Q3 = test_df['close'].quantile(0.75)
        IQR = Q3 - Q1

        # Create outlier just beyond upper bound
        outlier_value = Q3 + 2.0 * IQR
        test_df.loc[10, 'close'] = outlier_value

        # Detect outliers
        _, outlier_masks = self.cleaner.detect_outliers(
            test_df, method='iqr', threshold=1.5, action='flag'
        )

        # Test outlier detection
        assert outlier_masks['close'][10] == True

        # Test IQR bounds calculation
        expected_lower = Q1 - 1.5 * IQR
        expected_upper = Q3 + 1.5 * IQR

        # Our outlier should be beyond upper bound
        assert outlier_value > expected_upper

    def test_custom_outlier_detection_financial_logic(self):
        """Test custom financial outlier detection logic."""
        test_df = self.test_data.copy()

        # Test zero price detection
        test_df.loc[10, 'close'] = 0.0
        test_df.loc[11, 'close'] = -1.0

        # Test extreme price detection
        test_df.loc[12, 'close'] = 200000  # Extreme price

        # Test negative volume
        test_df.loc[15, 'volume'] = -1000

        # Test extreme volume spike
        median_volume = test_df['volume'].median()
        test_df.loc[16, 'volume'] = median_volume * 15  # 15x median

        # Detect outliers using custom method
        _, outlier_masks = self.cleaner.detect_outliers(
            test_df, method='custom', action='flag'
        )

        # Test financial logic detection
        assert outlier_masks['close'][10] == True  # Zero price
        assert outlier_masks['close'][11] == True  # Negative price
        assert outlier_masks['close'][12] == True  # Extreme price
        assert outlier_masks['volume'][15] == True  # Negative volume
        assert outlier_masks['volume'][16] == True  # Extreme volume

    def test_time_gap_detection_mathematics(self):
        """Test mathematical correctness of time gap detection."""
        test_df = self.test_data.copy()

        # Remove some rows to create gaps
        test_df = test_df.drop([15, 16, 17])  # Create 3-day gap

        # Detect gaps
        gap_info = self.cleaner.detect_time_gaps(test_df, expected_frequency='1D')

        # Test gap detection
        assert gap_info['is_gap'].any()

        # Test time difference calculation
        expected_diff = pd.Timedelta(days=1)
        gap_positions = gap_info[gap_info['is_gap']]

        for idx in gap_positions.index:
            actual_diff = gap_info.loc[idx, 'time_diff']
            assert actual_diff > expected_diff

    def test_data_quality_score_calculation(self):
        """Test data quality score mathematical calculation."""
        test_df = self.test_data.copy()

        # Base score for clean data
        base_score = self.cleaner.get_data_quality_score(test_df)
        assert base_score > 0.9  # Should be high for clean data

        # Add missing values
        test_df.loc[10:15, 'close'] = np.nan
        score_with_missing = self.cleaner.get_data_quality_score(test_df)
        assert score_with_missing < base_score

        # Add duplicates
        test_df = pd.concat([test_df, test_df.iloc[[10, 11, 12]]], ignore_index=True)
        score_with_duplicates = self.cleaner.get_data_quality_score(test_df)
        assert score_with_duplicates < score_with_missing

        # Add invalid values
        test_df.loc[20, 'close'] = -100  # Negative price
        score_with_invalid = self.cleaner.get_data_quality_score(test_df)
        assert score_with_invalid < score_with_duplicates

    def test_outlier_clipping_mathematics(self):
        """Test mathematical correctness of outlier clipping."""
        test_df = self.test_data.copy()

        # Add extreme outliers
        test_df.loc[10, 'close'] = 10000
        test_df.loc[11, 'close'] = 0.01

        # Clip outliers
        clipped, _ = self.cleaner.detect_outliers(
            test_df, method='iqr', action='clip'
        )

        # Test clipping bounds
        Q1 = test_df['close'].quantile(0.25)
        Q3 = test_df['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Clipped values should be within bounds
        assert clipped.loc[10, 'close'] <= upper_bound
        assert clipped.loc[11, 'close'] >= lower_bound

    def test_windowed_interpolation(self):
        """Test windowed interpolation mathematical properties."""
        test_df = self.test_data.copy()

        # Create gap larger than window
        test_df.loc[20:30, 'close'] = np.nan  # 11 missing values

        # Apply interpolation with limited window
        interpolated = self.cleaner.handle_missing_values(
            test_df, method='interpolation', window_size=5
        )

        # Test that window limits interpolation
        # Values beyond window should be filled with last known value
        window_limit = 20 + 5  # start + window_size
        for i in range(20, min(26, len(test_df))):
            if i <= window_limit:
                assert not pd.isna(interpolated.loc[i, 'close'])
            else:
                # Should be filled with forward fill
                assert interpolated.loc[i, 'close'] == interpolated.loc[window_limit, 'close']

    def test_statistical_properties_preservation(self):
        """Test that cleaning preserves important statistical properties."""
        original_mean = self.test_data['close'].mean()
        original_std = self.test_data['close'].std()

        # Apply gentle cleaning
        cleaned = self.cleaner.handle_missing_values(
            self.test_data, method='forward_fill'
        )

        # Statistical properties should be preserved
        cleaned_mean = cleaned['close'].mean()
        cleaned_std = cleaned['close'].std()

        # Allow small differences due to floating point precision
        assert abs(cleaned_mean - original_mean) < 1e-10
        assert abs(cleaned_std - original_std) < 1e-10

    def test_edge_cases_mathematical_handling(self):
        """Test mathematical handling of edge cases."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = self.cleaner.handle_missing_values(empty_df)
        assert result.empty

        # All NaN values
        all_nan = pd.DataFrame({'col1': [np.nan, np.nan, np.nan]})
        result = self.cleaner.handle_missing_values(all_nan, method='mean')
        # Should handle gracefully without errors

        # Constant values
        constant_df = pd.DataFrame({'col1': [1.0, 1.0, 1.0]})
        result = self.cleaner.handle_missing_values(constant_df, method='mean')
        assert result['col1'].iloc[0] == 1.0

    def test_reproducibility_with_seed(self):
        """Test that operations are reproducible with proper seed."""
        # Create test data with known random seed
        np.random.seed(123)
        test_data1 = pd.DataFrame({
            'values': np.random.normal(0, 1, 100)
        })

        np.random.seed(123)
        test_data2 = pd.DataFrame({
            'values': np.random.normal(0, 1, 100)
        })

        # Results should be identical for identical data
        result1 = self.cleaner.handle_missing_values(test_data1, method='mean')
        result2 = self.cleaner.handle_missing_values(test_data2, method='mean')

        pd.testing.assert_series_equal(result1['values'], result2['values'])

    def test_missing_data_threshold_logic(self):
        """Test mathematical logic of missing data thresholds."""
        test_df = self.test_data.copy()

        # Calculate threshold for 10% missing
        threshold = 0.1
        max_missing = int(len(test_df) * threshold)

        # Add exactly threshold missing values
        missing_indices = range(max_missing)
        for idx in missing_indices:
            test_df.loc[idx, 'close'] = np.nan

        # Should not warn (exactly at threshold)
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            self.cleaner.handle_missing_values(test_df, threshold=threshold)
            # Check no warnings were issued
            threshold_warnings = [w for w in warning_list if "exceeds threshold" in str(w.message)]
            assert len(threshold_warnings) == 0

        # Add one more missing value
        test_df.loc[max_missing, 'close'] = np.nan

        # Should warn (exceeds threshold)
        with pytest.warns(UserWarning):
            self.cleaner.handle_missing_values(test_df, threshold=threshold)