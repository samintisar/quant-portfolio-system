"""
Unit tests for data validation library logic.

Tests the validation rules, error detection, and consistency checking
for financial market data with focus on mathematical correctness.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the data src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

from lib.validation import DataValidator


class TestDataValidator:
    """Test DataValidator validation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()

        # Create clean test data
        np.random.seed(42)
        self.n_samples = 100
        dates = pd.date_range(start='2023-01-01', periods=self.n_samples, freq='D')

        base_prices = np.random.lognormal(4.0, 0.2, self.n_samples)

        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'high': base_prices * np.random.uniform(1.00, 1.05, self.n_samples),
            'low': base_prices * np.random.uniform(0.95, 1.00, self.n_samples),
            'close': base_prices * np.random.uniform(0.99, 1.01, self.n_samples),
            'volume': np.random.lognormal(15.0, 0.5, self.n_samples).astype(int)
        })

        # Ensure proper OHLC relationships
        self.test_data['high'] = np.maximum(
            self.test_data['high'],
            self.test_data[['open', 'close']].max(axis=1) * 1.001
        )
        self.test_data['low'] = np.minimum(
            self.test_data['low'],
            self.test_data[['open', 'close']].min(axis=1) * 0.999
        )

    def test_dataframe_structure_validation_logic(self):
        """Test DataFrame structure validation logic."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        optional_columns = ['symbol', 'adj_close']

        # Test with complete data
        result = self.validator.validate_dataframe_structure(
            self.test_data, required_columns, optional_columns
        )
        assert result['is_valid'] == True
        assert len(result['missing_required']) == 0

        # Test missing required column
        incomplete_data = self.test_data.drop(columns=['volume'])
        result = self.validator.validate_dataframe_structure(
            incomplete_data, required_columns, optional_columns
        )
        assert result['is_valid'] == False
        assert 'volume' in result['missing_required']

        # Test unexpected column
        extra_data = self.test_data.copy()
        extra_data['unexpected_column'] = 1
        result = self.validator.validate_dataframe_structure(
            extra_data, required_columns, optional_columns
        )
        assert 'unexpected_column' in result['unexpected_columns']

    def test_price_data_validation_mathematical_logic(self):
        """Test price data validation mathematical logic."""
        # Test negative prices
        test_df = self.test_data.copy()
        test_df.loc[10, 'close'] = -100

        result = self.validator.validate_price_data(test_df)
        assert result['is_valid'] == False
        assert 'Found 1 negative prices' in result['validation_issues']
        assert result['statistics']['close']['negative'] == 1

        # Test zero prices
        test_df = self.test_data.copy()
        test_df.loc[15, 'close'] = 0

        result = self.validator.validate_price_data(test_df)
        assert 'Found 1 zero prices' in result['validation_issues']
        assert result['statistics']['close']['zero'] == 1

        # Test extreme values
        test_df = self.test_data.copy()
        q99 = test_df['close'].quantile(0.99)
        test_df.loc[20, 'close'] = q99 * 15  # 15x 99th percentile

        result = self.validator.validate_price_data(test_df)
        assert result['statistics']['close']['extreme'] == 1

    def test_ohlc_relationship_validation_logic(self):
        """Test OHLC relationship validation mathematical logic."""
        # Test high < low violation
        test_df = self.test_data.copy()
        test_df.loc[10, 'high'] = 50
        test_df.loc[10, 'low'] = 60

        result = self.validator.validate_price_data(test_df)
        assert 'Found 1 rows where high < low' in result['validation_issues']
        assert result['is_valid'] == False

        # Test high < open violation
        test_df = self.test_data.copy()
        test_df.loc[15, 'high'] = test_df.loc[15, 'open'] - 1

        result = self.validator.validate_price_data(test_df)
        assert 'Found 1 rows where high < open' in result['validation_issues']

        # Test low > close violation
        test_df = self.test_data.copy()
        test_df.loc[20, 'low'] = test_df.loc[20, 'close'] + 1

        result = self.validator.validate_price_data(test_df)
        assert 'Found 1 rows where low > close' in result['validation_issues']

    def test_volume_validation_logic(self):
        """Test volume validation mathematical logic."""
        # Test negative volume
        test_df = self.test_data.copy()
        test_df.loc[10, 'volume'] = -1000

        result = self.validator.validate_volume_data(test_df)
        assert result['is_valid'] == False
        assert 'Found 1 negative volume values' in result['validation_issues']
        assert result['statistics']['negative'] == 1

        # Test zero volume with significant price move
        test_df = self.test_data.copy()
        test_df.loc[15, 'volume'] = 0
        test_df.loc[15, 'close'] = test_df.loc[15, 'open'] * 1.02  # 2% move

        result = self.validator.validate_volume_data(test_df)
        assert 'zero-volume days with >1% price move' in result['validation_issues']

        # Test extreme volume
        test_df = self.test_data.copy()
        q99 = test_df['volume'].quantile(0.99)
        test_df.loc[20, 'volume'] = q99 * 200  # 200x 99th percentile

        result = self.validator.validate_volume_data(test_df)
        assert result['statistics']['extreme'] == 1

    def test_time_series_validation_mathematical_logic(self):
        """Test time series validation mathematical logic."""
        # Test duplicate timestamps
        test_df = self.test_data.copy()
        test_df = pd.concat([test_df, test_df.iloc[[10, 20]]], ignore_index=True)

        result = self.validator.validate_time_series(test_df)
        assert 'Found 2 duplicate timestamps' in result['validation_issues']

        # Test time gaps
        test_df = self.test_data.copy()
        # Remove row to create gap
        test_df = test_df.drop(15).reset_index(drop=True)

        result = self.validator.validate_time_series(test_df, expected_frequency='1D')
        assert 'Found 1 time gaps' in result['validation_issues']

        # Test datetime conversion
        test_df = self.test_data.copy()
        test_df['timestamp'] = test_df['timestamp'].astype(str)

        result = self.validator.validate_time_series(test_df)
        # Should convert successfully and validate
        assert result['is_valid'] == True

    def test_financial_ratios_validation_logic(self):
        """Test financial ratios validation mathematical logic."""
        # Test extreme daily range
        test_df = self.test_data.copy()
        test_df.loc[10, 'high'] = test_df.loc[10, 'low'] * 1.6  # 60% range

        result = self.validator.validate_financial_ratios(test_df)
        assert 'Extreme daily ranges detected' in result['validation_issues']

        # Test extreme returns
        test_df = self.test_data.copy()
        test_df.loc[15, 'close'] = test_df.loc[15, 'open'] * 1.25  # 25% return

        result = self.validator.validate_financial_ratios(test_df)
        assert 'Extreme returns detected' in result['validation_issues']

        # Test price-volume correlation
        test_df = self.test_data.copy()
        # Create perfect correlation
        test_df['volume'] = test_df['close'] * 1000 + np.random.normal(0, 100, len(test_df))

        result = self.validator.validate_financial_ratios(test_df)
        correlation = result['ratio_statistics']['price_volume_correlation']
        assert abs(correlation) > 0.8  # Should detect high correlation

    def test_cross_asset_consistency_logic(self):
        """Test cross-asset consistency validation logic."""
        # Create multi-asset data
        test_df = pd.concat([
            self.test_data.assign(symbol='AAPL'),
            self.test_data.assign(symbol='GOOGL')
        ], ignore_index=True)

        # Test inconsistent frequency for one asset
        # Remove some data points for AAPL
        test_df = test_df[~((test_df['symbol'] == 'AAPL') & test_df.index.isin([15, 16, 17]))]

        result = self.validator.validate_cross_asset_consistency(test_df)
        assert result['is_valid'] == False
        assert 'Inconsistent data frequency' in str(result['validation_issues'])

    def test_data_quality_score_calculation_logic(self):
        """Test data quality score calculation mathematical logic."""
        # Perfect data should have high score
        perfect_result = {
            'overall_validity': True,
            'validation_summary': {
                'structure_valid': True,
                'prices_valid': True,
                'volume_valid': True,
                'time_series_valid': True
            }
        }
        score = self.validator.get_data_quality_score(perfect_result)
        assert score == 1.0

        # Missing required columns (major penalty)
        missing_columns_result = {
            'overall_validity': False,
            'validation_summary': {
                'structure_valid': False,
                'prices_valid': True,
                'volume_valid': True,
                'time_series_valid': True
            },
            'detailed_results': {
                'structure': {'missing_required': ['close']}
            }
        }
        score = self.validator.get_data_quality_score(missing_columns_result)
        assert score <= 0.7  # Should have major penalty

        # Invalid rows (minor penalty)
        invalid_rows_result = {
            'overall_validity': False,
            'validation_summary': {
                'structure_valid': True,
                'prices_valid': False,
                'volume_valid': True,
                'time_series_valid': True
            },
            'detailed_results': {
                'prices': {'invalid_rows': 100}
            }
        }
        score = self.validator.get_data_quality_score(invalid_rows_result)
        assert score < 1.0 and score > 0.7  # Should have minor penalty

    def test_comprehensive_validation_logic(self):
        """Test comprehensive validation logic."""
        # Test with clean data
        result = self.validator.run_comprehensive_validation(self.test_data)
        assert result['overall_validity'] == True

        # Test with multiple issues
        test_df = self.test_data.copy()
        test_df.loc[10, 'close'] = -100  # Negative price
        test_df.loc[15, 'volume'] = -1000  # Negative volume
        test_df = test_df.drop(20)  # Create time gap

        result = self.validator.run_comprehensive_validation(test_df)
        assert result['overall_validity'] == False
        assert len(result['recommendations']) > 0

        # Test that all validation categories are checked
        expected_categories = ['structure', 'prices', 'volume', 'time_series']
        for category in expected_categories:
            assert category in result['detailed_results']

    def test_recommendation_generation_logic(self):
        """Test recommendation generation logic."""
        # Test structure recommendations
        detailed_results = {
            'structure': {'missing_required': ['volume', 'close']},
            'prices': {'statistics': {'close': {'negative': 5, 'extreme': 2}}},
            'volume': {'statistics': {'negative': 3}},
            'time_series': {'statistics': {'duplicate_timestamps': 2, 'gaps_found': 1}}
        }

        recommendations = self.validator._generate_recommendations(detailed_results)

        # Should have recommendations for each issue
        rec_text = ' '.join(recommendations)
        assert 'volume' in rec_text  # Missing column
        assert 'close' in rec_text  # Missing column
        assert 'negative prices' in rec_text
        assert 'extreme values' in rec_text
        assert 'negative volume' in rec_text
        assert 'duplicate timestamps' in rec_text
        assert 'time gaps' in rec_text

    def test_edge_case_validation_logic(self):
        """Test validation logic for edge cases."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = self.validator.validate_dataframe_structure(empty_df, ['close'])
        assert result['is_valid'] == False
        assert 'Missing required column: close' in result['errors']

        # Single row DataFrame
        single_row = self.test_data.iloc[[0]].copy()
        result = self.validator.validate_price_data(single_row)
        # Should handle single row gracefully
        assert isinstance(result, dict)

        # All NaN values
        all_nan = self.test_data.copy()
        all_nan[['open', 'high', 'low', 'close']] = np.nan

        result = self.validator.validate_price_data(all_nan)
        assert result['is_valid'] == False
        assert result['invalid_rows'] > 0

    def test_statistical_consistency_checks(self):
        """Test statistical consistency validation."""
        # Test correlation between price and volume
        test_df = self.test_data.copy()
        # Create unrealistic correlation
        np.random.seed(42)
        noise = np.random.normal(0, 1, len(test_df))
        test_df['volume'] = test_df['close'] * 1000 + noise * 100

        result = self.validator.validate_financial_ratios(test_df)
        correlation = result['ratio_statistics']['price_volume_correlation']
        assert abs(correlation) > 0.9  # Should detect high correlation

        # Test daily range consistency
        test_df = self.test_data.copy()
        # Create unrealistic ranges
        test_df['daily_range'] = (test_df['high'] - test_df['low']) / test_df['close']
        test_df.loc[10, 'high'] = test_df.loc[10, 'low'] * 2  # 100% range

        result = self.validator.validate_financial_ratios(test_df)
        assert 'Extreme daily ranges detected' in result['validation_issues']

    def test_validation_error_accumulation(self):
        """Test that validation errors accumulate correctly."""
        test_df = self.test_data.copy()

        # Add multiple issues
        issues = [
            (10, 'close', -100),  # Negative price
            (15, 'volume', -1000),  # Negative volume
            (20, 'high', 50),  # High < low will be created
            (20, 'low', 60),
        ]

        for row, col, value in issues:
            test_df.loc[row, col] = value

        result = self.validator.run_comprehensive_validation(test_df)

        # Should detect multiple issues
        total_issues = 0
        for category_result in result['detailed_results'].values():
            if 'validation_issues' in category_result:
                total_issues += len(category_result['validation_issues'])

        assert total_issues >= len(issues)

    def test_validation_type_handling(self):
        """Test validation handles different data types correctly."""
        # Test with mixed data types
        test_df = self.test_data.copy()
        test_df['symbol'] = 'AAPL'  # String column
        test_df['flag'] = True  # Boolean column
        test_df['category'] = 1  # Integer column

        result = self.validator.validate_dataframe_structure(
            test_df, ['timestamp', 'close']
        )

        # Should handle different types
        assert 'symbol' in result['column_types']
        assert 'flag' in result['column_types']
        assert 'category' in result['column_types']

    def test_validation_config_flexibility(self):
        """Test validation configuration flexibility."""
        # Test with custom config
        config = {
            'required_columns': ['timestamp', 'close'],
            'expected_frequency': '1D',
            'validate_ratios': False,
            'validate_cross_asset': False
        }

        result = self.validator.run_comprehensive_validation(self.test_data, config)

        # Should respect config
        assert 'structure' in result['detailed_results']
        assert 'prices' in result['detailed_results']
        assert 'volume' in result['detailed_results']
        assert 'time_series' in result['detailed_results']
        assert 'ratios' not in result['detailed_results']  # Disabled in config
        assert 'cross_asset' not in result['detailed_results']  # Disabled in config