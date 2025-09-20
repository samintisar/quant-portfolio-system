"""
Test data quality and preprocessing validation for financial features.

This module contains tests for data quality validation and preprocessing
including missing value handling, outlier detection, and data integrity checks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add data/src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'src'))

try:
    from services.validation_service import DataValidator
    from lib.cleaning import detect_outliers, handle_missing_values
    from lib.normalization import normalize_data, validate_normalization
    from models.price_data import PriceData
    from models.financial_instrument import FinancialInstrument
except ImportError:
    pass


class TestDataQualityValidation:
    """Test suite for data quality validation and preprocessing."""

    @pytest.fixture
    def clean_price_data(self):
        """Clean price data fixture."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100
        volumes = np.random.lognormal(10, 1, 100)

        return pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'date': dates
        }).set_index('date')

    @pytest.fixture
    def dirty_price_data(self):
        """Dirty price data with various quality issues."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        # Create prices with various issues
        base_prices = np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100

        # Add issues
        base_prices[10] = np.nan  # Missing value
        base_prices[20] = -50.0   # Negative price (invalid)
        base_prices[30] = 1000000.0  # Extreme outlier
        base_prices[40] = 0.0    # Zero price (potentially invalid)

        # Volume with issues
        volumes = np.random.lognormal(10, 1, 100)
        volumes[15] = -1000  # Negative volume (invalid)
        volumes[25] = np.nan  # Missing volume
        volumes[35] = 0  # Zero volume (potentially invalid)

        return pd.DataFrame({
            'price': base_prices,
            'volume': volumes,
            'date': dates
        }).set_index('date')

    @pytest.fixture
    def time_gapped_data(self):
        """Data with time gaps."""
        # Create irregular dates with gaps
        dates = []
        current_date = datetime(2023, 1, 1)
        for i in range(100):
            dates.append(current_date)
            # Add random gaps
            gap_days = int(np.random.choice([1, 1, 1, 2, 3, 7]))  # Most 1 day, some larger gaps
            current_date += timedelta(days=gap_days)

        prices = np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100
        volumes = np.random.lognormal(10, 1, 100)

        return pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'date': dates
        }).set_index('date')

    def test_data_validator_exists(self):
        """Test that DataValidator class exists."""
        try:
            DataValidator
        except NameError:
            pytest.fail("DataValidator class not implemented")

    def test_data_validator_instantiation(self):
        """Test DataValidator instantiation."""
        try:
            validator = DataValidator()
            assert hasattr(validator, 'validate_data')
            assert hasattr(validator, 'generate_quality_report')
        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_missing_value_detection_exists(self):
        """Test that missing value detection exists."""
        try:
            handle_missing_values
        except NameError:
            pytest.fail("handle_missing_values function not implemented")

    def test_missing_value_detection(self, dirty_price_data):
        """Test missing value detection functionality."""
        try:
            result = handle_missing_values(dirty_price_data)

            # Should handle NaN values appropriately
            assert result.isna().sum().sum() == 0

            # Should maintain same structure
            assert result.shape[1] == dirty_price_data.shape[1]

        except (NameError, AttributeError):
            pytest.fail("Function not yet implemented")

    def test_outlier_detection_exists(self):
        """Test that outlier detection exists."""
        try:
            detect_outliers
        except NameError:
            pytest.fail("detect_outliers function not implemented")

    def test_outlier_detection(self, dirty_price_data):
        """Test outlier detection functionality."""
        try:
            outliers = detect_outliers(dirty_price_data['price'])

            # Should detect obvious outliers
            assert isinstance(outliers, (pd.Series, np.ndarray, list))
            assert len(outliers) > 0

            # Should mark extreme values as outliers
            extreme_positions = [20, 30]  # Positions we know have extreme values
            for pos in extreme_positions:
                if pos < len(outliers):
                    assert outliers[pos] == True

        except (NameError, AttributeError):
            pytest.fail("Function not yet implemented")

    def test_outlier_detection_methods(self, dirty_price_data):
        """Test different outlier detection methods."""
        try:
            methods = ['iqr', 'zscore', 'modified_zscore']

            for method in methods:
                outliers = detect_outliers(dirty_price_data['price'], method=method)
                assert isinstance(outliers, (pd.Series, np.ndarray, list))

        except (NameError, AttributeError):
            pytest.fail("Function not yet implemented")

    def test_data_normalization_exists(self):
        """Test that data normalization exists."""
        try:
            normalize_data
        except NameError:
            pytest.fail("normalize_data function not implemented")

    def test_data_normalization_basic(self, clean_price_data):
        """Test basic data normalization."""
        try:
            # Test z-score normalization
            normalized = normalize_data(clean_price_data['price'], method='zscore')

            # Z-score should have mean ~0 and std ~1
            assert abs(normalized.mean()) < 0.1
            assert abs(normalized.std() - 1.0) < 0.1

            # Test min-max normalization
            normalized_minmax = normalize_data(clean_price_data['price'], method='minmax')

            # Min-max should be in [0, 1]
            assert normalized_minmax.min() >= 0
            assert normalized_minmax.max() <= 1

        except (NameError, AttributeError):
            pytest.fail("Function not yet implemented")

    def test_normalization_validation_exists(self):
        """Test that normalization validation exists."""
        try:
            validate_normalization
        except NameError:
            pytest.fail("validate_normalization function not implemented")

    def test_normalization_validation(self, clean_price_data):
        """Test normalization validation."""
        try:
            normalized = normalize_data(clean_price_data['price'], method='zscore')
            validation_result = validate_normalization(normalized, method='zscore')

            # Should return validation metrics
            assert isinstance(validation_result, dict)
            assert 'mean' in validation_result
            assert 'std' in validation_result
            assert 'is_valid' in validation_result

            # Z-score validation should pass
            assert validation_result['is_valid'] == True

        except (NameError, AttributeError):
            pytest.fail("Function not yet implemented")

    def test_price_data_model_exists(self):
        """Test that PriceData model exists."""
        try:
            PriceData
        except NameError:
            pytest.fail("PriceData class not implemented")

    def test_price_data_instantiation(self):
        """Test PriceData model instantiation."""
        try:
            dates = pd.date_range('2023-01-01', periods=10, freq='D')
            prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0,
                               105.0, 106.0, 107.0, 108.0, 109.0], index=dates)

            price_data = PriceData(
                prices=prices,
                instrument="TEST",
                frequency='daily'
            )

            assert price_data.instrument == "TEST"
            assert price_data.frequency == 'daily'
            assert len(price_data.prices) == 10

        except (NameError, AttributeError):
            pytest.fail("PriceData class not yet implemented")

    def test_financial_instrument_model_exists(self):
        """Test that FinancialInstrument model exists."""
        try:
            FinancialInstrument
        except NameError:
            pytest.fail("FinancialInstrument class not implemented")

    def test_comprehensive_data_validation(self, dirty_price_data):
        """Test comprehensive data validation."""
        try:
            validator = DataValidator()
            validation_report = validator.validate_data(dirty_price_data)

            # Should return comprehensive validation report
            assert isinstance(validation_report, dict)
            assert 'missing_values' in validation_report
            assert 'outliers' in validation_report
            assert 'invalid_values' in validation_report
            assert 'quality_score' in validation_report

            # Quality score should be between 0 and 1
            assert 0 <= validation_report['quality_score'] <= 1

            # Should detect issues in dirty data
            assert validation_report['missing_values'] > 0
            assert validation_report['outliers'] > 0

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_time_gap_detection(self, time_gapped_data):
        """Test time gap detection."""
        try:
            validator = DataValidator()
            time_analysis = validator.analyze_time_gaps(time_gapped_data)

            # Should detect time gaps
            assert isinstance(time_analysis, dict)
            assert 'gaps_detected' in time_analysis
            assert 'max_gap' in time_analysis
            assert 'avg_gap' in time_analysis

            # Should find gaps in our test data
            assert time_analysis['gaps_detected'] > 0

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_data_integrity_checks(self, clean_price_data):
        """Test data integrity validation."""
        try:
            validator = DataValidator()
            integrity_checks = validator.check_integrity(clean_price_data)

            # Should perform various integrity checks
            assert isinstance(integrity_checks, dict)
            assert 'price_integrity' in integrity_checks
            assert 'volume_integrity' in integrity_checks
            assert 'chronological_order' in integrity_checks

            # Clean data should pass integrity checks
            assert integrity_checks['price_integrity'] == True
            assert integrity_checks['volume_integrity'] == True
            assert integrity_checks['chronological_order'] == True

        except (NameNameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_preprocessing_pipeline(self, dirty_price_data):
        """Test complete preprocessing pipeline."""
        try:
            validator = DataValidator()

            # Step 1: Initial validation
            initial_report = validator.validate_data(dirty_price_data)
            initial_score = initial_report['quality_score']

            # Step 2: Clean data
            cleaned_data = handle_missing_values(dirty_price_data)

            # Step 3: Remove outliers
            outliers = detect_outliers(cleaned_data['price'])
            outlier_free_data = cleaned_data[~outliers]

            # Step 4: Final validation
            final_report = validator.validate_data(outlier_free_data)
            final_score = final_report['quality_score']

            # Quality should improve after preprocessing
            assert final_score >= initial_score

        except (NameError, AttributeError):
            pytest.fail("Preprocessing functions not yet implemented")

    def test_financial_data_validation(self, dirty_price_data):
        """Test financial-specific data validation."""
        try:
            validator = DataValidator()
            financial_validation = validator.validate_financial_data(dirty_price_data)

            # Should check financial constraints
            assert isinstance(financial_validation, dict)
            assert 'price_constraints' in financial_validation
            assert 'volume_constraints' in financial_validation
            assert 'return_constraints' in financial_validation

            # Should detect financial data issues
            assert financial_validation['price_constraints']['negative_prices'] > 0
            assert financial_validation['volume_constraints']['negative_volumes'] > 0

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_data_quality_scoring(self, clean_price_data, dirty_price_data):
        """Test data quality scoring system."""
        try:
            validator = DataValidator()

            # Clean data should have high quality score
            clean_score = validator.validate_data(clean_price_data)['quality_score']
            assert clean_score > 0.8

            # Dirty data should have lower quality score
            dirty_score = validator.validate_data(dirty_price_data)['quality_score']
            assert dirty_score < clean_score

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_validation_reporting(self, dirty_price_data):
        """Test validation reporting functionality."""
        try:
            validator = DataValidator()
            report = validator.generate_quality_report(dirty_price_data)

            # Should generate detailed report
            assert isinstance(report, dict)
            assert 'summary' in report
            assert 'detailed_issues' in report
            assert 'recommendations' in report
            assert 'timestamp' in report

            # Report should be comprehensive
            assert len(report['detailed_issues']) > 0
            assert len(report['recommendations']) > 0

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_edge_cases_handling(self):
        """Test handling of edge cases."""
        try:
            validator = DataValidator()

            # Empty dataset
            empty_data = pd.DataFrame()
            empty_report = validator.validate_data(empty_data)
            assert empty_report['quality_score'] == 0

            # Single data point
            single_data = pd.DataFrame({'price': [100.0], 'volume': [1000]})
            single_report = validator.validate_data(single_data)
            assert 'quality_score' in single_report

            # All missing data
            all_missing = pd.DataFrame({'price': [np.nan, np.nan], 'volume': [np.nan, np.nan]})
            missing_report = validator.validate_data(all_missing)
            assert missing_report['quality_score'] == 0

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")

    def test_data_preprocessing_performance(self, clean_price_data):
        """Test preprocessing performance with larger datasets."""
        try:
            # Create larger dataset
            large_dates = pd.date_range('2020-01-01', periods=10000, freq='D')
            large_prices = np.cumsum(np.random.normal(0.001, 0.02, 10000)) + 100
            large_volumes = np.random.lognormal(10, 1, 10000)

            large_data = pd.DataFrame({
                'price': large_prices,
                'volume': large_volumes,
                'date': large_dates
            }).set_index('date')

            validator = DataValidator()

            # Performance test (should complete in reasonable time)
            import time
            start_time = time.time()
            report = validator.validate_data(large_data)
            end_time = time.time()

            # Should process 10K data points in under 5 seconds
            assert end_time - start_time < 5.0
            assert 'quality_score' in report

        except (NameError, AttributeError):
            pytest.fail("DataValidator class not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])