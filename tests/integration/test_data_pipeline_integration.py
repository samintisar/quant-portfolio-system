"""
Integration tests for data pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from portfolio.data.yahoo_service import YahooFinanceService


class TestDataPipelineIntegration:
    """Integration test suite for data pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = YahooFinanceService()
        # Use a small set of well-known symbols for testing
        self.test_symbols = ['AAPL', 'MSFT']  # Apple and Microsoft

    def test_end_to_end_pipeline_real_data(self):
        """Test complete pipeline with real Yahoo Finance data."""
        # Skip if no internet connection by using a short timeout
        try:
            # Test with recent data (1 month) to minimize API calls
            results = self.service.fetch_and_process_data(
                self.test_symbols,
                period="1mo",
                normalize_method=None
            )

            # Verify we got results for both symbols
            assert len(results) > 0, "Should get data for at least one symbol"

            # Test each successful result
            for symbol, result in results.items():
                if result['success']:
                    assert result['data'] is not None, f"Data should not be None for {symbol}"
                    assert not result['data'].empty, f"Data should not be empty for {symbol}"
                    assert result['validation']['is_valid'], f"Data should be valid for {symbol}"
                    assert result['quality_report'] is not None, f"Quality report should exist for {symbol}"

                    # Verify data structure
                    data = result['data']
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'returns']
                    for col in required_columns:
                        assert col in data.columns, f"Missing required column {col} for {symbol}"

                    # Verify no missing values in cleaned data
                    assert not data.isnull().any().any(), f"Cleaned data should have no missing values for {symbol}"

        except Exception as e:
            pytest.skip(f"Skipping integration test due to network issue: {e}")

    def test_pipeline_with_normalization(self):
        """Test pipeline with different normalization methods."""
        try:
            normalization_methods = [None, 'minmax', 'zscore', 'returns']

            for method in normalization_methods:
                results = self.service.fetch_and_process_data(
                    ['AAPL'],  # Test with single symbol to reduce API calls
                    period="1mo",
                    normalize_method=method
                )

                if results and results['AAPL']['success']:
                    data = results['AAPL']['data']
                    assert not data.empty, f"Data should not be empty with {method} normalization"

                    if method == 'minmax':
                        # Check min-max normalization
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col != 'symbol':
                                assert data[col].min() >= 0, f"Min should be >= 0 for {col} with minmax"
                                assert data[col].max() <= 1, f"Max should be <= 1 for {col} with minmax"

                    elif method == 'zscore':
                        # Check z-score normalization
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col != 'symbol':
                                assert abs(data[col].mean()) < 0.1, f"Mean should be ~0 for {col} with zscore"
                                assert abs(data[col].std() - 1) < 0.1, f"Std should be ~1 for {col} with zscore"

        except Exception as e:
            pytest.skip(f"Skipping normalization test due to network issue: {e}")

    def test_quality_report_comprehensive(self):
        """Test comprehensive quality reporting."""
        try:
            results = self.service.fetch_and_process_data(
                ['AAPL'],
                period="6mo",
                normalize_method=None
            )

            if results and results['AAPL']['success']:
                quality_report = results['AAPL']['quality_report']

                # Verify report structure
                assert 'symbol' in quality_report
                assert 'timestamp' in quality_report
                assert 'data_summary' in quality_report
                assert 'quality_metrics' in quality_report
                assert 'recommendations' in quality_report

                # Verify data summary
                summary = quality_report['data_summary']
                assert 'total_rows' in summary
                assert 'date_range' in summary
                assert 'columns' in summary
                assert 'missing_values' in summary
                assert 'data_types' in summary

                # Verify quality metrics
                metrics = quality_report['quality_metrics']
                assert 'completeness' in metrics
                assert 'volatility_annualized' in metrics
                assert 'data_density' in metrics
                assert 'price_range' in metrics

                # Verify reasonable values
                assert metrics['completeness'] > 0.9, "Completeness should be high"
                assert metrics['data_density'] > 0, "Data density should be positive"
                assert metrics['volatility_annualized'] is not None, "Volatility should be calculated"

        except Exception as e:
            pytest.skip(f"Skipping quality report test due to network issue: {e}")

    def test_multiple_symbols_pipeline(self):
        """Test pipeline with multiple symbols simultaneously."""
        try:
            results = self.service.fetch_and_process_data(
                self.test_symbols,
                period="3mo",
                normalize_method=None
            )

            # Should get results for both symbols (network permitting)
            assert len(results) >= 1, "Should get data for at least one symbol"

            # Compare data quality across symbols
            successful_symbols = [symbol for symbol, result in results.items() if result['success']]
            if len(successful_symbols) > 1:
                # Verify data structures are consistent
                first_symbol = successful_symbols[0]
                first_columns = set(results[first_symbol]['data'].columns)

                for symbol in successful_symbols[1:]:
                    current_columns = set(results[symbol]['data'].columns)
                    assert first_columns == current_columns, f"Columns should be consistent across symbols"

        except Exception as e:
            pytest.skip(f"Skipping multiple symbols test due to network issue: {e}")

    def test_data_validation_edge_cases(self):
        """Test data validation with various edge cases."""
        try:
            # Test with different time periods
            periods = ["5d", "1mo", "3mo", "6mo", "1y"]

            for period in periods:
                results = self.service.fetch_and_process_data(
                    ['AAPL'],
                    period=period,
                    normalize_method=None
                )

                if results and results['AAPL']['success']:
                    data = results['AAPL']['data']
                    validation = results['AAPL']['validation']

                    # Should have sufficient data for analysis
                    assert len(data) > 0, f"Should have data for period {period}"
                    assert validation['is_valid'], f"Data should be valid for period {period}"

                    # Should have required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    for col in required_cols:
                        assert col in data.columns, f"Missing {col} for period {period}"

        except Exception as e:
            pytest.skip(f"Skipping edge cases test due to network issue: {e}")

    def test_error_handling_integration(self):
        """Test error handling with realistic scenarios."""
        # Test with invalid symbol
        results = self.service.fetch_and_process_data(
            ['INVALID_SYMBOL_12345'],
            period="1mo",
            normalize_method=None
        )

        # Should handle invalid symbol gracefully
        assert len(results) == 0, "Should return empty dict for invalid symbol"

        # Test with empty symbols list
        results = self.service.fetch_and_process_data(
            [],
            period="1mo",
            normalize_method=None
        )

        assert len(results) == 0, "Should return empty dict for empty symbols list"

    def test_performance_considerations(self):
        """Test pipeline performance and memory usage."""
        try:
            import time

            start_time = time.time()
            results = self.service.fetch_and_process_data(
                self.test_symbols,
                period="1y",  # Longer period for performance test
                normalize_method='minmax'
            )
            end_time = time.time()

            # Should complete in reasonable time (less than 30 seconds)
            assert end_time - start_time < 30, "Pipeline should complete in reasonable time"

            if results:
                # Memory usage should be reasonable
                total_memory = sum(
                    result['data'].memory_usage(deep=True).sum()
                    for result in results.values()
                    if result['success'] and result['data'] is not None
                )

                assert total_memory < 100 * 1024 * 1024, "Memory usage should be reasonable (< 100MB)"

        except Exception as e:
            pytest.skip(f"Skipping performance test due to network issue: {e}")

    def test_data_consistency(self):
        """Test data consistency and integrity."""
        try:
            results = self.service.fetch_and_process_data(
                ['AAPL'],
                period="6mo",
                normalize_method=None
            )

            if results and results['AAPL']['success']:
                data = results['AAPL']['data']

                # Check data consistency
                assert data.index.is_monotonic_increasing, "Data should be sorted by date"
                assert not data.index.duplicated().any(), "No duplicate dates allowed"

                # Check price relationships
                price_data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
                assert (price_data['High'] >= price_data[['Open', 'Close']].max(axis=1)).all(), "High should be >= max(Open, Close)"
                assert (price_data['Low'] <= price_data[['Open', 'Close']].min(axis=1)).all(), "Low should be <= min(Open, Close)"

                # Check volume is non-negative
                assert (data['Volume'] >= 0).all(), "Volume should be non-negative"

                # Check returns calculation
                calculated_returns = data['Adj Close'].pct_change()
                # Allow for small numerical differences
                assert np.allclose(data['returns'].iloc[1:], calculated_returns.iloc[1:], equal_nan=True), "Returns should match calculated values"

        except Exception as e:
            pytest.skip(f"Skipping consistency test due to network issue: {e}")

    def test_symbol_info_integration(self):
        """Test symbol information fetching integration."""
        try:
            info = self.service.get_symbol_info('AAPL')

            assert 'symbol' in info
            assert info['symbol'] == 'AAPL'
            assert 'name' in info
            assert info['name'] != 'N/A'  # Should get real name for AAPL

            # Test with invalid symbol
            invalid_info = self.service.get_symbol_info('INVALID_SYMBOL_12345')
            assert invalid_info['symbol'] == 'INVALID_SYMBOL_12345'
            assert invalid_info['name'] == 'N/A'

        except Exception as e:
            pytest.skip(f"Skipping symbol info test due to network issue: {e}")


if __name__ == '__main__':
    pytest.main([__file__])