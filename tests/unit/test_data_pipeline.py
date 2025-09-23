"""
Unit tests for data pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from portfolio.data.yahoo_service import YahooFinanceService


class TestYahooFinanceService:
    """Test suite for YahooFinanceService data pipeline functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = YahooFinanceService()
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self):
        """Create sample financial data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        # Remove weekends
        dates = dates[dates.dayofweek < 5]

        n_days = len(dates)
        np.random.seed(42)  # For reproducible tests

        data = pd.DataFrame({
            'Open': 100 + np.random.normal(0, 2, n_days).cumsum(),
            'High': 0,
            'Low': 0,
            'Close': 0,
            'Adj Close': 100 + np.random.normal(0, 2, n_days).cumsum(),
            'Volume': np.random.randint(1000000, 5000000, n_days)
        }, index=dates)

        # Ensure High >= Low and Open/Close are within range
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 1, n_days)
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 1, n_days)
        data['Close'] = data['Adj Close']

        return data

    def test_clean_data_basic(self):
        """Test basic data cleaning functionality."""
        # Create test data with issues
        dirty_data = self.sample_data.copy()

        # Add duplicates
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0, 1]]])

        # Add missing values
        dirty_data.loc[dirty_data.index[10:15], 'Volume'] = np.nan

        # Clean the data
        cleaned_data = self.service.clean_data(dirty_data)

        # Verify duplicates removed
        assert len(cleaned_data) < len(dirty_data)

        # Verify missing values handled
        assert not cleaned_data['Volume'].isna().any()

        # Verify data is sorted
        assert cleaned_data.index.is_monotonic_increasing

    def test_clean_data_empty(self):
        """Test cleaning empty DataFrame."""
        empty_data = pd.DataFrame()
        cleaned_data = self.service.clean_data(empty_data)

        assert cleaned_data.empty

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        # Add returns column for validation
        test_data = self.sample_data.copy()
        test_data['returns'] = test_data['Adj Close'].pct_change()

        is_valid, report = self.service.validate_data(test_data, 'AAPL')

        assert is_valid is True
        assert report['symbol'] == 'AAPL'
        assert len(report['issues']) == 0
        assert report['total_rows'] == len(test_data)

    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        empty_data = pd.DataFrame()
        is_valid, report = self.service.validate_data(empty_data, 'AAPL')

        assert is_valid is False
        assert 'No data available' in report['issues']

    def test_validate_data_missing_columns(self):
        """Test data validation with missing required columns."""
        incomplete_data = pd.DataFrame({'Open': [100, 101]}, index=pd.date_range('2023-01-01', periods=2))

        is_valid, report = self.service.validate_data(incomplete_data, 'AAPL')

        assert is_valid is False
        assert any('Missing required columns' in issue for issue in report['issues'])

    def test_validate_data_extreme_returns(self):
        """Test data validation detects extreme returns."""
        test_data = self.sample_data.copy()
        test_data['returns'] = test_data['Adj Close'].pct_change()

        # Add extreme return
        test_data.loc[test_data.index[50], 'returns'] = 0.6  # 60% daily return

        is_valid, report = self.service.validate_data(test_data, 'AAPL')

        assert is_valid is False
        assert any('extreme returns' in issue for issue in report['issues'])

    def test_validate_data_zero_prices(self):
        """Test data validation detects zero/negative prices."""
        test_data = self.sample_data.copy()
        test_data.loc[test_data.index[20], 'Adj Close'] = 0

        is_valid, report = self.service.validate_data(test_data, 'AAPL')

        assert is_valid is False
        assert any('zero/negative prices' in issue for issue in report['issues'])

    def test_normalize_data_minmax(self):
        """Test min-max normalization."""
        normalized = self.service.normalize_data(self.sample_data, 'minmax')

        # Check that numeric columns (except symbol) are in [0, 1]
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'symbol':
                assert normalized[col].min() >= 0
                assert normalized[col].max() <= 1

    def test_normalize_data_zscore(self):
        """Test z-score normalization."""
        normalized = self.service.normalize_data(self.sample_data, 'zscore')

        # Check that numeric columns have mean ~0 and std ~1
        numeric_cols = normalized.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'symbol':
                assert abs(normalized[col].mean()) < 1e-10  # Approximately 0
                assert abs(normalized[col].std() - 1) < 1e-10  # Approximately 1

    def test_normalize_data_returns(self):
        """Test returns-based normalization."""
        normalized = self.service.normalize_data(self.sample_data, 'returns')

        # Check that first value is 0 (normalized to itself)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            if col in normalized.columns:
                assert abs(normalized[col].iloc[0]) < 1e-10

    def test_normalize_data_empty(self):
        """Test normalization of empty DataFrame."""
        empty_data = pd.DataFrame()
        normalized = self.service.normalize_data(empty_data, 'minmax')

        assert normalized.empty

    def test_generate_quality_report_basic(self):
        """Test basic quality report generation."""
        test_data = self.sample_data.copy()
        test_data['returns'] = test_data['Adj Close'].pct_change()

        report = self.service.generate_quality_report(test_data, 'AAPL')

        assert report['symbol'] == 'AAPL'
        assert 'timestamp' in report
        assert 'data_summary' in report
        assert 'quality_metrics' in report
        assert 'recommendations' in report

        # Check data summary
        summary = report['data_summary']
        assert summary['total_rows'] == len(test_data)
        assert 'date_range' in summary
        assert 'columns' in summary
        assert 'missing_values' in summary

    def test_generate_quality_report_completeness(self):
        """Test quality report completeness calculation."""
        test_data = self.sample_data.copy()
        test_data.loc[test_data.index[10:15], 'Volume'] = np.nan

        report = self.service.generate_quality_report(test_data, 'AAPL')

        completeness = report['quality_metrics']['completeness']
        assert 0 < completeness < 1  # Should be less than 1 due to missing values

    def test_generate_quality_report_volatility(self):
        """Test quality report volatility calculation."""
        test_data = self.sample_data.copy()
        test_data['returns'] = test_data['Adj Close'].pct_change()

        report = self.service.generate_quality_report(test_data, 'AAPL')

        volatility = report['quality_metrics']['volatility_annualized']
        assert volatility is not None
        assert volatility > 0

    def test_generate_quality_report_empty(self):
        """Test quality report for empty data."""
        report = self.service.generate_quality_report(pd.DataFrame(), 'AAPL')

        assert report['symbol'] == 'AAPL'
        assert report['data_summary']['total_rows'] == 0

    @patch('yfinance.download')
    def test_fetch_historical_data_success(self, mock_download):
        """Test successful historical data fetching."""
        mock_download.return_value = self.sample_data

        # Force online mode to test actual API call
        data = self.service.fetch_historical_data('AAPL', '1y', force_online=True)

        assert not data.empty
        assert 'returns' in data.columns
        assert 'symbol' in data.columns
        assert data['symbol'].iloc[0] == 'AAPL'
        mock_download.assert_called_once_with('AAPL', period='1y', auto_adjust=False)

    @patch('yfinance.download')
    def test_fetch_historical_data_empty(self, mock_download):
        """Test handling of empty historical data."""
        mock_download.return_value = pd.DataFrame()

        data = self.service.fetch_historical_data('INVALID', '1y')

        assert data.empty

    @patch('yfinance.download')
    def test_fetch_historical_data_error(self, mock_download):
        """Test error handling in historical data fetching."""
        mock_download.side_effect = Exception("Network error")

        # Force online mode and expect exception when no offline data available
        with pytest.raises(Exception):
            self.service.fetch_historical_data('AAPL', '1y', force_online=True)

    @patch('yfinance.download')
    def test_fetch_multiple_symbols(self, mock_download):
        """Test fetching multiple symbols."""
        mock_download.return_value = self.sample_data

        results = self.service.fetch_multiple_symbols(['AAPL', 'GOOGL'], '1y')

        assert len(results) == 2
        assert 'AAPL' in results
        assert 'GOOGL' in results
        assert not results['AAPL'].empty

    def test_fetch_multiple_symbols_with_failures(self):
        """Test fetching multiple symbols with some failures."""
        with patch.object(self.service, 'fetch_historical_data') as mock_fetch:
            # Make AAPL succeed and GOOGL fail
            mock_fetch.side_effect = [self.sample_data, Exception("Failed")]

            results = self.service.fetch_multiple_symbols(['AAPL', 'GOOGL'], '1y')

            assert len(results) == 1
            assert 'AAPL' in results
            assert 'GOOGL' not in results

    @patch('yfinance.download')
    def test_fetch_price_data_single_symbol(self, mock_download):
        """Test fetching price data for single symbol."""
        mock_data = self.sample_data[['Adj Close']].copy()
        mock_download.return_value = mock_data

        # Force online mode to test actual API call
        prices = self.service.fetch_price_data(['AAPL'], '1y', force_online=True)

        assert not prices.empty
        assert 'AAPL' in prices.columns
        mock_download.assert_called_once_with(['AAPL'], period='1y', auto_adjust=False)

    @patch('yfinance.Ticker')
    def test_get_symbol_info_success(self, mock_ticker):
        """Test successful symbol info fetching."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 2000000000000,
            'currency': 'USD',
            'exchange': 'NASDAQ'
        }
        mock_ticker.return_value = mock_ticker_instance

        info = self.service.get_symbol_info('AAPL')

        assert info['symbol'] == 'AAPL'
        assert info['name'] == 'Apple Inc.'
        assert info['sector'] == 'Technology'

    @patch('yfinance.Ticker')
    def test_get_symbol_info_error(self, mock_ticker):
        """Test error handling in symbol info fetching."""
        mock_ticker.side_effect = Exception("Failed to get info")

        info = self.service.get_symbol_info('AAPL')

        assert info['symbol'] == 'AAPL'
        assert info['name'] == 'N/A'

    @patch.object(YahooFinanceService, 'fetch_historical_data')
    @patch.object(YahooFinanceService, 'clean_data')
    @patch.object(YahooFinanceService, 'validate_data')
    @patch.object(YahooFinanceService, 'normalize_data')
    @patch.object(YahooFinanceService, 'generate_quality_report')
    def test_fetch_and_process_data_success(self, mock_quality, mock_normalize,
                                           mock_validate, mock_clean, mock_fetch):
        """Test successful end-to-end data pipeline."""
        # Setup mocks
        mock_fetch.return_value = self.sample_data
        mock_clean.return_value = self.sample_data
        mock_validate.return_value = (True, {'is_valid': True, 'issues': []})
        mock_normalize.return_value = self.sample_data
        mock_quality.return_value = {'symbol': 'AAPL', 'quality_metrics': {}}

        results = self.service.fetch_and_process_data(['AAPL'], '1y', 'minmax')

        assert 'AAPL' in results
        assert results['AAPL']['success'] is True
        assert results['AAPL']['data'] is not None
        assert results['AAPL']['validation']['is_valid'] is True
        assert results['AAPL']['quality_report'] is not None

        # Verify all methods were called
        mock_fetch.assert_called_once_with('AAPL', '1y')
        mock_clean.assert_called_once()
        mock_validate.assert_called_once()
        mock_normalize.assert_called_once()
        mock_quality.assert_called_once()

    @patch.object(YahooFinanceService, 'fetch_historical_data')
    def test_fetch_and_process_data_fetch_failure(self, mock_fetch):
        """Test data pipeline handling of fetch failure."""
        mock_fetch.return_value = pd.DataFrame()  # Empty data

        results = self.service.fetch_and_process_data(['AAPL'], '1y')

        assert 'AAPL' not in results  # Should be skipped

    @patch.object(YahooFinanceService, 'fetch_historical_data')
    @patch.object(YahooFinanceService, 'clean_data')
    @patch.object(YahooFinanceService, 'validate_data')
    def test_fetch_and_process_data_validation_failure(self, mock_validate,
                                                       mock_clean, mock_fetch):
        """Test data pipeline handling of validation failure."""
        mock_fetch.return_value = self.sample_data
        mock_clean.return_value = self.sample_data
        mock_validate.return_value = (False, {'is_valid': False, 'issues': ['Invalid data']})

        results = self.service.fetch_and_process_data(['AAPL'], '1y')

        assert 'AAPL' in results
        assert results['AAPL']['success'] is False
        assert results['AAPL']['data'] is None
        assert results['AAPL']['validation']['is_valid'] is False

    def test_service_string_representation(self):
        """Test string representation of service."""
        assert str(self.service) == "YahooFinanceService"

    def test_initialization(self):
        """Test service initialization."""
        service = YahooFinanceService()
        assert service.data_quality_report is None
        assert str(service) == "YahooFinanceService"


if __name__ == '__main__':
    pytest.main([__file__])