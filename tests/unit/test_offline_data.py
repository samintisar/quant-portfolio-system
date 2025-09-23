"""
Unit tests for offline data functionality.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from portfolio.data.yahoo_service import YahooFinanceService


class TestOfflineData:
    """Test suite for offline data functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.service = YahooFinanceService(
            use_offline_data=True,
            offline_data_dir=self.temp_dir
        )
        self.sample_data = self._create_sample_data()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_sample_data(self):
        """Create sample financial data for testing."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]  # Remove weekends

        n_days = len(dates)
        np.random.seed(42)

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

    def test_load_offline_data_no_file(self):
        """Test loading offline data when no file exists."""
        result = self.service.load_offline_data('AAPL', '1y')
        assert result is None

    def test_load_offline_data_processed_file(self):
        """Test loading offline data from processed file."""
        # Create processed data file
        processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(processed_dir)
        data_file = os.path.join(processed_dir, 'AAPL_1y_processed.csv')
        self.sample_data.to_csv(data_file)

        result = self.service.load_offline_data('AAPL', '1y')
        assert result is not None
        assert len(result) == len(self.sample_data)
        assert list(result.columns) == list(self.sample_data.columns)

    def test_load_offline_data_raw_file(self):
        """Test loading offline data from raw file when processed doesn't exist."""
        # Create raw data file
        raw_dir = os.path.join(self.temp_dir, 'raw')
        os.makedirs(raw_dir)
        data_file = os.path.join(raw_dir, 'AAPL_1y_raw.csv')
        self.sample_data.to_csv(data_file)

        result = self.service.load_offline_data('AAPL', '1y')
        assert result is not None
        assert len(result) == len(self.sample_data)

    def test_load_offline_data_prefer_processed(self):
        """Test that processed data is preferred over raw data."""
        # Create both raw and processed files
        raw_dir = os.path.join(self.temp_dir, 'raw')
        processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)

        # Create different data for each file
        raw_data = self.sample_data.iloc[:100]  # Subset
        processed_data = self.sample_data  # Full data

        raw_data.to_csv(os.path.join(raw_dir, 'AAPL_1y_raw.csv'))
        processed_data.to_csv(os.path.join(processed_dir, 'AAPL_1y_processed.csv'))

        result = self.service.load_offline_data('AAPL', '1y')
        # Should load processed data (full dataset)
        assert len(result) == len(processed_data)

    def test_load_offline_data_disabled(self):
        """Test that offline data is not loaded when disabled."""
        service = YahooFinanceService(use_offline_data=False)
        result = service.load_offline_data('AAPL', '1y')
        assert result is None

    def test_save_offline_data_raw(self):
        """Test saving raw data."""
        result = self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')
        assert result is True

        # Verify file exists
        raw_dir = os.path.join(self.temp_dir, 'raw')
        expected_file = os.path.join(raw_dir, 'AAPL_1y_raw.csv')
        assert os.path.exists(expected_file)

        # Verify data contents
        loaded_data = pd.read_csv(expected_file, index_col=0, parse_dates=True)
        assert len(loaded_data) == len(self.sample_data)

    def test_save_offline_data_processed(self):
        """Test saving processed data."""
        result = self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'processed')
        assert result is True

        # Verify file exists
        processed_dir = os.path.join(self.temp_dir, 'processed')
        expected_file = os.path.join(processed_dir, 'AAPL_1y_processed.csv')
        assert os.path.exists(expected_file)

    def test_save_offline_data_disabled(self):
        """Test that data is not saved when offline is disabled."""
        service = YahooFinanceService(use_offline_data=False)
        result = service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')
        assert result is False

    def test_save_offline_data_creates_directory(self):
        """Test that save_offline_data creates directory if it doesn't exist."""
        result = self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')
        assert result is True

        # Verify directory was created
        raw_dir = os.path.join(self.temp_dir, 'raw')
        assert os.path.exists(raw_dir)

    @patch('yfinance.download')
    def test_fetch_historical_data_uses_offline(self, mock_download):
        """Test that fetch_historical_data uses offline data when available."""
        # Create offline data
        self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')

        # Mock online fetch to return different data
        mock_online_data = self.sample_data.copy()
        mock_online_data['Close'] = 999  # Different value
        mock_download.return_value = mock_online_data

        result = self.service.fetch_historical_data('AAPL', '1y')

        # Should return offline data (not 999)
        assert result['Close'].iloc[0] != 999
        # Online fetch should not be called
        mock_download.assert_not_called()

    @patch('yfinance.download')
    def test_fetch_historical_data_force_online(self, mock_download):
        """Test that force_online parameter works."""
        # Create offline data
        self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')

        # Mock online fetch
        mock_online_data = self.sample_data.copy()
        mock_online_data['Close'] = 999  # Different value
        mock_download.return_value = mock_online_data

        result = self.service.fetch_historical_data('AAPL', '1y', force_online=True)

        # Should return online data (999)
        assert result['Close'].iloc[0] == 999
        # Online fetch should be called
        mock_download.assert_called_once()

    @patch('yfinance.download')
    def test_fetch_historical_data_saves_offline(self, mock_download):
        """Test that fetch_historical_data saves data offline."""
        mock_download.return_value = self.sample_data

        result = self.service.fetch_historical_data('AAPL', '1y')

        # Verify data was saved
        raw_dir = os.path.join(self.temp_dir, 'raw')
        expected_file = os.path.join(raw_dir, 'AAPL_1y_raw.csv')
        assert os.path.exists(expected_file)

    @patch('yfinance.download')
    def test_fetch_historical_data_fallback_to_offline(self, mock_download):
        """Test fallback to offline data when online fetch fails."""
        # Create offline data
        self.service.save_offline_data(self.sample_data, 'AAPL', '1y', 'raw')

        # Mock online fetch to fail
        mock_download.side_effect = Exception("Network error")

        result = self.service.fetch_historical_data('AAPL', '1y')

        # Should return offline data despite error
        assert result is not None
        assert len(result) == len(self.sample_data)

    def test_list_available_offline_data_empty(self):
        """Test listing offline data when no files exist."""
        result = self.service.list_available_offline_data()
        assert result == {'raw': [], 'processed': [], 'combined': []}

    def test_list_available_offline_data_with_files(self):
        """Test listing offline data when files exist."""
        # Create test files
        raw_dir = os.path.join(self.temp_dir, 'raw')
        processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)

        # Create files
        open(os.path.join(raw_dir, 'AAPL_1y_raw.csv'), 'w').close()
        open(os.path.join(raw_dir, 'MSFT_1y_raw.csv'), 'w').close()
        open(os.path.join(processed_dir, 'AAPL_1y_processed.csv'), 'w').close()
        open(os.path.join(processed_dir, 'combined_1y_prices.csv'), 'w').close()

        result = self.service.list_available_offline_data()

        assert len(result['raw']) == 2
        assert 'AAPL_1y_raw.csv' in result['raw']
        assert 'MSFT_1y_raw.csv' in result['raw']

        assert len(result['processed']) == 2
        assert 'AAPL_1y_processed.csv' in result['processed']
        assert 'combined_1y_prices.csv' in result['processed']

        assert len(result['combined']) == 1
        assert 'combined_1y_prices.csv' in result['combined']

    def test_list_available_offline_data_disabled(self):
        """Test listing offline data when disabled."""
        service = YahooFinanceService(use_offline_data=False)
        result = service.list_available_offline_data()
        assert result == {'raw': [], 'processed': [], 'combined': []}

    def test_clear_offline_data_all(self):
        """Test clearing all offline data."""
        # Create test files
        raw_dir = os.path.join(self.temp_dir, 'raw')
        processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)

        open(os.path.join(raw_dir, 'AAPL_1y_raw.csv'), 'w').close()
        open(os.path.join(processed_dir, 'AAPL_1y_processed.csv'), 'w').close()

        result = self.service.clear_offline_data()

        assert result is True
        assert not os.path.exists(raw_dir)
        assert not os.path.exists(processed_dir)

    def test_clear_offline_data_specific_type(self):
        """Test clearing specific type of offline data."""
        # Create test files
        raw_dir = os.path.join(self.temp_dir, 'raw')
        processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)

        open(os.path.join(raw_dir, 'AAPL_1y_raw.csv'), 'w').close()
        open(os.path.join(processed_dir, 'AAPL_1y_processed.csv'), 'w').close()

        result = self.service.clear_offline_data('raw')

        assert result is True
        assert not os.path.exists(raw_dir)
        assert os.path.exists(processed_dir)  # Should still exist

    def test_clear_offline_data_disabled(self):
        """Test clearing offline data when disabled."""
        service = YahooFinanceService(use_offline_data=False)
        result = service.clear_offline_data()
        assert result is False

    def test_fetch_price_data_uses_offline(self):
        """Test that fetch_price_data uses offline combined data."""
        # Create combined offline data
        combined_data = pd.DataFrame({
            'AAPL': [100, 101, 102],
            'MSFT': [200, 201, 202]
        }, index=pd.date_range('2023-01-01', periods=3))

        combined_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(combined_dir)
        combined_file = os.path.join(combined_dir, 'combined_1y_prices.csv')
        combined_data.to_csv(combined_file)

        result = self.service.fetch_price_data(['AAPL', 'MSFT'], '1y')

        assert result is not None
        assert len(result) == 3
        assert list(result.columns) == ['AAPL', 'MSFT']

    def test_fetch_price_data_force_online(self):
        """Test that fetch_price_data can be forced online."""
        # Create combined offline data
        combined_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(combined_dir)
        combined_file = os.path.join(combined_dir, 'combined_1y_prices.csv')
        pd.DataFrame().to_csv(combined_file)  # Empty file

        # This should fail offline and fall back to online or force online
        with patch('yfinance.download') as mock_download:
            mock_data = pd.DataFrame({
                'Adj Close': {
                    ('AAPL',): [100, 101],
                    ('MSFT',): [200, 201]
                }
            })
            mock_download.return_value = mock_data

            result = self.service.fetch_price_data(['AAPL', 'MSFT'], '1y', force_online=True)

            assert result is not None
            mock_download.assert_called_once()

    def test_fetch_price_data_saves_offline(self):
        """Test that fetch_price_data saves combined data offline."""
        with patch('yfinance.download') as mock_download:
            mock_data = pd.DataFrame({
                'Adj Close': {
                    ('AAPL',): [100, 101],
                    ('MSFT',): [200, 201]
                }
            })
            mock_download.return_value = mock_data

            result = self.service.fetch_price_data(['AAPL', 'MSFT'], '1y')

            # Verify combined data was saved
            combined_dir = os.path.join(self.temp_dir, 'processed')
            expected_file = os.path.join(combined_dir, 'combined_1y_prices.csv')
            assert os.path.exists(expected_file)


if __name__ == '__main__':
    pytest.main([__file__])