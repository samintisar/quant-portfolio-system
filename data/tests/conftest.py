"""
Pytest configuration and fixtures for data preprocessing tests.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    data = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(95, 115, 100),
        'low': np.random.uniform(85, 105, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000000, 10000000, 100),
        'symbol': 'TEST'
    }, index=dates)

    # Ensure high >= low >= close >= open
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_missing_data():
    """Sample data with missing values for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000000, 10000000, 100),
        'returns': np.random.normal(0, 0.02, 100)
    })

    # Add missing values
    data.iloc[10:15, 0] = np.nan
    data.iloc[20:25, 1] = np.nan
    data.iloc[30:35, 2] = np.nan

    return data


@pytest.fixture
def sample_outlier_data():
    """Sample data with outliers for testing."""
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.normal(100, 5, 100),
        'volume': np.random.normal(5000000, 1000000, 100)
    })

    # Add outliers
    data.loc[10, 'price'] = 1000  # Extreme outlier
    data.loc[11, 'price'] = -50   # Negative outlier
    data.loc[20, 'volume'] = 500000000  # Extreme volume outlier

    return data