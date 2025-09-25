"""
Data management configuration settings.

This module defines the data fetching and storage policies to prevent
redundant data downloads and maintain clean data storage.
"""

from typing import List, Dict, Any
import os

# Core symbols for portfolio optimization (19 large cap US stocks)
CORE_SYMBOLS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet
    'AMZN',  # Amazon
    'TSLA',  # Tesla
    'META',  # Meta Platforms
    'NVDA',  # NVIDIA
    'JPM',   # JPMorgan Chase
    'JNJ',   # Johnson & Johnson
    'V',     # Visa
    'PG',    # Procter & Gamble
    'UNH',   # UnitedHealth Group
    'HD',    # Home Depot
    'MA',    # Mastercard
    'PYPL',  # PayPal
    'DIS',   # Disney
    'NFLX',  # Netflix
    'ADBE',  # Adobe
    'CRM',   # Salesforce
]

# Benchmark symbols for market-relative analysis
BENCHMARK_SYMBOLS = [
    '^GSPC',  # S&P 500 - market benchmark
    '^TNX',   # 10-year Treasury yield - risk-free rate
]

# ESSENTIAL TIME PERIODS ONLY - Prevents redundant data storage
#
# Rationale:
# - 5y: Default period for most portfolio optimization tasks
# - 10y: Long-term analysis and trend identification
# - Removed: 1y, 3y (redundant with 5y data for most use cases)
# - Removed: 1mo, 3mo, 6mo, 2y (short-term noise, not needed for long-term optimization)
ESSENTIAL_TIME_PERIODS = ['5y', '10y']

# Data storage settings
DATA_STORAGE_SETTINGS = {
    'keep_raw_data': True,        # Keep raw files as backup
    'keep_processed_data': True,  # Keep processed files (primary usage)
    'keep_combined_data': True,   # Keep combined multi-symbol files
    'keep_report_files': False,   # Don't keep individual report files (unused)
    'cleanup_old_data': True,     # Remove old redundant files
}

# Data quality requirements
DATA_QUALITY_REQUIREMENTS = {
    'min_data_points': 252,       # 1 year of trading days
    'max_missing_pct': 0.05,      # 5% maximum missing data
    'require_adj_close': True,    # Adjusted close required
    'require_volume': True,        # Volume data required
    'allow_negative_prices': False, # Negative prices invalid
    'validate_price_logic': True, # High >= Low, etc.
}

# File naming conventions
FILE_NAMING = {
    'raw': '{symbol}_{period}_raw.csv',
    'processed': '{symbol}_{period}_processed.csv',
    'combined': 'combined_{period}_prices.csv',
    'features': '{symbol}_{period}_features.csv'
}

# Directory structure
DATA_DIRECTORIES = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'features': 'data/features',
    'cache': 'data/cache'
}

def get_data_storage_path(data_type: str, symbol: str = None, period: str = None) -> str:
    """
    Generate the proper file path for data storage.

    Args:
        data_type: Type of data ('raw', 'processed', 'combined', 'features')
        symbol: Stock symbol (not needed for combined data)
        period: Time period

    Returns:
        Full file path for storage
    """
    if data_type not in DATA_DIRECTORIES:
        raise ValueError(f"Unknown data type: {data_type}")

    if data_type == 'combined':
        filename = FILE_NAMING['combined'].format(period=period)
    elif symbol and period:
        filename = FILE_NAMING[data_type].format(symbol=symbol, period=period)
    else:
        raise ValueError(f"Symbol and period required for {data_type} data")

    return os.path.join(DATA_DIRECTORIES[data_type], filename)

def validate_time_period(period: str) -> bool:
    """
    Validate that a time period is allowed (prevents unnecessary data downloads).

    Args:
        period: Time period string (e.g., '5y', '10y', '1y')

    Returns:
        True if period is allowed, False otherwise
    """
    return period in ESSENTIAL_TIME_PERIODS

def get_allowed_periods() -> List[str]:
    """Get list of allowed time periods."""
    return ESSENTIAL_TIME_PERIODS.copy()

def get_restricted_periods() -> List[str]:
    """Get list of restricted time periods that should not be used."""
    all_common_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y', '10y', 'ytd', 'max']
    return [p for p in all_common_periods if p not in ESSENTIAL_TIME_PERIODS]

# Data fetching configuration
FETCHING_CONFIG = {
    'default_period': '5y',        # Default to 5-year data
    'timeout_seconds': 30,        # API timeout
    'retry_attempts': 3,          # Number of retry attempts for failed fetches
    'batch_size': 10,             # Symbols to fetch in parallel
    'use_offline_fallback': True,  # Use offline data if online fails
}

# Data cleanup settings
CLEANUP_CONFIG = {
    'remove_report_files': True,   # Remove unused *_report.txt files
    'remove_redundant_periods': True,  # Remove 1y, 3y, etc.
    'keep_min_period': '5y',     # Minimum period to keep
    'consolidate_duplicates': True,  # Remove duplicate data
}

# Configuration validation
def validate_config() -> Dict[str, Any]:
    """
    Validate the data management configuration.

    Returns:
        Validation report with any issues found
    """
    issues = []

    # Check for duplicate symbols
    all_symbols = CORE_SYMBOLS + BENCHMARK_SYMBOLS
    if len(all_symbols) != len(set(all_symbols)):
        issues.append("Duplicate symbols found in CORE_SYMBOLS or BENCHMARK_SYMBOLS")

    # Check time periods are reasonable
    if len(ESSENTIAL_TIME_PERIODS) == 0:
        issues.append("No essential time periods defined")

    # Check directories exist or can be created
    for dir_name, dir_path in DATA_DIRECTORIES.items():
        parent_dir = os.path.dirname(dir_path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {parent_dir}: {e}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'symbols_count': len(CORE_SYMBOLS),
        'benchmarks_count': len(BENCHMARK_SYMBOLS),
        'periods_count': len(ESSENTIAL_TIME_PERIODS)
    }

if __name__ == "__main__":
    # Test configuration validation
    validation = validate_config()
    print("Data Configuration Validation:")
    print(f"Valid: {validation['valid']}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    print(f"Core symbols: {validation['symbols_count']}")
    print(f"Benchmark symbols: {validation['benchmarks_count']}")
    print(f"Essential periods: {validation['periods_count']}")
    print(f"Allowed periods: {get_allowed_periods()}")