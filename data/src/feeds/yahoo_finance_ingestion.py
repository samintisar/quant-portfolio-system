"""
Yahoo Finance data ingestion module for quantitative trading system.

Provides unified interface for fetching historical data from Yahoo Finance API
for equities, ETFs, FX pairs, and bonds with comprehensive validation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import validation framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'unit'))
try:
    from test_data_validation import DataValidator, ValidationResult
except ImportError:
    # If validation module not available, create basic validation
    from dataclasses import dataclass
    from typing import List, Dict, Any, Optional

    @dataclass
    class ValidationResult:
        is_valid: bool
        errors: List[str]
        warnings: List[str]
        metrics: Dict[str, Any]

    class DataValidator:
        @staticmethod
        def validate_price_data(prices, min_price=0.01, max_price=1000000):
            errors = []
            warnings = []
            metrics = {'count': len(prices) if prices else 0}

            if not prices:
                errors.append("Price data is empty")
                return ValidationResult(False, errors, warnings, metrics)

            import numpy as np
            prices_array = np.array(prices)

            negative_count = np.sum(prices_array < 0)
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative prices")

            metrics.update({
                'mean': float(np.mean(prices_array)),
                'std': float(np.std(prices_array)),
                'min': float(np.min(prices_array)),
                'max': float(np.max(prices_array))
            })

            return ValidationResult(len(errors) == 0, errors, warnings, metrics)


class AssetClass(Enum):
    """Asset classes supported by the data ingestion system."""
    EQUITY = "equity"
    ETF = "etf"
    FX = "fx"
    BOND = "bond"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass
class DataRequest:
    """Data request configuration."""
    symbol: str
    asset_class: AssetClass
    start_date: datetime
    end_date: datetime
    interval: str = "1d"  # 1d, 1wk, 1mo
    include_dividends: bool = False
    include_splits: bool = False
    include_actions: bool = False


@dataclass
class IngestionResult:
    """Result of data ingestion operation."""
    success: bool
    data: Optional[pd.DataFrame]
    metadata: Dict[str, any]
    validation_result: ValidationResult
    warnings: List[str]
    error_message: Optional[str] = None


class YahooFinanceIngestion:
    """Yahoo Finance data ingestion engine with comprehensive validation."""

    def __init__(self, max_workers: int = 5, rate_limit: float = 0.1):
        """
        Initialize Yahoo Finance ingestion engine.

        Args:
            max_workers: Maximum concurrent requests
            rate_limit: Minimum time between requests (seconds)
        """
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        self.last_request_time = 0

        # Yahoo Finance specific mappings
        self._init_yahoo_mappings()

    def _init_yahoo_mappings(self):
        """Initialize Yahoo Finance specific symbol mappings."""
        self.fx_mappings = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'NZDUSD': 'NZDUSD=X',
            'USDCAD': 'USDCAD=X',
            'EURGBP': 'EURGBP=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X'
        }

        self.bond_mappings = {
            'US10Y': '^TNX',
            'US2Y': '^FVX',
            'US5Y': '^FVX',
            'US30Y': '^TYX',
            'BUND10Y': '^GDAXI',
            'GILT10Y': '^FTSE'
        }

        self.commodity_mappings = {
            'GOLD': 'GC=F',
            'SILVER': 'SI=F',
            'OIL': 'CL=F',
            'NATGAS': 'NG=F',
            'COPPER': 'HG=F',
            'CORN': 'ZC=F',
            'WHEAT': 'ZW=F'
        }

    def _get_yahoo_symbol(self, symbol: str, asset_class: AssetClass) -> str:
        """Convert internal symbol to Yahoo Finance symbol."""
        if asset_class == AssetClass.FX:
            return self.fx_mappings.get(symbol.upper(), symbol + '=X')
        elif asset_class == AssetClass.BOND:
            return self.bond_mappings.get(symbol.upper(), symbol)
        elif asset_class == AssetClass.COMMODITY:
            return self.commodity_mappings.get(symbol.upper(), symbol)
        else:
            return symbol

    def _respect_rate_limit(self):
        """Ensure rate limiting for Yahoo Finance API."""
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit:
            time.sleep(self.rate_limit - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def _fetch_single_asset(self, request: DataRequest) -> IngestionResult:
        """Fetch data for a single asset with validation."""
        try:
            self._respect_rate_limit()

            # Get Yahoo Finance symbol
            yahoo_symbol = self._get_yahoo_symbol(request.symbol, request.asset_class)

            # Create ticker object
            ticker = yf.Ticker(yahoo_symbol)

            # Fetch historical data
            data = ticker.history(
                start=request.start_date.strftime('%Y-%m-%d'),
                end=request.end_date.strftime('%Y-%m-%d'),
                interval=request.interval
            )

            if data.empty:
                return IngestionResult(
                    success=False,
                    data=None,
                    metadata={'symbol': request.symbol, 'yahoo_symbol': yahoo_symbol},
                    validation_result=ValidationResult(False, ['No data returned'], [], {}),
                    warnings=[],
                    error_message=f"No data found for {request.symbol}"
                )

            # Clean and validate data
            cleaned_data = self._clean_data(data, request)
            validation_result = self._validate_data(cleaned_data, request)

            # Create metadata
            metadata = {
                'symbol': request.symbol,
                'yahoo_symbol': yahoo_symbol,
                'asset_class': request.asset_class.value,
                'start_date': request.start_date.isoformat(),
                'end_date': request.end_date.isoformat(),
                'interval': request.interval,
                'data_points': len(cleaned_data),
                'columns': list(cleaned_data.columns),
                'data_range': {
                    'min_date': cleaned_data.index.min().isoformat(),
                    'max_date': cleaned_data.index.max().isoformat()
                }
            }

            return IngestionResult(
                success=validation_result.is_valid,
                data=cleaned_data if validation_result.is_valid else None,
                metadata=metadata,
                validation_result=validation_result,
                warnings=validation_result.warnings,
                error_message=None if validation_result.is_valid else "; ".join(validation_result.errors)
            )

        except Exception as e:
            self.logger.error(f"Error fetching data for {request.symbol}: {str(e)}")
            return IngestionResult(
                success=False,
                data=None,
                metadata={'symbol': request.symbol},
                validation_result=ValidationResult(False, [str(e)], [], {}),
                warnings=[],
                error_message=str(e)
            )

    def _clean_data(self, data: pd.DataFrame, request: DataRequest) -> pd.DataFrame:
        """Clean and standardize the fetched data."""
        # Make a copy to avoid modifying original
        cleaned = data.copy()

        # Standardize column names (lowercase)
        cleaned.columns = [col.lower().replace(' ', '_') for col in cleaned.columns]

        # Remove any rows with all NaN values
        cleaned = cleaned.dropna(how='all')

        # Ensure datetime index
        cleaned.index = pd.to_datetime(cleaned.index)

        # Sort by date
        cleaned = cleaned.sort_index()

        # Remove duplicates
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]

        # Add returns columns if OHLC data is available
        if all(col in cleaned.columns for col in ['open', 'high', 'low', 'close']):
            cleaned['return'] = cleaned['close'].pct_change()
            cleaned['log_return'] = np.log(cleaned['close'] / cleaned['close'].shift(1))

        # Add volume-based metrics if available
        if 'volume' in cleaned.columns:
            cleaned['dollar_volume'] = cleaned['close'] * cleaned['volume']

        return cleaned

    def _validate_data(self, data: pd.DataFrame, request: DataRequest) -> ValidationResult:
        """Validate the ingested data using the validation framework."""
        errors = []
        warnings = []
        metrics = {}

        # Basic validation
        if data.empty:
            errors.append("Data is empty after cleaning")
            return ValidationResult(False, errors, warnings, metrics)

        # Validate price data
        if 'close' in data.columns:
            price_validation = self.validator.validate_price_data(data['close'].tolist())
            errors.extend(price_validation.errors)
            warnings.extend(price_validation.warnings)
            metrics.update(price_validation.metrics)

        # Validate returns data
        if 'return' in data.columns:
            returns_data = data['return'].dropna().tolist()
            if returns_data:
                returns_validation = self.validator.validate_returns_data(returns_data)
                errors.extend(returns_validation.errors)
                warnings.extend(returns_validation.warnings)
                metrics.update({f'returns_{k}': v for k, v in returns_validation.metrics.items()})

        # Validate time series consistency
        dates = data.index.tolist()
        if 'close' in data.columns:
            values = data['close'].tolist()
            time_series_validation = self.validator.validate_time_series_data(dates, values)
            errors.extend(time_series_validation.errors)
            warnings.extend(time_series_validation.warnings)
            metrics.update({f'ts_{k}': v for k, v in time_series_validation.metrics.items()})

        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100
        if missing_pct > 10:
            warnings.append(f"High missing data percentage: {missing_pct:.2f}%")

        metrics['missing_data_pct'] = missing_pct
        metrics['rows'] = len(data)
        metrics['columns'] = len(data.columns)

        return ValidationResult(len(errors) == 0, errors, warnings, metrics)

    def fetch_equity_data(self, symbol: str, start_date: datetime, end_date: datetime,
                         interval: str = "1d") -> IngestionResult:
        """Fetch historical equity data."""
        request = DataRequest(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        return self._fetch_single_asset(request)

    def fetch_etf_data(self, symbol: str, start_date: datetime, end_date: datetime,
                      interval: str = "1d") -> IngestionResult:
        """Fetch historical ETF data."""
        request = DataRequest(
            symbol=symbol,
            asset_class=AssetClass.ETF,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        return self._fetch_single_asset(request)

    def fetch_fx_data(self, symbol: str, start_date: datetime, end_date: datetime,
                     interval: str = "1d") -> IngestionResult:
        """Fetch historical FX data."""
        request = DataRequest(
            symbol=symbol,
            asset_class=AssetClass.FX,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        return self._fetch_single_asset(request)

    def fetch_bond_data(self, symbol: str, start_date: datetime, end_date: datetime,
                       interval: str = "1d") -> IngestionResult:
        """Fetch historical bond data."""
        request = DataRequest(
            symbol=symbol,
            asset_class=AssetClass.BOND,
            start_date=end_date,
            end_date=end_date,
            interval=interval
        )
        return self._fetch_single_asset(request)

    def fetch_multiple_assets(self, requests: List[DataRequest]) -> Dict[str, IngestionResult]:
        """Fetch data for multiple assets concurrently."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(self._fetch_single_asset, request): request.symbol
                for request in requests
            }

            # Collect results as they complete
            for future in as_completed(future_to_request):
                symbol = future_to_request[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    results[symbol] = IngestionResult(
                        success=False,
                        data=None,
                        metadata={'symbol': symbol},
                        validation_result=ValidationResult(False, [str(e)], [], {}),
                        warnings=[],
                        error_message=str(e)
                    )

        return results

    def get_asset_info(self, symbol: str, asset_class: AssetClass) -> Dict[str, any]:
        """Get basic information about an asset."""
        try:
            yahoo_symbol = self._get_yahoo_symbol(symbol, asset_class)
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'yahoo_symbol': yahoo_symbol,
                'asset_class': asset_class.value,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'last_price': info.get('regularMarketPrice', 0),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'yahoo_symbol': self._get_yahoo_symbol(symbol, asset_class),
                'asset_class': asset_class.value,
                'error': str(e)
            }

    def create_batch_requests(self, symbols: List[str], asset_class: AssetClass,
                            start_date: datetime, end_date: datetime,
                            interval: str = "1d") -> List[DataRequest]:
        """Create a batch of data requests."""
        return [
            DataRequest(
                symbol=symbol,
                asset_class=asset_class,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            for symbol in symbols
        ]


# Import time module for rate limiting
import time


def create_default_ingestion() -> YahooFinanceIngestion:
    """Create a default Yahoo Finance ingestion instance."""
    return YahooFinanceIngestion(max_workers=5, rate_limit=0.1)


# Example usage and convenience functions
def fetch_sp500_components(start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
    """Fetch data for S&P 500 components (simplified - using major stocks)."""
    # Simplified list of major S&P 500 components
    sp500_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM', 'BAC'
    ]

    ingestion = create_default_ingestion()
    requests = ingestion.create_batch_requests(
        sp500_symbols, AssetClass.EQUITY, start_date, end_date
    )

    return ingestion.fetch_multiple_assets(requests)


def fetch_major_etfs(start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
    """Fetch data for major ETFs."""
    etf_symbols = [
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'GLD', 'TLT', 'XLF', 'XLK',
        'EFA', 'EEM', 'VWO', 'AGG', 'LQD', 'HYG', 'XLE', 'XLU', 'XLV', 'XLP'
    ]

    ingestion = create_default_ingestion()
    requests = ingestion.create_batch_requests(
        etf_symbols, AssetClass.ETF, start_date, end_date
    )

    return ingestion.fetch_multiple_assets(requests)


def fetch_major_fx_pairs(start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
    """Fetch data for major FX pairs."""
    fx_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']

    ingestion = create_default_ingestion()
    requests = ingestion.create_batch_requests(
        fx_symbols, AssetClass.FX, start_date, end_date
    )

    return ingestion.fetch_multiple_assets(requests)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    ingestion = create_default_ingestion()

    # Test single equity fetch
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("Testing single equity fetch...")
    result = ingestion.fetch_equity_data('AAPL', start_date, end_date)
    print(f"Success: {result.success}")
    print(f"Data points: {result.metadata.get('data_points', 0)}")
    print(f"Validation errors: {len(result.validation_result.errors)}")
    print(f"Validation warnings: {len(result.validation_result.warnings)}")

    if result.success and result.data is not None:
        print(f"Sample data shape: {result.data.shape}")
        print(f"Columns: {list(result.data.columns)}")

    print("\nTesting batch ETF fetch...")
    etf_results = fetch_major_etfs(start_date, end_date)
    successful_fetches = sum(1 for r in etf_results.values() if r.success)
    print(f"Successful fetches: {successful_fetches}/{len(etf_results)}")