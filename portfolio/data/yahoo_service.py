"""
Yahoo Finance data service for portfolio optimization system.

Handles fetching historical data from Yahoo Finance API.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import time

from portfolio.logging_config import get_logger, DataError
from portfolio.config import get_config

logger = get_logger(__name__)


class YahooFinanceService:
    """
    Yahoo Finance data service for fetching historical market data.

    Simple implementation with caching and error handling.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the Yahoo Finance service."""
        self.config = get_config()
        self.cache_dir = cache_dir or self.config.data.cache_dir
        self._cache = {}

        logger.info(f"Initialized YahooFinanceService with cache_dir: {self.cache_dir}")

    def download_historical_data(self, symbols: List[str],
                               period: str = "5y",
                               interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols.

        Args:
            symbols: List of asset symbols
            period: Time period ("1y", "2y", "5y", "10y", etc.)
            interval: Data interval ("1d", "1wk", "1mo")

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not symbols:
            raise DataError("No symbols provided for data download")

        logger.info(f"Downloading historical data for {len(symbols)} symbols, period: {period}")

        # Check cache first
        cache_key = f"{'_'.join(sorted(symbols))}_{period}_{interval}"
        if cache_key in self._cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self._cache[cache_key]

        # Download data
        try:
            data = {}

            # Download in batches to avoid overwhelming the API
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                logger.debug(f"Downloading batch {i//batch_size + 1}: {batch_symbols}")

                try:
                    # Use yfinance to download data
                    ticker_data = yf.download(
                        ' '.join(batch_symbols),
                        period=period,
                        interval=interval,
                        progress=False,
                        group_by='ticker'
                    )

                    # Process the data
                    if isinstance(ticker_data, pd.DataFrame):
                        if len(batch_symbols) == 1:
                            # Single symbol - data might have MultiIndex columns
                            if isinstance(ticker_data.columns, pd.MultiIndex):
                                # Get the second level which contains the actual OHLCV column names
                                ticker_data.columns = ticker_data.columns.get_level_values(1).str.lower()
                            data[batch_symbols[0]] = ticker_data
                        else:
                            # Multiple symbols - need to process multi-level columns
                            for symbol in batch_symbols:
                                if symbol in ticker_data.columns.levels[0]:
                                    symbol_data = ticker_data[symbol].copy()

                                    # Handle MultiIndex columns
                                    if isinstance(symbol_data.columns, pd.MultiIndex):
                                        symbol_data.columns = symbol_data.columns.get_level_values(0).str.lower()
                                    else:
                                        symbol_data.columns = symbol_data.columns.str.lower()

                                    data[symbol] = symbol_data
                                else:
                                    logger.warning(f"No data found for symbol: {symbol}")
                                    data[symbol] = pd.DataFrame()

                except Exception as e:
                    logger.error(f"Error downloading batch {batch_symbols}: {e}")
                    # Create empty DataFrames for failed symbols
                    for symbol in batch_symbols:
                        data[symbol] = pd.DataFrame()

                # Rate limiting
                time.sleep(0.5)

            # Clean and standardize data
            data = self._clean_data_frames(data)

            # Cache the result
            self._cache[cache_key] = data

            logger.info(f"Successfully downloaded data for {len([d for d in data.values() if not d.empty])} symbols")
            return data

        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            raise DataError(f"Failed to download historical data: {e}")

    def fetch_historical_data(self, symbol: str,
                              period: str = "5y",
                              interval: str = "1d",
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.

        Args:
            symbol: Asset symbol
            period: Time period ("1y", "2y", "5y", "10y", etc.)
            interval: Data interval ("1d", "1wk", "1mo")
            start_date: Start date in YYYY-MM-DD format (overrides period)
            end_date: End date in YYYY-MM-DD format (overrides period)

        Returns:
            DataFrame with historical data including returns
        """
        try:
            # Use download_historical_data for single symbol
            data_dict = self.download_historical_data([symbol], period=period, interval=interval)

            if symbol in data_dict and not data_dict[symbol].empty:
                df = data_dict[symbol].copy()

                # Filter by date range if provided
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

                # Calculate returns if we have close prices
                if 'close' in df.columns and not df['close'].empty:
                    df['returns'] = df['close'].pct_change().dropna()

                logger.info(f"Successfully fetched historical data for {symbol}: {len(df)} data points")
                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def download_asset_info(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Download basic asset information.

        Args:
            symbols: List of asset symbols

        Returns:
            Dictionary mapping symbols to asset info
        """
        logger.info(f"Downloading asset info for {len(symbols)} symbols")

        try:
            asset_info = {}

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # Extract relevant information
                    relevant_info = {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', None),
                        'currency': info.get('currency', 'USD'),
                        'exchange': info.get('exchange', 'Unknown')
                    }

                    asset_info[symbol] = relevant_info

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Error getting info for {symbol}: {e}")
                    asset_info[symbol] = {
                        'symbol': symbol,
                        'name': symbol,
                        'sector': 'Unknown',
                        'market_cap': None
                    }

            return asset_info

        except Exception as e:
            logger.error(f"Error downloading asset info: {e}")
            return {symbol: {'symbol': symbol, 'name': symbol, 'sector': 'Unknown'} for symbol in symbols}

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for symbols.

        Args:
            symbols: List of asset symbols

        Returns:
            Dictionary mapping symbols to latest prices
        """
        logger.info(f"Getting latest prices for {len(symbols)} symbols")

        try:
            # Get recent data (last 5 days)
            recent_data = self.download_historical_data(symbols, period="5d", interval="1d")

            latest_prices = {}
            for symbol, df in recent_data.items():
                if not df.empty and 'close' in df.columns:
                    latest_prices[symbol] = df['close'].iloc[-1]
                else:
                    logger.warning(f"No price data available for {symbol}")
                    latest_prices[symbol] = None

            return latest_prices

        except Exception as e:
            logger.error(f"Error getting latest prices: {e}")
            return {symbol: None for symbol in symbols}

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has data.

        Args:
            symbol: Asset symbol to validate

        Returns:
            True if symbol is valid
        """
        try:
            # Try to get a small amount of data
            data = self.download_historical_data([symbol], period="5d", interval="1d")
            return len(data.get(symbol, pd.DataFrame())) > 0

        except Exception as e:
            logger.debug(f"Symbol {symbol} validation failed: {e}")
            return False

    def get_historical_returns(self, symbols: List[str],
                             period: str = "5y",
                             return_type: str = "simple") -> Dict[str, pd.Series]:
        """
        Get historical returns for symbols.

        Args:
            symbols: List of asset symbols
            period: Time period
            return_type: "simple" or "log"

        Returns:
            Dictionary mapping symbols to return series
        """
        logger.info(f"Calculating {return_type} returns for {len(symbols)} symbols")

        try:
            # Download price data
            price_data = self.download_historical_data(symbols, period=period)

            returns = {}
            for symbol, df in price_data.items():
                if not df.empty and 'close' in df.columns:
                    prices = df['close']

                    if return_type == "simple":
                        returns[symbol] = prices.pct_change().dropna()
                    elif return_type == "log":
                        returns[symbol] = np.log(prices / prices.shift(1)).dropna()
                    else:
                        raise ValueError(f"Invalid return_type: {return_type}")
                else:
                    logger.warning(f"No price data for {symbol}")
                    returns[symbol] = pd.Series()

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return {symbol: pd.Series() for symbol in symbols}

    def _clean_data_frames(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean and standardize DataFrame data.

        Args:
            data_dict: Dictionary of DataFrames to clean

        Returns:
            Cleaned DataFrames
        """
        cleaned_data = {}

        for symbol, df in data_dict.items():
            if df.empty:
                cleaned_data[symbol] = df
                continue

            try:
                # Standardize column names
                df = df.copy()

                # Handle MultiIndex columns (from yfinance group_by='ticker')
                if isinstance(df.columns, pd.MultiIndex):
                    # For MultiIndex, we only care about the first level (the column names)
                    df.columns = df.columns.get_level_values(0).str.lower()
                else:
                    # For regular Index, just convert to lowercase
                    df.columns = df.columns.str.lower()

                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.warning(f"Missing columns {missing_columns} for {symbol}")
                    logger.warning(f"Available columns: {list(df.columns)}")
                    # Create missing columns with NaN values
                    for col in missing_columns:
                        df[col] = np.nan

                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Converting index to datetime for {symbol}")
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        logger.error(f"Failed to convert index for {symbol}: {e}")
                        df = pd.DataFrame()

                # Sort by date
                df = df.sort_index()

                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]

                # Basic data cleaning
                df = self._clean_individual_dataframe(df)

                cleaned_data[symbol] = df

            except Exception as e:
                logger.error(f"Error cleaning data for {symbol}: {e}")
                cleaned_data[symbol] = pd.DataFrame()

        return cleaned_data

    def _clean_individual_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean individual DataFrame.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        try:
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Remove zero or negative prices (for most assets)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    df = df[df[col] > 0]

            # Remove zero volume
            if 'volume' in df.columns:
                df = df[df['volume'] >= 0]

            # Handle outliers (basic filter)
            for col in price_columns:
                if col in df.columns:
                    # Remove extreme price changes (>50% in one day)
                    if len(df) > 1:
                        returns = df[col].pct_change().abs()
                        df = df[returns <= 0.5]

            return df

        except Exception as e:
            logger.warning(f"Error cleaning DataFrame: {e}")
            return df

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.info("Cleared data cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "cache_dir": self.cache_dir
        }

    def get_market_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Get number of market days between two dates.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Number of market days
        """
        try:
            # Get SPY data as market proxy
            spy_data = self.download_historical_data(['SPY'],
                                                   period=f"{(end_date - start_date).days}d",
                                                   interval="1d")

            if 'SPY' in spy_data and not spy_data['SPY'].empty:
                return len(spy_data['SPY'])
            else:
                # Fallback to weekday calculation
                return len(pd.bdate_range(start_date, end_date))

        except Exception as e:
            logger.warning(f"Error calculating market days: {e}")
            # Fallback to weekday calculation
            return len(pd.bdate_range(start_date, end_date))

    def get_trading_days_in_year(self, year: int) -> int:
        """
        Get number of trading days in a year.

        Args:
            year: Year to calculate for

        Returns:
            Number of trading days
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        return self.get_market_days(start_date, end_date)

    def __str__(self) -> str:
        """String representation."""
        return f"YahooFinanceService(cache_size={len(self._cache)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"YahooFinanceService(cache_dir='{self.cache_dir}', cache_size={len(self._cache)})"