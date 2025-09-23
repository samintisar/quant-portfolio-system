"""
Simple Yahoo Finance data service for fetching market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Simple service for fetching data from Yahoo Finance."""

    def __init__(self):
        """Initialize the service."""
        logger.info("Initialized YahooFinanceService")

    def fetch_historical_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """
        Fetch historical price data for a single symbol.

        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

        Returns:
            DataFrame with price data and calculated returns
        """
        try:
            ticker_data = yf.download(symbol, period=period)

            if ticker_data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()

            # Calculate returns
            ticker_data['returns'] = ticker_data['Adj Close'].pct_change()

            # Add basic metadata
            ticker_data['symbol'] = symbol

            logger.info(f"Fetched {len(ticker_data)} data points for {symbol}")
            return ticker_data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def fetch_multiple_symbols(self, symbols: List[str], period: str = "5y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Time period

        Returns:
            Dictionary mapping symbols to their data
        """
        try:
            all_data = {}

            for symbol in symbols:
                try:
                    data = self.fetch_historical_data(symbol, period)
                    if not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue

            logger.info(f"Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
            return all_data

        except Exception as e:
            logger.error(f"Error fetching multiple symbols: {e}")
            return {}

    def fetch_price_data(self, symbols: List[str], period: str = "5y") -> pd.DataFrame:
        """
        Fetch adjusted close prices for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Time period

        Returns:
            DataFrame with adjusted close prices
        """
        try:
            data = yf.download(symbols, period=period)['Adj Close']

            if len(symbols) == 1:
                data = data.to_frame()
                data.columns = symbols

            logger.info(f"Fetched price data for {len(symbols)} symbols")
            return data.dropna()

        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get basic information about a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with symbol information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            relevant_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A')
            }

            return relevant_info

        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return {'symbol': symbol, 'name': 'N/A'}

    def __str__(self):
        return "YahooFinanceService"