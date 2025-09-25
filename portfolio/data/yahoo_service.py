"""
Enhanced Yahoo Finance data service with offline data storage support.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Enhanced service for fetching and processing data from Yahoo Finance with offline storage."""

    def __init__(self, use_offline_data: bool = True, offline_data_dir: str = "data"):
        """
        Initialize the service.

        Args:
            use_offline_data: Whether to use offline data when available
            offline_data_dir: Directory containing offline data files
        """
        logger.info("Initialized YahooFinanceService")
        self.data_quality_report = None
        self.use_offline_data = use_offline_data
        self.offline_data_dir = offline_data_dir

    def load_offline_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Load data from offline storage.

        Args:
            symbol: Stock symbol
            period: Time period

        Returns:
            DataFrame with offline data or None if not found
        """
        if not self.use_offline_data:
            return None

        try:
            # Try processed data first, then raw data
            processed_file = os.path.join(self.offline_data_dir, "processed", f"{symbol}_{period}_processed.csv")
            raw_file = os.path.join(self.offline_data_dir, "raw", f"{symbol}_{period}_raw.csv")

            file_to_load = None
            if os.path.exists(processed_file):
                file_to_load = processed_file
            elif os.path.exists(raw_file):
                file_to_load = raw_file

            if file_to_load:
                data = pd.read_csv(file_to_load, index_col=0, parse_dates=True)
                logger.info(f"Loaded offline data for {symbol} ({period}): {len(data)} rows")
                return data
            else:
                logger.debug(f"No offline data found for {symbol} ({period})")
                return None

        except Exception as e:
            logger.warning(f"Error loading offline data for {symbol} ({period}): {e}")
            return None

    def save_offline_data(self, data: pd.DataFrame, symbol: str, period: str, data_type: str = "raw") -> bool:
        """
        Save data to offline storage.

        Args:
            data: DataFrame to save
            symbol: Stock symbol
            period: Time period
            data_type: Type of data ('raw' or 'processed')

        Returns:
            True if successful, False otherwise
        """
        if not self.use_offline_data:
            logger.debug("Offline data is disabled, not saving data")
            return False

        try:
            # Ensure directory exists
            save_dir = os.path.join(self.offline_data_dir, data_type)
            os.makedirs(save_dir, exist_ok=True)

            # Save data
            filename = os.path.join(save_dir, f"{symbol}_{period}_{data_type}.csv")
            data.to_csv(filename)
            logger.info(f"Saved {data_type} data for {symbol} ({period}) to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving offline data for {symbol} ({period}): {e}")
            return False

    def fetch_historical_data(self, symbol: str, period: str = "5y", force_online: bool = False) -> pd.DataFrame:
        """
        Fetch historical price data for a single symbol.

        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
                   NOTE: Default is 5y to prevent redundant data storage

        Returns:
            DataFrame with price data and calculated returns
        """
        # Try offline data first (unless forced online)
        if not force_online:
            offline_data = self.load_offline_data(symbol, period)
            if offline_data is not None:
                return offline_data

        # Fetch from online API
        try:
            logger.info(f"Fetching online data for {symbol} ({period})")
            ticker_data = yf.download(symbol, period=period, auto_adjust=False)

            if ticker_data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(ticker_data.columns, pd.MultiIndex):
                ticker_data.columns = ticker_data.columns.get_level_values(0)

            # Calculate returns
            ticker_data['returns'] = ticker_data['Adj Close'].pct_change()

            # Add basic metadata
            ticker_data['symbol'] = symbol

            # Save offline data for future use
            if self.use_offline_data:
                self.save_offline_data(ticker_data, symbol, period, "raw")

            logger.info(f"Fetched {len(ticker_data)} data points for {symbol}")
            return ticker_data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            # If offline data exists and online fetch fails, return offline data as fallback
            if not force_online:
                offline_data = self.load_offline_data(symbol, period)
                if offline_data is not None:
                    logger.info(f"Falling back to offline data for {symbol}")
                    return offline_data
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

    def fetch_price_data(self, symbols: List[str], period: str = "5y", force_online: bool = False) -> pd.DataFrame:
        """
        Fetch adjusted close prices for multiple symbols.

        Args:
            symbols: List of stock symbols
            period: Time period
            force_online: Force fetch from online API even if offline data exists

        Returns:
            DataFrame with adjusted close prices
        """
        # Try offline combined data first
        if not force_online and self.use_offline_data:
            combined_file = os.path.join(self.offline_data_dir, "processed", f"combined_{period}_prices.csv")
            if os.path.exists(combined_file):
                try:
                    data = pd.read_csv(combined_file, index_col=0, parse_dates=True)
                    # Filter to requested symbols
                    available_symbols = [s for s in symbols if s in data.columns]
                    if available_symbols:
                        data = data[available_symbols]
                        logger.info(f"Loaded offline combined price data for {len(available_symbols)}/{len(symbols)} symbols")
                        return data.dropna()
                except Exception as e:
                    logger.warning(f"Error loading offline combined data: {e}")

        # Fetch from online API
        try:
            logger.info(f"Fetching online price data for {len(symbols)} symbols")
            data = yf.download(symbols, period=period, auto_adjust=False)['Adj Close']

            # Robust single-symbol handling: Series or single-column DataFrame
            if len(symbols) == 1:
                sym = symbols[0]
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=sym)
                else:
                    # Already a DataFrame; ensure column name is the symbol
                    if data.shape[1] == 1:
                        data.columns = [sym]

            # Save combined data offline
            if self.use_offline_data:
                filename = f"combined_{period}_prices"
                save_dir = os.path.join(self.offline_data_dir, "processed")
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, f"{filename}.csv")
                data.to_csv(filepath)
                logger.info(f"Saved combined price data to {filepath}")

            logger.info(f"Fetched price data for {len(symbols)} symbols")
            return data.dropna()

        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            raise

    def list_available_offline_data(self) -> Dict[str, List[str]]:
        """
        List all available offline data files.

        Returns:
            Dictionary with keys 'raw', 'processed', 'combined' and lists of available files
        """
        available_data = {'raw': [], 'processed': [], 'combined': []}

        if not self.use_offline_data or not os.path.exists(self.offline_data_dir):
            return available_data

        try:
            # Check raw data
            raw_dir = os.path.join(self.offline_data_dir, 'raw')
            if os.path.exists(raw_dir):
                available_data['raw'] = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]

            # Check processed data
            processed_dir = os.path.join(self.offline_data_dir, 'processed')
            if os.path.exists(processed_dir):
                available_data['processed'] = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
                # Identify combined files
                available_data['combined'] = [f for f in available_data['processed'] if 'combined' in f]

        except Exception as e:
            logger.warning(f"Error listing offline data: {e}")

        return available_data

    def clear_offline_data(self, data_type: str = None) -> bool:
        """
        Clear offline data files.

        Args:
            data_type: Type of data to clear ('raw', 'processed', 'combined', or None for all)

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.use_offline_data:
                logger.warning("Offline data is disabled")
                return False

            if data_type is None:
                # Clear all data
                import shutil
                if os.path.exists(self.offline_data_dir):
                    shutil.rmtree(self.offline_data_dir)
                    os.makedirs(self.offline_data_dir)
                logger.info("Cleared all offline data")
                return True
            else:
                # Clear specific type (don't recreate directory)
                target_dir = os.path.join(self.offline_data_dir, data_type)
                if os.path.exists(target_dir):
                    import shutil
                    shutil.rmtree(target_dir)
                logger.info(f"Cleared offline {data_type} data")
                return True

        except Exception as e:
            logger.error(f"Error clearing offline data: {e}")
            return False

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

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess financial data.

        Args:
            data: Raw price data DataFrame

        Returns:
            Cleaned DataFrame
        """
        try:
            if data.empty:
                return data

            cleaned_data = data.copy()

            # Remove duplicate rows
            cleaned_data = cleaned_data.drop_duplicates()

            # Fix negative Low values - replace with Close or Open values
            if 'Low' in cleaned_data.columns:
                negative_low_mask = cleaned_data['Low'] < 0
                if negative_low_mask.any():
                    logger.warning(f"Found {negative_low_mask.sum()} negative Low values - fixing with Close/Open values")
                    # Try to fix with Close value first, then Open if Close is also negative
                    cleaned_data.loc[negative_low_mask, 'Low'] = np.where(
                        cleaned_data.loc[negative_low_mask, 'Close'] > 0,
                        cleaned_data.loc[negative_low_mask, 'Close'],
                        cleaned_data.loc[negative_low_mask, 'Open']
                    )
                    # If still negative, use absolute value as last resort
                    still_negative = cleaned_data['Low'] < 0
                    if still_negative.any():
                        cleaned_data.loc[still_negative, 'Low'] = cleaned_data.loc[still_negative, 'Low'].abs()

            # Fix other negative price columns (Open, High, Close, Adj Close)
            price_columns = ['Open', 'High', 'Close', 'Adj Close']
            for col in price_columns:
                if col in cleaned_data.columns:
                    negative_mask = cleaned_data[col] < 0
                    if negative_mask.any():
                        logger.warning(f"Found {negative_mask.sum()} negative {col} values - converting to positive")
                        cleaned_data.loc[negative_mask, col] = cleaned_data.loc[negative_mask, col].abs()

            # Handle missing values
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Forward fill missing values, then backward fill
                cleaned_data[col] = cleaned_data[col].ffill().bfill()

            # Remove rows with all NaN values
            cleaned_data = cleaned_data.dropna(how='all')

            # Ensure data is sorted by date
            if 'Date' in cleaned_data.columns:
                cleaned_data = cleaned_data.sort_values('Date')
            elif cleaned_data.index.name == 'Date' or isinstance(cleaned_data.index, pd.DatetimeIndex):
                cleaned_data = cleaned_data.sort_index()

            logger.info(f"Cleaned data: {len(data)} -> {len(cleaned_data)} rows")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data

    def validate_data(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, any]]:
        """
        Validate data quality and completeness.

        Args:
            data: DataFrame to validate
            symbol: Stock symbol for validation context

        Returns:
            Tuple of (is_valid, validation_report)
        """
        try:
            validation_report = {
                'symbol': symbol,
                'total_rows': len(data),
                'is_valid': True,
                'issues': [],          # critical issues that invalidate the dataset
                'warnings': []         # non-fatal quality warnings
            }

            if data.empty:
                validation_report['is_valid'] = False
                validation_report['issues'].append('No data available')
                return False, validation_report

            # Check required columns
            required_columns = ['Adj Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_report['is_valid'] = False
                validation_report['issues'].append(f'Missing required columns: {missing_columns}')

            # Check for negative prices in all price columns
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = data[data[col] < 0]
                    if len(negative_prices) > 0:
                        validation_report['warnings'].append(
                            f'Found {len(negative_prices)} negative {col} prices'
                        )

            # Check for zero prices (critical issue)
            if 'Adj Close' in data.columns:
                zero_prices = data[data['Adj Close'] <= 0]
                if len(zero_prices) > 0:
                    validation_report['issues'].append(
                        f'Found {len(zero_prices)} zero/negative Adj Close prices'
                    )

            # Check for illogical price relationships
            if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
                # High should be >= Low
                invalid_hl = data[data['High'] < data['Low']]
                if len(invalid_hl) > 0:
                    validation_report['issues'].append(
                        f'Found {len(invalid_hl)} rows where High < Low'
                    )

            # Check for data gaps using business days expectation (trading days)
            if len(data) > 1 and isinstance(data.index, pd.DatetimeIndex):
                start_dt = data.index.min()
                end_dt = data.index.max()
                expected_bdays = len(pd.bdate_range(start=start_dt, end=end_dt))
                actual_days = len(data)
                # Flag as warning if observed trading days < 95% of expected business days
                if expected_bdays > 0 and actual_days < expected_bdays * 0.95:
                    validation_report['warnings'].append(
                        f'Potential data gaps: {actual_days}/{expected_bdays} trading days'
                    )

            # Check for outliers in returns (warning only)
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                if len(returns) > 0:
                    extreme_returns = returns[abs(returns) > 0.5]  # 50% daily move
                    if len(extreme_returns) > 0:
                        validation_report['warnings'].append(
                            f'Found {len(extreme_returns)} extreme returns (>50%)'
                        )

            # Dataset is valid unless there are critical issues
            validation_report['is_valid'] = len(validation_report['issues']) == 0
            return validation_report['is_valid'], validation_report

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False, {'symbol': symbol, 'is_valid': False, 'issues': [str(e)]}

    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize price data using specified method.

        Args:
            data: DataFrame with price data
            method: Normalization method ('minmax', 'zscore', 'returns')

        Returns:
            Normalized DataFrame
        """
        try:
            if data.empty:
                return data

            normalized_data = data.copy()
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if method == 'minmax':
                # Min-Max normalization to [0, 1]
                for col in numeric_columns:
                    if col != 'symbol':  # Don't normalize symbol column
                        min_val = data[col].min()
                        max_val = data[col].max()
                        if max_val != min_val:
                            normalized_data[col] = (data[col] - min_val) / (max_val - min_val)

            elif method == 'zscore':
                # Z-score normalization
                for col in numeric_columns:
                    if col != 'symbol':
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        if std_val != 0:
                            normalized_data[col] = (data[col] - mean_val) / std_val

            elif method == 'returns':
                # Convert to cumulative returns from first observation
                for col in numeric_columns:
                    if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                        first_val = data[col].iloc[0]
                        if first_val != 0:
                            normalized_data[col] = data[col] / first_val - 1

            logger.info(f"Normalized data using {method} method")
            return normalized_data

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data

    def generate_quality_report(self, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """
        Generate comprehensive data quality report.

        Args:
            data: DataFrame to analyze
            symbol: Stock symbol

        Returns:
            Quality report dictionary
        """
        try:
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'data_summary': {},
                'quality_metrics': {},
                'recommendations': []
            }

            # Basic data summary
            report['data_summary'] = {
                'total_rows': len(data),
                'date_range': {
                    'start': data.index.min().isoformat() if len(data) > 0 else None,
                    'end': data.index.max().isoformat() if len(data) > 0 else None
                },
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.astype(str).to_dict()
            }

            # Quality metrics
            if not data.empty:
                # Completeness
                total_cells = data.shape[0] * data.shape[1]
                missing_cells = data.isnull().sum().sum()
                completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0

                # Volatility (if returns available)
                volatility = None
                if 'returns' in data.columns:
                    returns = data['returns'].dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252)  # Annualized

                report['quality_metrics'] = {
                    'completeness': completeness,
                    'volatility_annualized': volatility,
                    'data_density': len(data) / max(1, (data.index.max() - data.index.min()).days),
                    'price_range': {
                        'min': data['Adj Close'].min() if 'Adj Close' in data.columns else None,
                        'max': data['Adj Close'].max() if 'Adj Close' in data.columns else None
                    }
                }

                # Generate recommendations
                if completeness < 0.9:
                    report['recommendations'].append('Consider investigating missing data patterns')
                if volatility and volatility > 0.5:
                    report['recommendations'].append('High volatility detected - ensure risk metrics account for this')
                if len(data) < 252:  # Less than 1 year of trading days
                    report['recommendations'].append('Limited data history - may affect optimization reliability')

            # Store report
            self.data_quality_report = report
            logger.info(f"Generated quality report for {symbol}")
            return report

        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def fetch_and_process_data(self, symbols: List[str], period: str = "5y",
                             normalize_method: str = None) -> Dict[str, Dict[str, any]]:
        """
        Complete data pipeline: fetch, clean, validate, and optionally normalize data.

        Args:
            symbols: List of stock symbols
            period: Time period
            normalize_method: Normalization method (None, 'minmax', 'zscore', 'returns')

        Returns:
            Dictionary with processed data and reports for each symbol
        """
        try:
            pipeline_results = {}

            for symbol in symbols:
                logger.info(f"Processing data for {symbol}")

                # Step 1: Fetch raw data
                raw_data = self.fetch_historical_data(symbol, period)
                if raw_data.empty:
                    logger.warning(f"No data fetched for {symbol}")
                    continue

                # Step 2: Clean data
                cleaned_data = self.clean_data(raw_data)

                # Step 3: Validate data
                is_valid, validation_report = self.validate_data(cleaned_data, symbol)
                if not is_valid:
                    logger.warning(f"Data validation failed for {symbol}: {validation_report.get('issues')}")
                    # Proceed if only warnings exist and no critical issues
                    if len(validation_report.get('issues', [])) == 0:
                        logger.info(f"Proceeding with {symbol} despite warnings: {validation_report.get('warnings')}")
                    else:
                        pipeline_results[symbol] = {
                            'data': None,
                            'validation': validation_report,
                            'quality_report': None,
                            'success': False
                        }
                        continue

                # Step 4: Normalize if requested
                processed_data = cleaned_data
                if normalize_method:
                    processed_data = self.normalize_data(cleaned_data, normalize_method)

                # Step 5: Generate quality report
                quality_report = self.generate_quality_report(processed_data, symbol)

                pipeline_results[symbol] = {
                    'data': processed_data,
                    'validation': validation_report,
                    'quality_report': quality_report,
                    'success': True
                }

                logger.info(f"Successfully processed data for {symbol}")

            logger.info(f"Data pipeline completed for {len([r for r in pipeline_results.values() if r['success']])}/{len(symbols)} symbols")
            return pipeline_results

        except Exception as e:
            logger.error(f"Error in data pipeline: {e}")
            return {}

    def __str__(self):
        return "YahooFinanceService"