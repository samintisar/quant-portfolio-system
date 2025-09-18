"""
Unified data ingestion interface for quantitative trading system.

Provides a high-level interface for ingesting data from multiple sources
with standardized validation and quality control.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .yahoo_finance_ingestion import (
    YahooFinanceIngestion, AssetClass, DataRequest, IngestionResult,
    fetch_sp500_components, fetch_major_etfs, fetch_major_fx_pairs
)


class DataSource(Enum):
    """Supported data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    QUANDL = "quandl"
    ALPHA_VANTAGE = "alpha_vantage"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"
    FRED = "fred"


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion operations."""
    data_source: DataSource
    asset_class: AssetClass
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    interval: str = "1d"
    max_workers: int = 5
    rate_limit: float = 0.1
    validate_data: bool = True
    clean_data: bool = True
    include_metadata: bool = True


class DataIngestionInterface(ABC):
    """Abstract base class for data ingestion sources."""

    @abstractmethod
    def fetch_data(self, config: DataIngestionConfig) -> Dict[str, IngestionResult]:
        """Fetch data based on configuration."""
        pass

    @abstractmethod
    def get_asset_info(self, symbol: str, asset_class: AssetClass) -> Dict[str, Any]:
        """Get information about an asset."""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to data source."""
        pass


class YahooFinanceDataIngestion(DataIngestionInterface):
    """Yahoo Finance implementation of data ingestion interface."""

    def __init__(self, max_workers: int = 5, rate_limit: float = 0.1):
        """Initialize Yahoo Finance data ingestion."""
        self.ingestion_engine = YahooFinanceIngestion(max_workers, rate_limit)
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, config: DataIngestionConfig) -> Dict[str, IngestionResult]:
        """Fetch data from Yahoo Finance."""
        if config.data_source != DataSource.YAHOO_FINANCE:
            raise ValueError("This instance is configured for Yahoo Finance only")

        try:
            # Create batch requests
            requests = [
                DataRequest(
                    symbol=symbol,
                    asset_class=config.asset_class,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    interval=config.interval
                )
                for symbol in config.symbols
            ]

            # Fetch data
            results = self.ingestion_engine.fetch_multiple_assets(requests)

            # Log summary
            successful = sum(1 for r in results.values() if r.success)
            self.logger.info(
                f"Fetched {successful}/{len(results)} assets from Yahoo Finance "
                f"({config.asset_class.value})"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
            return {
                symbol: IngestionResult(
                    success=False,
                    data=None,
                    metadata={'symbol': symbol, 'asset_class': config.asset_class.value},
                    validation_result=None,
                    warnings=[],
                    error_message=str(e)
                )
                for symbol in config.symbols
            }

    def get_asset_info(self, symbol: str, asset_class: AssetClass) -> Dict[str, Any]:
        """Get asset information from Yahoo Finance."""
        return self.ingestion_engine.get_asset_info(symbol, asset_class)

    def validate_connection(self) -> bool:
        """Validate Yahoo Finance connection."""
        try:
            # Test with a well-known symbol
            test_result = self.ingestion_engine.fetch_equity_data(
                'AAPL',
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            return test_result.success
        except Exception as e:
            self.logger.error(f"Yahoo Finance connection test failed: {str(e)}")
            return False


class UnifiedDataIngestion:
    """Unified data ingestion system supporting multiple sources."""

    def __init__(self):
        """Initialize unified data ingestion system."""
        self.logger = logging.getLogger(__name__)
        self.data_sources = {
            DataSource.YAHOO_FINANCE: YahooFinanceDataIngestion()
        }
        self.active_source = DataSource.YAHOO_FINANCE

    def add_data_source(self, source: DataSource, ingestion: DataIngestionInterface):
        """Add a new data source."""
        self.data_sources[source] = ingestion

    def set_active_source(self, source: DataSource):
        """Set the active data source."""
        if source not in self.data_sources:
            raise ValueError(f"Data source {source} not available")
        self.active_source = source

    def fetch_market_data(self, symbols: List[str], asset_class: AssetClass,
                         start_date: datetime, end_date: datetime,
                         data_source: Optional[DataSource] = None) -> Dict[str, IngestionResult]:
        """
        Fetch market data with simplified interface.

        Args:
            symbols: List of asset symbols
            asset_class: Asset class (equity, etf, fx, bond, etc.)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            data_source: Data source to use (defaults to active source)

        Returns:
            Dictionary mapping symbols to ingestion results
        """
        source = data_source or self.active_source

        config = DataIngestionConfig(
            data_source=source,
            asset_class=asset_class,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        return self.data_sources[source].fetch_data(config)

    def fetch_equities(self, symbols: List[str], start_date: datetime,
                      end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch equity data with simplified interface."""
        return self.fetch_market_data(symbols, AssetClass.EQUITY, start_date, end_date)

    def fetch_etfs(self, symbols: List[str], start_date: datetime,
                   end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch ETF data with simplified interface."""
        return self.fetch_market_data(symbols, AssetClass.ETF, start_date, end_date)

    def fetch_fx_pairs(self, symbols: List[str], start_date: datetime,
                      end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch FX data with simplified interface."""
        return self.fetch_market_data(symbols, AssetClass.FX, start_date, end_date)

    def fetch_bonds(self, symbols: List[str], start_date: datetime,
                    end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch bond data with simplified interface."""
        return self.fetch_market_data(symbols, AssetClass.BOND, start_date, end_date)

    def fetch_sp500_data(self, start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch S&P 500 component data."""
        if self.active_source == DataSource.YAHOO_FINANCE:
            return fetch_sp500_components(start_date, end_date)
        else:
            # For other sources, we'd need to implement similar convenience functions
            raise NotImplementedError(f"SP500 fetch not implemented for {self.active_source}")

    def fetch_major_etf_data(self, start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch major ETF data."""
        if self.active_source == DataSource.YAHOO_FINANCE:
            return fetch_major_etfs(start_date, end_date)
        else:
            raise NotImplementedError(f"Major ETF fetch not implemented for {self.active_source}")

    def fetch_major_fx_data(self, start_date: datetime, end_date: datetime) -> Dict[str, IngestionResult]:
        """Fetch major FX pair data."""
        if self.active_source == DataSource.YAHOO_FINANCE:
            return fetch_major_fx_pairs(start_date, end_date)
        else:
            raise NotImplementedError(f"Major FX fetch not implemented for {self.active_source}")

    def get_asset_info(self, symbol: str, asset_class: AssetClass,
                      data_source: Optional[DataSource] = None) -> Dict[str, Any]:
        """Get asset information."""
        source = data_source or self.active_source
        return self.data_sources[source].get_asset_info(symbol, asset_class)

    def validate_all_connections(self) -> Dict[DataSource, bool]:
        """Validate connections to all data sources."""
        results = {}
        for source, ingestion in self.data_sources.items():
            try:
                results[source] = ingestion.validate_connection()
            except Exception as e:
                self.logger.error(f"Connection test failed for {source}: {str(e)}")
                results[source] = False
        return results

    def get_data_availability_summary(self, results: Dict[str, IngestionResult]) -> Dict[str, Any]:
        """Get summary of data availability from ingestion results."""
        total_symbols = len(results)
        successful_symbols = sum(1 for r in results.values() if r.success)

        summary = {
            'total_symbols': total_symbols,
            'successful_symbols': successful_symbols,
            'success_rate': successful_symbols / total_symbols if total_symbols > 0 else 0,
            'failed_symbols': total_symbols - successful_symbols,
            'total_data_points': sum(r.metadata.get('data_points', 0) for r in results.values()),
            'date_range': {
                'earliest': None,
                'latest': None
            },
            'asset_classes': set(),
            'failed_symbols_list': []
        }

        # Collect additional statistics
        for symbol, result in results.items():
            if result.success:
                data_range = result.metadata.get('data_range', {})
                if data_range:
                    earliest = data_range.get('min_date')
                    latest = data_range.get('max_date')

                    if earliest and (summary['date_range']['earliest'] is None or earliest < summary['date_range']['earliest']):
                        summary['date_range']['earliest'] = earliest

                    if latest and (summary['date_range']['latest'] is None or latest > summary['date_range']['latest']):
                        summary['date_range']['latest'] = latest

                asset_class = result.metadata.get('asset_class')
                if asset_class:
                    summary['asset_classes'].add(asset_class)
            else:
                summary['failed_symbols_list'].append(symbol)

        summary['asset_classes'] = list(summary['asset_classes'])

        return summary

    def save_results_to_storage(self, results: Dict[str, IngestionResult],
                               storage_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Save ingestion results to persistent storage.

        Args:
            results: Dictionary of ingestion results
            storage_path: Custom storage path (defaults to data/storage)

        Returns:
            Dictionary mapping symbols to save success status
        """
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'storage'))
        from market_data_storage import create_default_storage

        # Create storage instance
        if storage_path:
            storage = MarketDataStorage(base_path=storage_path)
        else:
            storage = create_default_storage()

        # Save all results
        save_results = storage.save_multiple_results(results)

        successful_saves = sum(1 for success in save_results.values() if success)
        self.logger.info(f"Saved {successful_saves}/{len(results)} results to storage")

        return save_results

    def export_results_to_dataframe(self, results: Dict[str, IngestionResult],
                                   combine: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Export ingestion results to pandas DataFrames.

        Args:
            results: Dictionary of ingestion results
            combine: Whether to combine all data into one DataFrame

        Returns:
            Combined DataFrame or dictionary of symbol-specific DataFrames
        """
        dataframes = {}

        for symbol, result in results.items():
            if result.success and result.data is not None:
                # Add symbol column for identification
                df = result.data.copy()
                df['symbol'] = symbol
                dataframes[symbol] = df

        if not dataframes:
            return pd.DataFrame() if combine else {}

        if combine:
            # Combine all dataframes
            combined_df = pd.concat(dataframes.values(), ignore_index=False)
            return combined_df.sort_index()
        else:
            return dataframes


def create_default_ingestion_system() -> UnifiedDataIngestion:
    """Create a default unified data ingestion system."""
    return UnifiedDataIngestion()


# Convenience functions for common use cases
def fetch_historical_market_data(symbols: List[str], asset_class: AssetClass,
                               days_back: int = 365) -> Dict[str, IngestionResult]:
    """
    Convenience function to fetch historical market data.

    Args:
        symbols: List of symbols to fetch
        asset_class: Asset class
        days_back: Number of days to look back (default: 1 year)

    Returns:
        Dictionary of ingestion results
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    ingestion = create_default_ingestion_system()
    return ingestion.fetch_market_data(symbols, asset_class, start_date, end_date)


def fetch_sp500_historical_data(days_back: int = 365) -> Dict[str, IngestionResult]:
    """Fetch historical S&P 500 component data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    ingestion = create_default_ingestion_system()
    return ingestion.fetch_sp500_data(start_date, end_date)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create ingestion system
    ingestion = create_default_ingestion_system()

    # Test connections
    print("Testing data source connections...")
    connections = ingestion.validate_all_connections()
    for source, connected in connections.items():
        print(f"{source}: {'✓' if connected else '✗'}")

    # Fetch sample data
    if connections[DataSource.YAHOO_FINANCE]:
        print("\nFetching sample equity data...")
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        results = ingestion.fetch_equities(symbols, start_date, end_date)

        # Print summary
        summary = ingestion.get_data_availability_summary(results)
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Total data points: {summary['total_data_points']:,}")

        # Export to DataFrame
        df = ingestion.export_results_to_dataframe(results)
        print(f"Combined DataFrame shape: {df.shape}")
        if not df.empty:
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Unique symbols: {df['symbol'].nunique()}")