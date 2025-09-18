"""
Data feeds module for quantitative trading system.

Provides unified interfaces for ingesting market data from multiple sources
with comprehensive validation and quality control.
"""

from .yahoo_finance_ingestion import (
    YahooFinanceIngestion,
    AssetClass,
    DataRequest,
    IngestionResult,
    create_default_ingestion,
    fetch_sp500_components,
    fetch_major_etfs,
    fetch_major_fx_pairs
)

from .data_ingestion_interface import (
    DataIngestionInterface,
    YahooFinanceDataIngestion,
    UnifiedDataIngestion,
    create_default_ingestion_system,
    fetch_historical_market_data,
    fetch_sp500_historical_data
)

__all__ = [
    'YahooFinanceIngestion',
    'AssetClass',
    'DataRequest',
    'IngestionResult',
    'create_default_ingestion',
    'fetch_sp500_components',
    'fetch_major_etfs',
    'fetch_major_fx_pairs',
    'DataIngestionInterface',
    'YahooFinanceDataIngestion',
    'UnifiedDataIngestion',
    'create_default_ingestion_system',
    'fetch_historical_market_data',
    'fetch_sp500_historical_data'
]