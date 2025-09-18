#!/usr/bin/env python3
"""
Script to initialize multi-year historical data for quantitative trading system.

Downloads and stores 5-10 years of historical data for specified universes
across multiple asset classes using Yahoo Finance.
"""

import sys
import os
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.src.storage.market_data_storage import MarketDataStorage, create_default_storage
from data.src.feeds.yahoo_finance_ingestion import YahooFinanceIngestion, AssetClass, create_default_ingestion


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup handlers
    handlers = []
    handlers.append(logging.StreamHandler(sys.stdout))

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set formatter for all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def progress_callback(progress):
    """Progress callback function for ingestion."""
    if hasattr(progress, 'completed_symbols') and progress.completed_symbols % 10 == 0:  # Log every 10 symbols
        progress_pct = (progress.completed_symbols / progress.total_symbols) * 100 if progress.total_symbols > 0 else 0
        print(f"\rProgress: {progress_pct:.1f}% ({progress.completed_symbols}/{progress.total_symbols}) | "
              f"Success: {progress.successful_symbols} | Failed: {progress.failed_symbols}", end='', flush=True)


def main():
    """Main function to initialize historical data."""
    parser = argparse.ArgumentParser(description='Initialize historical data for quantitative trading')
    parser.add_argument('--universe', type=str, default='sp500_large_cap',
                       choices=['sp500_large_cap', 'global_diversified'],
                       help='Asset universe to initialize')
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of historical data to download')
    parser.add_argument('--base-path', type=str, default='data/storage',
                       help='Base path for data storage')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (optional)')
    parser.add_argument('--max-workers', type=int, default=8,
                       help='Maximum number of concurrent workers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be downloaded without actually downloading')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only show storage estimates without downloading')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("HISTORICAL DATA INITIALIZATION")
    logger.info("=" * 60)
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Years: {args.years}")
    logger.info(f"Base path: {args.base_path}")
    logger.info(f"Max workers: {args.max_workers}")

    try:
        # Initialize storage and ingestion
        storage = create_default_storage()
        ingestion = create_default_ingestion(max_workers=args.max_workers)

        # Define universes directly in the script
        universes = {
            'sp500_large_cap': {
                AssetClass.EQUITY: [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
                    'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM', 'BAC',
                    'XOM', 'CVX', 'LLY', 'ABBV', 'MRK', 'PEP', 'KO', 'AVGO', 'COST', 'LIN',
                    'TMO', 'ABT', 'ACN', 'DHR', 'WFC', 'MCD', 'NKE', 'TXN', 'NEE', 'PM',
                    'HON', 'UNP', 'RTX', 'LOW', 'INTU', 'QCOM', 'CAT', 'CSCO', 'DE', 'GE',
                    'SCHW', 'BKNG', 'AMT', 'MS', 'BLK', 'GS', 'BA', 'AMGN', 'IBM', 'PLD',
                    'CVS', 'MDT', 'C', 'LMT', 'AXP', 'CI', 'ETN', 'CB', 'T', 'ADP',
                    'MMC', 'ISRG', 'MO', 'PFE', 'COP', 'DUK', 'ORCL', 'SO', 'BSX', 'SYK'
                ],
                AssetClass.ETF: [
                    'SPY', 'VOO', 'IVV', 'VTI', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'XLF',
                    'XLK', 'XLV', 'XLU', 'XLE', 'XLP', 'XLY', 'XLB', 'XLI', 'XLRE', 'XME'
                ],
                AssetClass.FX: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'],
                AssetClass.BOND: ['US10Y', 'US2Y', 'US5Y', 'US30Y'],
                AssetClass.COMMODITY: ['GOLD', 'SILVER', 'OIL', 'NATGAS', 'COPPER'],
                AssetClass.INDEX: ['^GSPC', '^DJI', '^IXIC', '^VIX']
            }
        }

        # Get universe information
        universe_symbols = universes.get(args.universe)
        if universe_symbols is None:
            print(f"Universe '{args.universe}' not found. Available universes: {list(universes.keys())}")
            return 1

        # Display universe summary
        print(f"\nUniverse Summary for '{args.universe}':")
        for asset_class, symbols in universe_symbols.items():
            if symbols:
                print(f"  {asset_class.value.upper()}: {len(symbols)} symbols")
                if asset_class == AssetClass.EQUITY and len(symbols) <= 10:
                    print(f"    Symbols: {', '.join(symbols)}")
                elif len(symbols) > 10:
                    print(f"    First 10: {', '.join(symbols[:10])}...")

        total_symbols = sum(len(symbols) for symbols in universe_symbols.values())
        print(f"\nTotal symbols to process: {total_symbols}")

        # Estimate storage requirements
        print(f"\nStorage Requirements:")
        storage_estimate = storage.estimate_storage_requirements(universe_symbols, args.years)
        print(f"  Estimated total storage: {storage_estimate['total_estimated_gb']:.2f} GB")

        for asset_class, estimate in storage_estimate['by_asset_class'].items():
            print(f"  {asset_class}: {estimate['estimated_gb']:.2f} GB "
                  f"({estimate['symbol_count']} symbols, "
                  f"{estimate['estimated_data_points']:,} data points)")

        if args.estimate_only:
            print("\nStorage estimation complete. Use --estimate-only=False to download data.")
            return 0

        if args.dry_run:
            print(f"\nDry run mode - would download {args.years} years of data for {total_symbols} symbols")
            print(f"   Estimated storage: {storage_estimate['total_estimated_gb']:.2f} GB")
            return 0

        # Get current data coverage
        print(f"\nCurrent Data Coverage:")
        current_coverage = storage.get_data_coverage_summary()
        print(f"  Total unique symbols: {current_coverage['total_unique_symbols']}")

        for asset_class, symbols in current_coverage['symbols_by_asset_class'].items():
            count = len(symbols)
            universe_count = len(universe_symbols.get(AssetClass(asset_class), []))
            if universe_count > 0:
                coverage_pct = (count / universe_count) * 100
                print(f"  {asset_class}: {count}/{universe_count} symbols ({coverage_pct:.1f}%)")

        # Confirm before proceeding
        if args.years > 3:
            print(f"\nWARNING: Downloading {args.years} years of data may take a long time")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled by user")
                return 1

        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.years * 365)

        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Perform ingestion
        print(f"\nStarting data ingestion...")
        results = ingestion.fetch_large_dataset(
            universe_symbols,
            start_date,
            end_date,
            batch_size=50,
            max_workers=args.max_workers,
            progress_callback=progress_callback
        )

        # Store results
        print(f"\nStoring ingested data...")
        successful_results = {
            symbol: result for symbol, result in results['detailed_results'].items()
            if result.success and result.data is not None
        }

        storage_results = storage.save_multiple_results(successful_results)

        successful_saves = sum(1 for success in storage_results.values() if success)
        print(f"Successfully saved: {successful_saves}/{len(successful_results)} datasets")

        # Validate data quality
        print(f"\nValidating data quality...")
        quality_report = ingestion.validate_data_quality(results['detailed_results'])

        print(f"Data Quality Summary:")
        print(f"  Completeness score: {quality_report['completeness_score']:.1%}")
        print(f"  Quality issues: {len(quality_report['data_quality_issues'])}")

        if quality_report['data_quality_issues']:
            print(f"  Top issues:")
            for issue in quality_report['data_quality_issues'][:5]:
                print(f"    {issue['symbol']}: {issue['issue']}")

        # Final summary
        print(f"\nFinal Summary:")
        print(f"  Total symbols processed: {results['total_symbols']}")
        print(f"  Successful downloads: {results['successful_symbols']}")
        print(f"  Failed downloads: {results['failed_symbols']}")
        print(f"  Overall success rate: {results['overall_success_rate']:.1%}")
        print(f"  Processing time: {results['total_time_seconds']:.1f} seconds")
        print(f"  Symbols per second: {results['symbols_per_second']:.2f}")

        if results['failed_symbols'] > 0:
            print(f"\nFailed symbols:")
            for asset_class, summary in results['results_by_asset_class'].items():
                if summary['failed_symbols'] > 0:
                    print(f"  {asset_class}: {summary['failed_symbols']} failed")
                    for symbol in summary['failed_symbols'][:5]:  # Show first 5
                        print(f"    - {symbol}")

        print(f"\nHistorical data initialization complete!")
        print(f"   Data saved to: {args.base_path}")

        return 0

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())