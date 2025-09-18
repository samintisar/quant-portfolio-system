"""
Basic usage examples for data ingestion and storage system.

This script shows simple examples of how to use the data ingestion
and storage system for quantitative trading.
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.src.feeds import create_default_ingestion_system, AssetClass
from data.src.storage import create_default_storage


def example_fetch_single_equity():
    """Example: Fetch data for a single equity."""
    print("=== Single Equity Fetch Example ===")

    # Create ingestion system
    ingestion = create_default_ingestion_system()

    # Define date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch Apple data
    results = ingestion.fetch_equities(['AAPL'], start_date, end_date)

    for symbol, result in results.items():
        if result.success:
            print(f"✓ Fetched {symbol}: {result.metadata['data_points']} data points")
            if result.data is not None:
                print(f"  Latest price: ${result.data['close'].iloc[-1]:.2f}")
                print(f"  Date range: {result.data.index.min()} to {result.data.index.max()}")
        else:
            print(f"✗ Failed to fetch {symbol}: {result.error_message}")


def example_fetch_multiple_assets():
    """Example: Fetch data for multiple asset classes."""
    print("\n=== Multiple Asset Classes Example ===")

    ingestion = create_default_ingestion_system()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch different asset types
    equity_results = ingestion.fetch_equities(['AAPL', 'GOOGL'], start_date, end_date)
    etf_results = ingestion.fetch_etfs(['SPY', 'QQQ'], start_date, end_date)
    fx_results = ingestion.fetch_fx_pairs(['EURUSD', 'GBPUSD'], start_date, end_date)

    # Print summary
    all_results = {**equity_results, **etf_results, **fx_results}
    successful = sum(1 for r in all_results.values() if r.success)
    print(f"Successfully fetched {successful}/{len(all_results)} assets")


def example_store_and_retrieve():
    """Example: Store data and retrieve it later."""
    print("\n=== Store and Retrieve Example ===")

    ingestion = create_default_ingestion_system()
    storage = create_default_storage()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch and store data
    print("Fetching and storing AAPL data...")
    results = ingestion.fetch_equities(['AAPL'], start_date, end_date)
    save_results = ingestion.save_results_to_storage(results)

    if save_results.get('AAPL', False):
        print("✓ Data stored successfully")

        # Retrieve the data
        print("Retrieving stored data...")
        aapl_data = storage.load_data('AAPL', AssetClass.EQUITY)

        if aapl_data is not None:
            print(f"✓ Retrieved {len(aapl_data)} data points")
            print(f"  Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
        else:
            print("✗ Failed to retrieve data")
    else:
        print("✗ Failed to store data")


def example_export_to_dataframe():
    """Example: Export results to pandas DataFrame."""
    print("\n=== DataFrame Export Example ===")

    ingestion = create_default_ingestion_system()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Fetch multiple symbols
    results = ingestion.fetch_equities(['AAPL', 'GOOGL', 'MSFT'], start_date, end_date)

    # Export to combined DataFrame
    df = ingestion.export_results_to_dataframe(results, combine=True)

    if not df.empty:
        print(f"✓ Created DataFrame with shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Unique symbols: {df['symbol'].nunique()}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print("\nSample data:")
        print(df[['symbol', 'close']].head())
    else:
        print("✗ No data to export")


def example_check_storage():
    """Example: Check what's available in storage."""
    print("\n=== Storage Check Example ===")

    storage = create_default_storage()

    # Get storage information
    info = storage.get_storage_info()
    print(f"Storage location: {info['base_path']}")
    print(f"Total files: {info['total_files']}")
    print(f"Total size: {info['total_size_mb']:.2f} MB")

    # Get available symbols
    symbols = storage.get_available_symbols()
    if symbols:
        print(f"Available symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    else:
        print("No data in storage yet")


def main():
    """Run all basic usage examples."""
    print("Data Ingestion and Storage - Basic Usage Examples")
    print("=" * 60)

    try:
        example_fetch_single_equity()
        example_fetch_multiple_assets()
        example_store_and_retrieve()
        example_export_to_dataframe()
        example_check_storage()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("For more advanced usage, see scripts/demo_data_ingestion_and_storage.py")

    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()