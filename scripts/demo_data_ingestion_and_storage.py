"""
Complete demonstration of data ingestion and storage system.

This script demonstrates how to:
1. Ingest historical data from Yahoo Finance
2. Store the data persistently
3. Retrieve the data later
4. Manage the storage system
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(__file__))

from data.src.feeds import (
    UnifiedDataIngestion,
    AssetClass,
    create_default_ingestion_system,
    fetch_historical_market_data
)
from data.src.storage import (
    MarketDataStorage,
    StorageFormat,
    CompressionType,
    create_default_storage
)

def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def demonstrate_data_ingestion_and_storage():
    """Demonstrate complete data ingestion and storage workflow."""
    print("=" * 70)
    print("DATA INGESTION AND STORAGE DEMONSTRATION")
    print("=" * 70)

    # Create ingestion and storage systems
    ingestion = create_default_ingestion_system()
    storage = create_default_storage()

    # Define test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Test symbols for different asset classes
    test_symbols = {
        AssetClass.EQUITY: ['AAPL', 'GOOGL', 'MSFT'],
        AssetClass.ETF: ['SPY', 'QQQ', 'IWM'],
        AssetClass.FX: ['EURUSD', 'GBPUSD']
    }

    all_results = {}

    # Ingest data for each asset class
    for asset_class, symbols in test_symbols.items():
        print(f"\n📊 Ingesting {asset_class.value.upper()} data: {symbols}")

        results = ingestion.fetch_market_data(
            symbols, asset_class, start_date, end_date
        )

        all_results.update(results)

        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        print(f"  ✓ Successfully fetched: {successful}/{len(symbols)} symbols")

    # Save all results to storage
    print(f"\n💾 Saving {len(all_results)} results to storage...")
    save_results = ingestion.save_results_to_storage(all_results)

    successful_saves = sum(1 for success in save_results.values() if success)
    print(f"  ✓ Successfully saved: {successful_saves}/{len(save_results)} symbols")

    # Show storage information
    print(f"\n📁 Storage Information:")
    storage_info = storage.get_storage_info()
    print(f"  Total files: {storage_info['total_files']}")
    print(f"  Total size: {storage_info['total_size_mb']:.2f} MB")
    print(f"  Asset classes: {list(storage_info['asset_class_distribution'].keys())}")

    return all_results, storage

def demonstrate_data_retrieval():
    """Demonstrate data retrieval from storage."""
    print("\n" + "=" * 70)
    print("DATA RETRIEVAL DEMONSTRATION")
    print("=" * 70)

    storage = create_default_storage()

    # Get available symbols
    available_symbols = storage.get_available_symbols()
    print(f"Available symbols in storage: {len(available_symbols)}")
    print(f"  Symbols: {available_symbols[:10]}{'...' if len(available_symbols) > 10 else ''}")

    if available_symbols:
        # Test retrieval for AAPL
        print(f"\n🔍 Retrieving AAPL equity data...")
        aapl_data = storage.load_data('AAPL', AssetClass.EQUITY)

        if aapl_data is not None:
            print(f"  ✓ Successfully retrieved AAPL data")
            print(f"  - Shape: {aapl_data.shape}")
            print(f"  - Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
            print(f"  - Columns: {list(aapl_data.columns)}")
            print(f"  - Latest price: ${aapl_data['close'].iloc[-1]:.2f}")
        else:
            print(f"  ✗ Failed to retrieve AAPL data")

        # Test retrieval with date filtering
        print(f"\n🔍 Retrieving SPY data for last 7 days...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        spy_data = storage.load_data('SPY', AssetClass.ETF, start_date, end_date)

        if spy_data is not None:
            print(f"  ✓ Successfully retrieved filtered SPY data")
            print(f"  - Shape: {spy_data.shape}")
            print(f"  - Date range: {spy_data.index.min()} to {spy_data.index.max()}")
        else:
            print(f"  ✗ Failed to retrieve SPY data")

def demonstrate_storage_management():
    """Demonstrate storage management features."""
    print("\n" + "=" * 70)
    print("STORAGE MANAGEMENT DEMONSTRATION")
    print("=" * 70)

    storage = create_default_storage()

    # Export storage summary
    print("📋 Exporting storage summary...")
    summary_df = storage.export_storage_summary()

    if not summary_df.empty:
        print(f"  ✓ Storage summary created with {len(summary_df)} records")
        print(f"  - Asset classes: {summary_df['asset_class'].unique().tolist()}")
        print(f"  - File formats: {summary_df['file_format'].unique().tolist()}")

        # Show sample records
        print(f"\n  Sample records:")
        for _, row in summary_df.head(3).iterrows():
            print(f"    {row['symbol']} ({row['asset_class']}): {row['data_points']} points, {row['file_size_mb']:.2f} MB")
    else:
        print(f"  ✗ No data in storage")

    # Show detailed storage info
    print(f"\n📊 Detailed storage information:")
    info = storage.get_storage_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")

def demonstrate_file_formats():
    """Demonstrate different file format support."""
    print("\n" + "=" * 70)
    print("FILE FORMAT DEMONSTRATION")
    print("=" * 70)

    formats_to_test = [
        (StorageFormat.CSV, CompressionType.NONE),
        (StorageFormat.PARQUET, CompressionType.NONE),
        (StorageFormat.PARQUET, CompressionType.GZIP),
    ]

    ingestion = create_default_ingestion_system()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    # Fetch test data
    results = ingestion.fetch_equities(['AAPL'], start_date, end_date)
    aapl_result = results.get('AAPL')

    if aapl_result and aapl_result.success:
        for fmt, compression in formats_to_test:
            print(f"\n📁 Testing {fmt.value} format with {compression.value} compression...")

            # Create storage with specific format
            storage = MarketDataStorage(
                base_path=f"data/storage_test/{fmt.value}",
                default_format=fmt,
                compression=compression
            )

            # Save data
            success = storage.save_ingestion_result(aapl_result)
            if success:
                print(f"  ✓ Successfully saved in {fmt.value} format")

                # Check file size
                info = storage.get_storage_info()
                if info['total_files'] > 0:
                    print(f"  - File size: {info['total_size_mb']:.3f} MB")

                    # Test retrieval
                    retrieved_data = storage.load_data('AAPL', AssetClass.EQUITY)
                    if retrieved_data is not None:
                        print(f"  ✓ Successfully retrieved data")
                        print(f"  - Data integrity: {len(retrieved_data)} points")
                    else:
                        print(f"  ✗ Failed to retrieve data")
                else:
                    print(f"  ✗ No files created")
            else:
                print(f"  ✗ Failed to save in {fmt.value} format")

def demonstrate_batch_operations():
    """Demonstrate batch operations and performance."""
    print("\n" + "=" * 70)
    print("BATCH OPERATIONS DEMONSTRATION")
    print("=" * 70)

    ingestion = create_default_ingestion_system()
    storage = create_default_storage()

    # Fetch larger batch of data
    print("🚀 Fetching batch data for multiple symbols...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    import time
    start_time = time.time()

    results = ingestion.fetch_equities(symbols, start_date, end_date)

    fetch_time = time.time() - start_time
    print(f"  ✓ Fetched {len(results)} symbols in {fetch_time:.2f} seconds")

    # Save batch to storage
    start_time = time.time()
    save_results = ingestion.save_results_to_storage(results)
    save_time = time.time() - start_time

    successful_saves = sum(1 for success in save_results.values() if success)
    print(f"  ✓ Saved {successful_saves}/{len(results)} symbols in {save_time:.2f} seconds")

    # Retrieve batch data
    print(f"\n📖 Retrieving batch data...")
    start_time = time.time()

    retrieved_data = {}
    for symbol in symbols:
        data = storage.load_data(symbol, AssetClass.EQUITY)
        if data is not None:
            retrieved_data[symbol] = data

    retrieve_time = time.time() - start_time
    print(f"  ✓ Retrieved {len(retrieved_data)} symbols in {retrieve_time:.2f} seconds")

    # Performance summary
    print(f"\n⚡ Performance Summary:")
    print(f"  - Fetch: {fetch_time:.2f}s ({fetch_time/len(results):.3f}s per symbol)")
    print(f"  - Save: {save_time:.2f}s ({save_time/len(results):.3f}s per symbol)")
    print(f"  - Retrieve: {retrieve_time:.2f}s ({retrieve_time/len(retrieved_data):.3f}s per symbol)")

def main():
    """Main demonstration function."""
    setup_logging()

    print("COMPLETE DATA INGESTION AND STORAGE DEMONSTRATION")
    print("=" * 70)
    print("This script demonstrates the complete workflow of:")
    print("1. Ingesting historical market data from Yahoo Finance")
    print("2. Storing data persistently with multiple format options")
    print("3. Retrieving and managing stored data")
    print("4. Performance testing with batch operations")
    print("=" * 70)

    try:
        # Run all demonstrations
        all_results, storage = demonstrate_data_ingestion_and_storage()
        demonstrate_data_retrieval()
        demonstrate_storage_management()
        demonstrate_file_formats()
        demonstrate_batch_operations()

        print("\n" + "=" * 70)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYour data is now stored in:")
        print(f"  - Base storage: data/storage/")
        print(f"  - Test formats: data/storage_test/")
        print("\nYou can now use the stored data for:")
        print("  - Portfolio analysis and optimization")
        print("  - Backtesting trading strategies")
        print("  - Risk management calculations")
        print("  - Statistical modeling")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()