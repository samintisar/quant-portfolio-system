"""
Demonstration of offline data storage functionality.

This script shows how the offline data storage works:
1. Fetch data from Yahoo Finance
2. Save it locally
3. Load it offline
4. Manage offline data
"""

import sys
import os
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.data.yahoo_service import YahooFinanceService

def main():
    """Demonstrate offline data functionality."""
    print("=== Offline Data Storage Demo ===\n")

    # Initialize service with offline data enabled
    service = YahooFinanceService(use_offline_data=True, offline_data_dir="data")

    # Test symbols
    symbols = ['AAPL', 'MSFT']
    period = '6mo'

    print(f"Testing with symbols: {', '.join(symbols)}")
    print(f"Period: {period}\n")

    # Step 1: Check initial state (no offline data)
    print("1. Checking initial offline data availability...")
    available_data = service.list_available_offline_data()
    print(f"   Raw files: {len(available_data['raw'])}")
    print(f"   Processed files: {len(available_data['processed'])}")
    print(f"   Combined files: {len(available_data['combined'])}\n")

    # Step 2: Fetch data (will save offline)
    print("2. Fetching data from Yahoo Finance (will save offline)...")
    for symbol in symbols:
        try:
            # First fetch (online)
            print(f"   Fetching {symbol}...")
            data = service.fetch_historical_data(symbol, period)
            print(f"   ✓ {symbol}: {len(data)} data points fetched and saved")
        except Exception as e:
            print(f"   ✗ {symbol}: Error - {e}")

    # Step 3: Check offline data was created
    print("\n3. Checking offline data after fetch...")
    available_data = service.list_available_offline_data()
    print(f"   Raw files: {len(available_data['raw'])}")
    print(f"   Processed files: {len(available_data['processed'])}")
    if available_data['raw']:
        print(f"   Files: {', '.join(available_data['raw'][:3])}")  # Show first 3

    # Step 4: Load data offline
    print("\n4. Loading data from offline storage...")
    for symbol in symbols:
        try:
            # This should load from offline (no API call)
            print(f"   Loading {symbol} from offline...")
            offline_data = service.load_offline_data(symbol, period)
            if offline_data is not None:
                print(f"   ✓ {symbol}: {len(offline_data)} data points loaded from offline")
            else:
                print(f"   ✗ {symbol}: No offline data found")
        except Exception as e:
            print(f"   ✗ {symbol}: Error - {e}")

    # Step 5: Fetch combined price data
    print("\n5. Fetching combined price data...")
    try:
        combined_data = service.fetch_price_data(symbols, period)
        print(f"   ✓ Combined data: {len(combined_data)} rows, {len(combined_data.columns)} symbols")
        print(f"   Symbols: {', '.join(combined_data.columns)}")
    except Exception as e:
        print(f"   ✗ Combined data error: {e}")

    # Step 6: Demonstrate data management
    print("\n6. Data management utilities...")
    print("   Available commands:")
    print("   - List data: python scripts/manage_data.py list")
    print("   - Validate data: python scripts/manage_data.py validate")
    print("   - Clear data: python scripts/manage_data.py clear --type raw")
    print("   - Refresh data: python scripts/manage_data.py refresh --symbols AAPL,MSFT")

    # Step 7: Show configuration
    print("\n7. Configuration settings...")
    print("   Offline data is enabled by default")
    print("   Can be disabled: YahooFinanceService(use_offline_data=False)")
    print("   Custom directory: YahooFinanceService(offline_data_dir='./my_data')")
    print("   Force online: fetch_historical_data('AAPL', '1y', force_online=True)")

    print("\n=== Demo Complete ===")
    print("✓ Data is fetched from Yahoo Finance")
    print("✓ Data is saved locally for offline use")
    print("✓ Offline data loads quickly without API calls")
    print("✓ Combined data supports multi-asset analysis")
    print("✓ Data management utilities available")

    print("\nNext steps:")
    print("1. Run 'python scripts/fetch_market_data.py' to download more data")
    print("2. Use 'python scripts/manage_data.py list' to see available data")
    print("3. Try the portfolio optimization with offline data")

if __name__ == "__main__":
    main()