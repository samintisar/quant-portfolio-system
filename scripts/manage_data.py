"""
Data management utilities for the portfolio optimization system.

This script provides utilities for managing offline market data including
listing, cleaning, refreshing, and validating data.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def list_data(service):
    """List available offline data."""
    print("=== Available Offline Data ===\n")

    available_data = service.list_available_offline_data()

    print(f"Raw data files: {len(available_data['raw'])}")
    if available_data['raw']:
        print("  Raw files:")
        for f in sorted(available_data['raw']):
            print(f"    {f}")

    print(f"\nProcessed data files: {len(available_data['processed'])}")
    if available_data['processed']:
        print("  Processed files:")
        for f in sorted(available_data['processed']):
            print(f"    {f}")

    print(f"\nCombined data files: {len(available_data['combined'])}")
    if available_data['combined']:
        print("  Combined files:")
        for f in sorted(available_data['combined']):
            print(f"    {f}")

    if not any(available_data.values()):
        print("No offline data found.")
        print("Run 'python scripts/fetch_market_data.py' to download data.")

def validate_data(service):
    """Validate offline data files."""
    print("=== Validating Offline Data ===\n")

    available_data = service.list_available_offline_data()
    all_files = available_data['raw'] + available_data['processed']

    if not all_files:
        print("No data files to validate.")
        return

    valid_files = 0
    invalid_files = 0

    for file_type in ['raw', 'processed']:
        directory = os.path.join(service.offline_data_dir, file_type)
        if not os.path.exists(directory):
            continue

        for filename in available_data[file_type]:
            filepath = os.path.join(directory, filename)
            try:
                # Load and validate data
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)

                # Basic validation checks
                issues = []

                # Check if data is empty
                if data.empty:
                    issues.append("Empty dataset")

                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    issues.append(f"Missing columns: {missing_cols}")

                # Check for missing values
                missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                if missing_pct > 0.05:  # More than 5% missing
                    issues.append(f"High missing data: {missing_pct:.1%}")

                # Check date range
                if len(data) > 1:
                    date_range = data.index.max() - data.index.min()
                    if date_range.days < 30:  # Less than 30 days
                        issues.append(f"Limited date range: {date_range.days} days")

                if issues:
                    print(f"✗ {filename}: {', '.join(issues)}")
                    invalid_files += 1
                else:
                    print(f"✓ {filename}: Valid ({len(data)} rows)")
                    valid_files += 1

            except Exception as e:
                print(f"✗ {filename}: Error reading file - {e}")
                invalid_files += 1

    print(f"\nValidation Summary:")
    print(f"  Valid files: {valid_files}")
    print(f"  Invalid files: {invalid_files}")
    print(f"  Total files: {valid_files + invalid_files}")

def clear_data(service, data_type=None):
    """Clear offline data files."""
    print(f"=== Clearing Offline Data ===\n")

    if data_type:
        print(f"Clearing {data_type} data...")
    else:
        print("Clearing ALL offline data...")

    # Show what will be deleted
    available_data = service.list_available_offline_data()
    if data_type == 'raw':
        files_to_delete = available_data['raw']
    elif data_type == 'processed':
        files_to_delete = available_data['processed']
    elif data_type == 'combined':
        files_to_delete = available_data['combined']
    else:
        files_to_delete = available_data['raw'] + available_data['processed']

    if files_to_delete:
        print(f"Files to be deleted: {len(files_to_delete)}")
        for f in sorted(files_to_delete)[:10]:  # Show first 10
            print(f"  {f}")
        if len(files_to_delete) > 10:
            print(f"  ... and {len(files_to_delete) - 10} more")

        try:
            response = input("\nAre you sure you want to delete these files? (y/N): ")
            if response.lower() == 'y':
                success = service.clear_offline_data(data_type)
                if success:
                    print("✓ Data cleared successfully")
                else:
                    print("✗ Failed to clear data")
            else:
                print("Operation cancelled")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
    else:
        print("No files to delete")

def refresh_data(service, symbols=None, period='1y'):
    """Refresh data for specific symbols."""
    print("=== Refreshing Data ===\n")

    if not symbols:
        print("No symbols specified. Available options:")
        print("  -a, --all: Refresh all symbols")
        print("  -s SYMBOLS: Comma-separated list of symbols")
        return

    print(f"Refreshing data for: {', '.join(symbols)}")
    print(f"Period: {period}\n")

    success_count = 0
    for symbol in symbols:
        try:
            print(f"Refreshing {symbol}...")
            # Force online fetch to get fresh data
            data = service.fetch_historical_data(symbol, period, force_online=True)

            if not data.empty:
                print(f"  ✓ {symbol}: {len(data)} data points")
                success_count += 1
            else:
                print(f"  ✗ {symbol}: No data fetched")

        except Exception as e:
            print(f"  ✗ {symbol}: Error - {e}")

    print(f"\nRefresh Summary: {success_count}/{len(symbols)} symbols updated")

def show_data_summary(service):
    """Show summary of offline data."""
    print("=== Data Summary ===\n")

    available_data = service.list_available_offline_data()

    # Analyze symbols and periods
    symbols = set()
    periods = set()

    for filename in available_data['raw'] + available_data['processed']:
        if 'combined' not in filename:
            # Extract symbol and period from filename
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 2:
                symbols.add(parts[0])
                periods.add(parts[1])

    print(f"Symbols available: {len(symbols)}")
    if symbols:
        print(f"  {', '.join(sorted(list(symbols)))}")

    print(f"\nPeriods available: {len(periods)}")
    if periods:
        print(f"  {', '.join(sorted(list(periods)))}")

    print(f"\nTotal files: {len(available_data['raw'] + available_data['processed'])}")
    print(f"Raw files: {len(available_data['raw'])}")
    print(f"Processed files: {len(available_data['processed'])}")
    print(f"Combined files: {len(available_data['combined'])}")

    # Estimate storage size
    total_size = 0
    for data_type in ['raw', 'processed']:
        directory = os.path.join(service.offline_data_dir, data_type)
        if os.path.exists(directory):
            for filename in available_data[data_type]:
                filepath = os.path.join(directory, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)

    print(f"\nEstimated storage: {total_size / (1024*1024):.1f} MB")

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Manage offline market data")
    parser.add_argument('command', choices=['list', 'validate', 'clear', 'refresh', 'summary'],
                       help='Command to execute')
    parser.add_argument('--type', choices=['raw', 'processed', 'combined'],
                       help='Data type for clear command')
    parser.add_argument('--symbols', '-s', help='Comma-separated list of symbols for refresh')
    parser.add_argument('--period', '-p', default='1y', help='Time period for refresh')
    parser.add_argument('--all', '-a', action='store_true', help='Refresh all available symbols')

    args = parser.parse_args()

    # Initialize service with offline data support
    config = Config()
    service = YahooFinanceService(
        use_offline_data=config.get('data', {}).get('use_offline_data', True),
        offline_data_dir=config.get('data', {}).get('offline_data_dir', 'data')
    )

    try:
        if args.command == 'list':
            list_data(service)
        elif args.command == 'validate':
            validate_data(service)
        elif args.command == 'clear':
            clear_data(service, args.type)
        elif args.command == 'refresh':
            if args.all:
                # Get all available symbols from offline data
                available_data = service.list_available_offline_data()
                symbols = set()
                for filename in available_data['raw'] + available_data['processed']:
                    if 'combined' not in filename:
                        parts = filename.replace('.csv', '').split('_')
                        if len(parts) >= 2:
                            symbols.add(parts[0])
                refresh_data(service, list(symbols), args.period)
            elif args.symbols:
                symbols = [s.strip() for s in args.symbols.split(',')]
                refresh_data(service, symbols, args.period)
            else:
                print("Please specify symbols with --symbols or use --all")
        elif args.command == 'summary':
            show_data_summary(service)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()