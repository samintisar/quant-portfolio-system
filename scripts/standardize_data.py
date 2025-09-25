#!/usr/bin/env python3
"""
Optimized data standardization script to fix date ranges and ensure consistent data quality.

This script will:
1. Standardize date ranges across all symbols
2. Apply the enhanced data cleaning logic to fix negative values
3. Regenerate processed data with consistent date ranges
4. Only fetch essential time periods (5y, 10y) to prevent redundant data

UPDATED: Optimized to prevent downloading unnecessary data periods.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.data.yahoo_service import YahooFinanceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Core symbols for portfolio optimization
CORE_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM'
]

# Benchmark symbols
BENCHMARK_SYMBOLS = ['^GSPC', '^TNX']

# Time periods to standardize - optimized to prevent redundant data
TIME_PERIODS = ['5y', '10y']  # Essential periods only

def standardize_date_ranges():
    """Standardize date ranges for all symbols."""
    print("=== Data Standardization Script ===\n")

    service = YahooFinanceService()

    print("Standardizing date ranges for all symbols...")
    print(f"Core symbols: {len(CORE_SYMBOLS)}")
    print(f"Benchmark symbols: {len(BENCHMARK_SYMBOLS)}")
    print(f"Time periods: {', '.join(TIME_PERIODS)}\n")

    # Process core symbols
    success_count = 0
    total_count = len(CORE_SYMBOLS) * len(TIME_PERIODS)

    for symbol in CORE_SYMBOLS:
        print(f"Processing {symbol}...")

        for period in TIME_PERIODS:
            try:
                # Force online fetch to get fresh data
                raw_data = service.fetch_historical_data(symbol, period, force_online=True)

                if raw_data.empty:
                    print(f"  [FAIL] {period}: No data fetched")
                    continue

                # Apply enhanced cleaning
                cleaned_data = service.clean_data(raw_data)

                # Validate data
                is_valid, validation_report = service.validate_data(cleaned_data, symbol)

                if not is_valid:
                    print(f"  [FAIL] {period}: Validation failed - {validation_report['issues']}")
                    continue

                # Save processed data
                processed_filename = f"data/processed/{symbol}_{period}_processed.csv"
                cleaned_data.to_csv(processed_filename)

                # Save raw data (with original timestamp for reproducibility)
                raw_filename = f"data/raw/{symbol}_{period}_raw.csv"
                raw_data.to_csv(raw_filename)

                success_count += 1
                status_msg = f"  [OK] {period}: {len(cleaned_data)} days"
                if validation_report['warnings']:
                    status_msg += f" (Warnings: {len(validation_report['warnings'])})"
                print(status_msg)

            except Exception as e:
                print(f"  [ERROR] {period}: {e}")
                logger.error(f"Error processing {symbol} {period}: {e}")

        print()

    # Process benchmark symbols
    print("Processing benchmark symbols...")
    benchmark_count = 0
    benchmark_total = len(BENCHMARK_SYMBOLS) * len(TIME_PERIODS)

    for symbol in BENCHMARK_SYMBOLS:
        print(f"Processing {symbol}...")

        for period in TIME_PERIODS:
            try:
                # Force online fetch to get fresh data
                raw_data = service.fetch_historical_data(symbol, period, force_online=True)

                if raw_data.empty:
                    print(f"  [FAIL] {period}: No data fetched")
                    continue

                # Apply enhanced cleaning
                cleaned_data = service.clean_data(raw_data)

                # Save processed data
                processed_filename = f"data/processed/{symbol}_{period}_processed.csv"
                cleaned_data.to_csv(processed_filename)

                # Save raw data
                raw_filename = f"data/raw/{symbol}_{period}_raw.csv"
                raw_data.to_csv(raw_filename)

                benchmark_count += 1
                print(f"  [OK] {period}: {len(cleaned_data)} days")

            except Exception as e:
                print(f"  [ERROR] {period}: {e}")
                logger.error(f"Error processing benchmark {symbol} {period}: {e}")

        print()

    # Generate combined price data
    print("Generating combined price data...")
    for period in TIME_PERIODS:
        try:
            combined_data = service.fetch_price_data(CORE_SYMBOLS, period, force_online=True)
            if not combined_data.empty:
                combined_filename = f"data/processed/combined_{period}_prices.csv"
                combined_data.to_csv(combined_filename)
                print(f"  [OK] Combined {period}: {len(combined_data)} rows, {len(combined_data.columns)} symbols")
            else:
                print(f"  [FAIL] Combined {period}: No data")
        except Exception as e:
            print(f"  [ERROR] Combined {period}: {e}")

    # Update summary file
    summary_filename = "data/data_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("Market Data Summary (Standardized)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Core Symbols: {len(CORE_SYMBOLS)}\n")
        f.write(f"Benchmark Symbols: {len(BENCHMARK_SYMBOLS)}\n")
        f.write(f"Time Periods: {', '.join(TIME_PERIODS)}\n")
        f.write(f"Core Success Rate: {success_count}/{total_count} ({success_count/total_count:.1%})\n")
        f.write(f"Benchmark Success Rate: {benchmark_count}/{benchmark_total} ({benchmark_count/benchmark_total:.1%})\n\n")
        f.write("Symbols Processed:\n")
        for symbol in CORE_SYMBOLS:
            f.write(f"  {symbol}\n")
        f.write("\nBenchmarks:\n")
        for symbol in BENCHMARK_SYMBOLS:
            f.write(f"  {symbol}\n")

    print("\n=== Standardization Complete ===")
    print(f"Core symbols: {success_count}/{total_count} datasets")
    print(f"Benchmark symbols: {benchmark_count}/{benchmark_total} datasets")
    print(f"Summary file: {summary_filename}")

    return success_count == total_count and benchmark_count == benchmark_total

def verify_standardization():
    """Verify that date ranges are now consistent."""
    print("\n=== Verification ===")

    data_dir = 'data/processed'
    period_groups = {}

    # Group files by period
    for file in os.listdir(data_dir):
        if file.endswith('_processed.csv') and not file.startswith('combined'):
            parts = file.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                period = parts[1]
                if period not in period_groups:
                    period_groups[period] = []
                period_groups[period].append(symbol)

    # Check date ranges for each period
    for period, symbols in period_groups.items():
        print(f"\n{period} period:")
        date_ranges = []

        for symbol in sorted(symbols):
            try:
                file_path = os.path.join(data_dir, f"{symbol}_{period}_processed.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    if not df.empty:
                        start_date = df.index.min()
                        end_date = df.index.max()
                        date_ranges.append((start_date, end_date, len(df)))
            except Exception as e:
                print(f"  Error reading {symbol}: {e}")

        # Check if all date ranges are consistent
        if date_ranges:
            unique_ranges = set((start, end) for start, end, _ in date_ranges)
            if len(unique_ranges) == 1:
                start, end, days = date_ranges[0]
                print(f"  ✅ Consistent: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)")
            else:
                print(f"  ⚠️  Inconsistent date ranges found:")
                for i, (start, end, days) in enumerate(date_ranges[:3]):  # Show first 3
                    print(f"      {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)")
                if len(date_ranges) > 3:
                    print(f"      ... and {len(date_ranges) - 3} more")

if __name__ == "__main__":
    print("Data Standardization Script")
    print("=" * 50)

    try:
        success = standardize_date_ranges()
        verify_standardization()

        if success:
            print("\n✅ Data standardization completed successfully!")
        else:
            print("\n⚠️  Data standardization completed with some issues.")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Standardization failed: {e}")

    print("\nScript completed.")