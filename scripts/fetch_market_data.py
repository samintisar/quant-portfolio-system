"""
One-time market data fetching script for offline data storage.

This script downloads market data from Yahoo Finance and saves it locally
for reliable offline access during portfolio optimization.
"""

import sys
import os
import pandas as pd
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

# Core symbols for portfolio optimization (19 large cap US stocks)
CORE_SYMBOLS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet
    'AMZN',  # Amazon
    'TSLA',  # Tesla
    'META',  # Meta Platforms
    'NVDA',  # NVIDIA
    'JPM',   # JPMorgan Chase
    'JNJ',   # Johnson & Johnson
    'V',     # Visa
    'PG',    # Procter & Gamble
    'UNH',   # UnitedHealth Group
    'HD',    # Home Depot
    'MA',    # Mastercard
    'PYPL',  # PayPal
    'DIS',   # Disney
    'NFLX',  # Netflix
    'ADBE',  # Adobe
    'CRM',   # Salesforce
]

# Benchmark data for ML enhancement and market-relative features
BENCHMARK_SYMBOLS = [
    '^GSPC',  # S&P 500 - market benchmark
    '^TNX',   # 10-year Treasury yield - risk-free rate
]

# Time periods to fetch
TIME_PERIODS = ['1y', '3y', '5y']

def fetch_and_save_data():
    """Fetch market data and save to local files."""
    print("=== Market Data Fetching Script ===\n")

    # Initialize service
    service = YahooFinanceService()

    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    print(f"Fetching data for {len(CORE_SYMBOLS)} symbols...")
    print(f"Symbols: {', '.join(CORE_SYMBOLS[:10])}{'...' if len(CORE_SYMBOLS) > 10 else ''}")
    print(f"Time periods: {', '.join(TIME_PERIODS)}\n")

    success_count = 0
    total_count = len(CORE_SYMBOLS) * len(TIME_PERIODS)

    for symbol in CORE_SYMBOLS:
        print(f"Processing {symbol}...")

        for period in TIME_PERIODS:
            try:
                # Fetch and process data
                results = service.fetch_and_process_data(
                    symbols=[symbol],
                    period=period,
                    normalize_method=None  # Keep raw data
                )

                if results and results[symbol]['success']:
                    data = results[symbol]['data']

                    # Save raw data
                    raw_filename = f"data/raw/{symbol}_{period}_raw.csv"
                    data.to_csv(raw_filename)

                    # Save processed data (just cleaned, no normalization)
                    processed_data = service.clean_data(data)
                    processed_filename = f"data/processed/{symbol}_{period}_processed.csv"
                    processed_data.to_csv(processed_filename)

                    # Save quality report
                    quality_report = results[symbol]['quality_report']
                    report_filename = f"data/raw/{symbol}_{period}_report.txt"
                    with open(report_filename, 'w') as f:
                        f.write(f"Data Quality Report for {symbol} ({period})\n")
                        f.write(f"Generated: {quality_report['timestamp']}\n")
                        f.write(f"Data Points: {quality_report['data_summary']['total_rows']}\n")
                        f.write(f"Completeness: {quality_report['quality_metrics']['completeness']:.2%}\n")
                        f.write(f"Volatility: {quality_report['quality_metrics']['volatility_annualized']:.2%}\n")
                        if quality_report['recommendations']:
                            f.write(f"Recommendations: {', '.join(quality_report['recommendations'])}\n")

                    success_count += 1
                    print(f"  [OK] {period}: {len(data)} data points saved")

                else:
                    print(f"  [FAIL] {period}: Failed to fetch data")

            except Exception as e:
                print(f"  [ERROR] {period}: Error - {e}")

        print()  # Empty line between symbols

    # Create a summary file
    summary_filename = "data/data_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("Market Data Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbols: {len(CORE_SYMBOLS)}\n")
        f.write(f"Time Periods: {', '.join(TIME_PERIODS)}\n")
        f.write(f"Success Rate: {success_count}/{total_count} ({success_count/total_count:.1%})\n\n")
        f.write("Symbols Processed:\n")
        for symbol in CORE_SYMBOLS:
            f.write(f"  {symbol}\n")

    print(f"\n=== Fetching Complete ===")
    print(f"Successfully processed: {success_count}/{total_count} datasets")
    print(f"Data saved to: data/ directory")
    print(f"Summary file: {summary_filename}")

    return success_count == total_count

def fetch_combined_data():
    """Fetch combined price data for all symbols (useful for optimization)."""
    print("=== Fetching Combined Data ===\n")

    service = YahooFinanceService()

    for period in TIME_PERIODS:
        try:
            print(f"Fetching combined data for {period}...")

            # Fetch price data for all symbols
            price_data = service.fetch_price_data(CORE_SYMBOLS, period)

            if not price_data.empty:
                # Save combined data
                filename = f"data/processed/combined_{period}_prices.csv"
                price_data.to_csv(filename)
                print(f"  [OK] Saved: {filename} ({len(price_data)} rows, {len(price_data.columns)} symbols)")
            else:
                print(f"  [FAIL] Failed to fetch combined data for {period}")

        except Exception as e:
            print(f"  [ERROR] Error fetching combined data for {period}: {e}")

    print("\n=== Combined Data Complete ===")

def check_existing_data():
    """Check what data already exists."""
    print("=== Existing Data Check ===\n")

    raw_files = []
    processed_files = []

    if os.path.exists('data/raw'):
        raw_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]

    if os.path.exists('data/processed'):
        processed_files = [f for f in os.listdir('data/processed') if f.endswith('.csv')]

    print(f"Raw data files: {len(raw_files)}")
    print(f"Processed data files: {len(processed_files)}")

    if raw_files:
        print("\nRaw files:")
        for f in sorted(raw_files):
            print(f"  {f}")

    if processed_files:
        print("\nProcessed files:")
        for f in sorted(processed_files):
            print(f"  {f}")

def fetch_benchmark_data():
    """Fetch benchmark data for market-relative features."""
    print("=== Fetching Benchmark Data ===\n")

    service = YahooFinanceService()

    for symbol in BENCHMARK_SYMBOLS:
        print(f"Processing {symbol}...")

        for period in TIME_PERIODS:
            try:
                # Fetch benchmark data
                data = service.fetch_historical_data(symbol, period)

                if not data.empty:
                    # Save benchmark data
                    raw_filename = f"data/raw/{symbol}_{period}_raw.csv"
                    data.to_csv(raw_filename)

                    # Save processed benchmark data
                    processed_data = service.clean_data(data)
                    processed_filename = f"data/processed/{symbol}_{period}_processed.csv"
                    processed_data.to_csv(processed_filename)

                    print(f"  [OK] {period}: {len(data)} data points saved")
                else:
                    print(f"  [FAIL] {period}: Failed to fetch data")

            except Exception as e:
                print(f"  [ERROR] {period}: Error - {e}")

        print()

if __name__ == "__main__":
    """Main execution."""
    print("Offline Market Data Storage Script")
    print("=" * 50)

    # Check what data already exists
    check_existing_data()
    print("\n" + "=" * 50)

    # Fetch all data automatically
    print("Fetching all market data (core + benchmark)...")
    print("=" * 50)

    try:
        # Fetch core symbols data
        success = fetch_and_save_data()

        # Fetch benchmark data
        fetch_benchmark_data()

        # Fetch combined data
        if success:
            fetch_combined_data()

        print("\n" + "=" * 50)
        print("All data fetching completed!")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")

    print("\nScript completed.")