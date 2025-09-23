"""
Example demonstrating the enhanced data pipeline functionality.
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from portfolio.data.yahoo_service import YahooFinanceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate the complete data pipeline."""
    print("=== Enhanced Data Pipeline Demo ===\n")

    # Initialize the service
    service = YahooFinanceService()

    # Define symbols to process
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Apple, Microsoft, Google

    print(f"Processing symbols: {', '.join(symbols)}\n")

    try:
        # Run complete data pipeline
        print("1. Running complete data pipeline (fetch -> clean -> validate -> normalize)...")
        results = service.fetch_and_process_data(
            symbols=symbols,
            period="6mo",  # 6 months of data
            normalize_method="minmax"  # Apply min-max normalization
        )

        # Process results
        successful_symbols = []
        failed_symbols = []

        for symbol, result in results.items():
            if result['success']:
                successful_symbols.append(symbol)
                print(f"✓ {symbol}: Successfully processed")
            else:
                failed_symbols.append(symbol)
                print(f"✗ {symbol}: Failed - {result['validation']['issues']}")

        print(f"\nPipeline Summary:")
        print(f"  Successfully processed: {len(successful_symbols)}/{len(symbols)} symbols")
        print(f"  Failed: {len(failed_symbols)} symbols\n")

        # Demonstrate data analysis for successful symbols
        if successful_symbols:
            print("2. Data Quality Analysis:")
            for symbol in successful_symbols:
                result = results[symbol]
                data = result['data']
                quality_report = result['quality_report']

                print(f"\n  {symbol}:")
                print(f"    Data points: {len(data)}")
                print(f"    Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                print(f"    Data completeness: {quality_report['quality_metrics']['completeness']:.2%}")
                print(f"    Annualized volatility: {quality_report['quality_metrics']['volatility_annualized']:.2%}")

                # Show sample data
                print(f"    Sample data (last 3 rows):")
                sample_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'returns']
                sample_data = data[sample_cols].tail(3)
                print(f"    {sample_data.to_string(index=False)}")

                # Show recommendations
                if quality_report['recommendations']:
                    print(f"    Recommendations: {', '.join(quality_report['recommendations'])}")
                else:
                    print(f"    Recommendations: No issues detected")

        # Demonstrate normalization comparison
        if successful_symbols:
            print("\n3. Normalization Comparison:")
            symbol = successful_symbols[0]
            print(f"  Comparing normalization methods for {symbol}:")

            # Fetch fresh data for comparison
            raw_data = service.fetch_historical_data(symbol, period="3mo")
            cleaned_data = service.clean_data(raw_data)

            # Test different normalization methods
            methods = ['minmax', 'zscore', 'returns']
            for method in methods:
                normalized_data = service.normalize_data(cleaned_data.copy(), method)
                price_range = normalized_data['Adj Close'].min(), normalized_data['Adj Close'].max()
                print(f"    {method}: [{price_range[0]:.3f}, {price_range[1]:.3f}]")

        # Demonstrate symbol information
        print("\n4. Symbol Information:")
        for symbol in symbols[:2]:  # Show info for first 2 symbols
            info = service.get_symbol_info(symbol)
            print(f"  {symbol}: {info['name']} ({info['sector']})")

        # Demonstrate error handling
        print("\n5. Error Handling Demo:")
        invalid_symbols = ['INVALID_SYMBOL_1', 'INVALID_SYMBOL_2']
        invalid_results = service.fetch_and_process_data(invalid_symbols, period="1mo")

        print(f"  Attempted to process invalid symbols: {invalid_symbols}")
        print(f"  Results: {len(invalid_results)} symbols processed (graceful handling)")

        print("\n=== Demo Complete ===")
        print("The data pipeline successfully:")
        print("  ✓ Fetches data from Yahoo Finance")
        print("  ✓ Cleans and validates data quality")
        print("  ✓ Normalizes data using different methods")
        print("  ✓ Generates comprehensive quality reports")
        print("  ✓ Handles errors gracefully")
        print("  ✓ Provides detailed recommendations")

    except Exception as e:
        print(f"Error during demo: {e}")
        print("This may be due to network connectivity issues.")

if __name__ == "__main__":
    main()