"""
Example usage of the simplified portfolio optimization system.

This script demonstrates how to use the cleaned-up, non-overengineered system.
"""

from portfolio_simple import SimplePortfolioOptimizer
from performance_simple import SimplePerformanceCalculator
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run portfolio optimization example."""
    print("=== Simple Portfolio Optimization Example ===")
    print()

    # Initialize optimizer
    optimizer = SimplePortfolioOptimizer()
    performance_calc = SimplePerformanceCalculator()

    # Example portfolio
    symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'AMZN']

    print(f"Optimizing portfolio for: {', '.join(symbols)}")
    print()

    # Optimize portfolio
    try:
        result = optimizer.optimize_portfolio(symbols)

        # Display results
        print("Optimization Results:")
        print(f"Assets: {result['assets']}")
        print()

        print("Optimal Weights:")
        for asset, weight in result['optimization']['weights'].items():
            print(f"  {asset}: {weight:.2%}")
        print()

        print("Expected Performance:")
        print(f"  Annual Return: {result['optimization']['expected_return']:.2%}")
        print(f"  Annual Volatility: {result['optimization']['expected_volatility']:.2%}")
        print(f"  Sharpe Ratio: {result['optimization']['sharpe_ratio']:.2f}")
        print()

        print("Historical Performance:")
        print(f"  Total Return: {result['performance']['total_return']:.2%}")
        print(f"  Max Drawdown: {result['performance']['max_drawdown']:.2%}")
        print(f"  Win Rate: {result['performance']['win_rate']:.1%}")
        print()

        # Generate performance report
        print("Performance Report:")
        print("-" * 40)
        report = performance_calc.generate_report(result['performance'])
        print(report)

    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Efficient Frontier
    print("\n" + "=" * 50)
    print("Efficient Frontier Example")
    print("=" * 50)

    try:
        # Use fewer assets for frontier calculation
        frontier_symbols = ['SPY', 'AAPL', 'GOOGL']
        frontier = optimizer.get_efficient_frontier(frontier_symbols, n_points=5)

        print(f"Efficient Frontier for: {', '.join(frontier_symbols)}")
        print()
        print("Return    Volatility    Sharpe")
        print("-" * 35)

        for point in frontier:
            print(f"{point['return']:8.2%}    {point['volatility']:8.2%}    {point['sharpe_ratio']:.2f}")

    except Exception as e:
        print(f"Error calculating efficient frontier: {e}")

    # Example 3: Custom Portfolio Analysis
    print("\n" + "=" * 50)
    print("Custom Portfolio Analysis")
    print("=" * 50)

    try:
        # Define custom weights
        custom_weights = {
            'SPY': 0.4,
            'AAPL': 0.3,
            'GOOGL': 0.2,
            'MSFT': 0.1
        }

        print("Analyzing custom portfolio:")
        for asset, weight in custom_weights.items():
            print(f"  {asset}: {weight:.2%}")
        print()

        # Fetch data and calculate performance
        prices = optimizer.fetch_data(list(custom_weights.keys()))
        portfolio_returns = performance_calc.calculate_portfolio_returns(prices, custom_weights)
        metrics = performance_calc.calculate_metrics(portfolio_returns)

        print("Performance Metrics:")
        print(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        print(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")

    except Exception as e:
        print(f"Error analyzing custom portfolio: {e}")


if __name__ == "__main__":
    main()