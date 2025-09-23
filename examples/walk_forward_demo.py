"""
Walk-forward backtesting demonstration script.

This script demonstrates how to use the walk-forward backtesting system
to evaluate different portfolio optimization strategies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.backtesting.walk_forward import (
    WalkForwardBacktester,
    BacktestConfig,
    run_walk_forward_backtest
)
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    logger.info("Starting walk-forward backtesting demonstration")

    # Define test symbols (mix of large-cap stocks)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']

    # Define backtest period
    start_date = '2019-01-01'
    end_date = '2023-12-31'

    # Create custom configuration
    config = BacktestConfig(
        train_years=1,           # 1 year training
        test_quarters=1,         # 1 quarter testing
        transaction_cost_bps=7.5, # 7.5 bps per trade
        rebalance_frequency='quarterly',
        include_equal_weight_baseline=True,
        include_ml_overlay=True
    )

    logger.info(f"Running backtest for {len(symbols)} symbols from {start_date} to {end_date}")
    logger.info(f"Configuration: {config.train_years}y train, {config.test_quarters}q test, "
                f"{config.transaction_cost_bps}bps costs")

    try:
        # Run walk-forward backtest
        results = run_walk_forward_backtest(symbols, start_date, end_date, config)

        # Generate and display report
        report = WalkForwardBacktester(config).generate_report(results)
        print("\n" + report)

        # Detailed analysis
        logger.info("\n=== DETAILED ANALYSIS ===")

        for strategy_name, result in results.items():
            if strategy_name == 'comparison':
                continue

            logger.info(f"\n{strategy_name.upper()} Strategy:")
            logger.info(f"  Total Return: {result.metrics.get('total_return', 0):.2%}")
            logger.info(f"  Annual Return: {result.metrics.get('annual_return', 0):.2%}")
            logger.info(f"  Annual Volatility: {result.metrics.get('annual_volatility', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Sortino Ratio: {result.metrics.get('sortino_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"  Win Rate: {result.metrics.get('win_rate', 0):.1%}")
            logger.info(f"  Transaction Costs: {result.transaction_costs:.4%}")
            logger.info(f"  Annual Turnover: {result.turnover:.2%}")

        # Strategy comparison
        if 'comparison' in results and 'rankings' in results['comparison']:
            logger.info("\n=== STRATEGY RANKINGS ===")
            rankings = results['comparison']['rankings']

            for metric, ranking in rankings.items():
                logger.info(f"{metric.replace('_', ' ').title()}: {' > '.join(ranking)}")

        # Performance insights
        logger.info("\n=== PERFORMANCE INSIGHTS ===")
        optimized_result = results['optimized']
        equal_weight_result = results['equal_weight']

        # Calculate outperformance
        excess_return = optimized_result.metrics.get('annual_return', 0) - \
                       equal_weight_result.metrics.get('annual_return', 0)

        logger.info(f"Optimized vs Equal Weight Annual Return: {excess_return:+.2%}")

        if excess_return > 0:
            logger.info("✓ Optimization added value over equal weight benchmark")
        else:
            logger.info("✗ Equal weight outperformed optimization")

        # Risk-adjusted performance comparison
        excess_sharpe = optimized_result.metrics.get('sharpe_ratio', 0) - \
                       equal_weight_result.metrics.get('sharpe_ratio', 0)

        logger.info(f"Optimized vs Equal Weight Sharpe Ratio: {excess_sharpe:+.2f}")

        # Transaction cost impact
        logger.info(f"Total transaction costs: {optimized_result.transaction_costs:.4%}")
        logger.info(f"Annual turnover: {optimized_result.turnover:.2%}")

        # Save results to CSV for further analysis
        save_results_to_csv(results, symbols)

        logger.info("\nBacktesting demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        raise


def save_results_to_csv(results, symbols):
    """Save backtesting results to CSV files for further analysis."""
    try:
        # Save returns data
        returns_data = {}
        for strategy_name, result in results.items():
            if strategy_name != 'comparison' and not result.returns.empty:
                returns_data[strategy_name] = result.returns

        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            returns_df.to_csv('backtest_returns.csv')
            logger.info("Returns data saved to backtest_returns.csv")

        # Save weights history
        for strategy_name, result in results.items():
            if strategy_name != 'comparison' and not result.weights_history.empty:
                result.weights_history.to_csv(f'backtest_weights_{strategy_name}.csv')
                logger.info(f"Weights history saved to backtest_weights_{strategy_name}.csv")

        # Save metrics summary
        metrics_data = {}
        for strategy_name, result in results.items():
            if strategy_name != 'comparison':
                metrics_data[strategy_name] = result.metrics

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data).T
            metrics_df.to_csv('backtest_metrics.csv')
            logger.info("Metrics summary saved to backtest_metrics.csv")

    except Exception as e:
        logger.warning(f"Error saving results to CSV: {e}")


if __name__ == '__main__':
    main()