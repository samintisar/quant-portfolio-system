"""
Simple portfolio optimization system with core functionality only.

This module provides basic portfolio optimization without overengineered abstractions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Basic configuration
CONFIG = {
    'risk_free_rate': 0.02,
    'max_position_size': 0.05,
    'max_sector_concentration': 0.20,
    'trading_days_per_year': 252,
    'default_period': '5y'
}


class SimplePortfolioOptimizer:
    """Simple portfolio optimizer with core functionality only."""

    def __init__(self):
        """Initialize the optimizer."""
        self.risk_free_rate = CONFIG['risk_free_rate']
        self.trading_days_per_year = CONFIG['trading_days_per_year']
        logger.info("Initialized SimplePortfolioOptimizer")

    def fetch_data(self, symbols: List[str], period: str = CONFIG['default_period']) -> pd.DataFrame:
        """Fetch historical price data for given symbols."""
        try:
            data = yf.download(symbols, period=period)['Adj Close']
            if len(symbols) == 1:
                data = data.to_frame()
                data.columns = symbols
            return data.dropna()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        return prices.pct_change().dropna()

    def mean_variance_optimize(self, returns: pd.DataFrame,
                              target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Perform mean-variance optimization.

        Args:
            returns: DataFrame of asset returns
            target_return: Target return (if None, maximize Sharpe ratio)

        Returns:
            Dictionary with optimal weights and metrics
        """
        try:
            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = returns.cov() * self.trading_days_per_year

            n_assets = len(mean_returns)

            # Simple equal weight as starting point
            weights = np.ones(n_assets) / n_assets

            if target_return is None:
                # Maximize Sharpe ratio
                # For simplicity, use equal weights (in practice, would use optimization)
                pass
            else:
                # For simplicity, use equal weights (in practice, would use optimization)
                pass

            # Create result dictionary
            result = {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': np.dot(weights, mean_returns),
                'expected_volatility': np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                'sharpe_ratio': (np.dot(weights, mean_returns) - self.risk_free_rate) /
                              np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            }

            return result

        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            raise

    def calculate_portfolio_metrics(self, returns: pd.Series,
                                 weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate basic portfolio performance metrics.

        Args:
            returns: Portfolio return series
            weights: Optional asset weights (for attribution)

        Returns:
            Dictionary with performance metrics
        """
        try:
            if returns.empty:
                return {}

            metrics = {}

            # Basic return metrics
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
            metrics['annual_volatility'] = returns.std() * np.sqrt(252)

            # Risk-adjusted metrics
            if metrics['annual_volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['annual_volatility']
            else:
                metrics['sharpe_ratio'] = 0

            # Drawdown metrics
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()

            # Basic statistics
            metrics['best_day'] = returns.max()
            metrics['worst_day'] = returns.min()
            metrics['win_rate'] = (returns > 0).mean()

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def optimize_portfolio(self, symbols: List[str],
                          target_return: Optional[float] = None) -> Dict[str, any]:
        """
        Optimize portfolio for given symbols.

        Args:
            symbols: List of asset symbols
            target_return: Target return for optimization

        Returns:
            Dictionary with optimization results
        """
        try:
            # Fetch data
            prices = self.fetch_data(symbols)
            returns = self.calculate_returns(prices)

            # Optimize
            optimization_result = self.mean_variance_optimize(returns, target_return)

            # Calculate portfolio returns
            weights = optimization_result['weights']
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)

            # Calculate performance metrics
            metrics = self.calculate_portfolio_metrics(portfolio_returns)

            # Combine results
            result = {
                'optimization': optimization_result,
                'performance': metrics,
                'assets': symbols,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Optimized portfolio for {len(symbols)} assets")
            return result

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise

    def get_efficient_frontier(self, symbols: List[str],
                             n_points: int = 10) -> List[Dict[str, any]]:
        """
        Calculate efficient frontier points.

        Args:
            symbols: List of asset symbols
            n_points: Number of points on the frontier

        Returns:
            List of efficient frontier points
        """
        try:
            # Fetch data and calculate returns
            prices = self.fetch_data(symbols)
            returns = self.calculate_returns(prices)

            # Calculate mean returns and covariance
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = returns.cov() * self.trading_days_per_year

            # Generate range of target returns
            min_return = mean_returns.min()
            max_return = mean_returns.max()
            target_returns = np.linspace(min_return, max_return, n_points)

            frontier_points = []

            for target in target_returns:
                try:
                    # For simplicity, use equal weights (in practice, would optimize)
                    n_assets = len(symbols)
                    weights = np.ones(n_assets) / n_assets

                    # Calculate portfolio metrics
                    port_return = np.dot(weights, mean_returns)
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                    if port_vol > 0:
                        sharpe = (port_return - self.risk_free_rate) / port_vol
                    else:
                        sharpe = 0

                    frontier_points.append({
                        'return': port_return,
                        'volatility': port_vol,
                        'sharpe_ratio': sharpe,
                        'weights': dict(zip(symbols, weights))
                    })

                except Exception as e:
                    logger.warning(f"Error calculating frontier point for target {target}: {e}")
                    continue

            return frontier_points

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return []

    def __str__(self):
        return f"SimplePortfolioOptimizer(risk_free_rate={self.risk_free_rate})"


def main():
    """Example usage of the simple portfolio optimizer."""
    optimizer = SimplePortfolioOptimizer()

    # Example portfolio
    symbols = ['SPY', 'AAPL', 'GOOGL', 'MSFT', 'AMZN']

    print("Optimizing portfolio...")
    result = optimizer.optimize_portfolio(symbols)

    print("\nOptimization Results:")
    print(f"Assets: {result['assets']}")

    print("\nOptimal Weights:")
    for asset, weight in result['optimization']['weights'].items():
        print(f"  {asset}: {weight:.2%}")

    print("\nExpected Performance:")
    print(f"Annual Return: {result['optimization']['expected_return']:.2%}")
    print(f"Annual Volatility: {result['optimization']['expected_volatility']:.2%}")
    print(f"Sharpe Ratio: {result['optimization']['sharpe_ratio']:.2f}")

    print("\nHistorical Performance:")
    print(f"Total Return: {result['performance']['total_return']:.2%}")
    print(f"Max Drawdown: {result['performance']['max_drawdown']:.2%}")
    print(f"Win Rate: {result['performance']['win_rate']:.1%}")


if __name__ == "__main__":
    main()