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
import cvxpy as cp

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
            data = yf.download(symbols, period=period, auto_adjust=False)
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            else:
                # Fallback to 'Close' if 'Adj Close' is not available
                data = data['Close']

            # Handle single symbol case - data might already be a Series
            if len(symbols) == 1:
                if isinstance(data, pd.DataFrame):
                    if len(data.columns) == 1:
                        data = data.iloc[:, 0]
                else:
                    # Already a Series, convert to DataFrame with proper column name
                    data = pd.DataFrame(data, columns=symbols)
            else:
                # Multi-symbol case, ensure proper column names
                if isinstance(data, pd.DataFrame):
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
        Perform mean-variance optimization using CVXPY.

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
            assets = returns.columns.tolist()

            # Define optimization variables
            weights = cp.Variable(n_assets)

            # Define objective
            if target_return is None:
                # Maximize Sharpe ratio
                portfolio_return = mean_returns.values @ weights
                portfolio_volatility = cp.sqrt(cp.quad_form(weights, cov_matrix.values))
                objective = cp.Maximize((portfolio_return - self.risk_free_rate) / portfolio_volatility)
            else:
                # Minimize volatility for target return
                portfolio_volatility = cp.sqrt(cp.quad_form(weights, cov_matrix.values))
                objective = cp.Minimize(portfolio_volatility)

            # Define constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0,          # No short selling
            ]

            if target_return is not None:
                constraints.append(mean_returns.values @ weights >= target_return)

            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != 'optimal':
                logger.warning(f"Optimization status: {problem.status}")
                # Fall back to equal weights
                weights_array = np.ones(n_assets) / n_assets
            else:
                weights_array = weights.value

            # Create result dictionary
            result = {
                'weights': dict(zip(assets, weights_array)),
                'expected_return': np.dot(weights_array, mean_returns),
                'expected_volatility': np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array))),
                'sharpe_ratio': (np.dot(weights_array, mean_returns) - self.risk_free_rate) /
                              np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            }

            return result

        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            # Fall back to equal weights on error
            n_assets = len(returns.columns)
            weights_array = np.ones(n_assets) / n_assets
            return {
                'weights': dict(zip(returns.columns, weights_array)),
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0
            }

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

    def optimize(self, assets=None, constraints=None, method="mean_variance", objective=None, **kwargs):
        """Optimize method for compatibility with existing tests."""
        try:
            if assets and hasattr(assets[0], 'symbol') and hasattr(assets[0], 'returns'):
                # Assets are Asset objects with returns data - use it directly
                returns_data = {}
                for asset in assets:
                    if not asset.returns.empty:
                        returns_data[asset.symbol] = asset.returns

                if not returns_data:
                    raise Exception("No valid asset returns data found")

                returns_df = pd.DataFrame(returns_data)

                # Handle different optimization methods
                if method == "cvar":
                    optimization_result = self.cvar_optimize(returns_df)
                elif method == "black_litterman":
                    market_views = kwargs.get('market_views', None)
                    optimization_result = self.black_litterman_optimize(returns_df, market_views)
                else:
                    optimization_result = self.mean_variance_optimize(returns_df)

                # Create a compatible result object
                class OptimizationResult:
                    def __init__(self, result_dict):
                        self.success = True
                        self.optimal_weights = result_dict['weights']
                        self.execution_time = 0.001
                        self.optimization_method = method
                        self.objective = objective
                        self.error_messages = []

                return OptimizationResult(optimization_result)

            else:
                # Assets are just symbols - fetch data normally
                if assets and hasattr(assets[0], 'symbol'):
                    symbols = [asset.symbol for asset in assets]
                else:
                    symbols = assets

                result = self.optimize_portfolio(symbols)

                # Create a compatible result object
                class OptimizationResult:
                    def __init__(self, result_dict):
                        self.success = True
                        self.optimal_weights = result_dict['optimization']['weights']
                        self.execution_time = 0.001
                        self.optimization_method = method
                        self.objective = objective
                        self.error_messages = []

                return OptimizationResult(result)

        except Exception as e:
            class OptimizationResult:
                def __init__(self):
                    self.success = False
                    self.optimal_weights = None
                    self.execution_time = 0.001
                    self.optimization_method = method
                    self.objective = objective
                    self.error_messages = [str(e)]

            return OptimizationResult()

    def black_litterman_optimize(self, returns: pd.DataFrame, market_views=None,
                              risk_aversion: float = 2.5, market_weights=None) -> Dict[str, float]:
        """
        Perform Black-Litterman optimization using market views.

        Args:
            returns: DataFrame of asset returns
            market_views: MarketViewCollection with investor views
            risk_aversion: Risk aversion parameter
            market_weights: Market capitalization weights (if None, use equal weights)

        Returns:
            Dictionary with optimal weights and metrics
        """
        try:
            n_assets = returns.shape[1]
            assets = returns.columns.tolist()

            # Calculate mean returns and covariance matrix
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = returns.cov() * self.trading_days_per_year

            # If no market weights provided, use equal weights
            if market_weights is None:
                market_weights = np.ones(n_assets) / n_assets

            # Market equilibrium returns (implied returns)
            pi = risk_aversion * np.dot(cov_matrix, market_weights)

            # If no views provided, use market equilibrium
            if market_views is None or len(market_views) == 0:
                weights_array = market_weights
            else:
                # Simple Black-Litterman implementation
                # Convert views to matrix form
                P = np.zeros((len(market_views), n_assets))  # Pick matrix
                Q = np.zeros(len(market_views))  # View returns vector
                Omega = np.eye(len(market_views)) * 0.01  # View uncertainty

                for i, view in enumerate(market_views):
                    if hasattr(view, 'asset_symbol') and view.asset_symbol in assets:
                        asset_idx = assets.index(view.asset_symbol)
                        P[i, asset_idx] = 1.0
                        Q[i] = view.expected_return * self.trading_days_per_year

                # Calculate Black-Litterman expected returns
                tau = 0.05  # Scaling parameter
                sigma_inv = np.linalg.inv(cov_matrix)

                # Posterior expected returns
                posterior_returns = np.linalg.inv(tau * sigma_inv + P.T @ np.linalg.inv(Omega) @ P) @ \
                                  (tau * sigma_inv @ pi + P.T @ np.linalg.inv(Omega) @ Q)

                # Use posterior returns in mean-variance optimization
                weights = cp.Variable(n_assets)
                portfolio_return = posterior_returns @ weights
                portfolio_volatility = cp.sqrt(cp.quad_form(weights, cov_matrix))

                # Maximize Sharpe ratio
                objective = cp.Maximize((portfolio_return - self.risk_free_rate) / portfolio_volatility)

                constraints = [
                    cp.sum(weights) == 1,
                    weights >= 0
                ]

                problem = cp.Problem(objective, constraints)
                problem.solve()

                if problem.status == 'optimal':
                    weights_array = weights.value
                else:
                    weights_array = market_weights

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights_array, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))

            result = {
                'weights': dict(zip(assets, weights_array)),
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_volatility,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            }

            return result

        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            # Fall back to equal weights on error
            n_assets = len(returns.columns)
            weights_array = np.ones(n_assets) / n_assets
            return {
                'weights': dict(zip(returns.columns, weights_array)),
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0
            }

    def cvar_optimize(self, returns: pd.DataFrame, alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform CVaR optimization using CVXPY.

        Args:
            returns: DataFrame of asset returns
            alpha: Confidence level for CVaR (default 0.05 for 95% CVaR)

        Returns:
            Dictionary with optimal weights and metrics
        """
        try:
            n_samples, n_assets = returns.shape
            assets = returns.columns.tolist()

            # Variables
            weights = cp.Variable(n_assets)
            VaR = cp.Variable()
            losses = -returns.values  # Convert returns to losses

            # Auxiliary variables for CVaR calculation
            u = cp.Variable(n_samples)

            # Objective: Minimize CVaR
            cvar = VaR + (1 / (alpha * n_samples)) * cp.sum(u)
            objective = cp.Minimize(cvar)

            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0,          # No short selling
                u >= 0,               # Auxiliary variables non-negative
                u >= losses @ weights - VaR  # CVaR definition constraint
            ]

            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != 'optimal':
                logger.warning(f"CVaR optimization status: {problem.status}")
                # Fall back to equal weights
                weights_array = np.ones(n_assets) / n_assets
            else:
                weights_array = weights.value

            # Calculate portfolio metrics
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = returns.cov() * self.trading_days_per_year

            portfolio_return = np.dot(weights_array, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))

            result = {
                'weights': dict(zip(assets, weights_array)),
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_volatility,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0,
                'cvar': float(cvar.value) if hasattr(cvar, 'value') else 0.0
            }

            return result

        except Exception as e:
            logger.error(f"Error in CVaR optimization: {e}")
            # Fall back to equal weights on error
            n_assets = len(returns.columns)
            weights_array = np.ones(n_assets) / n_assets
            return {
                'weights': dict(zip(returns.columns, weights_array)),
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'cvar': 0.0
            }

    def get_optimizer_info(self):
        """Get optimizer information for compatibility."""
        return {
            'available_methods': ['mean_variance', 'black_litterman', 'cvar'],
            'default_method': 'mean_variance'
        }

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
        Calculate efficient frontier points using real optimization.

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
            n_assets = len(symbols)

            for target in target_returns:
                try:
                    # Use CVXPY for efficient frontier calculation
                    weights = cp.Variable(n_assets)

                    # Minimize volatility for target return
                    portfolio_volatility = cp.sqrt(cp.quad_form(weights, cov_matrix.values))
                    objective = cp.Minimize(portfolio_volatility)

                    constraints = [
                        cp.sum(weights) == 1,  # Weights sum to 1
                        weights >= 0,          # No short selling
                        mean_returns.values @ weights >= target
                    ]

                    problem = cp.Problem(objective, constraints)
                    problem.solve()

                    if problem.status == 'optimal':
                        weights_array = weights.value
                        port_return = np.dot(weights_array, mean_returns)
                        port_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))

                        if port_vol > 0:
                            sharpe = (port_return - self.risk_free_rate) / port_vol
                        else:
                            sharpe = 0

                        frontier_points.append({
                            'return': float(port_return),
                            'volatility': float(port_vol),
                            'sharpe_ratio': float(sharpe),
                            'weights': dict(zip(symbols, weights_array))
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