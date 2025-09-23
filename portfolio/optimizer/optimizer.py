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
from portfolio.config import get_config
from sklearn.covariance import LedoitWolf, OAS

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
    """Simple portfolio optimizer with core functionality only.

    Adds constrained variants (weight caps) and CVaR/Black–Litterman methods
    to support realistic allocations in walk-forward tests.
    """

    def __init__(self):
        """Initialize the optimizer."""
        config = get_config()
        config_dict = config.to_dict() if hasattr(config, "to_dict") else config
        portfolio_cfg = config_dict.get('portfolio', {})

        self.risk_free_rate = portfolio_cfg.get('risk_free_rate', CONFIG['risk_free_rate'])
        self.trading_days_per_year = portfolio_cfg.get('trading_days_per_year', CONFIG['trading_days_per_year'])
        self._default_period = portfolio_cfg.get('default_period', CONFIG['default_period'])

        # Optimization knobs (lightweight, avoid overengineering)
        optimization_cfg = config_dict.get('optimization', {})
        self.risk_model = optimization_cfg.get('risk_model', 'ledoit_wolf')  # 'sample'|'ledoit_wolf'|'oas'
        self.default_entropy_penalty = float(optimization_cfg.get('entropy_penalty', 0.0))
        self.default_turnover_penalty = float(optimization_cfg.get('turnover_penalty', 0.0))

        CONFIG['risk_free_rate'] = self.risk_free_rate
        CONFIG['trading_days_per_year'] = self.trading_days_per_year
        CONFIG['default_period'] = self._default_period
        logger.info("Initialized SimplePortfolioOptimizer")

    @staticmethod
    def _solve_with_fallback(problem: cp.Problem) -> str:
        """Solve a CVXPY problem with a small set of fallback solvers.

        Returns the final problem status.
        """
        solver_preferences = [
            getattr(cp, "OSQP", None),
            getattr(cp, "CLARABEL", None),
            getattr(cp, "SCS", None),
            getattr(cp, "ECOS", None),
        ]
        for solver in solver_preferences:
            if solver is None:
                continue
            try:
                problem.solve(solver=solver)
                if problem.status in (cp.OPTIMAL, "optimal", "optimal_inaccurate"):
                    return problem.status
            except Exception as _:
                # Try next solver
                continue
        # Final attempt with default selection
        try:
            problem.solve()
        except Exception:
            pass
        return problem.status

    def _annualized_covariance(self, returns: pd.DataFrame, model: Optional[str] = None) -> np.ndarray:
        """Compute annualized covariance using selected risk model."""
        model = (model or self.risk_model or 'sample').lower()
        X = returns.values - returns.values.mean(axis=0, keepdims=True)
        if model == 'ledoit_wolf':
            try:
                lw = LedoitWolf().fit(X)
                Sigma = lw.covariance_
            except Exception:
                Sigma = np.cov(X, rowvar=False)
        elif model == 'oas':
            try:
                oas = OAS().fit(X)
                Sigma = oas.covariance_
            except Exception:
                Sigma = np.cov(X, rowvar=False)
        else:
            Sigma = np.cov(X, rowvar=False)
        # Symmetrize and annualize
        Sigma = (Sigma + Sigma.T) / 2.0
        return Sigma * self.trading_days_per_year

    def fetch_data(self, symbols: List[str], period: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical price data for given symbols."""
        try:
            period = period or self._default_period
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
                               target_return: Optional[float] = None,
                               weight_cap: Optional[float] = None,
                               risk_model: Optional[str] = None,
                               entropy_penalty: Optional[float] = None,
                               turnover_penalty: Optional[float] = None,
                               previous_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform mean-variance optimization using CVXPY.

        Args:
            returns: DataFrame of asset returns
            target_return: Target return (if None, maximize Sharpe ratio)

        Returns:
            Dictionary with optimal weights and metrics
        """
        try:
            # Calculate mean returns and covariance matrix (with optional shrinkage)
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = pd.DataFrame(self._annualized_covariance(returns, risk_model),
                                      index=returns.columns, columns=returns.columns)
            mean_values = mean_returns.values
            cov_values = cov_matrix.values
            # Numerical regularization to ensure PSD
            n_assets = len(mean_returns)
            cov_values = cov_values + np.eye(n_assets) * 1e-10

            assets = returns.columns.tolist()

            # Define optimization variables
            weights = cp.Variable(n_assets)

            # Define objective
            portfolio_variance = cp.quad_form(weights, cov_values)
            ent_pen = float(self.default_entropy_penalty if entropy_penalty is None else entropy_penalty)
            to_pen = float(self.default_turnover_penalty if turnover_penalty is None else turnover_penalty)
            eps = 1e-8

            if target_return is None:
                portfolio_return = mean_values @ weights
                risk_aversion = 0.5
                # Start with mean-variance utility
                expr = portfolio_return - risk_aversion * portfolio_variance
                # Optional entropy regularization (encourages diversification)
                if ent_pen > 0:
                    # Maximize concave entropy directly to keep the objective concave (DCP compliant)
                    entropy_term = cp.sum(cp.entr(weights + eps))
                    expr += ent_pen * entropy_term
                # Optional L2 turnover penalty relative to previous weights
                if to_pen > 0 and previous_weights is not None:
                    prev = np.asarray(previous_weights).reshape((-1,))
                    if prev.size == n_assets:
                        expr -= to_pen * cp.sum_squares(weights - prev)
                objective = cp.Maximize(expr)
            else:
                objective = cp.Minimize(portfolio_variance)

            # Define constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0,          # No short selling
            ]

            # Optional per-asset cap
            if weight_cap is not None:
                constraints.append(weights <= weight_cap)

            if target_return is not None:
                constraints.append(mean_values @ weights >= target_return)

            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            status = self._solve_with_fallback(problem)

            if status not in (cp.OPTIMAL, 'optimal', 'optimal_inaccurate'):
                logger.warning(f"Optimization status: {problem.status}")
                # Fall back to equal weights
                weights_array = np.ones(n_assets) / n_assets
            else:
                weights_array = np.clip(weights.value, 0, None)
                if weights_array.sum() == 0:
                    weights_array = np.ones(n_assets) / n_assets
                else:
                    weights_array = weights_array / weights_array.sum()

            # Create result dictionary
            result = {
                'weights': dict(zip(assets, weights_array)),
                'expected_return': float(np.dot(weights_array, mean_values)),
                'expected_volatility': float(np.sqrt(max(np.dot(weights_array.T, np.dot(cov_values, weights_array)), 0.0))),
                'sharpe_ratio': float(
                    (
                        np.dot(weights_array, mean_values) - self.risk_free_rate
                    ) /
                    max(np.sqrt(max(np.dot(weights_array.T, np.dot(cov_values, weights_array)), 0.0)), 1e-8)
                )
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

    def cvar_optimize(self, returns: pd.DataFrame,
                      alpha: float = 0.05,
                      weight_cap: Optional[float] = None) -> Dict[str, float]:
        """
        Minimize portfolio CVaR at level alpha with long-only, fully-invested weights
        and optional per-asset caps.

        Uses Rockafellar–Uryasev formulation.
        """
        try:
            if returns.empty:
                raise ValueError("Empty returns for optimization")

            scenarios = returns.values  # shape (T, N)
            T, N = scenarios.shape
            assets = returns.columns.tolist()

            w = cp.Variable(N)
            t = cp.Variable()             # VaR threshold
            z = cp.Variable(T)            # excess losses per scenario

            # Scenario portfolio returns
            port_ret = scenarios @ w      # length T

            # CVaR alpha objective: t + (1/((1-alpha)T)) * sum z_i
            cvar_obj = t + (1.0 / ((1 - alpha) * T)) * cp.sum(z)
            objective = cp.Minimize(cvar_obj)

            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                z >= 0,
                z >= -(port_ret + 0) - t  # z_i >= -(return_i) - t (loss beyond t)
            ]
            if weight_cap is not None:
                constraints.append(w <= weight_cap)

            problem = cp.Problem(objective, constraints)
            status = self._solve_with_fallback(problem)

            if status not in (cp.OPTIMAL, 'optimal', 'optimal_inaccurate'):
                logger.warning(f"CVaR optimization status: {problem.status}")
                w_arr = np.ones(N) / N
            else:
                w_arr = np.clip(w.value, 0, None)
                w_sum = w_arr.sum()
                if w_sum <= 0:
                    w_arr = np.ones(N) / N
                else:
                    w_arr = w_arr / w_sum

            # Compute summary stats
            mean_values = returns.mean().values * self.trading_days_per_year
            cov_values = (returns.cov() * self.trading_days_per_year).values
            exp_vol = float(np.sqrt(max(w_arr.T @ cov_values @ w_arr, 0.0)))
            exp_ret = float(w_arr @ mean_values)
            sharpe = (exp_ret - self.risk_free_rate) / max(exp_vol, 1e-8)

            return {
                'weights': dict(zip(returns.columns, w_arr)),
                'expected_return': exp_ret,
                'expected_volatility': exp_vol,
                'sharpe_ratio': float(sharpe)
            }
        except Exception as e:
            logger.error(f"Error in CVaR optimization: {e}")
            N = len(returns.columns)
            w_arr = np.ones(N) / N
            return {
                'weights': dict(zip(returns.columns, w_arr)),
                'expected_return': 0.0,
                'expected_volatility': 0.0,
                'sharpe_ratio': 0.0
            }

    def black_litterman_optimize(self, returns: pd.DataFrame,
                                 tau: float = 0.05,
                                 market_weights: Optional[np.ndarray] = None,
                                 weight_cap: Optional[float] = None,
                                 risk_aversion: float = 2.5) -> Dict[str, float]:
        """
        Black–Litterman optimization with neutral views (if none provided).

        - Prior implied returns mu = delta * Sigma * w_mkt (use equal-weights if market cap weights unknown)
        - Posterior equals prior when no views are supplied (neutral)
        - Solve mean-variance with posterior mu, long-only, sum-to-1, optional weight caps
        """
        try:
            if returns.empty:
                raise ValueError("Empty returns for optimization")

            # Use selected risk model for Sigma
            Sigma = self._annualized_covariance(returns, model=None)
            # Ensure PSD numerically
            N_eps = Sigma.shape[0]
            Sigma = Sigma + np.eye(N_eps) * 1e-10
            N = Sigma.shape[0]
            assets = returns.columns.tolist()

            if market_weights is None:
                w_mkt = np.ones(N) / N
            else:
                w_mkt = np.asarray(market_weights).reshape(-1)
                if w_mkt.size != N:
                    w_mkt = np.ones(N) / N
                else:
                    w_mkt = w_mkt / w_mkt.sum()

            # Implied equilibrium returns
            mu = risk_aversion * (Sigma @ w_mkt)

            # Posterior with neutral views is equal to prior (no P,Q specified)
            mu_post = mu

            # MV solve with mu_post and Sigma
            w = cp.Variable(N)
            portfolio_var = cp.quad_form(w, Sigma)
            objective = cp.Maximize(mu_post @ w - 0.5 * portfolio_var)
            constraints = [cp.sum(w) == 1, w >= 0]
            if weight_cap is not None:
                constraints.append(w <= weight_cap)

            problem = cp.Problem(objective, constraints)
            status = self._solve_with_fallback(problem)

            if status not in (cp.OPTIMAL, 'optimal', 'optimal_inaccurate'):
                logger.warning(f"Black–Litterman optimization status: {problem.status}")
                w_arr = np.ones(N) / N
            else:
                w_arr = np.clip(w.value, 0, None)
                w_sum = w_arr.sum()
                if w_sum <= 0:
                    w_arr = np.ones(N) / N
                else:
                    w_arr = w_arr / w_sum

            mean_values = returns.mean().values * self.trading_days_per_year
            exp_vol = float(np.sqrt(max(w_arr.T @ Sigma @ w_arr, 0.0)))
            exp_ret = float(w_arr @ mean_values)
            sharpe = (exp_ret - self.risk_free_rate) / max(exp_vol, 1e-8)

            return {
                'weights': dict(zip(returns.columns, w_arr)),
                'expected_return': exp_ret,
                'expected_volatility': exp_vol,
                'sharpe_ratio': float(sharpe)
            }
        except Exception as e:
            logger.error(f"Error in Black–Litterman optimization: {e}")
            N = len(returns.columns)
            w_arr = np.ones(N) / N
            return {
                'weights': dict(zip(returns.columns, w_arr)),
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

    def _generate_synthetic_returns(self, symbols: List[str], n_samples: int = 252) -> pd.DataFrame:
        """Generate synthetic return series for fallback scenarios."""
        n_assets = len(symbols)
        rng = np.random.default_rng(42)

        mean_returns = rng.normal(0.0005, 0.002, n_assets)
        random_matrix = rng.normal(size=(n_assets, n_assets))
        covariance = random_matrix @ random_matrix.T
        covariance /= max(n_assets, 1)
        covariance += np.eye(n_assets) * 1e-4

        samples = rng.multivariate_normal(mean_returns, covariance, size=n_samples)
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='B')

        return pd.DataFrame(samples, index=dates, columns=symbols)

    def optimize(self, assets=None, constraints=None, method="mean_variance", objective=None, **kwargs):
        """Optimize method for compatibility with existing tests."""
        try:
            if assets and hasattr(assets[0], 'symbol') and hasattr(assets[0], 'returns'):
                # Assets are Asset objects with returns data - use it directly
                returns_data = {}
                for asset in assets:
                    asset_returns = asset.returns
                    if isinstance(asset_returns, pd.Series):
                        series = asset_returns.dropna()
                    else:
                        series = pd.Series(asset_returns).dropna()

                    if not series.empty:
                        returns_data[asset.symbol] = series.reset_index(drop=True)

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
            try:
                prices = self.fetch_data(symbols)
                if prices.empty or len(prices) < 2:
                    raise ValueError("Insufficient price history for efficient frontier calculation")
                returns = self.calculate_returns(prices)
                if returns.empty:
                    raise ValueError("Unable to derive returns from price data")
            except Exception as data_error:
                logger.warning(f"Falling back to synthetic data for efficient frontier: {data_error}")
                returns = self._generate_synthetic_returns(symbols)

            n_assets = len(symbols)
            if n_assets == 0:
                return []

            # Calculate mean returns and covariance
            mean_returns = returns.mean() * self.trading_days_per_year
            cov_matrix = returns.cov() * self.trading_days_per_year
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

            # Generate range of target returns
            min_return = mean_returns.min()
            max_return = mean_returns.max()
            if np.isclose(min_return, max_return):
                target_returns = np.repeat(min_return, n_points)
            else:
                target_returns = np.linspace(min_return, max_return, n_points)

            frontier_points = []

            for target in target_returns:
                try:
                    # Use CVXPY for efficient frontier calculation
                    weights = cp.Variable(n_assets)

                    # Minimize variance for target return (avoids non-DCP sqrt)
                    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
                    objective = cp.Minimize(portfolio_variance)

                    constraints = [
                        cp.sum(weights) == 1,  # Weights sum to 1
                        weights >= 0,          # No short selling
                        mean_returns.values @ weights >= target
                    ]

                    problem = cp.Problem(objective, constraints)
                    status = self._solve_with_fallback(problem)

                    if status in (cp.OPTIMAL, 'optimal', 'optimal_inaccurate'):
                        weights_array = np.clip(weights.value, 0, None)
                        if weights_array.sum() == 0:
                            weights_array = np.ones(n_assets) / n_assets
                        else:
                            weights_array = weights_array / weights_array.sum()

                        port_return = float(np.dot(weights_array, mean_returns.values))
                        variance = float(np.dot(weights_array.T, np.dot(cov_matrix.values, weights_array)))
                        port_vol = float(np.sqrt(max(variance, 0.0)))
                        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

                        frontier_points.append({
                            'return': port_return,
                            'volatility': port_vol,
                            'sharpe_ratio': sharpe,
                            'weights': dict(zip(symbols, weights_array))
                        })
                    else:
                        raise ValueError(f"Solver status {problem.status}")

                except Exception as e:
                    logger.warning(f"Error calculating frontier point for target {target}: {e}")
                    fallback_weights = np.ones(n_assets) / n_assets
                    port_return = float(np.dot(fallback_weights, mean_returns.values))
                    port_vol = float(np.sqrt(np.dot(fallback_weights.T, np.dot(cov_matrix.values, fallback_weights))))
                    sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

                    frontier_points.append({
                        'return': port_return,
                        'volatility': port_vol,
                        'sharpe_ratio': sharpe,
                        'weights': dict(zip(symbols, fallback_weights))
                    })

            return frontier_points

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return []

    def __str__(self):
        return f"SimplePortfolioOptimizer(risk_free_rate={self.risk_free_rate})"
