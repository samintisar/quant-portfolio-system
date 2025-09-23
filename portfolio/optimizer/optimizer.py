"""
Main PortfolioOptimizer class that orchestrates optimization methods.

Provides unified interface for portfolio optimization with multiple methods.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json

from portfolio.logging_config import get_logger, OptimizationError
from portfolio.optimizer.base import BaseOptimizer, OptimizerFactory
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.models.result import OptimizationResult
from portfolio.models.views import MarketViewCollection
from portfolio.models.performance import PortfolioPerformance
from portfolio.data.returns import ReturnCalculator, ReturnType
from portfolio.config import get_config

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Main portfolio optimizer that coordinates multiple optimization methods.

    Provides unified interface for portfolio construction with various optimization approaches.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().optimizer.to_dict() if hasattr(get_config(), 'optimizer') else {}
        self.return_calculator = ReturnCalculator()
        self.optimizers = {}
        self.default_method = self.config.get('default_method', 'mean_variance')

        logger.info(f"Initialized PortfolioOptimizer with default method: {self.default_method}")

    def register_optimizer(self, method: str, optimizer: BaseOptimizer) -> None:
        """
        Register an optimizer instance.

        Args:
            method: Optimization method name
            optimizer: Optimizer instance
        """
        self.optimizers[method] = optimizer
        logger.info(f"Registered optimizer for method: {method}")

    def get_optimizer(self, method: Optional[str] = None) -> BaseOptimizer:
        """
        Get optimizer instance for specified method.

        Args:
            method: Optimization method name

        Returns:
            Optimizer instance
        """
        method = method or self.default_method

        if method not in self.optimizers:
            # Create optimizer using factory
            self.optimizers[method] = OptimizerFactory.create_optimizer(method)

        return self.optimizers[method]

    def optimize(self,
                 assets: List[Asset],
                 constraints: Optional[PortfolioConstraints] = None,
                 method: Optional[str] = None,
                 objective: str = 'sharpe',
                 market_views: Optional[MarketViewCollection] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio using specified method.

        Args:
            assets: List of assets in the portfolio
            constraints: Portfolio constraints
            method: Optimization method
            objective: Optimization objective
            market_views: Market views (for methods that support them)
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with optimal weights
        """
        start_time = datetime.now()

        try:
            # Set default constraints if not provided
            if constraints is None:
                constraints = self._get_default_constraints()

            # Validate inputs
            self._validate_inputs(assets, constraints, method, objective)

            # Get optimizer
            optimizer = self.get_optimizer(method)

            # Perform optimization
            result = optimizer.optimize(
                assets, constraints, objective, market_views, **kwargs
            )

            # Add metadata
            result.method = method
            result.objective = objective
            result.timestamp = datetime.now()

            # Log results
            if result.success:
                logger.info(f"Portfolio optimization successful: {method}/{objective}")
                logger.info(f"Optimal weights: {result.optimal_weights}")
            else:
                logger.warning(f"Portfolio optimization failed: {result.error_messages}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Portfolio optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=execution_time,
                optimization_method=method or self.default_method,
                error_messages=[str(e)]
            )

    def optimize_portfolio(self,
                          assets: List[Asset],
                          constraints: Optional[PortfolioConstraints] = None,
                          method: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        High-level portfolio optimization interface.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            method: Optimization method
            **kwargs: Additional parameters

        Returns:
            Dictionary with optimization results
        """
        result = self.optimize(assets, constraints, method, **kwargs)

        if not result.success:
            return {
                'success': False,
                'error': result.error_messages
            }

        return {
            'success': True,
            'method': result.optimization_method,
            'weights': result.optimal_weights,
            'performance': result.performance.to_dict() if result.performance else {},
            'execution_time': result.execution_time,
            'objective_value': result.objective_value,
            'iterations': result.iterations
        }

    def compare_methods(self,
                        assets: List[Asset],
                        constraints: Optional[PortfolioConstraints] = None,
                        methods: Optional[List[str]] = None,
                        objective: str = 'sharpe',
                        market_views: Optional[MarketViewCollection] = None,
                        **kwargs) -> Dict[str, OptimizationResult]:
        """
        Compare multiple optimization methods.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            methods: List of methods to compare
            objective: Optimization objective
            market_views: Market views
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping method names to optimization results
        """
        if methods is None:
            methods = OptimizerFactory.get_available_methods()

        results = {}

        for method in methods:
            try:
                result = self.optimize(assets, constraints, method, objective, market_views, **kwargs)
                results[method] = result
            except Exception as e:
                logger.error(f"Error with method {method}: {e}")
                results[method] = OptimizationResult(
                    success=False,
                    optimization_method=method,
                    error_messages=[str(e)]
                )

        return results

    def get_efficient_frontier(self,
                              assets: List[Asset],
                              constraints: Optional[PortfolioConstraints] = None,
                              method: str = 'mean_variance',
                              num_points: int = 20,
                              **kwargs) -> Dict[str, Any]:
        """
        Calculate efficient frontier for specified method.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            method: Optimization method
            num_points: Number of points on efficient frontier
            **kwargs: Additional parameters

        Returns:
            Dictionary with efficient frontier data
        """
        try:
            optimizer = self.get_optimizer(method)

            if hasattr(optimizer, 'calculate_efficient_frontier'):
                return optimizer.calculate_efficient_frontier(assets, constraints, num_points, **kwargs)
            else:
                logger.warning(f"Method {method} does not support efficient frontier calculation")
                return {}

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return {}

    def analyze_portfolio(self,
                         weights: Dict[str, float],
                         assets: List[Asset],
                         benchmark_returns: Optional[pd.Series] = None,
                         risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Analyze portfolio performance and risk characteristics.

        Args:
            weights: Portfolio weights
            assets: List of assets
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with portfolio analysis
        """
        try:
            # Prepare returns data
            returns_data = {}
            for asset in assets:
                if asset.symbol in weights and not asset.returns.empty:
                    returns_data[asset.symbol] = asset.returns

            if not returns_data:
                return {'error': 'No valid asset returns found'}

            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if returns_df.empty:
                return {'error': 'No overlapping data found'}

            # Calculate portfolio returns
            portfolio_returns = self.return_calculator.calculate_portfolio_returns(
                returns_df, weights, ReturnType.SIMPLE
            )

            # Calculate performance metrics
            analysis = self.return_calculator.get_performance_summary(
                portfolio_returns, benchmark_returns, risk_free_rate
            )

            # Add portfolio characteristics
            analysis.update({
                'num_assets': len(weights),
                'concentration': self._calculate_concentration(weights),
                'turnover': 0.0,  # Would require previous weights
                'effective_number_assets': self._calculate_effective_number_assets(weights)
            })

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing portfolio: {e}")
            return {'error': str(e)}

    def backtest_portfolio(self,
                          weights: Dict[str, float],
                          assets: List[Asset],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          rebalance_freq: str = 'monthly') -> Dict[str, Any]:
        """
        Simple backtesting of portfolio strategy.

        Args:
            weights: Portfolio weights
            assets: List of assets
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalance_freq: Rebalancing frequency

        Returns:
            Dictionary with backtest results
        """
        try:
            # Prepare returns data
            returns_data = {}
            for asset in assets:
                if asset.symbol in weights and not asset.returns.empty:
                    returns_data[asset.symbol] = asset.returns

            if not returns_data:
                return {'error': 'No valid asset returns found'}

            returns_df = pd.DataFrame(returns_data)

            # Filter by date range if specified
            if start_date:
                returns_df = returns_df[returns_df.index >= start_date]
            if end_date:
                returns_df = returns_df[returns_df.index <= end_date]

            returns_df = returns_df.dropna()

            if returns_df.empty:
                return {'error': 'No data found for specified date range'}

            # Calculate portfolio returns
            portfolio_returns = self.return_calculator.calculate_portfolio_returns(
                returns_df, weights, ReturnType.SIMPLE
            )

            # Calculate backtest metrics
            cumulative_returns = (1 + portfolio_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0

            backtest_results = {
                'total_return': total_return,
                'annualized_return': self.return_calculator.annualize_returns(portfolio_returns),
                'annualized_volatility': self.return_calculator.calculate_volatility(portfolio_returns),
                'sharpe_ratio': self.return_calculator.calculate_sharpe_ratio(portfolio_returns),
                'max_drawdown': self.return_calculator.calculate_max_drawdown(portfolio_returns)[0],
                'start_date': returns_df.index[0].strftime('%Y-%m-%d'),
                'end_date': returns_df.index[-1].strftime('%Y-%m-%d'),
                'num_days': len(returns_df)
            }

            return backtest_results

        except Exception as e:
            logger.error(f"Error backtesting portfolio: {e}")
            return {'error': str(e)}

    def _get_default_constraints(self) -> PortfolioConstraints:
        """
        Get default portfolio constraints.

        Returns:
            Default constraints
        """
        return PortfolioConstraints(
            max_position_size=self.config.get('max_position_size', 0.2),
            max_sector_concentration=self.config.get('max_sector_concentration', 0.3),
            max_volatility=self.config.get('max_volatility', 0.25),
            min_return=self.config.get('min_return', 0.0),
            risk_free_rate=self.config.get('risk_free_rate', 0.02)
        )

    def _validate_inputs(self,
                        assets: List[Asset],
                        constraints: PortfolioConstraints,
                        method: Optional[str],
                        objective: str) -> None:
        """
        Validate optimization inputs.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            method: Optimization method
            objective: Optimization objective
        """
        if not assets:
            raise OptimizationError("No assets provided")

        if len(assets) < 2:
            raise OptimizationError("At least 2 assets required for optimization")

        if method and not OptimizerFactory.validate_method(method):
            raise OptimizationError(f"Unsupported optimization method: {method}")

        # Check if method requires market views
        if method:
            optimizer = self.get_optimizer(method)
            if optimizer.requires_market_views() and objective != 'sharpe':
                logger.warning(f"Method {method} typically requires market views for optimal results")

    def _calculate_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        weights_array = np.array(list(weights.values()))
        return np.sum(weights_array ** 2)

    def _calculate_effective_number_assets(self, weights: Dict[str, float]) -> float:
        """Calculate effective number of assets (inverse of concentration)."""
        concentration = self._calculate_concentration(weights)
        return 1.0 / concentration if concentration > 0 else 0

    def get_optimizer_info(self) -> Dict[str, Any]:
        """
        Get information about available optimizers.

        Returns:
            Dictionary with optimizer information
        """
        info = {
            'available_methods': OptimizerFactory.get_available_methods(),
            'default_method': self.default_method,
            'registered_optimizers': list(self.optimizers.keys()),
            'config': self.config
        }

        # Add information for each optimizer
        for method in info['available_methods']:
            try:
                optimizer = self.get_optimizer(method)
                info[f'{method}_info'] = optimizer.get_info()
            except Exception as e:
                logger.error(f"Error getting info for {method}: {e}")

        return info

    def save_results(self, result: OptimizationResult, filepath: str) -> None:
        """
        Save optimization results to file.

        Args:
            result: Optimization result
            filepath: File path to save results
        """
        try:
            results_dict = {
                'success': result.success,
                'method': result.optimization_method,
                'optimal_weights': result.optimal_weights,
                'objective_value': result.objective_value,
                'execution_time': result.execution_time,
                'iterations': result.iterations,
                'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                'performance': result.performance.to_dict() if result.performance else None,
                'error_messages': result.error_messages
            }

            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)

            logger.info(f"Results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def load_results(self, filepath: str) -> OptimizationResult:
        """
        Load optimization results from file.

        Args:
            filepath: File path to load results from

        Returns:
            OptimizationResult
        """
        try:
            with open(filepath, 'r') as f:
                results_dict = json.load(f)

            result = OptimizationResult(
                success=results_dict['success'],
                optimization_method=results_dict['method'],
                optimal_weights=results_dict['optimal_weights'],
                objective_value=results_dict['objective_value'],
                execution_time=results_dict['execution_time'],
                iterations=results_dict['iterations'],
                error_messages=results_dict.get('error_messages', [])
            )

            if results_dict.get('performance'):
                performance = PortfolioPerformance()
                for key, value in results_dict['performance'].items():
                    setattr(performance, key, value)
                result.performance = performance

            if results_dict.get('timestamp'):
                result.timestamp = datetime.fromisoformat(results_dict['timestamp'])

            return result

        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise OptimizationError(f"Failed to load results: {e}")

    def __str__(self) -> str:
        """String representation."""
        return f"PortfolioOptimizer(methods={self.get_optimizer_info()['available_methods']})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(default_method='{self.default_method}', registered_methods={list(self.optimizers.keys())})"