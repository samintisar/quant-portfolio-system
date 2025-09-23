"""
Mean-Variance portfolio optimization implementation.

Classic Markowitz portfolio optimization using CVXPY.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import cvxpy as cp
from datetime import datetime

from portfolio.logging_config import get_logger, OptimizationError
from portfolio.optimizer.base import BaseOptimizer
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.models.result import OptimizationResult
from portfolio.models.views import MarketViewCollection
from portfolio.models.performance import PortfolioPerformance

logger = get_logger(__name__)


class MeanVarianceOptimizer(BaseOptimizer):
    """
    Mean-Variance portfolio optimizer using CVXPY.

    Implements classic Markowitz optimization with various objectives.
    """

    def __init__(self):
        """Initialize the Mean-Variance optimizer."""
        super().__init__("MeanVariance")
        self.supported_objectives = ['sharpe', 'min_risk', 'max_return', 'utility']

    def optimize(self,
                 assets: List[Asset],
                 constraints: PortfolioConstraints,
                 objective: str = 'sharpe',
                 market_views: Optional[MarketViewCollection] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio using Mean-Variance optimization.

        Args:
            assets: List of assets in the portfolio
            constraints: Portfolio constraints
            objective: Optimization objective
            market_views: Not used for Mean-Variance
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with optimal weights
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            self.validate_inputs(assets, constraints, objective)

            # Prepare data
            returns_df, data_info = self.prepare_returns_data(assets)
            mean_returns = data_info['mean_returns']
            cov_matrix = data_info['cov_matrix']

            # Build constraints
            constraint_matrices = self.build_constraints_matrix(assets, constraints, len(assets))

            # Solve optimization
            result = self._solve_optimization(
                assets, returns_df, mean_returns, cov_matrix,
                constraint_matrices, constraints, objective, **kwargs
            )

            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()

            # Log results
            if result.success:
                logger.info(f"Mean-Variance optimization successful: {result}")
            else:
                logger.warning(f"Mean-Variance optimization failed: {result.error_messages}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Mean-Variance optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=execution_time,
                optimization_method="mean_variance",
                error_messages=[str(e)]
            )

    def _solve_optimization(self,
                           assets: List[Asset],
                           returns_df: pd.DataFrame,
                           mean_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           constraint_matrices: Dict[str, Any],
                           constraints: PortfolioConstraints,
                           objective: str,
                           **kwargs) -> OptimizationResult:
        """
        Solve the Mean-Variance optimization problem.

        Args:
            assets: List of assets
            returns_df: Returns DataFrame
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            constraint_matrices: Constraint matrices
            constraints: Portfolio constraints
            objective: Optimization objective
            **kwargs: Additional parameters

        Returns:
            OptimizationResult
        """
        try:
            num_assets = len(assets)
            symbols = [asset.symbol for asset in assets]

            # Define optimization variable
            weights = cp.Variable(num_assets)

            # Build constraints list
            opt_constraints = self._build_optimization_constraints(
                weights, symbols, constraint_matrices, constraints
            )

            # Define objective based on objective function
            if objective == 'sharpe':
                objective_expr, success_metric = self._build_sharpe_objective(
                    weights, mean_returns, cov_matrix, constraints.risk_free_rate
                )
            elif objective == 'min_risk':
                objective_expr, success_metric = self._build_min_risk_objective(
                    weights, cov_matrix
                )
            elif objective == 'max_return':
                objective_expr, success_metric = self._build_max_return_objective(
                    weights, mean_returns
                )
            elif objective == 'utility':
                risk_aversion = kwargs.get('risk_aversion', 1.0)
                objective_expr, success_metric = self._build_utility_objective(
                    weights, mean_returns, cov_matrix, risk_aversion
                )
            else:
                raise OptimizationError(f"Unsupported objective: {objective}")

            # Formulate and solve problem
            problem = cp.Problem(objective_expr, opt_constraints)

            # Try different solvers
            solvers = ['ECOS', 'SCS', 'OSQP']
            solution = None

            for solver in solvers:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in ['optimal', 'optimal_inaccurate']:
                        solution = weights.value
                        break
                except Exception as e:
                    logger.debug(f"Solver {solver} failed: {e}")
                    continue

            if solution is None:
                return OptimizationResult(
                    success=False,
                    optimization_method="mean_variance",
                    error_messages=["No solver found feasible solution"]
                )

            # Process results
            optimal_weights = {symbols[i]: float(solution[i]) for i in range(num_assets)}
            optimal_weights = {k: v for k, v in optimal_weights.items() if abs(v) > 1e-8}

            # Validate result
            if not self.validate_optimization_result(optimal_weights, constraints):
                return OptimizationResult(
                    success=False,
                    optimization_method="mean_variance",
                    error_messages=["Optimization result violates constraints"]
                )

            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(
                optimal_weights, mean_returns, cov_matrix, constraints.risk_free_rate
            )

            # Create performance object
            performance = PortfolioPerformance()
            performance.annual_return = portfolio_metrics.get('annual_return')
            performance.annual_volatility = portfolio_metrics.get('annual_volatility')
            performance.sharpe_ratio = portfolio_metrics.get('sharpe_ratio')

            return OptimizationResult(
                success=True,
                optimal_weights=optimal_weights,
                objective_value=float(success_metric),
                optimization_method="mean_variance",
                performance=performance,
                iterations=problem.solver_stats.num_iters if hasattr(problem, 'solver_stats') else 0
            )

        except Exception as e:
            logger.error(f"Error solving optimization: {e}")
            return OptimizationResult(
                success=False,
                optimization_method="mean_variance",
                error_messages=[str(e)]
            )

    def _build_optimization_constraints(self,
                                     weights: cp.Variable,
                                     symbols: List[str],
                                     constraint_matrices: Dict[str, Any],
                                     constraints: PortfolioConstraints) -> List[cp.Constraint]:
        """
        Build optimization constraints.

        Args:
            weights: CVXPY variable for weights
            symbols: Asset symbols
            constraint_matrices: Constraint matrices
            constraints: Portfolio constraints

        Returns:
            List of CVXPY constraints
        """
        opt_constraints = []

        # Sum of weights = 1
        opt_constraints.append(cp.sum(weights) == 1)

        # Weight bounds
        min_weights, max_weights = constraint_matrices['weight_bounds']
        opt_constraints.extend([weights >= min_weights, weights <= max_weights])

        # Sector constraints
        if 'sector_constraints' in constraint_matrices:
            for sector, sector_data in constraint_matrices['sector_constraints'].items():
                sector_weights = sector_data['matrix'] @ weights
                opt_constraints.append(sector_weights <= sector_data['max_weight'])

        # Minimum return constraint
        if 'min_return' in constraint_matrices:
            min_return = constraint_matrices['min_return']
            mean_returns_arr = np.array([mean_returns for mean_returns in symbols])
            opt_constraints.append(mean_returns_arr @ weights >= min_return)

        # Maximum volatility constraint
        if 'max_volatility' in constraint_matrices:
            max_vol = constraint_matrices['max_volatility']
            cov_matrix = constraint_matrices.get('cov_matrix')
            if cov_matrix is not None:
                # Annualized volatility constraint
                annual_vol_factor = np.sqrt(252)
                portfolio_variance = cp.quad_form(weights, cov_matrix)
                max_annual_vol = max_vol
                opt_constraints.append(cp.sqrt(portfolio_variance) * annual_vol_factor <= max_annual_vol)

        return opt_constraints

    def _build_sharpe_objective(self,
                              weights: cp.Variable,
                              mean_returns: pd.Series,
                              cov_matrix: pd.DataFrame,
                              risk_free_rate: float) -> Tuple[cp.Expression, float]:
        """
        Build Sharpe ratio maximization objective.

        Args:
            weights: CVXPY variable for weights
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Tuple of (objective expression, success metric)
        """
        # Portfolio return
        mean_returns_arr = mean_returns.values
        portfolio_return = mean_returns_arr @ weights

        # Portfolio risk
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        portfolio_risk = cp.sqrt(portfolio_variance)

        # Sharpe ratio (maximize)
        excess_return = portfolio_return - risk_free_rate / 252  # Daily risk-free rate

        # For numerical stability, use fractional form
        # Maximize (excess_return / risk) = Minimize (risk / excess_return)
        # This is a non-convex problem, so we use a convex approximation

        # Instead, maximize return - lambda*risk (approximation of Sharpe)
        risk_aversion = 1.0
        objective = cp.Maximize(excess_return - risk_aversion * portfolio_risk)

        return objective, 0.0  # Return a simple success metric

    def _build_min_risk_objective(self,
                                weights: cp.Variable,
                                cov_matrix: pd.DataFrame) -> Tuple[cp.Expression, float]:
        """
        Build minimum risk objective.

        Args:
            weights: CVXPY variable for weights
            cov_matrix: Covariance matrix

        Returns:
            Tuple of (objective expression, success metric)
        """
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)

        return objective, 0.0  # Return a simple success metric

    def _build_max_return_objective(self,
                                   weights: cp.Variable,
                                   mean_returns: pd.Series) -> Tuple[cp.Expression, float]:
        """
        Build maximum return objective.

        Args:
            weights: CVXPY variable for weights
            mean_returns: Mean returns

        Returns:
            Tuple of (objective expression, success metric)
        """
        mean_returns_arr = mean_returns.values
        portfolio_return = mean_returns_arr @ weights
        objective = cp.Maximize(portfolio_return)

        return objective, 0.0  # Return a simple success metric

    def _build_utility_objective(self,
                                weights: cp.Variable,
                                mean_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                risk_aversion: float) -> Tuple[cp.Expression, float]:
        """
        Build utility maximization objective.

        Args:
            weights: CVXPY variable for weights
            mean_returns: Mean returns
            cov_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter

        Returns:
            Tuple of (objective expression, success metric)
        """
        mean_returns_arr = mean_returns.values
        portfolio_return = mean_returns_arr @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)

        # Utility = Return - 0.5 * Risk Aversion * Variance
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        objective = cp.Maximize(utility)

        return objective, 0.0  # Return a simple success metric

    def calculate_efficient_frontier(self,
                                   assets: List[Asset],
                                   constraints: PortfolioConstraints,
                                   num_points: int = 20) -> Dict[str, Any]:
        """
        Calculate efficient frontier points.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            num_points: Number of points on efficient frontier

        Returns:
            Dictionary with efficient frontier data
        """
        try:
            # Prepare data
            returns_df, data_info = self.prepare_returns_data(assets)
            mean_returns = data_info['mean_returns']
            cov_matrix = data_info['cov_matrix']

            # Find min and max return portfolios
            min_var_result = self.optimize(assets, constraints, 'min_risk')
            max_ret_result = self.optimize(assets, constraints, 'max_return')

            if not min_var_result.success or not max_ret_result.success:
                logger.error("Failed to calculate efficient frontier endpoints")
                return {}

            # Get return range
            min_return = min_var_result.performance.annual_return if min_var_result.performance else 0
            max_return = max_ret_result.performance.annual_return if max_ret_result.performance else 0

            # Calculate points along efficient frontier
            frontier_points = []
            return_targets = np.linspace(min_return, max_return, num_points)

            for target_return in return_targets:
                # Temporarily set minimum return constraint
                temp_constraints = PortfolioConstraints.from_dict(constraints.to_dict())
                temp_constraints.min_return = target_return / 252  # Convert to daily

                try:
                    result = self.optimize(assets, temp_constraints, 'min_risk')
                    if result.success and result.performance:
                        frontier_points.append({
                            'return': result.performance.annual_return,
                            'volatility': result.performance.annual_volatility,
                            'sharpe_ratio': result.performance.sharpe_ratio,
                            'weights': result.optimal_weights
                        })
                except Exception as e:
                    logger.debug(f"Failed to calculate frontier point for return {target_return}: {e}")

            return {
                'min_return': min_return,
                'max_return': max_return,
                'min_volatility': min_var_result.performance.annual_volatility if min_var_result.performance else 0,
                'max_sharpe_portfolio': self._find_max_sharpe_portfolio(frontier_points),
                'frontier_points': frontier_points
            }

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return {}

    def _find_max_sharpe_portfolio(self, frontier_points: List[Dict]) -> Optional[Dict]:
        """Find portfolio with maximum Sharpe ratio on efficient frontier."""
        if not frontier_points:
            return None

        max_sharpe_portfolio = max(frontier_points, key=lambda x: x.get('sharpe_ratio', -float('inf')))
        return max_sharpe_portfolio