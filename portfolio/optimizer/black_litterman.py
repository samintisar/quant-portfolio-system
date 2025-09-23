"""
Black-Litterman portfolio optimization implementation.

Implements the Black-Litterman model for combining market equilibrium with investor views.
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


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman portfolio optimizer.

    Combines market equilibrium returns with investor views.
    """

    def __init__(self):
        """Initialize the Black-Litterman optimizer."""
        super().__init__("BlackLitterman")
        self.supported_objectives = ['sharpe', 'min_risk', 'max_return']
        self.confidence_scaling = 1.0  # Scaling factor for view confidence

    def requires_market_views(self) -> bool:
        """Return True since Black-Litterman requires market views."""
        return True

    def optimize(self,
                 assets: List[Asset],
                 constraints: PortfolioConstraints,
                 objective: str = 'sharpe',
                 market_views: Optional[MarketViewCollection] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio using Black-Litterman model.

        Args:
            assets: List of assets in the portfolio
            constraints: Portfolio constraints
            objective: Optimization objective
            market_views: Investor views
            **kwargs: Additional parameters (delta, risk_aversion, etc.)

        Returns:
            OptimizationResult with optimal weights
        """
        start_time = datetime.now()

        try:
            # Validate inputs
            self.validate_inputs(assets, constraints, objective)

            if market_views is None or not market_views.views:
                raise OptimizationError("Black-Litterman optimization requires market views")

            # Prepare data
            returns_df, data_info = self.prepare_returns_data(assets)
            mean_returns = data_info['mean_returns']
            cov_matrix = data_info['cov_matrix']

            # Build constraints
            constraint_matrices = self.build_constraints_matrix(assets, constraints, len(assets))

            # Calculate Black-Litterman expected returns
            bl_returns = self._calculate_bl_returns(
                assets, mean_returns, cov_matrix, market_views, **kwargs
            )

            # Solve optimization with BL returns
            result = self._solve_optimization(
                assets, returns_df, bl_returns, cov_matrix,
                constraint_matrices, constraints, objective, market_views, **kwargs
            )

            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()

            # Log results
            if result.success:
                logger.info(f"Black-Litterman optimization successful: {result}")
            else:
                logger.warning(f"Black-Litterman optimization failed: {result.error_messages}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Black-Litterman optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=execution_time,
                optimization_method="black_litterman",
                error_messages=[str(e)]
            )

    def _calculate_bl_returns(self,
                              assets: List[Asset],
                              market_returns: pd.Series,
                              cov_matrix: pd.DataFrame,
                              market_views: MarketViewCollection,
                              **kwargs) -> pd.Series:
        """
        Calculate Black-Litterman expected returns.

        Args:
            assets: List of assets
            market_returns: Market equilibrium returns
            cov_matrix: Covariance matrix
            market_views: Investor views
            **kwargs: Additional parameters

        Returns:
            Black-Litterman expected returns
        """
        try:
            symbols = [asset.symbol for asset in assets]
            num_assets = len(symbols)

            # Black-Litterman parameters
            delta = kwargs.get('delta', 2.5)  # Risk aversion parameter
            tau = kwargs.get('tau', 0.05)     # Scaling factor for uncertainty

            # Market weights (if not provided, assume equal weights)
            market_weights = kwargs.get('market_weights')
            if market_weights is None:
                market_weights = np.ones(num_assets) / num_assets

            # Calculate implied excess equilibrium returns
            implied_returns = delta * np.dot(cov_matrix, market_weights)

            # Prepare views in Black-Litterman format
            P, Q, Omega = market_views.to_bl_matrices(symbols)

            if P.shape[0] == 0:  # No views
                logger.warning("No valid market views provided, using market equilibrium")
                return market_returns

            # Calculate Black-Litterman expected returns
            # Formula: μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
            tau_cov = tau * cov_matrix

            # Calculate intermediate matrices
            inv_tau_cov = np.linalg.inv(tau_cov)
            inv_omega = np.linalg.inv(Omega)

            # Calculate Black-Litterman returns
            temp_matrix = inv_tau_cov + P.T @ inv_omega @ P
            inv_temp_matrix = np.linalg.inv(temp_matrix)

            temp_vector = inv_tau_cov @ implied_returns + P.T @ inv_omega @ Q
            bl_returns = inv_temp_matrix @ temp_vector

            # Convert to Series
            bl_returns_series = pd.Series(bl_returns, index=symbols)

            # Calculate posterior covariance matrix (for information)
            posterior_cov = np.linalg.inv(inv_temp_matrix)

            logger.info(f"Calculated Black-Litterman returns for {len(symbols)} assets with {len(market_views.views)} views")

            return bl_returns_series

        except Exception as e:
            logger.error(f"Error calculating Black-Litterman returns: {e}")
            raise OptimizationError(f"Failed to calculate Black-Litterman returns: {e}")

    def _solve_optimization(self,
                           assets: List[Asset],
                           returns_df: pd.DataFrame,
                           bl_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           constraint_matrices: Dict[str, Any],
                           constraints: PortfolioConstraints,
                           objective: str,
                           market_views: MarketViewCollection,
                           **kwargs) -> OptimizationResult:
        """
        Solve Black-Litterman optimization.

        Args:
            assets: List of assets
            returns_df: Returns DataFrame
            bl_returns: Black-Litterman expected returns
            cov_matrix: Covariance matrix
            constraint_matrices: Constraint matrices
            constraints: Portfolio constraints
            objective: Optimization objective
            market_views: Market views
            **kwargs: Additional parameters

        Returns:
            OptimizationResult
        """
        try:
            num_assets = len(assets)
            symbols = [asset.symbol for asset in assets]

            # Define optimization variable
            weights = cp.Variable(num_assets)

            # Build constraints (same as Mean-Variance)
            opt_constraints = self._build_optimization_constraints(
                weights, symbols, constraint_matrices, constraints
            )

            # Define objective using Black-Litterman returns
            bl_returns_arr = bl_returns[symbols].values

            if objective == 'sharpe':
                objective_expr, success_metric = self._build_sharpe_objective(
                    weights, bl_returns_arr, cov_matrix, constraints.risk_free_rate
                )
            elif objective == 'min_risk':
                objective_expr, success_metric = self._build_min_risk_objective(
                    weights, cov_matrix
                )
            elif objective == 'max_return':
                objective_expr, success_metric = self._build_max_return_objective(
                    weights, bl_returns_arr
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
                    optimization_method="black_litterman",
                    error_messages=["No solver found feasible solution"]
                )

            # Process results
            optimal_weights = {symbols[i]: float(solution[i]) for i in range(num_assets)}
            optimal_weights = {k: v for k, v in optimal_weights.items() if abs(v) > 1e-8}

            # Validate result
            if not self.validate_optimization_result(optimal_weights, constraints):
                return OptimizationResult(
                    success=False,
                    optimization_method="black_litterman",
                    error_messages=["Optimization result violates constraints"]
                )

            # Calculate portfolio metrics using Black-Litterman returns
            portfolio_metrics = self.calculate_portfolio_metrics(
                optimal_weights, bl_returns, cov_matrix, constraints.risk_free_rate
            )

            # Create performance object
            performance = PortfolioPerformance()
            performance.annual_return = portfolio_metrics.get('annual_return')
            performance.annual_volatility = portfolio_metrics.get('annual_volatility')
            performance.sharpe_ratio = portfolio_metrics.get('sharpe_ratio')

            # Calculate view tracking metrics
            view_metrics = self._calculate_view_tracking_metrics(
                optimal_weights, market_views, bl_returns
            )

            return OptimizationResult(
                success=True,
                optimal_weights=optimal_weights,
                objective_value=float(success_metric),
                optimization_method="black_litterman",
                performance=performance,
                iterations=problem.solver_stats.num_iters if hasattr(problem, 'solver_stats') else 0
            )

        except Exception as e:
            logger.error(f"Error solving Black-Litterman optimization: {e}")
            return OptimizationResult(
                success=False,
                optimization_method="black_litterman",
                error_messages=[str(e)]
            )

    def _build_optimization_constraints(self,
                                     weights: cp.Variable,
                                     symbols: List[str],
                                     constraint_matrices: Dict[str, Any],
                                     constraints: PortfolioConstraints) -> List[cp.Constraint]:
        """
        Build optimization constraints (same as Mean-Variance).

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
            bl_returns_arr = constraint_matrices.get('bl_returns', np.zeros(len(symbols)))
            opt_constraints.append(bl_returns_arr @ weights >= min_return)

        return opt_constraints

    def _build_sharpe_objective(self,
                              weights: cp.Variable,
                              bl_returns: np.ndarray,
                              cov_matrix: pd.DataFrame,
                              risk_free_rate: float) -> Tuple[cp.Expression, float]:
        """
        Build Sharpe ratio maximization objective with BL returns.

        Args:
            weights: CVXPY variable for weights
            bl_returns: Black-Litterman expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Tuple of (objective expression, success metric)
        """
        # Portfolio return
        portfolio_return = bl_returns @ weights

        # Portfolio risk
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        portfolio_risk = cp.sqrt(portfolio_variance)

        # Excess return
        excess_return = portfolio_return - risk_free_rate / 252  # Daily risk-free rate

        # Maximize return - lambda*risk (approximation of Sharpe)
        risk_aversion = 1.0
        objective = cp.Maximize(excess_return - risk_aversion * portfolio_risk)

        return objective, float(excess_return.value) if hasattr(excess_return, 'value') else 0.0

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

        return objective, float(portfolio_variance.value) if hasattr(portfolio_variance, 'value') else 0.0

    def _build_max_return_objective(self,
                                   weights: cp.Variable,
                                   bl_returns: np.ndarray) -> Tuple[cp.Expression, float]:
        """
        Build maximum return objective with BL returns.

        Args:
            weights: CVXPY variable for weights
            bl_returns: Black-Litterman expected returns

        Returns:
            Tuple of (objective expression, success metric)
        """
        portfolio_return = bl_returns @ weights
        objective = cp.Maximize(portfolio_return)

        return objective, float(portfolio_return.value) if hasattr(portfolio_return, 'value') else 0.0

    def _calculate_view_tracking_metrics(self,
                                       weights: Dict[str, float],
                                       market_views: MarketViewCollection,
                                       bl_returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate metrics for how well views are tracked.

        Args:
            weights: Optimal weights
            market_views: Market views
            bl_returns: Black-Litterman returns

        Returns:
            Dictionary with view tracking metrics
        """
        try:
            metrics = {}

            for view in market_views.views:
                if view.asset_symbol in weights and view.asset_symbol in bl_returns:
                    # Compare expected return with implied return from weights
                    weight = weights[view.asset_symbol]
                    expected_return = bl_returns[view.asset_symbol]

                    metrics[view.asset_symbol] = {
                        'view_return': view.expected_return,
                        'bl_return': expected_return * 252,  # Annualize
                        'portfolio_weight': weight,
                        'confidence': view.confidence,
                        'view_type': view.view_type.value
                    }

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating view tracking metrics: {e}")
            return {}

    def analyze_view_impact(self,
                           assets: List[Asset],
                           constraints: PortfolioConstraints,
                           market_views: MarketViewCollection,
                           **kwargs) -> Dict[str, Any]:
        """
        Analyze the impact of views on portfolio allocation.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            market_views: Market views
            **kwargs: Additional parameters

        Returns:
            Dictionary with view impact analysis
        """
        try:
            # Optimize with market equilibrium (no views)
            equilibrium_result = self.optimize(assets, constraints, 'sharpe', MarketViewCollection(), **kwargs)

            # Optimize with views
            views_result = self.optimize(assets, constraints, 'sharpe', market_views, **kwargs)

            if not equilibrium_result.success or not views_result.success:
                return {}

            # Compare allocations
            analysis = {
                'equilibrium_weights': equilibrium_result.optimal_weights,
                'bl_weights': views_result.optimal_weights,
                'weight_changes': {},
                'performance_comparison': {
                    'equilibrium': equilibrium_result.performance.to_dict() if equilibrium_result.performance else {},
                    'bl_views': views_result.performance.to_dict() if views_result.performance else {}
                }
            }

            # Calculate weight changes
            for asset in assets:
                symbol = asset.symbol
                eq_weight = equilibrium_result.optimal_weights.get(symbol, 0)
                bl_weight = views_result.optimal_weights.get(symbol, 0)
                analysis['weight_changes'][symbol] = bl_weight - eq_weight

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing view impact: {e}")
            return {}