"""
CVaR (Conditional Value at Risk) portfolio optimization implementation.

Implements CVaR optimization for robust portfolio construction with tail risk management.
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


class CVaROptimizer(BaseOptimizer):
    """
    CVaR portfolio optimizer using linear programming approach.

    Implements Conditional Value at Risk optimization for tail risk management.
    """

    def __init__(self):
        """Initialize the CVaR optimizer."""
        super().__init__("CVaR")
        self.supported_objectives = ['cvar', 'min_cvar', 'return_cvar']
        self.default_alpha = 0.05  # 95% confidence level
        self.default_num_scenarios = 1000

    def requires_market_views(self) -> bool:
        """Return False since CVaR optimization doesn't require market views."""
        return False

    def optimize(self,
                 assets: List[Asset],
                 constraints: PortfolioConstraints,
                 objective: str = 'cvar',
                 market_views: Optional[MarketViewCollection] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio using CVaR optimization.

        Args:
            assets: List of assets in the portfolio
            constraints: Portfolio constraints
            objective: Optimization objective
            market_views: Not used for CVaR
            **kwargs: Additional parameters (alpha, num_scenarios, etc.)

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

            # Solve CVaR optimization
            result = self._solve_cvar_optimization(
                assets, returns_df, mean_returns, cov_matrix,
                constraint_matrices, constraints, objective, **kwargs
            )

            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()

            # Log results
            if result.success:
                logger.info(f"CVaR optimization successful: {result}")
            else:
                logger.warning(f"CVaR optimization failed: {result.error_messages}")

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"CVaR optimization failed: {e}")

            return OptimizationResult(
                success=False,
                execution_time=execution_time,
                optimization_method="cvar",
                error_messages=[str(e)]
            )

    def _solve_cvar_optimization(self,
                                assets: List[Asset],
                                returns_df: pd.DataFrame,
                                mean_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                constraint_matrices: Dict[str, Any],
                                constraints: PortfolioConstraints,
                                objective: str,
                                **kwargs) -> OptimizationResult:
        """
        Solve CVaR optimization problem.

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

            # CVaR parameters
            alpha = kwargs.get('alpha', self.default_alpha)
            num_scenarios = kwargs.get('num_scenarios', self.default_num_scenarios)
            min_return_target = kwargs.get('min_return', 0.0)

            # Generate scenarios
            scenario_returns = self._generate_scenarios(returns_df, num_scenarios)

            # Define optimization variables
            weights = cp.Variable(num_assets)
            VaR = cp.Variable()  # Value at Risk
            z = cp.Variable(num_scenarios)  # Shortfall variables

            # Build constraints
            opt_constraints = self._build_cvar_constraints(
                weights, VaR, z, symbols, constraint_matrices, constraints, scenario_returns
            )

            # Build objective based on objective function
            if objective == 'cvar':
                objective_expr, success_metric = self._build_cvar_objective(
                    weights, VaR, z, scenario_returns, alpha
                )
            elif objective == 'min_cvar':
                objective_expr, success_metric = self._build_min_cvar_objective(
                    weights, VaR, z, scenario_returns, alpha
                )
            elif objective == 'return_cvar':
                objective_expr, success_metric = self._build_return_cvar_objective(
                    weights, VaR, z, scenario_returns, alpha, mean_returns, min_return_target
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
                    optimization_method="cvar",
                    error_messages=["No solver found feasible solution"]
                )

            # Process results
            optimal_weights = {symbols[i]: float(solution[i]) for i in range(num_assets)}
            optimal_weights = {k: v for k, v in optimal_weights.items() if abs(v) > 1e-8}

            # Validate result
            if not self.validate_optimization_result(optimal_weights, constraints):
                return OptimizationResult(
                    success=False,
                    optimization_method="cvar",
                    error_messages=["Optimization result violates constraints"]
                )

            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(
                optimal_weights, mean_returns, cov_matrix, constraints.risk_free_rate
            )

            # Calculate CVaR-specific metrics
            cvar_metrics = self._calculate_cvar_metrics(
                optimal_weights, scenario_returns, alpha, VaR.value
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
                optimization_method="cvar",
                performance=performance,
                iterations=problem.solver_stats.num_iters if hasattr(problem, 'solver_stats') else 0,
                cvar_metrics=cvar_metrics
            )

        except Exception as e:
            logger.error(f"Error solving CVaR optimization: {e}")
            return OptimizationResult(
                success=False,
                optimization_method="cvar",
                error_messages=[str(e)]
            )

    def _generate_scenarios(self, returns_df: pd.DataFrame, num_scenarios: int) -> np.ndarray:
        """
        Generate scenario returns for CVaR calculation.

        Args:
            returns_df: Historical returns
            num_scenarios: Number of scenarios to generate

        Returns:
            Scenario returns matrix
        """
        try:
            if len(returns_df) < num_scenarios:
                # Use historical data if insufficient for sampling
                return returns_df.values

            # Sample with replacement from historical returns
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(returns_df), num_scenarios, replace=True)
            scenario_returns = returns_df.iloc[indices].values

            return scenario_returns

        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            # Fallback to historical returns
            return returns_df.values

    def _build_cvar_constraints(self,
                                weights: cp.Variable,
                                VaR: cp.Variable,
                                z: cp.Variable,
                                symbols: List[str],
                                constraint_matrices: Dict[str, Any],
                                constraints: PortfolioConstraints,
                                scenario_returns: np.ndarray) -> List[cp.Constraint]:
        """
        Build CVaR optimization constraints.

        Args:
            weights: CVXPY variable for weights
            VaR: CVXPY variable for VaR
            z: CVXPY variable for shortfalls
            symbols: Asset symbols
            constraint_matrices: Constraint matrices
            constraints: Portfolio constraints
            scenario_returns: Scenario returns

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

        # CVaR-specific constraints
        # Portfolio returns for each scenario
        portfolio_returns = scenario_returns @ weights

        # Shortfall constraints: z_i >= -(portfolio_return_i - VaR)
        opt_constraints.extend([z[i] >= -(portfolio_returns[i] - VaR) for i in range(len(scenario_returns))])

        # Non-negativity of shortfalls
        opt_constraints.extend([z[i] >= 0 for i in range(len(scenario_returns))])

        return opt_constraints

    def _build_cvar_objective(self,
                              weights: cp.Variable,
                              VaR: cp.Variable,
                              z: cp.Variable,
                              scenario_returns: np.ndarray,
                              alpha: float) -> Tuple[cp.Expression, float]:
        """
        Build CVaR minimization objective.

        Args:
            weights: CVXPY variable for weights
            VaR: CVXPY variable for VaR
            z: CVXPY variable for shortfalls
            scenario_returns: Scenario returns
            alpha: Confidence level

        Returns:
            Tuple of (objective expression, success metric)
        """
        num_scenarios = len(scenario_returns)
        # CVaR = VaR + (1/(alpha*N)) * sum(z_i)
        cvar = VaR + (1 / (alpha * num_scenarios)) * cp.sum(z)

        objective = cp.Minimize(cvar)

        return objective, float(cvar.value) if hasattr(cvar, 'value') else 0.0

    def _build_min_cvar_objective(self,
                                 weights: cp.Variable,
                                 VaR: cp.Variable,
                                 z: cp.Variable,
                                 scenario_returns: np.ndarray,
                                 alpha: float) -> Tuple[cp.Expression, float]:
        """
        Build minimum CVaR objective (similar to standard CVaR).

        Args:
            weights: CVXPY variable for weights
            VaR: CVXPY variable for VaR
            z: CVXPY variable for shortfalls
            scenario_returns: Scenario returns
            alpha: Confidence level

        Returns:
            Tuple of (objective expression, success metric)
        """
        return self._build_cvar_objective(weights, VaR, z, scenario_returns, alpha)

    def _build_return_cvar_objective(self,
                                    weights: cp.Variable,
                                    VaR: cp.Variable,
                                    z: cp.Variable,
                                    scenario_returns: np.ndarray,
                                    alpha: float,
                                    mean_returns: pd.Series,
                                    min_return_target: float) -> Tuple[cp.Expression, float]:
        """
        Build return-CVaR optimization objective.

        Args:
            weights: CVXPY variable for weights
            VaR: CVXPY variable for VaR
            z: CVXPY variable for shortfalls
            scenario_returns: Scenario returns
            alpha: Confidence level
            mean_returns: Mean returns
            min_return_target: Minimum return target

        Returns:
            Tuple of (objective expression, success metric)
        """
        num_scenarios = len(scenario_returns)

        # Expected portfolio return
        expected_return = cp.sum(weights * mean_returns.values)

        # CVaR calculation
        cvar = VaR + (1 / (alpha * num_scenarios)) * cp.sum(z)

        # Maximize return - lambda * CVaR
        risk_aversion = 2.0
        objective_expr = cp.Maximize(expected_return - risk_aversion * cvar)

        # Add minimum return constraint
        return objective_expr, float(expected_return.value) if hasattr(expected_return, 'value') else 0.0

    def _calculate_cvar_metrics(self,
                               weights: Dict[str, float],
                               scenario_returns: np.ndarray,
                               alpha: float,
                               var_value: float) -> Dict[str, Any]:
        """
        Calculate CVaR-specific metrics.

        Args:
            weights: Optimal weights
            scenario_returns: Scenario returns
            alpha: Confidence level
            var_value: VaR value

        Returns:
            Dictionary with CVaR metrics
        """
        try:
            num_scenarios = len(scenario_returns)

            # Calculate portfolio returns for each scenario
            symbols = list(weights.keys())
            weight_array = np.array([weights[symbol] for symbol in symbols])
            portfolio_returns = scenario_returns @ weight_array

            # Calculate CVaR
            shortfalls = np.maximum(-(portfolio_returns - var_value), 0)
            cvar_value = var_value + (1 / (alpha * num_scenarios)) * np.sum(shortfalls)

            # Calculate VaR and CVaR from scenarios
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            cvar_99 = np.mean(portfolio_returns[portfolio_returns <= var_99])

            return {
                'var_value': float(var_value),
                'cvar_value': float(cvar_value),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'alpha': alpha,
                'num_scenarios': num_scenarios
            }

        except Exception as e:
            logger.error(f"Error calculating CVaR metrics: {e}")
            return {}

    def calculate_efficient_cvar_frontier(self,
                                        assets: List[Asset],
                                        constraints: PortfolioConstraints,
                                        num_points: int = 20,
                                        alpha: float = 0.05) -> Dict[str, Any]:
        """
        Calculate efficient CVaR frontier.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            num_points: Number of points on efficient frontier
            alpha: Confidence level for CVaR

        Returns:
            Dictionary with efficient CVaR frontier data
        """
        try:
            # Prepare data
            returns_df, data_info = self.prepare_returns_data(assets)
            mean_returns = data_info['mean_returns']
            cov_matrix = data_info['cov_matrix']

            # Find min CVaR and max return portfolios
            min_cvar_result = self.optimize(assets, constraints, 'min_cvar', alpha=alpha)

            # Get return range
            min_return = min_cvar_result.performance.annual_return if min_cvar_result.performance else 0
            max_return = mean_returns.max() * 252  # Annualized

            # Calculate points along efficient frontier
            frontier_points = []
            return_targets = np.linspace(min_return, max_return, num_points)

            for target_return in return_targets:
                try:
                    result = self.optimize(
                        assets, constraints, 'return_cvar',
                        min_return=target_return / 252,  # Convert to daily
                        alpha=alpha
                    )
                    if result.success and result.cvar_metrics:
                        frontier_points.append({
                            'return': result.performance.annual_return if result.performance else 0,
                            'cvar': result.cvar_metrics.get('cvar_value', 0),
                            'var': result.cvar_metrics.get('var_value', 0),
                            'weights': result.optimal_weights
                        })
                except Exception as e:
                    logger.debug(f"Failed to calculate frontier point for return {target_return}: {e}")

            return {
                'min_cvar': min_cvar_result.cvar_metrics.get('cvar_value', 0) if min_cvar_result.cvar_metrics else 0,
                'max_return': max_return,
                'frontier_points': frontier_points,
                'alpha': alpha
            }

        except Exception as e:
            logger.error(f"Error calculating efficient CVaR frontier: {e}")
            return {}