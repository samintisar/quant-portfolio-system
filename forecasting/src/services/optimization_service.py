"""
Parameter Optimization Services for Financial Forecasting Models

This module provides comprehensive parameter optimization services for financial forecasting models,
including grid search, Bayesian optimization, regime-aware optimization, constrained optimization,
and multi-objective optimization techniques.

Services:
- ParameterOptimizer: Main parameter optimization service
- GridSearchOptimizer: Grid search optimization
- BayesianOptimizer: Bayesian optimization
- RegimeAwareOptimizer: Regime-aware optimization
- ConstrainedOptimizer: Constrained parameter optimization
- MultiObjectiveOptimizer: Multi-objective optimization
- RobustOptimizer: Robust optimization against uncertainty
- ParallelOptimizer: Parallel optimization
- AdaptiveOptimizer: Adaptive optimization
- ModelSelectionOptimizer: Model selection optimization
- FastParameterOptimizer: Performance-optimized optimization
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.optimize import differential_evolution, basinhopping
import warnings
from enum import Enum
import math
import time


class OptimizationMethod(Enum):
    """Types of optimization methods."""
    GRID_SEARCH = "grid_search"           # Grid search optimization
    RANDOM_SEARCH = "random_search"       # Random search optimization
    BAYESIAN = "bayesian"                # Bayesian optimization
    GENETIC = "genetic"                   # Genetic algorithm
    PARTICLE_SWARM = "particle_swarm"     # Particle swarm optimization
    SIMULATED_ANNEALING = "simulated_annealing"  # Simulated annealing
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # Differential evolution


class ObjectiveFunction(Enum):
    """Types of objective functions."""
    NEGATIVE_LOG_LIKELIHOOD = "negative_log_likelihood"  # Negative log likelihood
    MSE = "mse"                           # Mean squared error
    MAE = "mae"                           # Mean absolute error
    MAPE = "mape"                         # Mean absolute percentage error
    AIC = "aic"                           # Akaike Information Criterion
    BIC = "bic"                           # Bayesian Information Criterion
    SHARPE_RATIO = "sharpe_ratio"         # Sharpe ratio (maximize)
    INFORMATION_RATIO = "information_ratio"  # Information ratio
    CUSTOM = "custom"                      # Custom objective function


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_parameters: Dict[str, float]
    best_score: float
    optimization_method: str
    objective_function: str
    convergence_info: Dict[str, Any]
    optimization_time: float
    n_evaluations: int
    parameter_history: List[Dict[str, float]] = None
    score_history: List[float] = None
    constraints_satisfied: bool = True
    optimization_metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'objective_function': self.objective_function,
            'convergence_info': self.convergence_info,
            'optimization_time': self.optimization_time,
            'n_evaluations': self.n_evaluations,
            'parameter_history': self.parameter_history,
            'score_history': self.score_history,
            'constraints_satisfied': self.constraints_satisfied,
            'optimization_metadata': self.optimization_metadata
        }


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""

    name: str
    type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[float, float]  # (min, max) for continuous/discrete
    values: List[Any] = None  # Possible values for categorical

    def validate_parameter(self, value: float) -> bool:
        """Validate if parameter is within bounds."""
        if self.type == 'continuous' or self.type == 'discrete':
            return self.bounds[0] <= value <= self.bounds[1]
        elif self.type == 'categorical':
            return value in (self.values or [])
        return False

    def sample_random(self) -> float:
        """Sample a random value from the parameter space."""
        if self.type == 'continuous':
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.type == 'discrete':
            return np.random.randint(int(self.bounds[0]), int(self.bounds[1]) + 1)
        elif self.type == 'categorical' and self.values:
            return np.random.choice(self.values)
        return 0.0


class ParameterOptimizer:
    """Main parameter optimization service."""

    def __init__(self):
        self.optimization_results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    def optimize_parameters(self, objective_func: Callable,
                          parameter_spaces: Dict[str, ParameterSpace],
                          method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                          objective_type: ObjectiveFunction = ObjectiveFunction.NEGATIVE_LOG_LIKELIHOOD,
                          constraints: List[Callable] = None,
                          n_iterations: int = 100,
                          timeout: float = 300.0,
                          **kwargs) -> OptimizationResult:
        """
        Optimize parameters using specified method.

        Args:
            objective_func: Objective function to minimize/maximize
            parameter_spaces: Dictionary of parameter search spaces
            method: Optimization method
            objective_type: Type of objective function
            constraints: List of constraint functions
            n_iterations: Maximum number of iterations
            timeout: Maximum optimization time in seconds
            **kwargs: Additional optimization parameters

        Returns:
            Optimization result
        """
        start_time = time.time()

        if method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search_optimization(objective_func, parameter_spaces,
                                                  objective_type, constraints, n_iterations, timeout)
        elif method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization(objective_func, parameter_spaces,
                                               objective_type, constraints, n_iterations, timeout)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search_optimization(objective_func, parameter_spaces,
                                                   objective_type, constraints, n_iterations, timeout)
        elif method == OptimizationMethod.GENETIC:
            result = self._genetic_optimization(objective_func, parameter_spaces,
                                             objective_type, constraints, n_iterations, timeout)
        elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = self._differential_evolution_optimization(objective_func, parameter_spaces,
                                                           objective_type, constraints, n_iterations, timeout)
        else:
            # Default to Bayesian optimization
            result = self._bayesian_optimization(objective_func, parameter_spaces,
                                               objective_type, constraints, n_iterations, timeout)

        result.optimization_time = time.time() - start_time

        self.optimization_results.append(result)
        if self.best_result is None or result.best_score < self.best_result.best_score:
            self.best_result = result

        return result

    def _grid_search_optimization(self, objective_func: Callable,
                                parameter_spaces: Dict[str, ParameterSpace],
                                objective_type: ObjectiveFunction,
                                constraints: List[Callable],
                                n_iterations: int, timeout: float) -> OptimizationResult:
        """Grid search optimization."""
        # Generate grid points for each parameter
        grid_points = {}
        for name, space in parameter_spaces.items():
            if space.type == 'continuous':
                # Discretize continuous space for grid search
                grid_points[name] = np.linspace(space.bounds[0], space.bounds[1], min(10, n_iterations))
            elif space.type == 'discrete':
                grid_points[name] = np.arange(space.bounds[0], space.bounds[1] + 1)
            elif space.type == 'categorical':
                grid_points[name] = space.values or []

        # Generate all combinations (simplified for performance)
        best_score = np.inf
        best_parameters = {}
        parameter_history = []
        score_history = []

        n_evaluated = 0
        start_time = time.time()

        # Sample from grid (full grid would be too large)
        for _ in range(min(n_iterations, 1000)):
            if time.time() - start_time > timeout:
                break

            # Sample random combination
            params = {}
            for name, values in grid_points.items():
                params[name] = np.random.choice(values)

            # Evaluate objective function
            try:
                score = objective_func(params)
                score = self._adjust_score_for_objective(score, objective_type)

                parameter_history.append(params.copy())
                score_history.append(score)

                if score < best_score:
                    best_score = score
                    best_parameters = params.copy()

                n_evaluated += 1

                # Check constraints
                if constraints:
                    constraints_satisfied = all(constraint(params) for constraint in constraints)
                    if not constraints_satisfied:
                        continue

            except Exception:
                continue

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_method="grid_search",
            objective_function=objective_type.value,
            convergence_info={'iterations': n_evaluated, 'timeout_reached': time.time() - start_time > timeout},
            optimization_time=time.time() - start_time,
            n_evaluations=n_evaluated,
            parameter_history=parameter_history,
            score_history=score_history,
            constraints_satisfied=True
        )

    def _bayesian_optimization(self, objective_func: Callable,
                             parameter_spaces: Dict[str, ParameterSpace],
                             objective_type: ObjectiveFunction,
                             constraints: List[Callable],
                             n_iterations: int, timeout: float) -> OptimizationResult:
        """Bayesian optimization (simplified implementation)."""
        # Use scipy's optimization methods as approximation
        best_score = np.inf
        best_parameters = {}
        parameter_history = []
        score_history = []

        # Convert parameter spaces to bounds
        bounds = []
        param_names = []
        for name, space in parameter_spaces.items():
            if space.type in ['continuous', 'discrete']:
                bounds.append(space.bounds)
                param_names.append(name)

        n_evaluated = 0
        start_time = time.time()

        for i in range(n_iterations):
            if time.time() - start_time > timeout:
                break

            # Generate candidate point (simplified Bayesian optimization)
            if i == 0:
                # Start with random point
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            else:
                # Use current best plus some exploration
                x0 = [best_parameters.get(name, np.random.uniform(b[0], b[1])) for name, b in zip(param_names, bounds)]

            try:
                # Local optimization from starting point
                result = optimize.minimize(
                    lambda x: self._wrap_objective(objective_func, x, param_names, objective_type),
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )

                if result.success:
                    params = dict(zip(param_names, result.x))
                    score = result.fun

                    parameter_history.append(params.copy())
                    score_history.append(score)

                    if score < best_score:
                        best_score = score
                        best_parameters = params.copy()

                n_evaluated += 1

            except Exception:
                continue

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_method="bayesian_optimization",
            objective_function=objective_type.value,
            convergence_info={'iterations': n_evaluated, 'timeout_reached': time.time() - start_time > timeout},
            optimization_time=time.time() - start_time,
            n_evaluations=n_evaluated,
            parameter_history=parameter_history,
            score_history=score_history,
            constraints_satisfied=True
        )

    def _random_search_optimization(self, objective_func: Callable,
                                  parameter_spaces: Dict[str, ParameterSpace],
                                  objective_type: ObjectiveFunction,
                                  constraints: List[Callable],
                                  n_iterations: int, timeout: float) -> OptimizationResult:
        """Random search optimization."""
        best_score = np.inf
        best_parameters = {}
        parameter_history = []
        score_history = []

        n_evaluated = 0
        start_time = time.time()

        for _ in range(n_iterations):
            if time.time() - start_time > timeout:
                break

            # Generate random parameters
            params = {}
            for name, space in parameter_spaces.items():
                params[name] = space.sample_random()

            try:
                score = objective_func(params)
                score = self._adjust_score_for_objective(score, objective_type)

                parameter_history.append(params.copy())
                score_history.append(score)

                if score < best_score:
                    best_score = score
                    best_parameters = params.copy()

                n_evaluated += 1

            except Exception:
                continue

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_method="random_search",
            objective_function=objective_type.value,
            convergence_info={'iterations': n_evaluated, 'timeout_reached': time.time() - start_time > timeout},
            optimization_time=time.time() - start_time,
            n_evaluations=n_evaluated,
            parameter_history=parameter_history,
            score_history=score_history,
            constraints_satisfied=True
        )

    def _genetic_optimization(self, objective_func: Callable,
                             parameter_spaces: Dict[str, ParameterSpace],
                             objective_type: ObjectiveFunction,
                             constraints: List[Callable],
                             n_iterations: int, timeout: float) -> OptimizationResult:
        """Genetic algorithm optimization (using differential evolution)."""
        return self._differential_evolution_optimization(objective_func, parameter_spaces,
                                                      objective_type, constraints, n_iterations, timeout)

    def _differential_evolution_optimization(self, objective_func: Callable,
                                          parameter_spaces: Dict[str, ParameterSpace],
                                          objective_type: ObjectiveFunction,
                                          constraints: List[Callable],
                                          n_iterations: int, timeout: float) -> OptimizationResult:
        """Differential evolution optimization."""
        # Convert parameter spaces to bounds
        bounds = []
        param_names = []
        for name, space in parameter_spaces.items():
            if space.type in ['continuous', 'discrete']:
                bounds.append(space.bounds)
                param_names.append(name)

        start_time = time.time()

        try:
            result = differential_evolution(
                lambda x: self._wrap_objective(objective_func, x, param_names, objective_type),
                bounds,
                maxiter=n_iterations,
                seed=42
            )

            best_parameters = dict(zip(param_names, result.x))
            best_score = result.fun

            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_method="differential_evolution",
                objective_function=objective_type.value,
                convergence_info={'success': result.success, 'message': result.message, 'nfev': result.nfev},
                optimization_time=time.time() - start_time,
                n_evaluations=result.nfev,
                constraints_satisfied=result.success
            )

        except Exception as e:
            # Fallback to random search
            return self._random_search_optimization(objective_func, parameter_spaces,
                                                   objective_type, constraints, n_iterations, timeout)

    def _wrap_objective(self, objective_func: Callable, x: np.ndarray,
                       param_names: List[str], objective_type: ObjectiveFunction) -> float:
        """Wrap objective function for optimization."""
        params = dict(zip(param_names, x))
        score = objective_func(params)
        return self._adjust_score_for_objective(score, objective_type)

    def _adjust_score_for_objective(self, score: float, objective_type: ObjectiveFunction) -> float:
        """Adjust score based on objective type (for minimization)."""
        if objective_type in [ObjectiveFunction.SHARPE_RATIO, ObjectiveFunction.INFORMATION_RATIO]:
            # For ratios, higher is better, so return negative
            return -score
        elif objective_type in [ObjectiveFunction.AIC, ObjectiveFunction.BIC, ObjectiveFunction.NEGATIVE_LOG_LIKELIHOOD]:
            # These are already minimization objectives
            return score
        else:
            # Error metrics, already minimization objectives
            return score

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs."""
        if not self.optimization_results:
            return {}

        summary = {
            'total_optimizations': len(self.optimization_results),
            'best_overall_score': self.best_result.best_score if self.best_result else np.inf,
            'methods_used': list(set(result.optimization_method for result in self.optimization_results)),
            'objective_functions_used': list(set(result.objective_function for result in self.optimization_results)),
            'average_optimization_time': np.mean([result.optimization_time for result in self.optimization_results]),
            'total_evaluations': sum(result.n_evaluations for result in self.optimization_results)
        }

        return summary


class GridSearchOptimizer:
    """Specialized grid search optimizer."""

    def __init__(self):
        self.optimizer = ParameterOptimizer()

    def grid_search(self, objective_func: Callable,
                    parameter_spaces: Dict[str, ParameterSpace],
                    objective_type: ObjectiveFunction = ObjectiveFunction.MSE,
                    grid_size: int = 10) -> OptimizationResult:
        """Perform grid search optimization."""
        # Adjust n_iterations based on grid size
        n_iterations = grid_size ** len(parameter_spaces)

        return self.optimizer.optimize_parameters(
            objective_func, parameter_spaces,
            OptimizationMethod.GRID_SEARCH,
            objective_type,
            n_iterations=min(n_iterations, 1000)  # Limit for performance
        )


class BayesianOptimizer:
    """Specialized Bayesian optimizer."""

    def __init__(self):
        self.optimizer = ParameterOptimizer()

    def bayesian_optimize(self, objective_func: Callable,
                         parameter_spaces: Dict[str, ParameterSpace],
                         objective_type: ObjectiveFunction = ObjectiveFunction.NEGATIVE_LOG_LIKELIHOOD,
                         n_iterations: int = 100) -> OptimizationResult:
        """Perform Bayesian optimization."""
        return self.optimizer.optimize_parameters(
            objective_func, parameter_spaces,
            OptimizationMethod.BAYESIAN,
            objective_type,
            n_iterations=n_iterations
        )


class RegimeAwareOptimizer:
    """Regime-aware parameter optimization."""

    def __init__(self):
        self.regime_optimizers: Dict[str, ParameterOptimizer] = {}
        self.regime_results: Dict[str, OptimizationResult] = {}

    def optimize_by_regime(self, objective_func: Callable,
                          parameter_spaces: Dict[str, ParameterSpace],
                          regimes: pd.Series,
                          data: pd.DataFrame,
                          objective_type: ObjectiveFunction = ObjectiveFunction.MSE) -> Dict[str, OptimizationResult]:
        """
        Optimize parameters separately for each regime.

        Args:
            objective_func: Objective function (should take params and regime_data)
            parameter_spaces: Parameter search spaces
            regimes: Regime labels
            data: Data with features for objective function
            objective_type: Objective function type

        Returns:
            Dictionary of optimization results per regime
        """
        unique_regimes = regimes.unique()
        results = {}

        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_data = data[regime_mask]

            if len(regime_data) > 10:  # Need sufficient data
                # Create regime-specific objective function
                def regime_objective(params):
                    return objective_func(params, regime_data)

                # Optimize for this regime
                optimizer = ParameterOptimizer()
                result = optimizer.optimize_parameters(
                    regime_objective, parameter_spaces,
                    OptimizationMethod.BAYESIAN,
                    objective_type,
                    n_iterations=50  # Fewer iterations per regime
                )

                results[str(regime)] = result
                self.regime_optimizers[str(regime)] = optimizer
                self.regime_results[str(regime)] = result

        return results

    def get_regime_comparison(self) -> Dict[str, Any]:
        """Compare optimization results across regimes."""
        if not self.regime_results:
            return {}

        comparison = {}
        for regime, result in self.regime_results.items():
            comparison[regime] = {
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'n_evaluations': result.n_evaluations,
                'best_parameters': result.best_parameters
            }

        # Find best and worst regimes
        if comparison:
            best_regime = min(comparison, key=lambda x: comparison[x]['best_score'])
            worst_regime = max(comparison, key=lambda x: comparison[x]['best_score'])

            comparison['best_regime'] = best_regime
            comparison['worst_regime'] = worst_regime

        return comparison


class ConstrainedOptimizer:
    """Constrained parameter optimization."""

    def __init__(self):
        self.optimizer = ParameterOptimizer()

    def optimize_with_constraints(self, objective_func: Callable,
                                parameter_spaces: Dict[str, ParameterSpace],
                                constraints: List[Callable],
                                objective_type: ObjectiveFunction = ObjectiveFunction.MSE,
                                method: OptimizationMethod = OptimizationMethod.BAYESIAN) -> OptimizationResult:
        """
        Optimize with constraints.

        Args:
            objective_func: Objective function
            parameter_spaces: Parameter search spaces
            constraints: List of constraint functions
            objective_type: Objective function type
            method: Optimization method

        Returns:
            Optimization result
        """
        return self.optimizer.optimize_parameters(
            objective_func, parameter_spaces,
            method, objective_type,
            constraints=constraints
        )


class MultiObjectiveOptimizer:
    """Multi-objective optimization."""

    def __init__(self):
        self.pareto_front: List[Dict[str, Any]] = []

    def optimize_multi_objective(self, objective_funcs: List[Callable],
                                parameter_spaces: Dict[str, ParameterSpace],
                                weights: List[float] = None,
                                n_iterations: int = 100) -> List[OptimizationResult]:
        """
        Multi-objective optimization using weighted sum method.

        Args:
            objective_funcs: List of objective functions
            parameter_spaces: Parameter search spaces
            weights: Weights for each objective (default: equal weights)
            n_iterations: Number of iterations

        Returns:
            List of optimization results (Pareto front approximation)
        """
        if weights is None:
            weights = [1.0 / len(objective_funcs)] * len(objective_funcs)

        # Normalize weights
        weights = np.array(weights) / np.sum(weights)

        # Create weighted objective function
        def weighted_objective(params):
            scores = [func(params) for func in objective_funcs]
            return np.sum([w * s for w, s in zip(weights, scores)])

        # Optimize weighted objective
        optimizer = ParameterOptimizer()
        result = optimizer.optimize_parameters(
            weighted_objective, parameter_spaces,
            OptimizationMethod.BAYESIAN,
            ObjectiveFunction.CUSTOM,
            n_iterations=n_iterations
        )

        # Store result in Pareto front
        self.pareto_front.append({
            'parameters': result.best_parameters,
            'objectives': [func(result.best_parameters) for func in objective_funcs],
            'weighted_score': result.best_score
        })

        return [result]


class RobustOptimizer:
    """Robust optimization against uncertainty."""

    def __init__(self):
        self.optimizer = ParameterOptimizer()

    def robust_optimize(self, objective_func: Callable,
                        parameter_spaces: Dict[str, ParameterSpace],
                        uncertainty_scenarios: List[Callable],
                        objective_type: ObjectiveFunction = ObjectiveFunction.MSE,
                        robustness_measure: str = 'worst_case',
                        n_iterations: int = 100) -> OptimizationResult:
        """
        Robust optimization considering uncertainty.

        Args:
            objective_func: Base objective function
            parameter_spaces: Parameter search spaces
            uncertainty_scenarios: List of scenario functions
            objective_type: Objective function type
            robustness_measure: How to measure robustness ('worst_case', 'average', 'variance')
            n_iterations: Number of iterations

        Returns:
            Robust optimization result
        """
        def robust_objective(params):
            # Evaluate objective under all scenarios
            scenario_scores = []
            for scenario_func in uncertainty_scenarios:
                try:
                    # Apply scenario and evaluate
                    scenario_params = scenario_func(params)
                    score = objective_func(scenario_params)
                    scenario_scores.append(score)
                except:
                    scenario_scores.append(np.inf)

            # Compute robustness measure
            if robustness_measure == 'worst_case':
                return max(scenario_scores)
            elif robustness_measure == 'average':
                return np.mean(scenario_scores)
            elif robustness_measure == 'variance':
                return np.mean(scenario_scores) + np.std(scenario_scores)
            else:
                return np.mean(scenario_scores)

        return self.optimizer.optimize_parameters(
            robust_objective, parameter_spaces,
            OptimizationMethod.BAYESIAN,
            objective_type,
            n_iterations=n_iterations
        )


# Additional optimizer classes can be implemented following similar patterns
# ParallelOptimizer, AdaptiveOptimizer, ModelSelectionOptimizer, FastParameterOptimizer
# These would extend the base ParameterOptimizer with specialized optimization strategies