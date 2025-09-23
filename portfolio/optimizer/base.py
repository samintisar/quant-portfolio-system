"""
Base optimizer interface for portfolio optimization system.

Defines the abstract interface for all optimization methods.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from portfolio.logging_config import get_logger, OptimizationError
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.models.result import OptimizationResult
from portfolio.models.views import MarketViewCollection

logger = get_logger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for portfolio optimizers.

    Defines the interface that all optimization methods must implement.
    """

    def __init__(self, name: str):
        """Initialize the optimizer with a name."""
        self.name = name
        self.supported_objectives = ['sharpe', 'min_risk', 'max_return']
        self.supported_constraints = ['position_size', 'sector_concentration', 'volatility']

        logger.info(f"Initialized {name} optimizer")

    @abstractmethod
    def optimize(self,
                 assets: List[Asset],
                 constraints: PortfolioConstraints,
                 objective: str = 'sharpe',
                 market_views: Optional[MarketViewCollection] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            assets: List of assets in the portfolio
            constraints: Portfolio constraints
            objective: Optimization objective ('sharpe', 'min_risk', 'max_return')
            market_views: Market views (for methods that support them)
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        pass

    def validate_inputs(self,
                       assets: List[Asset],
                       constraints: PortfolioConstraints,
                       objective: str) -> bool:
        """
        Validate optimization inputs.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            objective: Optimization objective

        Returns:
            True if inputs are valid
        """
        # Check assets
        if not assets:
            raise OptimizationError("No assets provided for optimization")

        if len(assets) < 2:
            raise OptimizationError("At least 2 assets required for optimization")

        # Check data sufficiency
        for asset in assets:
            if not asset.has_sufficient_data():
                raise OptimizationError(f"Insufficient data for asset {asset.symbol}")

        # Check constraints
        if not constraints.is_feasible():
            raise OptimizationError("Constraints are not feasible")

        # Check objective
        if objective not in self.supported_objectives:
            raise OptimizationError(f"Unsupported objective: {objective}. "
                                  f"Supported: {self.supported_objectives}")

        logger.debug(f"Input validation passed for {len(assets)} assets")
        return True

    def prepare_returns_data(self, assets: List[Asset]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare returns data for optimization.

        Args:
            assets: List of assets

        Returns:
            Tuple of (returns DataFrame, metadata dictionary)
        """
        try:
            # Collect returns data
            returns_data = {}
            metadata = {}

            for asset in assets:
                if not asset.returns.empty:
                    returns_data[asset.symbol] = asset.returns
                    metadata[asset.symbol] = {
                        'name': asset.name,
                        'sector': asset.sector,
                        'volatility': asset.volatility,
                        'data_points': len(asset.returns)
                    }

            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)

            # Align dates (remove assets with missing data on key dates)
            returns_df = returns_df.dropna()

            if returns_df.empty:
                raise OptimizationError("No overlapping data found for assets")

            # Calculate mean returns and covariance matrix
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()

            metadata['num_assets'] = len(assets)
            metadata['num_data_points'] = len(returns_df)
            metadata['date_range'] = {
                'start': returns_df.index[0].strftime('%Y-%m-%d'),
                'end': returns_df.index[-1].strftime('%Y-%m-%d')
            }

            logger.info(f"Prepared returns data: {len(assets)} assets, {len(returns_df)} data points")

            return returns_df, {
                'mean_returns': mean_returns,
                'cov_matrix': cov_matrix,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error preparing returns data: {e}")
            raise OptimizationError(f"Failed to prepare returns data: {e}")

    def build_constraints_matrix(self,
                               assets: List[Asset],
                               constraints: PortfolioConstraints,
                               num_assets: int) -> Dict[str, np.ndarray]:
        """
        Build constraint matrices for optimization.

        Args:
            assets: List of assets
            constraints: Portfolio constraints
            num_assets: Number of assets

        Returns:
            Dictionary with constraint matrices
        """
        try:
            constraint_matrices = {}

            # Weight bounds (position size constraints)
            min_weights = np.zeros(num_assets)
            max_weights = np.full(num_assets, constraints.max_position_size)

            constraint_matrices['weight_bounds'] = (min_weights, max_weights)

            # Sector constraints
            sector_constraints = self._build_sector_constraints(assets, constraints)
            constraint_matrices['sector_constraints'] = sector_constraints

            # Sum of weights = 1
            constraint_matrices['sum_weights'] = np.ones(num_assets)

            # Minimum return constraint
            if constraints.min_return > 0:
                constraint_matrices['min_return'] = constraints.min_return

            # Maximum volatility constraint
            if constraints.max_volatility < 1.0:
                constraint_matrices['max_volatility'] = constraints.max_volatility

            logger.debug(f"Built constraint matrices for {num_assets} assets")
            return constraint_matrices

        except Exception as e:
            logger.error(f"Error building constraint matrices: {e}")
            raise OptimizationError(f"Failed to build constraint matrices: {e}")

    def _build_sector_constraints(self,
                                 assets: List[Asset],
                                 constraints: PortfolioConstraints) -> Dict[str, Any]:
        """
        Build sector concentration constraints.

        Args:
            assets: List of assets
            constraints: Portfolio constraints

        Returns:
            Dictionary with sector constraint matrices
        """
        # Group assets by sector
        sector_groups = {}
        for i, asset in enumerate(assets):
            sector = asset.sector
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(i)

        # Build sector constraint matrices
        sector_matrices = {}
        for sector, asset_indices in sector_groups.items():
            if len(asset_indices) > 1:
                # Create matrix that sums weights in this sector
                sector_matrix = np.zeros(len(assets))
                sector_matrix[asset_indices] = 1.0
                sector_matrices[sector] = {
                    'matrix': sector_matrix,
                    'max_weight': constraints.max_sector_concentration
                }

        return sector_matrices

    def validate_optimization_result(self,
                                  weights: Dict[str, float],
                                  constraints: PortfolioConstraints) -> bool:
        """
        Validate optimization result against constraints.

        Args:
            weights: Optimal weights
            constraints: Portfolio constraints

        Returns:
            True if result is valid
        """
        try:
            # Check weights sum to 1
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(f"Weights sum to {weight_sum:.6f}, expected 1.0")
                return False

            # Check no negative weights
            for symbol, weight in weights.items():
                if weight < -0.001:  # Allow small negative due to numerical precision
                    logger.warning(f"Negative weight for {symbol}: {weight}")
                    return False

            # Check maximum position size
            for symbol, weight in weights.items():
                if weight > constraints.max_position_size + 0.001:
                    logger.warning(f"Weight {weight:.4f} for {symbol} exceeds max_position_size {constraints.max_position_size}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating optimization result: {e}")
            return False

    def calculate_portfolio_metrics(self,
                                   weights: Dict[str, float],
                                   mean_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   risk_free_rate: float) -> Dict[str, float]:
        """
        Calculate portfolio metrics for the optimal weights.

        Args:
            weights: Optimal weights
            mean_returns: Mean returns of assets
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            # Convert weights to array
            symbols = list(weights.keys())
            weight_array = np.array([weights[symbol] for symbol in symbols])

            # Calculate portfolio return
            portfolio_return = np.dot(weight_array, mean_returns[symbols])

            # Calculate portfolio variance
            relevant_cov = cov_matrix.loc[symbols, symbols]
            portfolio_variance = np.dot(weight_array.T, np.dot(relevant_cov, weight_array))
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate Sharpe ratio
            if portfolio_volatility > 0:
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            else:
                sharpe_ratio = 0.0

            metrics = {
                'annual_return': portfolio_return * 252,  # Annualize
                'annual_volatility': portfolio_volatility * np.sqrt(252),  # Annualize
                'sharpe_ratio': sharpe_ratio,
                'portfolio_variance': portfolio_variance
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def get_info(self) -> Dict[str, Any]:
        """
        Get optimizer information.

        Returns:
            Dictionary with optimizer information
        """
        return {
            'name': self.name,
            'supported_objectives': self.supported_objectives,
            'supported_constraints': self.supported_constraints,
            'requires_market_views': self.requires_market_views()
        }

    def requires_market_views(self) -> bool:
        """
        Check if optimizer requires market views.

        Returns:
            True if optimizer requires market views
        """
        return False

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}Optimizer"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class OptimizerFactory:
    """
    Factory class for creating optimizers.
    """

    @staticmethod
    def create_optimizer(method: str) -> BaseOptimizer:
        """
        Create optimizer instance based on method.

        Args:
            method: Optimization method ('mean_variance', 'black_litterman', 'cvar')

        Returns:
            Optimizer instance
        """
        try:
            if method == 'mean_variance':
                from portfolio.optimizer.mean_variance import MeanVarianceOptimizer
                return MeanVarianceOptimizer()
            elif method == 'black_litterman':
                from portfolio.optimizer.black_litterman import BlackLittermanOptimizer
                return BlackLittermanOptimizer()
            elif method == 'cvar':
                from portfolio.optimizer.cvar import CVaROptimizer
                return CVaROptimizer()
            else:
                raise OptimizationError(f"Unknown optimization method: {method}")

        except ImportError as e:
            logger.error(f"Failed to import optimizer for method {method}: {e}")
            raise OptimizationError(f"Optimizer not available for method: {method}")

    @staticmethod
    def get_available_methods() -> List[str]:
        """
        Get list of available optimization methods.

        Returns:
            List of available methods
        """
        return ['mean_variance', 'black_litterman', 'cvar']

    @staticmethod
    def validate_method(method: str) -> bool:
        """
        Validate if optimization method is available.

        Args:
            method: Optimization method

        Returns:
            True if method is available
        """
        return method in OptimizerFactory.get_available_methods()