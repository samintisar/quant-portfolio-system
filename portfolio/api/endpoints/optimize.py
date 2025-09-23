"""
Portfolio optimization endpoints.

Implements REST endpoints for portfolio optimization functionality.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from portfolio.api.models import (
    OptimizationRequest, OptimizationResponse, ErrorResponse, HealthResponse,
    AssetRequest, PortfolioConstraintsRequest, OptimizationMethod, OptimizationObjective
)
from portfolio.optimizer.optimizer import PortfolioOptimizer
from portfolio.models.asset import Asset
from portfolio.models.constraints import PortfolioConstraints
from portfolio.models.views import MarketViewCollection
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.logging_config import get_logger

router = APIRouter(tags=["portfolio optimization"])
logger = get_logger(__name__)


# Global instances (in production, use dependency injection)
optimizer = PortfolioOptimizer()
data_service = YahooFinanceService()


@router.post("/optimize")
async def optimize_portfolio(request: dict = None):
    """
    Optimize portfolio using specified method and objective.
    """
    try:
        start_time = datetime.now()

        if request is None or not isinstance(request, dict):
            raise HTTPException(
                status_code=422,
                detail="Request must be a JSON object"
            )

        # Get request parameters
        assets_data = request.get("assets", [])
        method = request.get("method", "mean_variance")
        objective = request.get("objective", "sharpe")
        constraints_dict = request.get("constraints", {})
        lookback_period = request.get("lookback_period", 252)

        # Validate optimization method
        available_methods = ['mean_variance', 'black_litterman', 'cvar']
        if method not in available_methods:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid optimization method: {method}. Available methods: {available_methods}"
            )

        # Extract asset symbols
        asset_symbols = []
        if isinstance(assets_data, list):
            for asset in assets_data:
                if isinstance(asset, dict):
                    asset_symbols.append(asset.get("symbol"))
                else:
                    asset_symbols.append(asset)
        elif isinstance(assets_data, str):
            asset_symbols = [assets_data]

        # Validate assets
        if not asset_symbols or len(asset_symbols) < 2:
            raise HTTPException(
                status_code=422,
                detail="At least 2 assets are required for optimization"
            )

        # Create domain assets
        assets = []
        for symbol in asset_symbols:
            asset = Asset(
                symbol=symbol,
                name=symbol,
                sector="Technology"  # Default sector
            )
            assets.append(asset)

        # Create domain constraints
        constraints = PortfolioConstraints(
            max_position_size=constraints_dict.get("max_position_size", 0.2),
            max_sector_concentration=constraints_dict.get("max_sector_concentration", 0.3),
            max_volatility=constraints_dict.get("max_volatility", 0.25),
            min_return=constraints_dict.get("min_return", 0.0),
            risk_free_rate=constraints_dict.get("risk_free_rate", 0.02)
        )

        # Fetch historical data for assets
        logger.info(f"Fetching data for {len(assets)} assets")
        for asset in assets:
            try:
                data = data_service.fetch_historical_data(
                    asset.symbol,
                    period=f"{lookback_period}d"
                )
                if not data.empty:
                    asset.returns = data['returns']
                    logger.info(f"Successfully fetched data for {asset.symbol}")
                else:
                    logger.warning(f"No data found for {asset.symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {asset.symbol}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not fetch data for {asset.symbol}: {str(e)}"
                )

        # Perform optimization
        logger.info(f"Starting {method} optimization")
        result = optimizer.optimize(
            assets=assets,
            constraints=constraints,
            method=method,
            objective=objective,
            market_views=None
        )

        # Convert result to response format
        response = {
            "success": result.success,
            "method": result.optimization_method,
            "objective": objective,
            "optimal_weights": result.optimal_weights,
            "execution_time": result.execution_time,
            "timestamp": (result.timestamp or datetime.now()).isoformat(),
            "error_messages": result.error_messages if result.error_messages else None
        }

        logger.info(f"Optimization completed in {result.execution_time:.2f} seconds")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.get("/methods")
async def get_optimization_methods():
    """Get available optimization methods and their capabilities."""
    try:
        optimizer_info = optimizer.get_optimizer_info()

        methods_info = {}
        for method in optimizer_info.get('available_methods', []):
            methods_info[method] = {
                'description': get_method_description(method),
                'objectives': get_method_objectives(method),
                'requires_market_views': method_requires_market_views(method)
            }

        return {
            'available_methods': methods_info,
            'default_method': optimizer_info.get('default_method'),
            'supported_objectives': list(OptimizationObjective.__members__.keys())
        }

    except Exception as e:
        logger.error(f"Error getting optimization methods: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving optimization methods: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for optimization service."""
    try:
        # Check if optimizer is available
        optimizer_info = optimizer.get_optimizer_info()

        services = {
            'optimizer': 'healthy' if optimizer_info else 'unhealthy',
            'data_service': 'healthy' if data_service else 'unhealthy'
        }

        overall_status = 'healthy' if all(status == 'healthy' for status in services.values()) else 'degraded'

        return HealthResponse(
            status=overall_status,
            services=services
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status='unhealthy',
            services={'optimizer': 'error', 'data_service': 'error'}
        )


def get_method_description(method: str) -> str:
    """Get description for optimization method."""
    descriptions = {
        'mean_variance': 'Classic Markowitz mean-variance optimization',
        'black_litterman': 'Black-Litterman model combining market equilibrium with investor views',
        'cvar': 'Conditional Value at Risk optimization for tail risk management'
    }
    return descriptions.get(method, 'Unknown method')


def get_method_objectives(method: str) -> List[str]:
    """Get supported objectives for optimization method."""
    objectives = {
        'mean_variance': ['sharpe', 'min_variance', 'max_return', 'utility'],
        'black_litterman': ['sharpe', 'min_variance', 'max_return', 'utility'],
        'cvar': ['cvar', 'min_cvar', 'return_cvar']
    }
    return objectives.get(method, [])


def method_requires_market_views(method: str) -> bool:
    """Check if method requires market views."""
    return method == 'black_litterman'