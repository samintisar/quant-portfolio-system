"""
API data models and portfolio optimization system.

Provides Pydantic models for request/response validation and serialization.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class OptimizationMethod(str, Enum):
    """Supported optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    CVAR = "cvar"


class OptimizationObjective(str, Enum):
    """Supported optimization objectives."""
    SHARPE = "sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    UTILITY = "utility"
    CVAR = "cvar"


class AssetRequest(BaseModel):
    """Asset model for API requests."""
    symbol: str = Field(..., description="Asset symbol/ticker")
    name: Optional[str] = Field(None, description="Asset name")
    asset_type: Optional[str] = Field("stock", description="Asset type (stock, bond, etf, etc.)")
    sector: Optional[str] = Field(None, description="Asset sector/industry")


class PortfolioConstraintsRequest(BaseModel):
    """Portfolio constraints model for API requests."""
    max_position_size: Optional[float] = Field(0.2, ge=0, le=1, description="Maximum position size")
    min_position_size: Optional[float] = Field(0.0, ge=0, le=1, description="Minimum position size")
    max_sector_concentration: Optional[float] = Field(0.3, ge=0, le=1, description="Maximum sector concentration")
    max_volatility: Optional[float] = Field(0.25, ge=0, description="Maximum portfolio volatility")
    min_return: Optional[float] = Field(0.0, description="Minimum expected return")
    risk_free_rate: Optional[float] = Field(0.02, ge=0, description="Risk-free rate")


class MarketViewRequest(BaseModel):
    """Market view model for API requests."""
    symbol: str = Field(..., description="Asset symbol")
    view_type: str = Field(..., description="Type of view (absolute, relative)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    expected_return: Optional[float] = Field(None, description="Expected return")
    benchmark_symbol: Optional[str] = Field(None, description="Benchmark symbol for relative views")


class OptimizationRequest(BaseModel):
    """Portfolio optimization request model."""
    assets: List[AssetRequest] = Field(..., description="List of assets", min_items=2)
    constraints: Optional[PortfolioConstraintsRequest] = Field(None, description="Portfolio constraints")
    method: OptimizationMethod = Field(OptimizationMethod.MEAN_VARIANCE, description="Optimization method")
    objective: OptimizationObjective = Field(OptimizationObjective.SHARPE, description="Optimization objective")
    market_views: Optional[List[MarketViewRequest]] = Field(None, description="Market views for Black-Litterman")
    lookback_period: Optional[int] = Field(252, ge=30, description="Lookback period in days")


class PortfolioAnalysisRequest(BaseModel):
    """Portfolio analysis request model."""
    weights: Dict[str, float] = Field(..., description="Portfolio weights")
    assets: List[AssetRequest] = Field(..., description="List of assets", min_items=1)
    benchmark_symbol: Optional[str] = Field(None, description="Benchmark symbol")
    risk_free_rate: Optional[float] = Field(0.02, ge=0, description="Risk-free rate")
    lookback_period: Optional[int] = Field(252, ge=30, description="Lookback period in days")


class AssetResponse(BaseModel):
    """Asset model for API responses."""
    symbol: str
    name: Optional[str]
    asset_type: Optional[str]
    sector: Optional[str]
    current_price: Optional[float]
    historical_return: Optional[float]
    volatility: Optional[float]


class OptimizationResponse(BaseModel):
    """Portfolio optimization response model."""
    success: bool
    method: str
    objective: str
    optimal_weights: Dict[str, float]
    performance: Optional[Dict[str, Any]]
    execution_time: float
    timestamp: datetime
    error_messages: Optional[List[str]]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    services: Dict[str, str] = Field(default_factory=dict)


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response model."""
    annual_return: Optional[float]
    annual_volatility: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    beta: Optional[float]


class RiskAnalysisResponse(BaseModel):
    """Risk analysis response model."""
    var_95: Optional[float]
    var_99: Optional[float]
    cvar_95: Optional[float]
    cvar_99: Optional[float]
    max_drawdown: Optional[float]


class PortfolioAnalysisResponse(BaseModel):
    """Portfolio analysis response model."""
    success: bool
    weights: Dict[str, float]
    performance_metrics: Optional[PerformanceMetricsResponse]
    risk_analysis: Optional[RiskAnalysisResponse]
    execution_time: float
    timestamp: datetime
    error_messages: Optional[List[str]]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str