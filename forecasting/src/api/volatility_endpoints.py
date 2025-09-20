"""
Advanced volatility forecasting API endpoints with asymmetric model support.

Implements REST endpoints for volatility forecasting using GARCH family models
including EGARCH, GJR-GARCH, and regime-switching variants for capturing
asymmetric volatility effects.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.models.garch_model import GARCHModel
from forecasting.services.forecast_service import ForecastOrchestrationService
from forecasting.services.data_service import DataPreprocessingService


router = APIRouter(prefix="/api/v1/volatility", tags=["volatility"])


class VolatilityForecastRequest(BaseModel):
    """Request model for volatility forecasting."""
    symbol: str = Field(..., description="Stock symbol to forecast")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    horizon: int = Field(30, description="Forecast horizon in days")
    model_type: str = Field("egarch", description="Model type: garch, egarch, gjr, rs-garch")
    include_asymmetric: bool = Field(True, description="Include asymmetric effects")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    include_regime: bool = Field(True, description="Include regime-switching")


class VolatilityResponse(BaseModel):
    """Response model for volatility forecasting."""
    symbol: str
    forecast_date: str
    horizon: int
    volatility_forecast: List[float]
    confidence_intervals: List[Dict[str, float]]
    model_info: Dict[str, Any]
    regime_info: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float]
    asymmetric_metrics: Optional[Dict[str, float]] = None


@router.post("/forecast", response_model=VolatilityResponse)
async def forecast_volatility(request: VolatilityForecastRequest):
    """
    Generate volatility forecasts using advanced GARCH family models.

    Features:
    - EGARCH for asymmetric volatility effects
    - GJR-GARCH for leverage effects
    - Regime-switching GARCH variants
    - Heavy-tail distribution support
    """
    try:
        # Initialize services
        data_service = DataPreprocessingService()
        forecast_service = ForecastOrchestrationService()

        # Fetch and preprocess data
        data = await data_service.fetch_historical_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Generate volatility forecast
        forecast_result = await forecast_service.generate_volatility_forecast(
            data=data,
            horizon=request.horizon,
            model_type=request.model_type,
            include_asymmetric=request.include_asymmetric,
            confidence_level=request.confidence_level,
            include_regime=request.include_regime
        )

        return VolatilityResponse(
            symbol=request.symbol,
            forecast_date=datetime.now().isoformat(),
            horizon=request.horizon,
            volatility_forecast=forecast_result['volatility_forecast'],
            confidence_intervals=forecast_result['confidence_intervals'],
            model_info=forecast_result['model_info'],
            regime_info=forecast_result.get('regime_info'),
            performance_metrics=forecast_result['performance_metrics'],
            asymmetric_metrics=forecast_result.get('asymmetric_metrics')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{symbol}/latest")
async def get_latest_volatility_forecast(
    symbol: str,
    horizon: int = Query(30, description="Forecast horizon in days"),
    model_type: str = Query("egarch", description="Model type")
):
    """Get the latest available volatility forecast for a symbol."""
    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        request = VolatilityForecastRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            model_type=model_type
        )

        return await forecast_volatility(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_volatility_models():
    """Get list of available volatility forecasting models."""
    return {
        "models": [
            {"name": "garch", "description": "Standard GARCH model"},
            {"name": "egarch", "description": "EGARCH with asymmetric effects"},
            {"name": "gjr", "description": "GJR-GARCH with leverage effects"},
            {"name": "rs-garch", "description": "Regime-switching GARCH"},
            {"name": "figarch", "description": "FIGARCH for long memory"}
        ],
        "default_model": "egarch"
    }


@router.post("/implied-volatility")
async def calculate_implied_volatility(
    symbol: str = Body(..., description="Stock symbol"),
    option_data: List[Dict] = Body(..., description="Option chain data"),
    method: str = Body("black-scholes", description="Calculation method")
):
    """Calculate implied volatility from option prices."""
    try:
        # This would implement Black-Scholes or other models
        # For now, return a mock response
        return {
            "symbol": symbol,
            "implied_volatility": 0.25,
            "calculation_method": method,
            "options_used": len(option_data),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/realized/{symbol}")
async def get_realized_volatility(
    symbol: str,
    window: int = Query(30, description="Lookback window in days"),
    method: str = Query("standard", description="Calculation method")
):
    """Calculate realized volatility from historical data."""
    try:
        # This would calculate realized volatility from price data
        return {
            "symbol": symbol,
            "realized_volatility": 0.20,
            "window": window,
            "method": method,
            "calculation_date": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-forecast")
async def batch_volatility_forecast(
    requests: List[VolatilityForecastRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent forecasts")
):
    """Generate volatility forecasts for multiple symbols."""
    try:
        results = []

        for request in requests:
            try:
                result = await forecast_volatility(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "symbol": request.symbol,
                    "error": str(e)
                })

        return {"forecasts": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for volatility forecasting service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }