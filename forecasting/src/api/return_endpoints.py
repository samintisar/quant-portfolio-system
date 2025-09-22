"""
Enhanced return forecasting API endpoints with regime-aware predictions.

Implements REST endpoints for return forecasting using advanced statistical models
including ARIMA with heavy-tail distributions, regime-switching considerations,
and confidence interval estimation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.models.arima_model import EnhancedARIMAModel
from forecasting.services.forecast_service import ForecastOrchestrationService
from forecasting.services.data_service import DataPreprocessingService


router = APIRouter(prefix="/api/v1/forecasts", tags=["returns"])


class ForecastRequest(BaseModel):
    """Request model for return forecasting."""
    symbol: str = Field(..., description="Stock symbol to forecast")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    horizon: int = Field(30, description="Forecast horizon in days")
    model_type: str = Field("auto", description="Model type: auto, arima, sarima, etc.")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    include_regime: bool = Field(True, description="Include regime-aware forecasting")


class ForecastResponse(BaseModel):
    """Response model for return forecasting."""
    symbol: str
    forecast_date: str
    horizon: int
    predictions: List[float]
    confidence_intervals: List[Dict[str, float]]
    model_info: Dict[str, Any]
    regime_info: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float]


@router.post("/returns", response_model=ForecastResponse)
async def forecast_returns(request: ForecastRequest):
    """
    Generate return forecasts using enhanced statistical models.

    Features:
    - Heavy-tail distribution handling
    - Regime-aware predictions
    - Confidence interval estimation
    - Multiple model variants (ARIMA, SARIMA, auto-ARIMA)
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

        # Generate forecast
        forecast_result = await forecast_service.generate_forecast(
            data=data,
            horizon=request.horizon,
            model_type=request.model_type,
            confidence_level=request.confidence_level,
            include_regime=request.include_regime
        )

        return ForecastResponse(
            symbol=request.symbol,
            forecast_date=datetime.now().isoformat(),
            horizon=request.horizon,
            predictions=forecast_result['predictions'],
            confidence_intervals=forecast_result['confidence_intervals'],
            model_info=forecast_result['model_info'],
            regime_info=forecast_result.get('regime_info'),
            performance_metrics=forecast_result['performance_metrics']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/returns/{symbol}/latest")
async def get_latest_forecast(
    symbol: str,
    horizon: int = Query(30, description="Forecast horizon in days"),
    model_type: str = Query("auto", description="Model type")
):
    """Get the latest available forecast for a symbol."""
    try:
        # Get latest data date for the symbol
        # In a real implementation, this would query a database
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        request = ForecastRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            horizon=horizon,
            model_type=model_type
        )

        return await forecast_returns(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/returns/{symbol}/history")
async def get_forecast_history(
    symbol: str,
    limit: int = Query(10, description="Number of historical forecasts to return")
):
    """Get historical forecast accuracy for a symbol."""
    try:
        # This would query a database for historical forecasts
        # For now, return mock data structure
        return {
            "symbol": symbol,
            "historical_forecasts": [],
            "accuracy_metrics": {
                "mape": 0.0,
                "rmse": 0.0,
                "directional_accuracy": 0.0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/returns/batch")
async def batch_forecast_returns(
    requests: List[ForecastRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent forecasts")
):
    """Generate return forecasts for multiple symbols."""
    try:
        results = []

        for request in requests:
            try:
                result = await forecast_returns(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "symbol": request.symbol,
                    "error": str(e)
                })

        return {"forecasts": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/returns/models")
async def get_available_models():
    """Get list of available forecasting models."""
    return {
        "models": [
            {"name": "auto", "description": "Automatic model selection"},
            {"name": "arima", "description": "ARIMA with heavy-tail distributions"},
            {"name": "sarima", "description": "Seasonal ARIMA"},
            {"name": "auto_arima", "description": "Automatic ARIMA parameter selection"},
            {"name": "prophet", "description": "Facebook Prophet"}
        ],
        "default_model": "auto"
    }


@router.get("/returns/health")
async def health_check():
    """Health check for return forecasting service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }