"""
Regime detection API endpoints with multiple emission models.

Implements REST endpoints for market regime detection using advanced Hidden Markov Models
with Student-t and mixture-of-Gaussian emissions for capturing financial market characteristics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.models.hmm_model import AdvancedHMMModel
from forecasting.services.regime_service import RegimeDetectionService
from forecasting.services.data_service import DataPreprocessingService


router = APIRouter(prefix="/api/v1/regimes", tags=["regimes"])


class RegimeDetectionRequest(BaseModel):
    """Request model for regime detection."""
    symbol: str = Field(..., description="Stock symbol or market index")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    emission_model: str = Field("student_t", description="Emission model: gaussian, student_t, mixture")
    n_regimes: int = Field(2, description="Number of regimes to detect")
    features: List[str] = Field(["returns", "volatility"], description="Features to use")
    include_transitions: bool = Field(True, description="Include transition probabilities")


class RegimeResponse(BaseModel):
    """Response model for regime detection."""
    symbol: str
    detection_date: str
    regimes: List[Dict[str, Any]]
    current_regime: int
    transition_matrix: Optional[Dict[str, List[float]]] = None
    model_info: Dict[str, Any]
    performance_metrics: Dict[str, float]


class RegimeForecastRequest(BaseModel):
    """Request model for regime forecasting."""
    symbol: str = Field(..., description="Stock symbol or market index")
    horizon: int = Field(30, description="Forecast horizon in days")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")


@router.post("/detect", response_model=RegimeResponse)
async def detect_regimes(request: RegimeDetectionRequest):
    """
    Detect market regimes using advanced Hidden Markov Models.

    Features:
    - Student-t emission models for heavy-tail financial data
    - Mixture-of-Gaussian emissions for complex distributions
    - Multiple feature support (returns, volatility, volume, etc.)
    - Transition probability estimation
    """
    try:
        # Initialize services
        data_service = DataPreprocessingService()
        regime_service = RegimeDetectionService()

        # Fetch and preprocess data
        data = await data_service.fetch_historical_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Detect regimes
        regime_result = await regime_service.detect_regimes(
            data=data,
            emission_model=request.emission_model,
            n_regimes=request.n_regimes,
            features=request.features,
            include_transitions=request.include_transitions
        )

        return RegimeResponse(
            symbol=request.symbol,
            detection_date=datetime.now().isoformat(),
            regimes=regime_result['regimes'],
            current_regime=regime_result['current_regime'],
            transition_matrix=regime_result.get('transition_matrix'),
            model_info=regime_result['model_info'],
            performance_metrics=regime_result['performance_metrics']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast")
async def forecast_regimes(request: RegimeForecastRequest):
    """
    Forecast future regime probabilities and expected transitions.

    Uses detected regime model to forecast future regime states and
    calculate confidence intervals for regime transitions.
    """
    try:
        regime_service = RegimeDetectionService()

        forecast_result = await regime_service.forecast_regimes(
            symbol=request.symbol,
            horizon=request.horizon,
            confidence_level=request.confidence_level
        )

        return {
            "symbol": request.symbol,
            "forecast_date": datetime.now().isoformat(),
            "horizon": request.horizon,
            "regime_forecasts": forecast_result['regime_forecasts'],
            "transition_forecasts": forecast_result['transition_forecasts'],
            "confidence_intervals": forecast_result['confidence_intervals']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current/{symbol}")
async def get_current_regime(
    symbol: str,
    emission_model: str = Query("student_t", description="Emission model"),
    lookback_days: int = Query(365, description="Lookback period in days")
):
    """Get the current market regime for a symbol."""
    try:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        request = RegimeDetectionRequest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            emission_model=emission_model,
            n_regimes=2,
            features=["returns", "volatility"],
            include_transitions=True
        )

        return await detect_regimes(request)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_emission_models():
    """Get list of available emission models for regime detection."""
    return {
        "emission_models": [
            {"name": "gaussian", "description": "Standard Gaussian emissions"},
            {"name": "student_t", "description": "Student-t emissions for heavy tails"},
            {"name": "mixture", "description": "Mixture-of-Gaussian emissions"},
            {"name": "skew_t", "description": "Skewed Student-t emissions"}
        ],
        "default_model": "student_t"
    }


@router.post("/optimize")
async def optimize_regime_model(
    symbol: str = Body(..., description="Stock symbol"),
    max_regimes: int = Body(5, description="Maximum number of regimes to test"),
    model_selection: str = Body("bic", description="Model selection criterion")
):
    """
    Automatically select optimal number of regimes and model parameters.

    Uses information criteria (BIC, AIC) to select the best model
    from different configurations.
    """
    try:
        regime_service = RegimeDetectionService()

        optimization_result = await regime_service.optimize_regime_model(
            symbol=symbol,
            max_regimes=max_regimes,
            model_selection=model_selection
        )

        return {
            "symbol": symbol,
            "optimal_n_regimes": optimization_result['n_regimes'],
            "optimal_model": optimization_result['model_type'],
            "selection_scores": optimization_result['scores'],
            "model_comparison": optimization_result['comparison']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_regime_history(
    symbol: str,
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """Get historical regime changes and durations."""
    try:
        regime_service = RegimeDetectionService()

        history_result = await regime_service.get_regime_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "symbol": symbol,
            "regime_history": history_result['regime_history'],
            "regime_durations": history_result['durations'],
            "transition_events": history_result['transitions']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-detect")
async def batch_regime_detection(
    requests: List[RegimeDetectionRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent detections")
):
    """Detect regimes for multiple symbols."""
    try:
        results = []

        for request in requests:
            try:
                result = await detect_regimes(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "symbol": request.symbol,
                    "error": str(e)
                })

        return {"detections": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for regime detection service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }