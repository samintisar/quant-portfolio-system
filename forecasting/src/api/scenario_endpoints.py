"""
Scenario modeling API endpoints with economic data lag handling.

Implements REST endpoints for economic scenario modeling using Bayesian networks
with comprehensive handling of data lags, revision uncertainty, and scenario impact analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.models.scenario_model import BayesianScenarioModel
from forecasting.services.scenario_service import ScenarioModelingService
from forecasting.services.data_service import DataPreprocessingService


router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])


class ScenarioRequest(BaseModel):
    """Request model for scenario modeling."""
    scenario_name: str = Field(..., description="Name of the scenario")
    economic_indicators: Dict[str, float] = Field(..., description="Economic indicator values")
    lag_adjustments: Optional[Dict[str, int]] = Field(None, description="Data lag adjustments")
    confidence_level: float = Field(0.95, description="Confidence level for intervals")
    time_horizon: int = Field(12, description="Time horizon in months")


class ScenarioResponse(BaseModel):
    """Response model for scenario modeling."""
    scenario_name: str
    analysis_date: str
    impact_assessment: Dict[str, Any]
    probability_estimate: float
    confidence_intervals: Dict[str, List[float]]
    lag_adjustments: Dict[str, int]
    model_info: Dict[str, Any]


class EconomicIndicatorRequest(BaseModel):
    """Request model for economic indicator data."""
    indicators: List[str] = Field(..., description="List of economic indicators")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    include_revisions: bool = Field(True, description="Include data revisions")


@router.post("/analyze", response_model=ScenarioResponse)
async def analyze_scenario(request: ScenarioRequest):
    """
    Analyze economic scenarios using Bayesian networks with lag handling.

    Features:
    - Economic data lag and revision uncertainty handling
    - Bayesian network inference for scenario probabilities
    - Impact assessment across multiple asset classes
    - Confidence interval estimation
    """
    try:
        # Initialize services
        scenario_service = ScenarioModelingService()

        # Analyze scenario
        scenario_result = await scenario_service.analyze_scenario(
            scenario_name=request.scenario_name,
            economic_indicators=request.economic_indicators,
            lag_adjustments=request.lag_adjustments,
            confidence_level=request.confidence_level,
            time_horizon=request.time_horizon
        )

        return ScenarioResponse(
            scenario_name=request.scenario_name,
            analysis_date=datetime.now().isoformat(),
            impact_assessment=scenario_result['impact_assessment'],
            probability_estimate=scenario_result['probability_estimate'],
            confidence_intervals=scenario_result['confidence_intervals'],
            lag_adjustments=scenario_result['lag_adjustments'],
            model_info=scenario_result['model_info']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators")
async def get_available_indicators():
    """Get list of available economic indicators."""
    return {
        "indicators": [
            {"name": "GDP_GROWTH", "description": "GDP Growth Rate (%)", "lag_months": 2},
            {"name": "INFLATION_CPI", "description": "Consumer Price Inflation (%)", "lag_months": 1},
            {"name": "UNEMPLOYMENT_RATE", "description": "Unemployment Rate (%)", "lag_months": 1},
            {"name": "FED_FUNDS_RATE", "description": "Federal Funds Rate (%)", "lag_months": 0},
            {"name": "TREASURY_YIELD_10Y", "description": "10-Year Treasury Yield (%)", "lag_months": 0},
            {"name": "VIX_INDEX", "description": "VIX Volatility Index", "lag_months": 0},
            {"name": "OIL_PRICE", "description": "Oil Price (USD)", "lag_months": 0},
            {"name": "USD_INDEX", "description": "US Dollar Index", "lag_months": 0}
        ]
    }


@router.get("/templates")
async def get_scenario_templates():
    """Get pre-defined scenario templates."""
    return {
        "templates": [
            {
                "name": "Recession",
                "description": "Economic recession scenario",
                "indicators": {"GDP_GROWTH": -2.0, "UNEMPLOYMENT_RATE": 8.0, "INFLATION_CPI": 1.5}
            },
            {
                "name": "Inflation_Spike",
                "description": "High inflation scenario",
                "indicators": {"INFLATION_CPI": 6.0, "FED_FUNDS_RATE": 5.0, "GDP_GROWTH": 1.0}
            },
            {
                "name": "Soft_Landing",
                "description": "Soft economic landing scenario",
                "indicators": {"GDP_GROWTH": 2.0, "INFLATION_CPI": 2.5, "UNEMPLOYMENT_RATE": 4.0}
            },
            {
                "name": "Growth_Surge",
                "description": "Strong economic growth scenario",
                "indicators": {"GDP_GROWTH": 4.0, "INFLATION_CPI": 3.0, "UNEMPLOYMENT_RATE": 3.5}
            }
        ]
    }


@router.post("/batch-analyze")
async def batch_analyze_scenarios(
    requests: List[ScenarioRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent analyses")
):
    """Analyze multiple economic scenarios."""
    try:
        results = []

        for request in requests:
            try:
                result = await analyze_scenario(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "scenario_name": request.scenario_name,
                    "error": str(e)
                })

        return {"scenarios": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-lags")
async def get_data_lag_info():
    """Get current data lag information for economic indicators."""
    return {
        "indicator_lags": {
            "GDP_GROWTH": {"typical_lag": 2, "revision_history": "High", "current_lag": 1},
            "INFLATION_CPI": {"typical_lag": 1, "revision_history": "Medium", "current_lag": 0},
            "UNEMPLOYMENT_RATE": {"typical_lag": 1, "revision_history": "Low", "current_lag": 0},
            "FED_FUNDS_RATE": {"typical_lag": 0, "revision_history": "None", "current_lag": 0},
            "TREASURY_YIELD_10Y": {"typical_lag": 0, "revision_history": "None", "current_lag": 0}
        },
        "last_updated": datetime.now().isoformat()
    }


@router.post("/simulate")
async def simulate_scenario_impact(
    scenario_name: str = Body(..., description="Scenario name"),
    base_portfolio: Dict[str, float] = Body(..., description="Base portfolio weights"),
    shock_magnitudes: Dict[str, float] = Body(..., description="Shock magnitudes for indicators"),
    simulation_runs: int = Body(1000, description="Number of Monte Carlo simulations")
):
    """
    Simulate scenario impact on portfolio using Monte Carlo methods.

    Runs multiple simulations to estimate portfolio impact under different
    realizations of the economic scenario.
    """
    try:
        scenario_service = ScenarioModelingService()

        simulation_result = await scenario_service.simulate_scenario_impact(
            scenario_name=scenario_name,
            base_portfolio=base_portfolio,
            shock_magnitudes=shock_magnitudes,
            simulation_runs=simulation_runs
        )

        return {
            "scenario_name": scenario_name,
            "simulation_date": datetime.now().isoformat(),
            "portfolio_impact": simulation_result['portfolio_impact'],
            "risk_metrics": simulation_result['risk_metrics'],
            "confidence_intervals": simulation_result['confidence_intervals'],
            "simulation_parameters": {
                "runs": simulation_runs,
                "base_portfolio": base_portfolio,
                "shock_magnitudes": shock_magnitudes
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical/{scenario_name}")
async def get_historical_scenarios(
    scenario_name: str,
    lookback_years: int = Query(10, description="Lookback period in years")
):
    """Get historical occurrences of similar scenarios."""
    try:
        scenario_service = ScenarioModelingService()

        historical_result = await scenario_service.get_historical_scenarios(
            scenario_name=scenario_name,
            lookback_years=lookback_years
        )

        return {
            "scenario_name": scenario_name,
            "historical_occurrences": historical_result['occurrences'],
            "similar_periods": historical_result['similar_periods'],
            "historical_impact": historical_result['historical_impact']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_scenario_response(
    risk_budget: float = Body(..., description="Risk budget"),
    return_target: float = Body(..., description="Return target"),
    scenario_weights: Dict[str, float] = Body(..., description="Scenario probabilities")
):
    """
    Optimize portfolio allocation across multiple scenarios.

    Uses mean-variance optimization with scenario constraints to find
    optimal portfolio weights given multiple economic scenarios.
    """
    try:
        scenario_service = ScenarioModelingService()

        optimization_result = await scenario_service.optimize_scenario_response(
            risk_budget=risk_budget,
            return_target=return_target,
            scenario_weights=scenario_weights
        )

        return {
            "optimization_date": datetime.now().isoformat(),
            "optimal_weights": optimization_result['optimal_weights'],
            "expected_metrics": optimization_result['expected_metrics'],
            "scenario_breakdown": optimization_result['scenario_breakdown'],
            "optimization_constraints": {
                "risk_budget": risk_budget,
                "return_target": return_target,
                "scenario_weights": scenario_weights
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for scenario modeling service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }