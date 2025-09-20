"""
Signal validation API endpoints with relative benchmark reporting.

Implements REST endpoints for signal validation including statistical significance testing,
relative benchmark comparisons, and performance metrics evaluation.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.models.validation import SignalValidationModel
from forecasting.services.validation_service import SignalValidationService


router = APIRouter(prefix="/api/v1/validation", tags=["validation"])


class SignalValidationRequest(BaseModel):
    """Request model for signal validation."""
    signal_name: str = Field(..., description="Name of the signal")
    signal_data: List[float] = Field(..., description="Signal values over time")
    target_data: List[float] = Field(..., description="Target values over time")
    benchmark_data: Optional[List[float]] = Field(None, description="Benchmark signal values")
    confidence_level: float = Field(0.95, description="Confidence level for tests")
    test_type: str = Field("comprehensive", description="Type of validation: comprehensive, statistical, financial")


class ValidationResponse(BaseModel):
    """Response model for signal validation."""
    signal_name: str
    validation_date: str
    statistical_significance: Dict[str, float]
    financial_metrics: Dict[str, float]
    benchmark_comparison: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    model_info: Dict[str, Any]


class BacktestRequest(BaseModel):
    """Request model for backtesting signals."""
    strategy_name: str = Field(..., description="Strategy name")
    signals: List[Dict[str, Any]] = Field(..., description="Signal data with timestamps")
    price_data: List[Dict[str, Any]] = Field(..., description="Price data for backtesting")
    benchmark_prices: Optional[List[Dict[str, Any]]] = Field(None, description="Benchmark price data")
    transaction_costs: float = Field(0.001, description="Transaction cost rate")


@router.post("/validate", response_model=ValidationResponse)
async def validate_signal(request: SignalValidationRequest):
    """
    Validate trading signals using comprehensive statistical and financial tests.

    Features:
    - Statistical significance testing (t-tests, non-parametric tests)
    - Financial metrics calculation (Sharpe, Sortino, drawdown)
    - Relative benchmark comparisons
    - Regime-aware validation
    """
    try:
        # Initialize services
        validation_service = SignalValidationService()

        # Validate signal
        validation_result = await validation_service.validate_signal(
            signal_name=request.signal_name,
            signal_data=request.signal_data,
            target_data=request.target_data,
            benchmark_data=request.benchmark_data,
            confidence_level=request.confidence_level,
            test_type=request.test_type
        )

        return ValidationResponse(
            signal_name=request.signal_name,
            validation_date=datetime.now().isoformat(),
            statistical_significance=validation_result['statistical_significance'],
            financial_metrics=validation_result['financial_metrics'],
            benchmark_comparison=validation_result.get('benchmark_comparison'),
            recommendations=validation_result['recommendations'],
            model_info=validation_result['model_info']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest trading strategies with comprehensive performance metrics.

    Calculates traditional and relative performance metrics including
    regime-aware analysis and benchmark comparisons.
    """
    try:
        validation_service = SignalValidationService()

        backtest_result = await validation_service.backtest_strategy(
            strategy_name=request.strategy_name,
            signals=request.signals,
            price_data=request.price_data,
            benchmark_prices=request.benchmark_prices,
            transaction_costs=request.transaction_costs
        )

        return {
            "strategy_name": request.strategy_name,
            "backtest_date": datetime.now().isoformat(),
            "performance_metrics": backtest_result['performance_metrics'],
            "benchmark_comparison": backtest_result['benchmark_comparison'],
            "regime_analysis": backtest_result['regime_analysis'],
            "risk_metrics": backtest_result['risk_metrics'],
            "trade_analysis": backtest_result['trade_analysis']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_available_metrics():
    """Get list of available validation metrics."""
    return {
        "statistical_metrics": [
            {"name": "t_test", "description": "Student's t-test for mean significance"},
            {"name": "mann_whitney", "description": "Mann-Whitney U test for non-normal data"},
            {"name": "correlation", "description": "Pearson/Spearman correlation tests"},
            {"name": "stationarity", "description": "ADF and KPSS stationarity tests"},
            {"name": "autocorrelation", "description": "Ljung-Box autocorrelation test"}
        ],
        "financial_metrics": [
            {"name": "sharpe_ratio", "description": "Risk-adjusted return metric"},
            {"name": "sortino_ratio", "description": "Downside risk-adjusted return"},
            {"name": "max_drawdown", "description": "Maximum peak-to-trough decline"},
            {"name": "calmar_ratio", "description": "Return over maximum drawdown"},
            {"name": "information_ratio", "description": "Active return over tracking error"},
            {"name": "treynor_ratio", "description": "Return over systematic risk"}
        ],
        "relative_metrics": [
            {"name": "alpha", "description": "Excess return over benchmark"},
            {"name": "beta", "description": "Systematic risk relative to benchmark"},
            {"name": "information_coefficient", "description": "Forecast skill metric"},
            {"name": "up_down_capture", "description": "Benchmark capture ratios"}
        ]
    }


@router.post("/batch-validate")
async def batch_validate_signals(
    requests: List[SignalValidationRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent validations")
):
    """Validate multiple signals."""
    try:
        results = []

        for request in requests:
            try:
                result = await validate_signal(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "signal_name": request.signal_name,
                    "error": str(e)
                })

        return {"validations": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stress-test")
async def stress_test_signal(
    signal_data: List[float] = Body(..., description="Signal values"),
    scenarios: Dict[str, Dict[str, float]] = Body(..., description="Stress test scenarios"),
    confidence_level: float = Body(0.95, description="Confidence level")
):
    """
    Stress test signal performance under extreme market scenarios.

    Applies various stress scenarios to evaluate signal robustness and
    identifies potential failure modes.
    """
    try:
        validation_service = SignalValidationService()

        stress_result = await validation_service.stress_test_signal(
            signal_data=signal_data,
            scenarios=scenarios,
            confidence_level=confidence_level
        )

        return {
            "test_date": datetime.now().isoformat(),
            "stress_scenarios": stress_result['scenarios'],
            "robustness_metrics": stress_result['robustness_metrics'],
            "failure_modes": stress_result['failure_modes'],
            "survival_analysis": stress_result['survival_analysis']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks")
async def get_available_benchmarks():
    """Get list of available benchmark indices."""
    return {
        "benchmarks": [
            {"name": "SPY", "description": "S&P 500 ETF"},
            {"name": "QQQ", "description": "Nasdaq-100 ETF"},
            {"name": "IWM", "description": "Russell 2000 ETF"},
            {"name": "TLT", "description": "20+ Year Treasury ETF"},
            {"name": "GLD", "description": "Gold ETF"},
            {"name": "VTI", "description": "Total Stock Market ETF"},
            {"name": "AGG", "description": "Total Bond Market ETF"}
        ],
        "default_benchmark": "SPY"
    }


@router.post("/regime-analysis")
async def analyze_regime_performance(
    strategy_name: str = Body(..., description="Strategy name"),
    returns: List[float] = Body(..., description="Strategy returns"),
    regimes: List[int] = Body(..., description="Regime classifications"),
    benchmark_returns: Optional[List[float]] = Body(None, description="Benchmark returns")
):
    """
    Analyze strategy performance across different market regimes.

    Calculates regime-specific performance metrics and identifies
    regime-dependent strengths and weaknesses.
    """
    try:
        validation_service = SignalValidationService()

        regime_result = await validation_service.analyze_regime_performance(
            strategy_name=strategy_name,
            returns=returns,
            regimes=regimes,
            benchmark_returns=benchmark_returns
        )

        return {
            "strategy_name": strategy_name,
            "analysis_date": datetime.now().isoformat(),
            "regime_performance": regime_result['regime_performance'],
            "regime_characteristics": regime_result['regime_characteristics'],
            "adaptation_score": regime_result['adaptation_score'],
            "recommendations": regime_result['recommendations']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_signal_weights(
    signals: Dict[str, List[float]] = Body(..., description="Multiple signals to combine"),
    target: List[float] = Body(..., description="Target values"),
    objective: str = Body("max_sharpe", description="Optimization objective"),
    constraints: Optional[Dict[str, Any]] = Body(None, description="Optimization constraints")
):
    """
    Optimize signal combination weights for maximum performance.

    Uses mean-variance optimization or other methods to find optimal
    weights for combining multiple signals.
    """
    try:
        validation_service = SignalValidationService()

        optimization_result = await validation_service.optimize_signal_weights(
            signals=signals,
            target=target,
            objective=objective,
            constraints=constraints
        )

        return {
            "optimization_date": datetime.now().isoformat(),
            "optimal_weights": optimization_result['optimal_weights'],
            "expected_performance": optimization_result['expected_performance'],
            "optimization_metrics": optimization_result['optimization_metrics'],
            "constraint_analysis": optimization_result.get('constraint_analysis')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for signal validation service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }