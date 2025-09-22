"""
ML baseline comparison API endpoints for model evaluation.

Implements REST endpoints for comparing statistical models against ML baselines
including XGBoost, LSTM, Transformer, and other machine learning approaches.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from forecasting.services.ml_benchmark_service import MLBenchmarkService
from forecasting.services.data_service import DataPreprocessingService


router = APIRouter(prefix="/api/v1/ml-benchmark", tags=["ml-benchmark"])


class BenchmarkRequest(BaseModel):
    """Request model for ML benchmarking."""
    task_type: str = Field(..., description="Task type: returns, volatility, regime")
    symbol: str = Field(..., description="Stock symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    features: List[str] = Field(..., description="Feature list for ML models")
    target_variable: str = Field(..., description="Target variable to predict")
    test_size: float = Field(0.2, description="Test set proportion")
    cv_folds: int = Field(5, description="Cross-validation folds")
    baselines: List[str] = Field(["xgboost", "lstm"], description="ML baselines to test")


class BenchmarkResponse(BaseModel):
    """Response model for ML benchmarking."""
    task_type: str
    symbol: str
    benchmark_date: str
    baseline_results: List[Dict[str, Any]]
    statistical_comparison: Dict[str, Any]
    performance_ranking: List[Dict[str, float]]
    model_recommendations: List[str]


class ModelTrainingRequest(BaseModel):
    """Request model for model training."""
    model_type: str = Field(..., description="Model type: xgboost, lstm, transformer")
    symbol: str = Field(..., description="Stock symbol")
    features: List[str] = Field(..., description="Feature list")
    target: str = Field(..., description="Target variable")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    validation_split: float = Field(0.2, description="Validation split")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """
    Run ML baseline comparison against statistical models.

    Features:
    - XGBoost gradient boosting
    - LSTM neural networks
    - Transformer architectures
    - Performance comparison with statistical baselines
    - Statistical significance testing
    """
    try:
        # Initialize services
        data_service = DataPreprocessingService()
        ml_service = MLBenchmarkService()

        # Fetch and preprocess data
        data = await data_service.fetch_historical_data(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )

        # Run benchmark
        benchmark_result = await ml_service.run_benchmark(
            task_type=request.task_type,
            data=data,
            features=request.features,
            target_variable=request.target_variable,
            test_size=request.test_size,
            cv_folds=request.cv_folds,
            baselines=request.baselines
        )

        return BenchmarkResponse(
            task_type=request.task_type,
            symbol=request.symbol,
            benchmark_date=datetime.now().isoformat(),
            baseline_results=benchmark_result['baseline_results'],
            statistical_comparison=benchmark_result['statistical_comparison'],
            performance_ranking=benchmark_result['performance_ranking'],
            model_recommendations=benchmark_result['model_recommendations']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model(request: ModelTrainingRequest):
    """
    Train a specific ML model with custom hyperparameters.

    Supports training of various ML architectures with optional
    hyperparameter optimization and cross-validation.
    """
    try:
        ml_service = MLBenchmarkService()

        training_result = await ml_service.train_model(
            model_type=request.model_type,
            symbol=request.symbol,
            features=request.features,
            target=request.target,
            hyperparameters=request.hyperparameters,
            validation_split=request.validation_split
        )

        return {
            "model_type": request.model_type,
            "symbol": request.symbol,
            "training_date": datetime.now().isoformat(),
            "model_id": training_result['model_id'],
            "training_metrics": training_result['training_metrics'],
            "validation_metrics": training_result['validation_metrics'],
            "hyperparameters": training_result['hyperparameters'],
            "feature_importance": training_result.get('feature_importance')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """Get list of available ML models."""
    return {
        "ml_models": [
            {"name": "xgboost", "description": "XGBoost Gradient Boosting", "task_type": ["returns", "volatility"]},
            {"name": "lstm", "description": "LSTM Neural Network", "task_type": ["returns", "volatility", "regime"]},
            {"name": "transformer", "description": "Transformer Architecture", "task_type": ["returns", "volatility"]},
            {"name": "random_forest", "description": "Random Forest", "task_type": ["returns", "volatility", "regime"]},
            {"name": "svm", "description": "Support Vector Machine", "task_type": ["regime"]},
            {"name": "prophet", "description": "Facebook Prophet", "task_type": ["returns"]}
        ],
        "statistical_baselines": [
            {"name": "arima", "description": "ARIMA Time Series Model"},
            {"name": "garch", "description": "GARCH Volatility Model"},
            {"name": "hmm", "description": "Hidden Markov Model"}
        ]
    }


@router.post("/predict")
async def predict_with_model(
    model_id: str = Body(..., description="Model ID"),
    features: Dict[str, float] = Body(..., description="Feature values"),
    horizon: int = Body(1, description="Prediction horizon")
):
    """
    Make predictions using a trained ML model.

    Uses previously trained models to generate forecasts
    for new data points.
    """
    try:
        ml_service = MLBenchmarkService()

        prediction_result = await ml_service.predict(
            model_id=model_id,
            features=features,
            horizon=horizon
        )

        return {
            "model_id": model_id,
            "prediction_date": datetime.now().isoformat(),
            "predictions": prediction_result['predictions'],
            "confidence_intervals": prediction_result.get('confidence_intervals'),
            "feature_contributions": prediction_result.get('feature_contributions')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperparameter-tuning")
async def tune_hyperparameters(
    model_type: str = Body(..., description="Model type"),
    symbol: str = Body(..., description="Stock symbol"),
    features: List[str] = Body(..., description="Feature list"),
    target: str = Body(..., description="Target variable"),
    search_space: Dict[str, List] = Body(..., description="Hyperparameter search space"),
    optimization_metric: str = Body("mse", description="Metric to optimize"),
    n_trials: int = Body(50, description="Number of optimization trials")
):
    """
    Perform hyperparameter optimization for ML models.

    Uses Bayesian optimization or grid search to find optimal
    hyperparameters for a given model.
    """
    try:
        ml_service = MLBenchmarkService()

        tuning_result = await ml_service.tune_hyperparameters(
            model_type=model_type,
            symbol=symbol,
            features=features,
            target=target,
            search_space=search_space,
            optimization_metric=optimization_metric,
            n_trials=n_trials
        )

        return {
            "model_type": model_type,
            "symbol": symbol,
            "tuning_date": datetime.now().isoformat(),
            "best_parameters": tuning_result['best_parameters'],
            "best_score": tuning_result['best_score'],
            "optimization_history": tuning_result['optimization_history'],
            "parameter_importance": tuning_result.get('parameter_importance')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance/{model_id}")
async def get_feature_importance(
    model_id: str,
    method: str = Query("shap", description="Method: shap, permutation, built-in")
):
    """Get feature importance analysis for a trained model."""
    try:
        ml_service = MLBenchmarkService()

        importance_result = await ml_service.get_feature_importance(
            model_id=model_id,
            method=method
        )

        return {
            "model_id": model_id,
            "analysis_date": datetime.now().isoformat(),
            "feature_importance": importance_result['feature_importance'],
            "method": method,
            "visualization_data": importance_result.get('visualization_data')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-benchmark")
async def batch_benchmark(
    requests: List[BenchmarkRequest],
    max_concurrent: int = Query(5, description="Maximum concurrent benchmarks")
):
    """Run benchmarks for multiple symbols and configurations."""
    try:
        results = []

        for request in requests:
            try:
                result = await run_benchmark(request)
                results.append(result)
            except Exception as e:
                results.append({
                    "symbol": request.symbol,
                    "task_type": request.task_type,
                    "error": str(e)
                })

        return {"benchmarks": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-explanation")
async def explain_model_predictions(
    model_id: str = Body(..., description="Model ID"),
    sample_data: List[Dict[str, float]] = Body(..., description="Sample data for explanation"),
    method: str = Body("shap", description="Explanation method")
):
    """
    Generate model explanations using SHAP or other methods.

    Provides interpretability insights for ML model predictions,
    helping to understand feature contributions and decision logic.
    """
    try:
        ml_service = MLBenchmarkService()

        explanation_result = await ml_service.explain_model(
            model_id=model_id,
            sample_data=sample_data,
            method=method
        )

        return {
            "model_id": model_id,
            "explanation_date": datetime.now().isoformat(),
            "method": method,
            "explanations": explanation_result['explanations'],
            "global_importance": explanation_result['global_importance'],
            "local_explanations": explanation_result['local_explanations']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-comparison")
async def compare_models(
    model_ids: List[str] = Body(..., description="List of model IDs to compare"),
    test_data: Dict[str, Any] = Body(..., description="Test data for comparison"),
    metrics: List[str] = Body(["mse", "mae", "r2"], description="Metrics to compare")
):
    """
    Compare multiple trained models on the same test data.

    Provides comprehensive comparison of model performance across
    multiple metrics and time periods.
    """
    try:
        ml_service = MLBenchmarkService()

        comparison_result = await ml_service.compare_models(
            model_ids=model_ids,
            test_data=test_data,
            metrics=metrics
        )

        return {
            "comparison_date": datetime.now().isoformat(),
            "model_comparison": comparison_result['model_comparison'],
            "performance_table": comparison_result['performance_table'],
            "statistical_significance": comparison_result['statistical_significance'],
            "recommendation": comparison_result['recommendation']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for ML benchmark service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }