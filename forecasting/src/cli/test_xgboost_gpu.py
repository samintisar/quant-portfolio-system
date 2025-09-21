#!/usr/bin/env python3
"""
Test script to verify XGBoost GPU acceleration in forecasting models.

This script tests:
1. XGBoost model with GPU enabled
2. XGBoost model with CPU only
3. Performance comparison
4. GPU integration with the forecasting system
"""

import sys
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgboost_model import XGBoostForecaster, XGBoostForecastConfig

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_data(n_samples: int = 1000) -> pd.Series:
    """Create sample time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Create synthetic time series with trend, seasonality, and noise
    trend = np.linspace(100, 200, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    noise = np.random.normal(0, 5, n_samples)

    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name='sample_prices')

def test_gpu_xgboost():
    """Test XGBoost with GPU acceleration."""
    logger = setup_logging()
    logger.info("Testing XGBoost with GPU acceleration...")

    # Create sample data
    data = create_sample_data(2000)

    # Test with GPU enabled
    gpu_config = XGBoostForecastConfig(
        n_estimators=200,
        max_depth=8,
        use_gpu=True,
        gpu_id=0
    )

    try:
        # Initialize GPU forecaster
        gpu_forecaster = XGBoostForecaster(gpu_config)

        # Train with timing
        start_time = time.time()
        gpu_results = gpu_forecaster.fit(data, validation_split=0.2)
        gpu_training_time = time.time() - start_time

        # Make predictions with timing
        start_time = time.time()
        gpu_predictions = gpu_forecaster.predict(data, steps=30)
        gpu_prediction_time = time.time() - start_time

        # Get model summary
        gpu_summary = gpu_forecaster.get_model_summary()

        logger.info(f"GPU Training Time: {gpu_training_time:.2f}s")
        logger.info(f"GPU Prediction Time (30 steps): {gpu_prediction_time:.4f}s")
        logger.info(f"GPU Model Summary: {gpu_summary['gpu_info']}")

        return {
            'gpu_enabled': True,
            'training_time': gpu_training_time,
            'prediction_time': gpu_prediction_time,
            'model_summary': gpu_summary,
            'predictions': gpu_predictions
        }

    except Exception as e:
        logger.error(f"GPU XGBoost test failed: {e}")
        return {'gpu_enabled': False, 'error': str(e)}

def test_cpu_xgboost():
    """Test XGBoost with CPU only."""
    logger = setup_logging()
    logger.info("Testing XGBoost with CPU only...")

    # Create sample data
    data = create_sample_data(2000)

    # Test with CPU only
    cpu_config = XGBoostForecastConfig(
        n_estimators=200,
        max_depth=8,
        use_gpu=False
    )

    try:
        # Initialize CPU forecaster
        cpu_forecaster = XGBoostForecaster(cpu_config)

        # Train with timing
        start_time = time.time()
        cpu_results = cpu_forecaster.fit(data, validation_split=0.2)
        cpu_training_time = time.time() - start_time

        # Make predictions with timing
        start_time = time.time()
        cpu_predictions = cpu_forecaster.predict(data, steps=30)
        cpu_prediction_time = time.time() - start_time

        # Get model summary
        cpu_summary = cpu_forecaster.get_model_summary()

        logger.info(f"CPU Training Time: {cpu_training_time:.2f}s")
        logger.info(f"CPU Prediction Time (30 steps): {cpu_prediction_time:.4f}s")
        logger.info(f"CPU Model Summary: {cpu_summary['gpu_info']}")

        return {
            'gpu_enabled': False,
            'training_time': cpu_training_time,
            'prediction_time': cpu_prediction_time,
            'model_summary': cpu_summary,
            'predictions': cpu_predictions
        }

    except Exception as e:
        logger.error(f"CPU XGBoost test failed: {e}")
        return {'gpu_enabled': False, 'error': str(e)}

def compare_performance(gpu_results, cpu_results):
    """Compare GPU vs CPU performance."""
    logger = setup_logging()
    logger.info("Comparing GPU vs CPU performance...")

    if gpu_results.get('error') or cpu_results.get('error'):
        logger.error("Cannot compare performance due to errors")
        return

    gpu_time = gpu_results['training_time']
    cpu_time = cpu_results['training_time']

    if cpu_time > 0:
        speedup = cpu_time / gpu_time
        logger.info(f"Training Speedup: {speedup:.2f}x")
    else:
        logger.info("CPU training time was too small to calculate speedup")

    # Compare prediction performance
    gpu_pred_time = gpu_results['prediction_time']
    cpu_pred_time = cpu_results['prediction_time']

    if cpu_pred_time > 0:
        pred_speedup = cpu_pred_time / gpu_pred_time
        logger.info(f"Prediction Speedup: {pred_speedup:.2f}x")
    else:
        logger.info("CPU prediction time was too small to calculate speedup")

def test_cross_validation():
    """Test cross-validation with GPU support."""
    logger = setup_logging()
    logger.info("Testing cross-validation with GPU support...")

    # Create sample data
    data = create_sample_data(1000)

    # Test GPU cross-validation
    gpu_config = XGBoostForecastConfig(
        n_estimators=100,
        max_depth=6,
        use_gpu=True
    )

    try:
        gpu_forecaster = XGBoostForecaster(gpu_config)
        cv_results = gpu_forecaster.cross_validate(data, n_splits=5)

        logger.info(f"GPU Cross-Validation Results:")
        logger.info(f"  RMSE: {np.mean(cv_results['rmse']):.4f} ± {np.std(cv_results['rmse']):.4f}")
        logger.info(f"  MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")

        return cv_results

    except Exception as e:
        logger.error(f"Cross-validation test failed: {e}")
        return None

def run_comprehensive_test():
    """Run comprehensive XGBoost GPU test."""
    logger = setup_logging()
    logger.info("Starting comprehensive XGBoost GPU test...")

    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test GPU XGBoost
    logger.info("\n=== GPU XGBoost Test ===")
    gpu_results = test_gpu_xgboost()
    test_results['tests']['gpu_xgboost'] = gpu_results

    # Test CPU XGBoost
    logger.info("\n=== CPU XGBoost Test ===")
    cpu_results = test_cpu_xgboost()
    test_results['tests']['cpu_xgboost'] = cpu_results

    # Compare performance
    logger.info("\n=== Performance Comparison ===")
    compare_performance(gpu_results, cpu_results)

    # Test cross-validation
    logger.info("\n=== Cross-Validation Test ===")
    cv_results = test_cross_validation()
    test_results['tests']['cross_validation'] = {'success': cv_results is not None}

    # Summary
    logger.info("\n=== Test Summary ===")
    gpu_success = gpu_results.get('gpu_enabled', False) and 'error' not in gpu_results
    cpu_success = 'error' not in cpu_results
    cv_success = cv_results is not None

    logger.info(f"GPU XGBoost: {'PASS' if gpu_success else 'FAIL'}")
    logger.info(f"CPU XGBoost: {'PASS' if cpu_success else 'FAIL'}")
    logger.info(f"Cross-Validation: {'PASS' if cv_success else 'FAIL'}")

    # Save results
    output_file = "xgboost_gpu_test_results.json"
    import json
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Test results saved to {output_file}")

    return test_results

if __name__ == "__main__":
    results = run_comprehensive_test()