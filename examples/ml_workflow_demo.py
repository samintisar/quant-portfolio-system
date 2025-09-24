#!/usr/bin/env python3
"""
ML Workflow Demo: End-to-end example with new technical indicators (RSI + MACD)
This demonstrates the complete ML pipeline from data to trained model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio.ml.predictor import RandomForestPredictor
from portfolio.data.yahoo_service import YahooFinanceService
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ml_workflow():
    """Run complete ML workflow demonstration."""

    # Initialize services
    yahoo_service = YahooFinanceService(use_offline_data=True)
    predictor = RandomForestPredictor(n_estimators=100, random_state=42)

    # Configuration
    symbol = 'AAPL'
    period = '5y'

    logger.info(f"=== ML Workflow Demo for {symbol} ===")

    try:
        # Step 1: Fetch data
        logger.info("Step 1: Fetching historical data...")
        data = yahoo_service.fetch_historical_data(symbol, period)
        logger.info(f"Fetched {len(data)} rows of data")

        # Step 2: Create features (including new RSI + MACD)
        logger.info("Step 2: Creating technical features...")
        features_df = predictor.create_features(data)

        # Show new indicators
        new_indicators = ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram']
        logger.info(f"New technical indicators created:")
        for indicator in new_indicators:
            if indicator in features_df.columns:
                logger.info(f"  ✓ {indicator}")

        # Step 3: Prepare training data
        logger.info("Step 3: Preparing training data...")
        X, y = predictor.prepare_features(features_df)
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")

        # Step 4: Train model
        logger.info("Step 4: Training Random Forest model...")
        metrics = predictor.train(X, y)

        # Step 5: Analyze results
        logger.info("Step 5: Analyzing results...")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Test MSE: {metrics['test_mse']:.6f}")
        logger.info(f"CV MSE: {metrics['cv_mse']:.6f}")

        # Step 6: Feature importance analysis
        logger.info("Step 6: Top 10 most important features:")
        feature_importance = metrics['feature_importance']
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")

        # Check if new indicators are in top features
        new_in_top = sum(1 for feat in new_indicators if feat in [f[0] for f in list(feature_importance.items())[:10]])
        logger.info(f"New technical indicators in top 10: {new_in_top}/4")

        # Step 7: Validation metrics
        logger.info("Step 7: Additional validation...")
        validation_metrics = predictor.validate_model(X, y)
        logger.info(f"Directional accuracy: {validation_metrics['directional_accuracy']:.4f}")
        logger.info(f"RMSE: {validation_metrics['rmse']:.6f}")

        # Create summary report
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {period}")
        logger.info(f"Data points: {len(features_df)}")
        logger.info(f"Total features: {len(feature_importance)}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Directional accuracy: {validation_metrics['directional_accuracy']:.1%}")
        logger.info(f"CV MSE: {metrics['cv_mse']:.6f}")

        # Save feature importance plot
        try:
            plot_feature_importance(feature_importance, symbol)
            logger.info("Feature importance plot saved")
        except Exception as e:
            logger.warning(f"Could not save plot: {e}")

        return True

    except Exception as e:
        logger.error(f"Error in ML workflow: {e}")
        return False

def plot_feature_importance(feature_importance, symbol):
    """Create and save feature importance plot."""
    # Take top 15 features
    top_features = dict(list(feature_importance.items())[:15])

    plt.figure(figsize=(10, 8))
    features = list(top_features.keys())
    importances = list(top_features.values())

    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importance - {symbol}')
    plt.tight_layout()

    # Highlight new indicators
    new_indicators = ['rsi_14', 'macd_line', 'macd_signal', 'macd_histogram']
    colors = ['red' if feat in new_indicators else 'blue' for feat in features]

    # Recreate with colors
    plt.clf()
    plt.barh(range(len(features)), importances, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importance - {symbol} (Red = New Indicators)')
    plt.tight_layout()

    # Save under examples/figures
    figures_dir = Path(__file__).resolve().parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f'feature_importance_{symbol}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    success = run_ml_workflow()
    if success:
        logger.info("✅ ML workflow demo completed successfully")
    else:
        logger.error("❌ ML workflow demo failed")

if __name__ == "__main__":
    run_ml_workflow()
