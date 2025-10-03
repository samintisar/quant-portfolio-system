"""
Model Comparison Script for ML Predictors

Systematically compares XGBoost, RandomForest, and Ensemble models.
Outputs comparative table with directional accuracy, R², and training time.
"""

import sys
import os
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.ml import RandomForestPredictor, XGBoostPredictor, EnsemblePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_models(symbols: list, period: str = "10y", device: str = "cpu"):
    """
    Compare ML models across multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        period: Data period
        device: Device for XGBoost ('cpu' or 'cuda')
    
    Returns:
        DataFrame with comparison results
    """
    service = YahooFinanceService(use_offline_data=True, offline_data_dir="data")
    
    results = []
    
    for sym in symbols:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {sym}")
            logger.info(f"{'='*60}")
            
            # Fetch data
            data = service.fetch_historical_data(sym, period=period)
            if data is None or data.empty:
                logger.warning(f"No data for {sym}, skipping")
                continue
            
            # XGBoost (improved)
            logger.info(f"\n[{sym}] Testing XGBoost...")
            xgb = XGBoostPredictor(device=device)
            feats_xgb = xgb.create_features(data)
            if feats_xgb is None or feats_xgb.empty:
                continue
            X_xgb, y_xgb = xgb.prepare_features(feats_xgb)
            
            t0 = time.time()
            metrics_xgb = xgb.train(X_xgb, y_xgb)
            val_xgb = xgb.validate_model(X_xgb, y_xgb)
            time_xgb = time.time() - t0
            
            # RandomForest (improved)
            logger.info(f"\n[{sym}] Testing RandomForest...")
            rf = RandomForestPredictor()
            feats_rf = rf.create_features(data)
            X_rf, y_rf = rf.prepare_features(feats_rf)
            
            t0 = time.time()
            metrics_rf = rf.train(X_rf, y_rf)
            val_rf = rf.validate_model(X_rf, y_rf)
            time_rf = time.time() - t0
            
            # Ensemble
            logger.info(f"\n[{sym}] Testing Ensemble...")
            ensemble = EnsemblePredictor(xgb_weight=0.6, device=device)
            feats_ens = ensemble.create_features(data)
            X_ens, y_ens = ensemble.prepare_features(feats_ens)
            
            t0 = time.time()
            metrics_ens = ensemble.train(X_ens, y_ens)
            val_ens = ensemble.validate_model(X_ens, y_ens)
            time_ens = time.time() - t0
            
            # Store results
            results.append({
                'symbol': sym,
                'model': 'XGBoost',
                'train_r2': metrics_xgb['train_r2'],
                'test_r2': metrics_xgb['test_r2'],
                'directional_accuracy': val_xgb['directional_accuracy'],
                'cv_mse': metrics_xgb['cv_mse'],
                'best_iteration': metrics_xgb.get('best_iteration', metrics_xgb.get('n_estimators', 0)),
                'stopped_early': metrics_xgb.get('stopped_early', False),
                'n_features': len(xgb.feature_names),
                'training_time_sec': time_xgb
            })
            
            results.append({
                'symbol': sym,
                'model': 'RandomForest',
                'train_r2': metrics_rf['train_r2'],
                'test_r2': metrics_rf['test_r2'],
                'directional_accuracy': val_rf['directional_accuracy'],
                'cv_mse': metrics_rf['cv_mse'],
                'best_iteration': None,
                'stopped_early': False,
                'n_features': len(rf.feature_names),
                'training_time_sec': time_rf
            })
            
            results.append({
                'symbol': sym,
                'model': 'Ensemble',
                'train_r2': None,  # Not directly available
                'test_r2': val_ens['r2'],
                'directional_accuracy': val_ens['directional_accuracy'],
                'cv_mse': None,
                'best_iteration': None,
                'stopped_early': False,
                'n_features': len(ensemble.feature_names),
                'training_time_sec': time_ens
            })
            
            # Log top features for XGBoost
            importance = xgb.get_feature_importance()
            top_features = list(importance.keys())[:5]
            logger.info(f"\n[{sym}] Top 5 features (XGBoost): {top_features}")
            
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}", exc_info=True)
    
    df = pd.DataFrame(results)
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics and rankings."""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Overall averages by model
    print("\n### AVERAGE METRICS BY MODEL ###")
    summary = df.groupby('model').agg({
        'test_r2': 'mean',
        'directional_accuracy': 'mean',
        'training_time_sec': 'mean',
        'n_features': 'mean'
    }).round(4)
    print(summary.to_string())
    
    # Best model per metric
    print("\n### BEST MODEL BY METRIC ###")
    best_r2 = df.loc[df['test_r2'].idxmax()]
    best_dir = df.loc[df['directional_accuracy'].idxmax()]
    print(f"Best Test R²: {best_r2['model']} ({best_r2['symbol']}) = {best_r2['test_r2']:.4f}")
    print(f"Best Directional Accuracy: {best_dir['model']} ({best_dir['symbol']}) = {best_dir['directional_accuracy']:.4f}")
    
    # Early stopping effectiveness
    xgb_stopped = df[df['model'] == 'XGBoost']['stopped_early'].sum()
    xgb_total = len(df[df['model'] == 'XGBoost'])
    print(f"\nXGBoost Early Stopping: {xgb_stopped}/{xgb_total} stopped early")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "JPM", "JNJ", "PG"]
    
    print("Starting ML Model Comparison...")
    print(f"Symbols: {symbols}")
    print(f"Period: 10y\n")
    
    # Run comparison
    results_df = compare_models(symbols, period="10y", device="cpu")
    
    # Print summary
    print_summary(results_df)
    
    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ml_model_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved detailed results to {output_file}")
    
    # Display full table
    print("\n### DETAILED RESULTS ###")
    print(results_df.to_string(index=False))
