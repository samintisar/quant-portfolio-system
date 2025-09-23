"""
Example usage of the ML prediction module for stock return forecasting.
"""

import pandas as pd
import numpy as np
from portfolio.ml.predictor import RandomForestPredictor
from portfolio.data.yahoo_service import YahooFinanceService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate ML prediction."""
    try:
        # Initialize services
        yahoo_service = YahooFinanceService()
        predictor = RandomForestPredictor(n_estimators=100, random_state=42)

        # Define symbols for demonstration
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        logger.info("Fetching historical data...")
        all_data = {}

        for symbol in symbols:
            try:
                data = yahoo_service.fetch_historical_data(symbol, period="2y")
                if not data.empty:
                    all_data[symbol] = data
                    logger.info(f"Loaded {len(data)} data points for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")

        # Train model for each symbol
        results = {}

        for symbol, data in all_data.items():
            logger.info(f"\n=== Training model for {symbol} ===")

            try:
                # Create features
                df_features = predictor.create_features(data)

                # Prepare features
                X, y = predictor.prepare_features(df_features)

                # Train model
                metrics = predictor.train(X, y)

                # Validate model
                validation_metrics = predictor.validate_model(X, y)

                # Get feature importance
                feature_importance = predictor.get_feature_importance()

                # Store results
                results[symbol] = {
                    'training_metrics': metrics,
                    'validation_metrics': validation_metrics,
                    'feature_importance': feature_importance,
                    'n_samples': len(df_features),
                    'n_features': X.shape[1]
                }

                # Print summary
                logger.info(f"Model trained for {symbol}:")
                logger.info(f"  - Samples: {len(df_features)}")
                logger.info(f"  - Features: {X.shape[1]}")
                logger.info(f"  - Test R²: {metrics['test_r2']:.4f}")
                logger.info(f"  - Directional Accuracy: {validation_metrics['directional_accuracy']:.4f}")
                logger.info(f"  - Top 3 features: {list(feature_importance.keys())[:3]}")

            except Exception as e:
                logger.error(f"Failed to train model for {symbol}: {e}")
                continue

        # Print overall summary
        logger.info("\n=== Overall Summary ===")
        logger.info(f"Successfully trained models for {len(results)}/{len(symbols)} symbols")

        for symbol, result in results.items():
            logger.info(f"\n{symbol}:")
            logger.info(f"  Test R²: {result['training_metrics']['test_r2']:.4f}")
            logger.info(f"  Directional Accuracy: {result['validation_metrics']['directional_accuracy']:.4f}")
            logger.info(f"  CV MSE: {result['training_metrics']['cv_mse']:.6f}")

        # Demonstrate prediction
        logger.info("\n=== Making Predictions ===")
        if results:
            symbol = list(results.keys())[0]
            data = all_data[symbol]
            df_features = predictor.create_features(data)

            # Use last few samples for prediction demo
            X, y = predictor.prepare_features(df_features)
            predictions = predictor.predict(X[-5:])  # Predict last 5 returns

            logger.info(f"Sample predictions for {symbol}:")
            for i, (pred, actual) in enumerate(zip(predictions, y[-5:])):
                logger.info(f"  Day {i+1}: Predicted={pred:.4f}, Actual={actual:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return None


if __name__ == "__main__":
    results = main()
    if results:
        logger.info("ML prediction example completed successfully!")
    else:
        logger.error("ML prediction example failed!")