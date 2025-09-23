"""
Machine Learning module for return prediction using Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RandomForestPredictor:
    """Random Forest model for predicting stock returns."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the Random Forest predictor.

        Args:
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic technical features from price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()

            # Basic price features
            df['price_change'] = df['Adj Close'].pct_change()
            df['log_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

            # Moving averages
            df['ma_5'] = df['Adj Close'].rolling(window=5).mean()
            df['ma_10'] = df['Adj Close'].rolling(window=10).mean()
            df['ma_20'] = df['Adj Close'].rolling(window=20).mean()

            # Moving average relationships
            df['ma_5_10_diff'] = df['ma_5'] - df['ma_10']
            df['ma_10_20_diff'] = df['ma_10'] - df['ma_20']
            df['price_vs_ma_5'] = (df['Adj Close'] / df['ma_5']) - 1
            df['price_vs_ma_20'] = (df['Adj Close'] / df['ma_20']) - 1

            # Volatility features
            df['volatility_5'] = df['log_returns'].rolling(window=5).std()
            df['volatility_10'] = df['log_returns'].rolling(window=10).std()
            df['volatility_20'] = df['log_returns'].rolling(window=20).std()

            # Volume features
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_5']

            # Price momentum
            df['momentum_5'] = (df['Adj Close'] / df['Adj Close'].shift(5)) - 1
            df['momentum_10'] = (df['Adj Close'] / df['Adj Close'].shift(10)) - 1
            df['momentum_20'] = (df['Adj Close'] / df['Adj Close'].shift(20)) - 1

            # High-Low range
            df['hl_range'] = (df['High'] - df['Low']) / df['Adj Close']
            df['hl_range_ma_5'] = df['hl_range'].rolling(window=5).mean()

            # RSI(14) - Relative Strength Index
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD(12,26,9) - Moving Average Convergence Divergence
            ema_12 = df['Adj Close'].ewm(span=12).mean()
            ema_26 = df['Adj Close'].ewm(span=26).mean()
            df['macd_line'] = ema_12 - ema_26
            df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']

            # Target: next day return
            df['target'] = df['price_change'].shift(-1)

            # Drop rows with NaN values
            df = df.dropna()

            logger.info(f"Created {len(df.columns)} features from {len(data)} rows")
            return df

        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training.

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (features, target)
        """
        try:
            # Define feature columns (exclude target and original OHLCV columns)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                          'returns', 'symbol', 'target']

            feature_cols = [col for col in df.columns if col not in exclude_cols]
            self.feature_names = feature_cols

            X = df[feature_cols].values
            y = df['target'].values

            logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary with training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, shuffle=False
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()

            self.is_trained = True

            metrics = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mse': cv_mse,
                'cv_std': cv_std,
                'feature_importance': self.get_feature_importance()
            }

            logger.info(f"Model trained successfully. Test R²: {test_r2:.4f}, CV MSE: {cv_mse:.6f}")
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        try:
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise

    def validate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Validate the model with additional metrics.

        Args:
            X: Feature matrix
            y: True values

        Returns:
            Dictionary with validation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")

        try:
            predictions = self.predict(X)

            # Calculate various metrics
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)

            # Directional accuracy (for returns)
            direction_correct = np.sum(np.sign(y) == np.sign(predictions)) / len(y)

            # Mean Absolute Error
            mae = np.mean(np.abs(y - predictions))

            validation_metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': direction_correct,
                'mae': mae
            }

            logger.info(f"Model validation completed. Directional accuracy: {direction_correct:.4f}")
            return validation_metrics

        except Exception as e:
            logger.error(f"Error validating model: {e}")
            raise

    def __str__(self):
        return f"RandomForestPredictor(n_estimators={self.n_estimators}, trained={self.is_trained})"