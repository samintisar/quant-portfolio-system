"""
Machine Learning module for return prediction using tree-based models.

Includes:
- RandomForestPredictor (CPU)
- XGBoostPredictor (CPU/GPU via device flag)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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
            max_depth=8,              # Limit tree depth to reduce overfitting
            min_samples_split=20,     # Require more samples to split
            min_samples_leaf=10,      # Require more samples in leaves
            max_features='sqrt',      # Use sqrt(n_features) per split
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.selected_features = None
        self.is_trained = False

    def create_features(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create robust factor-based, regime-aware, and cross-sectional features.

        Args:
            data: DataFrame with OHLCV data for the asset
            spy_data: Optional DataFrame with SPY data for market regime features

        Returns:
            DataFrame with comprehensive engineered features
        """
        try:
            df = data.copy()

            # Basic returns
            df['price_change'] = df['Adj Close'].pct_change()
            df['log_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

            # === FACTOR-BASED FEATURES ===
            # Momentum factors (12M, 6M, 3M, 1M)
            df['momentum_12m'] = (df['Adj Close'] / df['Adj Close'].shift(252)) - 1  # ~12 months
            df['momentum_6m'] = (df['Adj Close'] / df['Adj Close'].shift(126)) - 1   # ~6 months
            df['momentum_3m'] = (df['Adj Close'] / df['Adj Close'].shift(63)) - 1    # ~3 months
            df['momentum_1m'] = (df['Adj Close'] / df['Adj Close'].shift(21)) - 1    # ~1 month
            
            # Short-term momentum for mean reversion
            df['momentum_5d'] = (df['Adj Close'] / df['Adj Close'].shift(5)) - 1
            df['momentum_10d'] = (df['Adj Close'] / df['Adj Close'].shift(10)) - 1
            df['momentum_21d'] = (df['Adj Close'] / df['Adj Close'].shift(21)) - 1

            # Value proxies (using price-to-moving-average as simplified P/E proxy)
            df['price_to_ma_252'] = df['Adj Close'] / df['Adj Close'].rolling(252).mean()
            df['price_to_ma_126'] = df['Adj Close'] / df['Adj Close'].rolling(126).mean()
            df['price_to_ma_63'] = df['Adj Close'] / df['Adj Close'].rolling(63).mean()
            
            # Quality proxy: ROE trend approximated by return stability
            df['return_stability_63d'] = df['log_returns'].rolling(63).mean() / (df['log_returns'].rolling(63).std() + 1e-8)
            df['return_stability_126d'] = df['log_returns'].rolling(126).mean() / (df['log_returns'].rolling(126).std() + 1e-8)

            # === REGIME-AWARE FEATURES ===
            # Volatility regimes
            df['volatility_21d'] = df['log_returns'].rolling(21).std() * np.sqrt(252)
            df['volatility_63d'] = df['log_returns'].rolling(63).std() * np.sqrt(252)
            df['volatility_252d'] = df['log_returns'].rolling(252).std() * np.sqrt(252)
            df['vol_regime'] = df['volatility_21d'] / (df['volatility_252d'] + 1e-8)
            
            # Volume regime
            df['volume_ma_21'] = df['Volume'].rolling(21).mean()
            df['volume_ma_63'] = df['Volume'].rolling(63).mean()
            df['volume_regime'] = df['volume_ma_21'] / (df['volume_ma_63'] + 1e-8)
            
            # Market correlation with SPY (if available)
            if spy_data is not None and not spy_data.empty:
                spy_returns = spy_data['Adj Close'].pct_change()
                # Align indices
                aligned_df = df.join(spy_returns.rename('spy_returns'), how='left')
                if 'spy_returns' in aligned_df.columns:
                    # Rolling correlation with market
                    df['spy_corr_63d'] = aligned_df['log_returns'].rolling(63).corr(aligned_df['spy_returns'])
                    df['spy_corr_126d'] = aligned_df['log_returns'].rolling(126).corr(aligned_df['spy_returns'])
                    
                    # Beta stability
                    df['beta_63d'] = aligned_df['log_returns'].rolling(63).cov(aligned_df['spy_returns']) / (aligned_df['spy_returns'].rolling(63).var() + 1e-8)
                    df['beta_126d'] = aligned_df['log_returns'].rolling(126).cov(aligned_df['spy_returns']) / (aligned_df['spy_returns'].rolling(126).var() + 1e-8)
                    df['beta_stability'] = np.abs(df['beta_63d'] - df['beta_126d'])

            # === LAG FEATURES ===
            # Lagged momentum
            df['momentum_5d_lag5'] = df['momentum_5d'].shift(5)
            df['momentum_10d_lag10'] = df['momentum_10d'].shift(10)
            df['momentum_21d_lag21'] = df['momentum_21d'].shift(21)
            
            # Lagged volatility
            df['volatility_21d_lag5'] = df['volatility_21d'].shift(5)
            df['volatility_21d_lag10'] = df['volatility_21d'].shift(10)
            df['volatility_21d_lag21'] = df['volatility_21d'].shift(21)

            # === INTERACTION FEATURES ===
            # Volatility × Momentum (risk-adjusted momentum)
            df['vol_mom_21d'] = df['volatility_21d'] * df['momentum_21d']
            df['vol_mom_63d'] = df['volatility_63d'] * df['momentum_3m']
            
            # Volume × Price Change (volume-weighted momentum) - with proper lag to avoid leakage
            df['volume_price_change_lag1'] = df['volume_regime'].shift(1) * df['price_change'].shift(1)
            df['volume_momentum_21d'] = df['volume_regime'] * df['momentum_21d']
            
            # High-Low range features
            df['hl_range'] = (df['High'] - df['Low']) / (df['Adj Close'] + 1e-8)
            df['hl_range_ma_21'] = df['hl_range'].rolling(21).mean()
            df['hl_volatility'] = df['hl_range'].rolling(21).std()

            # === ADDITIONAL MOMENTUM FEATURES ===
            # Acceleration (change in momentum)
            df['momentum_accel_21d'] = df['momentum_21d'] - df['momentum_21d'].shift(21)
            df['momentum_accel_63d'] = df['momentum_3m'] - df['momentum_3m'].shift(63)
            
            # Moving average crossovers
            df['ma_21'] = df['Adj Close'].rolling(21).mean()
            df['ma_63'] = df['Adj Close'].rolling(63).mean()
            df['ma_cross_21_63'] = (df['ma_21'] / df['ma_63']) - 1

            # === NEW FEATURES: Volatility regime indicators ===
            # Market stress indicator
            df['vol_spike'] = (df['volatility_21d'] > df['volatility_252d'].rolling(63).quantile(0.75)).astype(int)
            
            # Trend strength
            df['trend_strength'] = np.abs(df['momentum_3m']) * (1.0 / (df['volatility_63d'] + 1e-8))
            
            # Volume surge (abnormal trading activity)
            df['volume_surge'] = (df['volume_ma_21'] > df['volume_ma_63'] * 1.5).astype(int)
            
            # Drawdown from 252-day high
            df['drawdown_from_high'] = (df['Adj Close'] / df['Adj Close'].rolling(252).max()) - 1
            
            # Rolling win rate (% of positive days)
            df['win_rate_21d'] = (df['price_change'] > 0).rolling(21).mean()
            df['win_rate_63d'] = (df['price_change'] > 0).rolling(63).mean()

            # === TARGET VARIABLE ===
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
            # Exclude raw OHLCV data, target, and intermediate calculations that cause leakage
            # Also remove noisy interaction features
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                          'returns', 'symbol', 'target', 'Date', 'spy_returns',
                          'price_change', 'log_returns', 'ma_21', 'ma_63',
                          # Remove noisy lagged features
                          'momentum_5d_lag5', 'momentum_10d_lag10', 'momentum_21d_lag21',
                          'volatility_21d_lag5', 'volatility_21d_lag10', 'volatility_21d_lag21',
                          # Remove weak interaction features
                          'vol_mom_21d', 'vol_mom_63d', 'volume_momentum_21d',
                          'momentum_accel_21d', 'momentum_accel_63d']

            feature_cols = [col for col in df.columns if col not in exclude_cols]
            self.feature_names = feature_cols

            X = df[feature_cols].values
            y = df['target'].values

            logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def select_features(self, X: np.ndarray, y: np.ndarray, top_n: int = 20) -> np.ndarray:
        """
        Select top N features using mutual information and remove redundant ones.
        
        Args:
            X: Feature matrix
            y: Target values
            top_n: Number of features to select
        
        Returns:
            Indices of selected features
        """
        from sklearn.feature_selection import mutual_info_regression, SelectKBest
        
        # Mutual information for feature relevance
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Get top features by MI score
        selector = SelectKBest(score_func=mutual_info_regression, k=min(top_n, X.shape[1]))
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        return selected_indices

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

            # Feature selection to reduce overfitting
            if len(self.feature_names) > 20:
                selected_idx = self.select_features(X_train, y_train, top_n=20)
                X_train = X_train[:, selected_idx]
                X_test = X_test[:, selected_idx]
                self.selected_features = selected_idx
                self.feature_names = [self.feature_names[i] for i in selected_idx]

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

            # Time-series cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error')
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
            # Apply feature selection if it was used during training
            if self.selected_features is not None:
                X = X[:, self.selected_features]
            
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


class XGBoostPredictor:
    """XGBoost model for predicting stock returns with optional GPU acceleration.

    Set device to "cuda" to use an NVIDIA GPU (e.g., RTX 3060) if available.
    """

    def __init__(
        self,
        n_estimators: int = 200,           # Increased for early stopping
        max_depth: int = 4,                # REDUCED from 5
        learning_rate: float = 0.05,       # REDUCED from 0.1
        subsample: float = 0.8,            # REDUCED from 0.85
        colsample_bytree: float = 0.8,     # REDUCED from 0.85
        min_child_weight: int = 5,         # INCREASED from 3
        gamma: float = 0.05,               # INCREASED from 0.01
        reg_alpha: float = 0.05,           # INCREASED from 0.01
        reg_lambda: float = 0.5,           # INCREASED from 0.1
        early_stopping_rounds: int = 20,   # NEW parameter
        random_state: int = 42,
        device: str = "cpu",
    ):
        """
        Initialize the XGBoost predictor.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (reduced to prevent overfitting)
            learning_rate: Learning rate (eta)
            subsample: Row subsampling rate
            colsample_bytree: Column subsampling per tree
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            early_stopping_rounds: Stop if no improvement for N rounds
            random_state: Random seed for reproducibility
            device: "cpu" or "cuda" for GPU acceleration
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.device = device

        # Use histogram tree method with regularization to prevent overfitting
        # Note: early_stopping_rounds is passed during fit(), not initialization
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            random_state=random_state,
        )

        # Scaling is not required for tree models but we keep it for consistency
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.selected_features = None
        self.is_trained = False

    def create_features(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create robust factor-based, regime-aware, and cross-sectional features.
        
        Args:
            data: DataFrame with OHLCV data for the asset
            spy_data: Optional DataFrame with SPY data for market regime features
            
        Returns:
            DataFrame with comprehensive engineered features
        """
        try:
            df = data.copy()

            # Basic returns
            df['price_change'] = df['Adj Close'].pct_change()
            df['log_returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

            # === FACTOR-BASED FEATURES ===
            # Momentum factors (12M, 6M, 3M, 1M)
            df['momentum_12m'] = (df['Adj Close'] / df['Adj Close'].shift(252)) - 1  # ~12 months
            df['momentum_6m'] = (df['Adj Close'] / df['Adj Close'].shift(126)) - 1   # ~6 months
            df['momentum_3m'] = (df['Adj Close'] / df['Adj Close'].shift(63)) - 1    # ~3 months
            df['momentum_1m'] = (df['Adj Close'] / df['Adj Close'].shift(21)) - 1    # ~1 month
            
            # Short-term momentum for mean reversion
            df['momentum_5d'] = (df['Adj Close'] / df['Adj Close'].shift(5)) - 1
            df['momentum_10d'] = (df['Adj Close'] / df['Adj Close'].shift(10)) - 1
            df['momentum_21d'] = (df['Adj Close'] / df['Adj Close'].shift(21)) - 1

            # Value proxies (using price-to-moving-average as simplified P/E proxy)
            df['price_to_ma_252'] = df['Adj Close'] / df['Adj Close'].rolling(252).mean()
            df['price_to_ma_126'] = df['Adj Close'] / df['Adj Close'].rolling(126).mean()
            df['price_to_ma_63'] = df['Adj Close'] / df['Adj Close'].rolling(63).mean()
            
            # Quality proxy: ROE trend approximated by return stability
            df['return_stability_63d'] = df['log_returns'].rolling(63).mean() / (df['log_returns'].rolling(63).std() + 1e-8)
            df['return_stability_126d'] = df['log_returns'].rolling(126).mean() / (df['log_returns'].rolling(126).std() + 1e-8)

            # === REGIME-AWARE FEATURES ===
            # Volatility regimes
            df['volatility_21d'] = df['log_returns'].rolling(21).std() * np.sqrt(252)
            df['volatility_63d'] = df['log_returns'].rolling(63).std() * np.sqrt(252)
            df['volatility_252d'] = df['log_returns'].rolling(252).std() * np.sqrt(252)
            df['vol_regime'] = df['volatility_21d'] / (df['volatility_252d'] + 1e-8)
            
            # Volume regime
            df['volume_ma_21'] = df['Volume'].rolling(21).mean()
            df['volume_ma_63'] = df['Volume'].rolling(63).mean()
            df['volume_regime'] = df['volume_ma_21'] / (df['volume_ma_63'] + 1e-8)
            
            # Market correlation with SPY (if available)
            if spy_data is not None and not spy_data.empty:
                spy_returns = spy_data['Adj Close'].pct_change()
                # Align indices
                aligned_df = df.join(spy_returns.rename('spy_returns'), how='left')
                if 'spy_returns' in aligned_df.columns:
                    # Rolling correlation with market
                    df['spy_corr_63d'] = aligned_df['log_returns'].rolling(63).corr(aligned_df['spy_returns'])
                    df['spy_corr_126d'] = aligned_df['log_returns'].rolling(126).corr(aligned_df['spy_returns'])
                    
                    # Beta stability
                    df['beta_63d'] = aligned_df['log_returns'].rolling(63).cov(aligned_df['spy_returns']) / (aligned_df['spy_returns'].rolling(63).var() + 1e-8)
                    df['beta_126d'] = aligned_df['log_returns'].rolling(126).cov(aligned_df['spy_returns']) / (aligned_df['spy_returns'].rolling(126).var() + 1e-8)
                    df['beta_stability'] = np.abs(df['beta_63d'] - df['beta_126d'])

            # === LAG FEATURES ===
            # Lagged momentum
            df['momentum_5d_lag5'] = df['momentum_5d'].shift(5)
            df['momentum_10d_lag10'] = df['momentum_10d'].shift(10)
            df['momentum_21d_lag21'] = df['momentum_21d'].shift(21)
            
            # Lagged volatility
            df['volatility_21d_lag5'] = df['volatility_21d'].shift(5)
            df['volatility_21d_lag10'] = df['volatility_21d'].shift(10)
            df['volatility_21d_lag21'] = df['volatility_21d'].shift(21)

            # === INTERACTION FEATURES ===
            # Volatility × Momentum (risk-adjusted momentum)
            df['vol_mom_21d'] = df['volatility_21d'] * df['momentum_21d']
            df['vol_mom_63d'] = df['volatility_63d'] * df['momentum_3m']
            
            # Volume × Price Change (volume-weighted momentum) - with proper lag to avoid leakage
            df['volume_price_change_lag1'] = df['volume_regime'].shift(1) * df['price_change'].shift(1)
            df['volume_momentum_21d'] = df['volume_regime'] * df['momentum_21d']
            
            # High-Low range features
            df['hl_range'] = (df['High'] - df['Low']) / (df['Adj Close'] + 1e-8)
            df['hl_range_ma_21'] = df['hl_range'].rolling(21).mean()
            df['hl_volatility'] = df['hl_range'].rolling(21).std()

            # === ADDITIONAL MOMENTUM FEATURES ===
            # Acceleration (change in momentum)
            df['momentum_accel_21d'] = df['momentum_21d'] - df['momentum_21d'].shift(21)
            df['momentum_accel_63d'] = df['momentum_3m'] - df['momentum_3m'].shift(63)
            
            # Moving average crossovers
            df['ma_21'] = df['Adj Close'].rolling(21).mean()
            df['ma_63'] = df['Adj Close'].rolling(63).mean()
            df['ma_cross_21_63'] = (df['ma_21'] / df['ma_63']) - 1

            # === NEW FEATURES: Volatility regime indicators ===
            # Market stress indicator
            df['vol_spike'] = (df['volatility_21d'] > df['volatility_252d'].rolling(63).quantile(0.75)).astype(int)
            
            # Trend strength
            df['trend_strength'] = np.abs(df['momentum_3m']) * (1.0 / (df['volatility_63d'] + 1e-8))
            
            # Volume surge (abnormal trading activity)
            df['volume_surge'] = (df['volume_ma_21'] > df['volume_ma_63'] * 1.5).astype(int)
            
            # Drawdown from 252-day high
            df['drawdown_from_high'] = (df['Adj Close'] / df['Adj Close'].rolling(252).max()) - 1
            
            # Rolling win rate (% of positive days)
            df['win_rate_21d'] = (df['price_change'] > 0).rolling(21).mean()
            df['win_rate_63d'] = (df['price_change'] > 0).rolling(63).mean()

            # === TARGET VARIABLE ===
            df['target'] = df['price_change'].shift(-1)

            # Drop rows with NaN values
            df = df.dropna()

            logger.info(f"Created {len(df.columns)} features from {len(data)} rows")
            return df
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target arrays.
        
        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (features, target)
        """
        try:
            # Exclude raw OHLCV data, target, and intermediate calculations that cause leakage
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                            'returns', 'symbol', 'target', 'Date', 'spy_returns',
                            'price_change', 'log_returns', 'ma_21', 'ma_63']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            self.feature_names = feature_cols
            X = df[feature_cols].values
            y = df['target'].values
            logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
            return X, y
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def select_features(self, X: np.ndarray, y: np.ndarray, top_n: int = 20) -> np.ndarray:
        """
        Select top N features using mutual information and remove redundant ones.
        
        Args:
            X: Feature matrix
            y: Target values
            top_n: Number of features to select
        
        Returns:
            Indices of selected features
        """
        from sklearn.feature_selection import mutual_info_regression, SelectKBest
        
        # Mutual information for feature relevance
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Get top features by MI score
        selector = SelectKBest(score_func=mutual_info_regression, k=min(top_n, X.shape[1]))
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:10]}...")
        return selected_indices

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost with early stopping and feature selection."""
        try:
            # Split: 60% train, 20% validation, 20% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=self.random_state, shuffle=False
            )

            # Feature selection to reduce overfitting
            if len(self.feature_names) > 20:
                selected_idx = self.select_features(X_train, y_train, top_n=20)
                X_train = X_train[:, selected_idx]
                X_val = X_val[:, selected_idx]
                X_test = X_test[:, selected_idx]
                self.selected_features = selected_idx
                self.feature_names = [self.feature_names[i] for i in selected_idx]

            # Train model (early stopping support depends on XGBoost version)
            try:
                # Try modern XGBoost API with callbacks
                from xgboost.callback import EarlyStopping
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=self.early_stopping_rounds, save_best=True)],
                    verbose=False
                )
                booster = self.model.get_booster()
                best_iteration = booster.best_iteration if hasattr(booster, 'best_iteration') else booster.num_boosted_rounds()
            except (TypeError, ImportError):
                # Fallback for older XGBoost versions - just train normally
                self.model.fit(X_train, y_train, verbose=False)
                best_iteration = self.n_estimators

            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Time-series cross-validation (use a fresh model without early stopping for CV)
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                tree_method="hist",
                device=self.device,
                random_state=self.random_state,
            )
            cv_scores = cross_val_score(cv_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
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
                'feature_importance': self.get_feature_importance(),
                'device': self.device,
                'best_iteration': best_iteration,
                'stopped_early': best_iteration < self.n_estimators,
            }

            logger.info(
                f"XGBoost trained (device={self.device}). Test R²: {test_r2:.4f}, CV MSE: {cv_mse:.6f}, Best iteration: {best_iteration}"
            )
            return metrics
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        try:
            # Apply feature selection if it was used during training
            if self.selected_features is not None:
                X = X[:, self.selected_features]
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        try:
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise

    def validate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained before validation")
        try:
            predictions = self.predict(X)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            direction_correct = np.sum(np.sign(y) == np.sign(predictions)) / len(y)
            mae = np.mean(np.abs(y - predictions))
            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': direction_correct,
                'mae': mae,
            }
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            raise

    def __str__(self):
        return (
            f"XGBoostPredictor(n_estimators={self.n_estimators}, device={self.device}, "
            f"trained={self.is_trained})"
        )


class EnsemblePredictor:
    """Simple ensemble combining XGBoost and RandomForest predictions."""
    
    def __init__(self, xgb_weight: float = 0.6, random_state: int = 42, device: str = "cpu"):
        """
        Initialize ensemble predictor.
        
        Args:
            xgb_weight: Weight for XGBoost (RF gets 1 - xgb_weight)
            random_state: Random seed
            device: Device for XGBoost
        """
        self.xgb = XGBoostPredictor(random_state=random_state, device=device)
        self.rf = RandomForestPredictor(random_state=random_state)
        self.xgb_weight = xgb_weight
        self.rf_weight = 1.0 - xgb_weight
        self.is_trained = False
        self.feature_names = []
    
    def create_features(self, data: pd.DataFrame, spy_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Delegate to XGBoost feature creation."""
        return self.xgb.create_features(data, spy_data)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to XGBoost feature preparation."""
        X, y = self.xgb.prepare_features(df)
        self.feature_names = self.xgb.feature_names
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train both models and return combined metrics."""
        xgb_metrics = self.xgb.train(X, y)
        rf_metrics = self.rf.train(X, y)
        self.is_trained = True
        
        return {
            'xgb_test_r2': xgb_metrics['test_r2'],
            'rf_test_r2': rf_metrics['test_r2'],
            'xgb_weight': self.xgb_weight,
            'ensemble_method': 'weighted_average'
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of both model predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        xgb_pred = self.xgb.predict(X)
        rf_pred = self.rf.predict(X)
        
        return self.xgb_weight * xgb_pred + self.rf_weight * rf_pred
    
    def validate_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Validate ensemble predictions."""
        predictions = self.predict(X)
        direction_correct = np.sum(np.sign(y) == np.sign(predictions)) / len(y)
        r2 = r2_score(y, predictions)
        
        return {
            'r2': r2,
            'directional_accuracy': direction_correct,
            'mae': np.mean(np.abs(y - predictions))
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from both models."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained to get feature importance")
        
        xgb_importance = self.xgb.get_feature_importance()
        rf_importance = self.rf.get_feature_importance()
        
        # Weight by ensemble weights
        combined = {}
        for feat in self.feature_names:
            xgb_val = xgb_importance.get(feat, 0.0)
            rf_val = rf_importance.get(feat, 0.0)
            combined[feat] = self.xgb_weight * xgb_val + self.rf_weight * rf_val
        
        # Sort by importance
        return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
    
    def __str__(self):
        return f"EnsemblePredictor(xgb_weight={self.xgb_weight}, trained={self.is_trained})"