"""
Unit tests for the ML prediction module.
"""

import pytest
import pandas as pd
import numpy as np
from portfolio.ml.predictor import RandomForestPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRandomForestPredictor:
    """Test cases for RandomForestPredictor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        n_samples = 100

        # Generate sample price data
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        base_price = 100

        data = {
            'Open': base_price + np.random.normal(0, 1, n_samples),
            'High': base_price + np.random.normal(1, 1, n_samples),
            'Low': base_price + np.random.normal(-1, 1, n_samples),
            'Close': base_price + np.random.normal(0, 1, n_samples),
            'Adj Close': base_price + np.random.normal(0, 1, n_samples),
            'Volume': np.random.randint(1000000, 10000000, n_samples)
        }

        # Ensure High >= Low and High >= Open, Close
        data['High'] = np.maximum(
            data['High'],
            np.maximum(data['Open'], np.maximum(data['Close'], data['Low']))
        )
        data['Low'] = np.minimum(
            data['Low'],
            np.minimum(data['Open'], np.minimum(data['Close'], data['High']))
        )

        df = pd.DataFrame(data, index=dates)
        return df

    @pytest.fixture
    def predictor(self):
        """Create a RandomForestPredictor instance for testing."""
        return RandomForestPredictor(n_estimators=10, random_state=42)

    def test_predictor_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.n_estimators == 10
        assert predictor.random_state == 42
        assert not predictor.is_trained
        assert predictor.feature_names == []

    def test_create_features(self, predictor, sample_data):
        """Test feature creation."""
        features_df = predictor.create_features(sample_data)

        # Check that features are created
        assert len(features_df) > 0
        assert 'target' in features_df.columns
        assert 'ma_5' in features_df.columns
        assert 'volatility_5' in features_df.columns
        assert 'momentum_5' in features_df.columns
        assert 'rsi_14' in features_df.columns
        assert 'macd_line' in features_df.columns
        assert 'macd_signal' in features_df.columns
        assert 'macd_histogram' in features_df.columns

        # Check that NaN values are removed
        assert not features_df.isnull().any().any()

        # Check RSI values are in valid range [0, 100]
        assert features_df['rsi_14'].min() >= 0
        assert features_df['rsi_14'].max() <= 100

        # Check MACD components have correct relationship
        assert (features_df['macd_histogram'] == features_df['macd_line'] - features_df['macd_signal']).all()

    def test_prepare_features(self, predictor, sample_data):
        """Test feature preparation."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        # Check shapes
        assert X.shape[0] == len(features_df)
        assert y.shape[0] == len(features_df)
        assert len(predictor.feature_names) > 0

    def test_train_model(self, predictor, sample_data):
        """Test model training."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        metrics = predictor.train(X, y)

        # Check that model is trained
        assert predictor.is_trained

        # Check that metrics are returned
        assert 'train_mse' in metrics
        assert 'test_mse' in metrics
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert 'cv_mse' in metrics
        assert 'feature_importance' in metrics

        # Check metric values are reasonable
        assert metrics['train_mse'] >= 0
        assert metrics['test_mse'] >= 0
        assert -1 <= metrics['test_r2'] <= 1

    def test_predict_before_training(self, predictor, sample_data):
        """Test that prediction fails before training."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            predictor.predict(X)

    def test_predict_after_training(self, predictor, sample_data):
        """Test prediction after training."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        # Train model
        predictor.train(X, y)

        # Make predictions
        predictions = predictor.predict(X[:5])

        # Check predictions
        assert len(predictions) == 5
        assert all(isinstance(pred, (int, float)) for pred in predictions)

    def test_get_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        # Train model
        predictor.train(X, y)

        # Get feature importance
        importance = predictor.get_feature_importance()

        # Check importance dictionary
        assert isinstance(importance, dict)
        assert len(importance) == len(predictor.feature_names)
        assert all(isinstance(v, (int, float)) for v in importance.values())

        # Check that values are sorted
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)

    def test_validate_model(self, predictor, sample_data):
        """Test model validation."""
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)

        # Train model
        predictor.train(X, y)

        # Validate model
        validation_metrics = predictor.validate_model(X, y)

        # Check validation metrics
        assert 'mse' in validation_metrics
        assert 'rmse' in validation_metrics
        assert 'r2' in validation_metrics
        assert 'directional_accuracy' in validation_metrics
        assert 'mae' in validation_metrics

        # Check metric values are reasonable
        assert validation_metrics['mse'] >= 0
        assert validation_metrics['rmse'] >= 0
        assert -1 <= validation_metrics['r2'] <= 1
        assert 0 <= validation_metrics['directional_accuracy'] <= 1
        assert validation_metrics['mae'] >= 0

    def test_string_representation(self, predictor, sample_data):
        """Test string representation of predictor."""
        str_repr = str(predictor)
        assert "RandomForestPredictor" in str_repr
        assert "n_estimators=10" in str_repr
        assert "trained=False" in str_repr

        # After training
        features_df = predictor.create_features(sample_data)
        X, y = predictor.prepare_features(features_df)
        predictor.train(X, y)

        str_repr = str(predictor)
        assert "trained=True" in str_repr

    def test_empty_data_handling(self, predictor):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()

        with pytest.raises(Exception):
            predictor.create_features(empty_df)

    def test_small_dataset_handling(self, predictor):
        """Test handling of very small datasets."""
        # Create minimal dataset
        small_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Adj Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start='2020-01-01', periods=3, freq='D'))

        # This should work but result in very few features
        features_df = predictor.create_features(small_data)
        # With moving windows of 20, most features will be NaN and dropped
        # So we expect the result to be empty or very small
        assert len(features_df) <= len(small_data)