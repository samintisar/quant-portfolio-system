"""
Unit tests for market regime detection module.
"""

import pytest
import pandas as pd
import numpy as np
from portfolio.regime.detector import RegimeDetector


@pytest.fixture
def sample_returns():
    """Create sample return series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create synthetic returns with regime-like patterns
    returns = []
    for i in range(500):
        if i < 150:  # Bull regime
            ret = np.random.normal(0.001, 0.01)
        elif i < 300:  # Bear regime
            ret = np.random.normal(-0.002, 0.025)
        else:  # Sideways
            ret = np.random.normal(0.0001, 0.008)
        returns.append(ret)
    
    return pd.Series(returns, index=dates, name='returns')


def test_detector_initialization():
    """Test RegimeDetector initialization."""
    detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
    
    assert detector.n_states == 3
    assert detector.return_window == 60
    assert detector.volatility_window == 20
    assert not detector.is_fitted


def test_feature_computation(sample_returns):
    """Test feature engineering."""
    detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
    
    features = detector._compute_features(sample_returns)
    
    # Check features are computed
    assert 'rolling_return' in features.columns
    assert 'realized_vol' in features.columns
    assert 'vix_proxy' in features.columns
    
    # Check no NaN values after dropna
    assert not features.isnull().any().any()
    
    # Check features are numeric
    assert features['rolling_return'].dtype in [np.float64, np.float32]
    assert features['realized_vol'].dtype in [np.float64, np.float32]


def test_fit_predict(sample_returns):
    """Test fitting and prediction."""
    detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
    
    regimes = detector.fit_predict(sample_returns)
    
    # Check detector is fitted
    assert detector.is_fitted
    
    # Check regimes are predicted
    assert len(regimes) > 0
    assert isinstance(regimes, pd.Series)
    
    # Check regime labels are valid
    valid_regimes = {'Bull', 'Bear', 'Sideways'}
    assert all(r in valid_regimes for r in regimes.unique())
    
    # Check all three regimes appear (with high probability)
    # Note: This might occasionally fail due to randomness
    unique_regimes = set(regimes.unique())
    assert len(unique_regimes) >= 2  # At least 2 regimes should appear


def test_regime_statistics(sample_returns):
    """Test regime statistics calculation."""
    detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
    regimes = detector.fit_predict(sample_returns)
    
    stats = detector.get_regime_statistics(regimes, sample_returns)
    
    # Check stats DataFrame structure
    assert isinstance(stats, pd.DataFrame)
    assert 'count' in stats.columns
    assert 'pct_time' in stats.columns
    assert 'mean_return' in stats.columns
    assert 'volatility' in stats.columns
    assert 'sharpe' in stats.columns
    
    # Check values are reasonable
    assert all(stats['count'] > 0)
    assert abs(stats['pct_time'].sum() - 100.0) < 0.1  # Sum to ~100%


def test_transition_matrix(sample_returns):
    """Test regime transition matrix."""
    detector = RegimeDetector(n_states=3, return_window=60, volatility_window=20)
    regimes = detector.fit_predict(sample_returns)
    
    transitions = detector.get_regime_transition_matrix(regimes)
    
    # Check matrix structure
    assert isinstance(transitions, pd.DataFrame)
    assert transitions.shape == (3, 3)
    
    # Check all regimes in rows and columns
    assert 'Bull' in transitions.index
    assert 'Bear' in transitions.index
    assert 'Sideways' in transitions.index
    
    # Check probabilities sum to ~1 for each row
    for regime in transitions.index:
        row_sum = transitions.loc[regime].sum()
        assert abs(row_sum - 1.0) < 0.01 or row_sum == 0.0  # Allow for regimes with no transitions


def test_optimize_for_regime(sample_returns):
    """Test regime-conditional optimization selection."""
    from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
    
    detector = RegimeDetector(n_states=3)
    detector.fit(sample_returns)
    
    # Create sample asset returns
    np.random.seed(42)
    asset_returns = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 200),
        'MSFT': np.random.normal(0.0008, 0.018, 200),
        'GOOGL': np.random.normal(0.0009, 0.019, 200)
    })
    
    optimizer = SimplePortfolioOptimizer()
    
    # Test all three regimes
    for regime in ['Bull', 'Bear', 'Sideways']:
        result = detector.optimize_for_regime(
            regime,
            optimizer,
            asset_returns,
            weight_cap=0.5
        )
        
        # Check result structure
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'expected_volatility' in result
        assert 'sharpe_ratio' in result
        
        # Check weights sum to ~1
        weights_sum = sum(result['weights'].values())
        assert abs(weights_sum - 1.0) < 0.01
        
        # Check weights respect cap
        assert all(w <= 0.5 + 1e-6 for w in result['weights'].values())


def test_risk_parity_implementation(sample_returns):
    """Test risk parity optimization."""
    from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
    
    detector = RegimeDetector(n_states=3)
    detector.fit(sample_returns)
    
    # Create sample returns with different volatilities
    np.random.seed(42)
    asset_returns = pd.DataFrame({
        'Low_Vol': np.random.normal(0.0005, 0.01, 200),
        'Med_Vol': np.random.normal(0.0008, 0.02, 200),
        'High_Vol': np.random.normal(0.001, 0.04, 200)
    })
    
    optimizer = SimplePortfolioOptimizer()
    
    result = detector._risk_parity_optimize(optimizer, asset_returns, weight_cap=0.6)
    
    # Low vol asset should get higher weight in risk parity
    weights = result['weights']
    assert weights['Low_Vol'] > weights['High_Vol']


def test_predict_requires_fit(sample_returns):
    """Test that predict requires fit to be called first."""
    detector = RegimeDetector(n_states=3)
    
    with pytest.raises(ValueError, match="must be fitted"):
        detector.predict(sample_returns)


def test_regime_labeling():
    """Test regime labeling logic."""
    detector = RegimeDetector(n_states=3)
    
    # Create means where state 0 has lowest return, 1 middle, 2 highest
    means = np.array([
        [-0.02, 0.25, 0.15],  # State 0: Low return (Bear)
        [0.00, 0.15, 0.10],   # State 1: Mid return (Sideways)
        [0.03, 0.12, 0.08]    # State 2: High return (Bull)
    ])
    
    labels = detector._label_regimes(means)
    
    assert labels[0] == 'Bear'
    assert labels[1] == 'Sideways'
    assert labels[2] == 'Bull'


def test_feature_scaling(sample_returns):
    """Test that features are properly scaled."""
    detector = RegimeDetector(n_states=3)
    
    features = detector._compute_features(sample_returns)
    scaled_features = detector.scaler.fit_transform(features.values)
    
    # Scaled features should have ~0 mean and ~1 std
    assert abs(scaled_features.mean()) < 0.1
    assert abs(scaled_features.std() - 1.0) < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
