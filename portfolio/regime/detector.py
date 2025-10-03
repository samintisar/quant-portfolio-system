"""
Market regime detection using Hidden Markov Models.

This module implements a 3-state HMM to classify market conditions:
- Bull: High returns, low volatility (growth-favorable)
- Bear: Negative returns, high volatility (defensive positioning)
- Sideways: Low returns, low volatility (balanced approach)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.
    
    Features used for classification:
    - Rolling 60-day SPY returns (momentum signal)
    - 20-day realized volatility (risk measure)
    - VIX proxy (rolling standard deviation)
    """
    
    def __init__(
        self,
        n_states: int = 3,
        return_window: int = 60,
        volatility_window: int = 20,
        random_state: int = 42
    ):
        """
        Initialize the regime detector.
        
        Args:
            n_states: Number of market regimes (default: 3 for Bull/Bear/Sideways)
            return_window: Window for rolling returns (default: 60 days)
            volatility_window: Window for volatility calculation (default: 20 days)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.return_window = return_window
        self.volatility_window = volatility_window
        self.random_state = random_state
        
        # Initialize HMM with Gaussian emissions
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_labels = {}  # Maps state indices to regime names
        
    def _compute_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Compute regime detection features from return series.
        
        Args:
            returns: Daily return series (preferably benchmark like SPY)
            
        Returns:
            DataFrame with features: rolling_return, realized_vol, vix_proxy
        """
        features = pd.DataFrame(index=returns.index)
        
        # Feature 1: Rolling N-day cumulative return
        features['rolling_return'] = returns.rolling(window=self.return_window).sum()
        
        # Feature 2: Realized volatility (annualized)
        features['realized_vol'] = returns.rolling(
            window=self.volatility_window
        ).std() * np.sqrt(252)
        
        # Feature 3: VIX proxy - rolling std of rolling volatility (vol-of-vol)
        features['vix_proxy'] = features['realized_vol'].rolling(
            window=self.volatility_window
        ).std()
        
        # Drop NaN rows created by rolling windows
        features = features.dropna()
        
        return features
    
    def _label_regimes(self, means: np.ndarray) -> Dict[int, str]:
        """
        Label HMM states as Bull/Bear/Sideways based on feature means.
        
        The labeling logic:
        - Bull: Highest rolling return mean
        - Bear: Lowest rolling return mean
        - Sideways: Middle rolling return mean
        
        Args:
            means: HMM state means (shape: n_states x n_features)
            
        Returns:
            Dictionary mapping state index to regime name
        """
        # Extract rolling return means (first feature)
        return_means = means[:, 0]
        
        # Sort states by return mean
        sorted_indices = np.argsort(return_means)
        
        labels = {}
        labels[sorted_indices[0]] = 'Bear'     # Lowest returns
        labels[sorted_indices[-1]] = 'Bull'    # Highest returns
        
        # Middle state is Sideways (handle both 3 and >3 states)
        for idx in sorted_indices[1:-1]:
            labels[idx] = 'Sideways'
            
        return labels
    
    def fit(self, benchmark_returns: pd.Series) -> 'RegimeDetector':
        """
        Fit the HMM to historical benchmark returns.
        
        Args:
            benchmark_returns: Daily return series (e.g., SPY)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting regime detector with {len(benchmark_returns)} observations")
        
        # Compute features
        features = self._compute_features(benchmark_returns)
        logger.info(f"Computed features: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features.values)
        
        # Fit HMM
        self.model.fit(features_scaled)
        
        # Label regimes based on learned means
        self.regime_labels = self._label_regimes(self.model.means_)
        logger.info(f"Regime labels: {self.regime_labels}")
        
        self.is_fitted = True
        return self
    
    def predict(self, benchmark_returns: pd.Series) -> pd.Series:
        """
        Predict regime states for given return series.
        
        Args:
            benchmark_returns: Daily return series
            
        Returns:
            Series with regime labels (Bull/Bear/Sideways)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Compute features
        features = self._compute_features(benchmark_returns)
        
        # Standardize using fitted scaler
        features_scaled = self.scaler.transform(features.values)
        
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Convert state indices to regime labels
        regimes = pd.Series(
            [self.regime_labels[s] for s in states],
            index=features.index,
            name='regime'
        )
        
        return regimes
    
    def fit_predict(self, benchmark_returns: pd.Series) -> pd.Series:
        """
        Fit the model and predict regimes in one step.
        
        Args:
            benchmark_returns: Daily return series
            
        Returns:
            Series with regime labels
        """
        self.fit(benchmark_returns)
        return self.predict(benchmark_returns)
    
    def get_regime_statistics(
        self,
        regimes: pd.Series,
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Compute descriptive statistics for each detected regime.
        
        Args:
            regimes: Series with regime labels
            returns: Corresponding return series
            
        Returns:
            DataFrame with statistics per regime
        """
        aligned_returns = returns.loc[regimes.index]
        
        stats = []
        for regime in ['Bull', 'Bear', 'Sideways']:
            mask = regimes == regime
            regime_returns = aligned_returns[mask]
            
            if len(regime_returns) > 0:
                stats.append({
                    'regime': regime,
                    'count': len(regime_returns),
                    'pct_time': len(regime_returns) / len(regimes) * 100,
                    'mean_return': regime_returns.mean() * 252,  # Annualized
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe': (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252) + 1e-8)
                })
        
        return pd.DataFrame(stats).set_index('regime')
    
    def get_regime_transition_matrix(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Compute transition probabilities between regimes.
        
        Args:
            regimes: Series with regime labels
            
        Returns:
            DataFrame with transition probabilities
        """
        regime_names = ['Bull', 'Bear', 'Sideways']
        transitions = pd.DataFrame(
            0.0,
            index=regime_names,
            columns=regime_names
        )
        
        # Count transitions
        for i in range(len(regimes) - 1):
            current = regimes.iloc[i]
            next_regime = regimes.iloc[i + 1]
            transitions.loc[current, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1)
        for regime in regime_names:
            if row_sums[regime] > 0:
                transitions.loc[regime] = transitions.loc[regime] / row_sums[regime]
        
        return transitions
    
    def optimize_for_regime(
        self,
        regime: str,
        optimizer,
        returns: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """
        Select and run appropriate optimization method based on regime.
        
        Regime-conditional strategy:
        - Bull: Black-Litterman (growth-tilted with market views)
        - Bear: CVaR (tail risk minimization)
        - Sideways: Risk Parity (balanced risk allocation)
        
        Args:
            regime: Current market regime ('Bull', 'Bear', or 'Sideways')
            optimizer: Portfolio optimizer instance
            returns: Asset return DataFrame
            **kwargs: Additional parameters for optimization
            
        Returns:
            Dictionary with optimization results
        """
        weight_cap = kwargs.get('weight_cap', 0.20)
        
        if regime == 'Bull':
            # Bull market: Use Black-Litterman for growth tilt
            logger.info("Bull regime detected → Using Black-Litterman optimization")
            return optimizer.black_litterman_optimize(
                returns,
                tau=0.05,
                weight_cap=weight_cap,
                risk_aversion=2.0  # Lower risk aversion in bull markets
            )
            
        elif regime == 'Bear':
            # Bear market: Use CVaR for downside protection
            logger.info("Bear regime detected → Using CVaR optimization")
            return optimizer.cvar_optimize(
                returns,
                alpha=0.05,
                weight_cap=weight_cap
            )
            
        else:  # Sideways
            # Sideways market: Use Risk Parity for balance
            logger.info("Sideways regime detected → Using Risk Parity")
            return self._risk_parity_optimize(optimizer, returns, weight_cap)
    
    def _risk_parity_optimize(
        self,
        optimizer,
        returns: pd.DataFrame,
        weight_cap: float
    ) -> Dict[str, float]:
        """
        Simple risk parity implementation using inverse volatility weighting.
        
        Args:
            optimizer: Portfolio optimizer instance
            returns: Asset return DataFrame
            weight_cap: Maximum weight per asset
            
        Returns:
            Dictionary with optimization results
        """
        # Calculate annualized volatilities
        vols = returns.std() * np.sqrt(252)
        
        # Inverse volatility weights
        inv_vols = 1.0 / (vols + 1e-8)
        weights = inv_vols / inv_vols.sum()
        
        # Apply weight cap
        weights = np.minimum(weights.values, weight_cap)
        weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        exp_return = float(weights @ mean_returns.values)
        exp_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
        sharpe = (exp_return - optimizer.risk_free_rate) / (exp_vol + 1e-8)
        
        return {
            'weights': dict(zip(returns.columns, weights)),
            'expected_return': exp_return,
            'expected_volatility': exp_vol,
            'sharpe_ratio': sharpe
        }
