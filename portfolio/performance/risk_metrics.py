"""
Simple risk metrics calculation for portfolio analysis.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats

from portfolio.logging_config import get_logger, ValidationError

logger = get_logger(__name__)


class RiskMetricsCalculator:
    """Simple risk metrics calculator for portfolios."""

    def __init__(self):
        """Initialize the risk metrics calculator."""
        self.trading_days_per_year = 252
        logger.info("Initialized RiskMetricsCalculator")

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate historical Value at Risk (VaR)."""
        if returns.empty:
            raise ValidationError("Returns series is empty")

        if not (0 < confidence_level < 1):
            raise ValidationError("Confidence level must be between 0 and 1")

        var_percentile = (1 - confidence_level) * 100
        var = -np.percentile(returns, var_percentile)
        return var

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate historical Conditional Value at Risk (CVaR)."""
        if returns.empty:
            raise ValidationError("Returns series is empty")

        if not (0 < confidence_level < 1):
            raise ValidationError("Confidence level must be between 0 and 1")

        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)
        tail_returns = returns[returns <= var_threshold]
        cvar = -tail_returns.mean() if len(tail_returns) > 0 else 0.0
        return cvar

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility."""
        if returns.empty:
            raise ValidationError("Returns series is empty")

        volatility = returns.std()
        if annualize:
            volatility = volatility * np.sqrt(self.trading_days_per_year)
        return volatility

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if prices.empty:
            raise ValidationError("Prices series is empty")

        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark."""
        if returns.empty or benchmark_returns.empty:
            raise ValidationError("Returns series is empty")

        # Align data
        aligned_returns = returns.align(benchmark_returns, join='inner')
        returns_aligned = aligned_returns[0]
        benchmark_aligned = aligned_returns[1]

        if len(returns_aligned) < 2:
            raise ValidationError("Insufficient data for beta calculation")

        # Calculate covariance and variance
        covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def calculate_correlation_matrix(self, returns_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for multiple assets."""
        if returns_matrix.empty:
            raise ValidationError("Returns matrix is empty")

        corr_matrix = returns_matrix.corr()

        # Convert to nested dictionary
        result = {}
        for col in corr_matrix.columns:
            result[col] = {}
            for row in corr_matrix.index:
                result[col][row] = corr_matrix.loc[row, col]

        return result

    def calculate_all_metrics(self, returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None,
                            prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate all available risk metrics."""
        if returns.empty:
            raise ValidationError("Returns series is empty")

        results = {}

        try:
            # VaR and CVaR at different confidence levels
            for confidence in [0.90, 0.95, 0.99]:
                results[f'var_{int(confidence*100)}'] = self.calculate_var(returns, confidence)
                results[f'cvar_{int(confidence*100)}'] = self.calculate_cvar(returns, confidence)

            # Volatility
            results['volatility'] = self.calculate_volatility(returns)

            # Max drawdown (if prices provided or calculated from returns)
            if prices is None:
                prices = (1 + returns).cumprod()
            results['max_drawdown'] = self.calculate_max_drawdown(prices)

            # Beta (if benchmark provided)
            if benchmark_returns is not None:
                results['beta'] = self.calculate_beta(returns, benchmark_returns)

            logger.info("Calculated all risk metrics")

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            raise ValidationError(f"Risk metrics calculation failed: {e}")

        return results