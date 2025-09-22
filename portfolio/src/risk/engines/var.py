from __future__ import annotations

from typing import Optional

import numpy as np


def historical_var(portfolio_returns: np.ndarray, confidence: float) -> float:
    _validate_confidence(confidence)
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("portfolio_returns must be one-dimensional")
    if returns.size == 0:
        raise ValueError("portfolio_returns must not be empty")
    quantile = float(np.quantile(returns, 1 - confidence))
    return max(0.0, -quantile)


def parametric_var(portfolio_returns: np.ndarray, confidence: float) -> float:
    _validate_confidence(confidence)
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("portfolio_returns must be one-dimensional")
    if returns.size < 2:
        raise ValueError("portfolio_returns must contain at least two observations")
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return max(0.0, -mean)
    from scipy.stats import norm

    z_score = norm.ppf(1 - confidence)
    value = -(mean + std * z_score)
    return max(0.0, value)


def monte_carlo_var(
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float,
    paths: int,
    seed: Optional[int] = None,
) -> float:
    _validate_confidence(confidence)
    if paths <= 0:
        raise ValueError("paths must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=mean_vector, cov=covariance_matrix, size=paths)
    portfolio_draws = draws @ weights
    quantile = float(np.quantile(portfolio_draws, 1 - confidence))
    return max(0.0, -quantile)


def _validate_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

