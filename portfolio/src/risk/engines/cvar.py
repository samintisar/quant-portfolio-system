from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm

from . import var as var_module


def historical_cvar(portfolio_returns: np.ndarray, confidence: float) -> float:
    var_module._validate_confidence(confidence)
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("portfolio_returns must be one-dimensional")
    if returns.size == 0:
        raise ValueError("portfolio_returns must not be empty")
    threshold = float(np.quantile(returns, 1 - confidence))
    tail = returns[returns <= threshold]
    if tail.size == 0:
        return 0.0
    return max(0.0, -float(np.mean(tail)))


def parametric_cvar(portfolio_returns: np.ndarray, confidence: float) -> float:
    var_module._validate_confidence(confidence)
    returns = np.asarray(portfolio_returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("portfolio_returns must be one-dimensional")
    if returns.size < 2:
        raise ValueError("portfolio_returns must contain at least two observations")
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return max(0.0, -mean)
    z_lower = norm.ppf(1 - confidence)
    es = -(mean + std * (norm.pdf(z_lower) / (1 - confidence)))
    return max(0.0, es)


def monte_carlo_cvar(
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: np.ndarray,
    confidence: float,
    paths: int,
    seed: Optional[int] = None,
) -> float:
    var_module._validate_confidence(confidence)
    if paths <= 0:
        raise ValueError("paths must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=mean_vector, cov=covariance_matrix, size=paths)
    portfolio_draws = draws @ weights
    threshold = float(np.quantile(portfolio_draws, 1 - confidence))
    tail = portfolio_draws[portfolio_draws <= threshold]
    if tail.size == 0:
        return 0.0
    return max(0.0, -float(np.mean(tail)))

