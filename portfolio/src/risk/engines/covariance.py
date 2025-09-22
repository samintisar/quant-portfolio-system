from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def compute_sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance with Bessel correction (ddof=1)."""
    data = _coerce_returns(returns)
    return np.cov(data, rowvar=False, ddof=1)


def compute_ledoit_wolf_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Shrinkage covariance estimate via Ledoit-Wolf."""
    data = _coerce_returns(returns)
    model = LedoitWolf().fit(data)
    return model.covariance_


def compute_ewma_covariance(returns: pd.DataFrame, decay_lambda: float) -> np.ndarray:
    """Exponentially weighted moving average covariance matrix."""
    if not (0 < decay_lambda < 1):
        raise ValueError("decay_lambda must fall within (0, 1)")
    data = _coerce_returns(returns)
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    n_assets = centered.shape[1]
    covariance = np.zeros((n_assets, n_assets))
    for row in centered:
        covariance = decay_lambda * covariance + (1 - decay_lambda) * np.outer(row, row)
    return covariance


def is_positive_semidefinite(matrix: np.ndarray, atol: float = 1e-8) -> bool:
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    return bool(np.all(eigenvalues >= -abs(atol)))


def _coerce_returns(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be provided as a pandas DataFrame")
    data = returns.to_numpy(dtype=float, copy=True)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("returns must contain at least two observations")
    return data

