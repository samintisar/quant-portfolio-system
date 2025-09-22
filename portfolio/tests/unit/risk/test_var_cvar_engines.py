from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from scipy.stats import norm

from portfolio.src.risk.engines import cvar, var


@pytest.fixture()
def portfolio_returns(unit_returns_df, unit_weights) -> np.ndarray:
    weights = unit_weights.reindex(unit_returns_df.columns).to_numpy()
    return unit_returns_df.to_numpy() @ weights


def test_var_engines_calculate_expected_losses(portfolio_returns) -> None:
    confidence = 0.95
    historical = var.historical_var(portfolio_returns, confidence)
    parametric = var.parametric_var(portfolio_returns, confidence)

    quantile = np.quantile(portfolio_returns, 1 - confidence)
    expected_historical = max(0.0, -quantile)
    mean = float(np.mean(portfolio_returns))
    std = float(np.std(portfolio_returns, ddof=1))
    z_lower = norm.ppf(1 - confidence)
    expected_parametric = max(0.0, -(mean + std * z_lower))

    assert historical == pytest.approx(expected_historical, rel=1e-6)
    assert parametric == pytest.approx(expected_parametric, rel=1e-6)


def test_monte_carlo_var_is_seed_deterministic(unit_returns_df, unit_weights) -> None:
    confidence = 0.99
    cov = np.cov(unit_returns_df.to_numpy(), rowvar=False, ddof=1)
    mean = unit_returns_df.to_numpy().mean(axis=0)
    weights = unit_weights.reindex(unit_returns_df.columns).to_numpy()

    first = var.monte_carlo_var(mean, cov, weights, confidence, paths=2000, seed=42)
    second = var.monte_carlo_var(mean, cov, weights, confidence, paths=2000, seed=42)
    third = var.monte_carlo_var(mean, cov, weights, confidence, paths=2000, seed=99)

    assert first == pytest.approx(second, rel=1e-9)
    assert third != pytest.approx(first, rel=1e-6)


def test_cvar_exceeds_var(portfolio_returns, unit_returns_df, unit_weights) -> None:
    confidence = 0.95
    var_value = var.historical_var(portfolio_returns, confidence)
    cvar_value = cvar.historical_cvar(portfolio_returns, confidence)
    assert cvar_value >= var_value

    cov = np.cov(unit_returns_df.to_numpy(), rowvar=False, ddof=1)
    mean = unit_returns_df.to_numpy().mean(axis=0)
    weights = unit_weights.reindex(unit_returns_df.columns).to_numpy()
    var_mc = var.monte_carlo_var(mean, cov, weights, confidence, paths=5000, seed=7)
    cvar_mc = cvar.monte_carlo_cvar(mean, cov, weights, confidence, paths=5000, seed=7)
    assert cvar_mc >= var_mc
