from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot


_UNIT_RETURNS = [
    [0.010, 0.020],
    [0.015, 0.018],
    [-0.005, 0.011],
    [0.020, -0.004],
]
_UNIT_ASSETS = ["EQ1", "EQ2"]
_UNIT_WEIGHTS = {"EQ1": 0.6, "EQ2": 0.4}

_INTEGRATION_RETURNS = [
    [0.012, 0.018, 0.004],
    [-0.007, 0.004, 0.003],
    [0.009, -0.003, 0.002],
    [0.011, 0.020, 0.001],
    [-0.004, 0.005, 0.002],
    [0.015, 0.012, 0.004],
]
_INTEGRATION_ASSETS = ["EQ1", "EQ2", "BOND"]
_INTEGRATION_WEIGHTS = {"EQ1": 0.4, "EQ2": 0.35, "BOND": 0.25}

_FACTOR_EXPOSURES = pd.DataFrame(
    {
        "VALUE": [0.8, 0.2, 0.1],
        "SIZE": [0.3, 0.6, -0.1],
        "MOMENTUM": [0.4, 0.5, 0.2],
    },
    index=_INTEGRATION_ASSETS,
).T

# Extended datasets for regression testing
_REGRESSION_RETURNS = []
_REGRESSION_ASSETS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "JPM", "BAC", "XOM", "CVX"]

# Generate 252 days of returns (1 year) with realistic patterns
np.random.seed(42)
base_returns = np.random.normal(0.001, 0.02, 252)  # Market returns
for i, asset in enumerate(_REGRESSION_ASSETS):
    beta = 0.8 + (i % 3) * 0.2  # Different betas
    alpha = (i % 5) * 0.0005  # Small alpha
    idiosyncratic = np.random.normal(0, 0.015, 252)
    asset_returns = alpha + beta * base_returns + idiosyncratic
    _REGRESSION_RETURNS.append(asset_returns)

_REGRESSION_RETURNS = np.array(_REGRESSION_RETURNS).T
_REGRESSION_WEIGHTS = {asset: 0.1 for asset in _REGRESSION_ASSETS}  # Equal weights

# Stress test dataset with extreme events
_STRESS_RETURNS = np.array([
    # Normal period
    [0.01, 0.015, 0.008, 0.012, 0.009, 0.02, 0.005, 0.006, 0.003, 0.004],
    # Market crash
    [-0.08, -0.12, -0.10, -0.15, -0.18, -0.20, -0.05, -0.06, -0.03, -0.04],
    # Volatility spike
    [-0.05, 0.08, -0.12, 0.15, -0.10, 0.18, -0.02, 0.03, -0.08, 0.12],
    # Recovery
    [0.12, 0.18, 0.15, 0.22, 0.25, 0.30, 0.08, 0.09, 0.06, 0.07],
    # New normal
    [0.005, 0.008, 0.004, 0.006, 0.003, 0.01, 0.002, 0.003, 0.001, 0.002],
])
_STRESS_ASSETS = ["TECH1", "TECH2", "TECH3", "TECH4", "TECH5", "GROWTH1", "GROWTH2", "FIN1", "FIN2", "ENERGY1"]
_STRESS_WEIGHTS = {
    "TECH1": 0.15, "TECH2": 0.12, "TECH3": 0.10, "TECH4": 0.08, "TECH5": 0.05,
    "GROWTH1": 0.15, "GROWTH2": 0.10, "FIN1": 0.10, "FIN2": 0.08, "ENERGY1": 0.07
}

# Large portfolio dataset for performance testing
_LARGE_PORTFOLIO_SIZE = 500
_LARGE_RETURNS = np.random.RandomState(123).normal(0.001, 0.02, (63, _LARGE_PORTFOLIO_SIZE))  # Quarter of data
_LARGE_ASSETS = [f"ASSET_{i:03d}" for i in range(_LARGE_PORTFOLIO_SIZE)]
_LARGE_WEIGHTS = {asset: 1.0 / _LARGE_PORTFOLIO_SIZE for asset in _LARGE_ASSETS}


def load_unit_returns() -> pd.DataFrame:
    """Return a deterministic two-asset return matrix for unit tests."""

    return pd.DataFrame(_UNIT_RETURNS, columns=_UNIT_ASSETS)


def load_unit_weights() -> pd.Series:
    """Return weights aligned to the unit return matrix."""

    return pd.Series(_UNIT_WEIGHTS)


def load_integration_returns() -> pd.DataFrame:
    """Return a deterministic three-asset return matrix used by integration suites."""

    return pd.DataFrame(_INTEGRATION_RETURNS, columns=_INTEGRATION_ASSETS)


def load_integration_weights() -> pd.Series:
    """Return weights aligned to the integration dataset."""

    return pd.Series(_INTEGRATION_WEIGHTS)


def load_factor_exposures() -> pd.DataFrame:
    """Provide synthetic factor exposures for stress and visualization tests."""

    return _FACTOR_EXPOSURES.copy()


def build_demo_configuration(**overrides: Any) -> RiskConfiguration:
    """Construct a `RiskConfiguration` seeded with defaults used across tests."""

    base: Dict[str, Any] = {
        "confidence_levels": [0.95, 0.99],
        "horizons": [1, 10],
        "decay_lambda": 0.94,
        "mc_paths": 5000,
        "seed": 123,
        "stress_scenarios": ["macro_recession", "inflation_spike"],
        "data_frequency": "daily",
        "reports": ["covariance", "var", "cvar", "stress", "visualizations"],
        "covariance_methods": ["sample", "ledoit_wolf", "ewma"],
        "var_methods": ["historical", "parametric", "monte_carlo"],
        "cvar_methods": ["historical", "parametric", "monte_carlo"],
        "reports_path": Path("data/storage/reports"),
        "logging_config": Path("config/logging/risk_logging.yaml"),
    }
    base.update(overrides)
    return RiskConfiguration(**base)


def build_demo_snapshot(include_factors: bool = False) -> PortfolioSnapshot:
    """Construct a `PortfolioSnapshot` for regression-style tests."""

    returns = load_integration_returns()
    weights = load_integration_weights()
    factor_exposures = load_factor_exposures() if include_factors else None

    return PortfolioSnapshot(
        asset_ids=list(returns.columns),
        returns=returns,
        weights=weights,
        factor_exposures=factor_exposures,
        timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
    )


def load_regression_returns() -> pd.DataFrame:
    """Return a realistic 10-asset, 252-day return series for regression testing."""
    return pd.DataFrame(_REGRESSION_RETURNS, columns=_REGRESSION_ASSETS)


def load_regression_weights() -> pd.Series:
    """Return equal weights for the regression dataset."""
    return pd.Series(_REGRESSION_WEIGHTS)


def load_stress_returns() -> pd.DataFrame:
    """Return a dataset with extreme market events for stress testing validation."""
    return pd.DataFrame(_STRESS_RETURNS, columns=_STRESS_ASSETS)


def load_stress_weights() -> pd.Series:
    """Return sector-concentrated weights for stress testing."""
    return pd.Series(_STRESS_WEIGHTS)


def load_large_portfolio_returns() -> pd.DataFrame:
    """Return a 500-asset portfolio for performance testing."""
    return pd.DataFrame(_LARGE_RETURNS, columns=_LARGE_ASSETS)


def load_large_portfolio_weights() -> pd.Series:
    """Return equal weights for the large portfolio dataset."""
    return pd.Series(_LARGE_WEIGHTS)


def build_regression_snapshot(include_factors: bool = False) -> PortfolioSnapshot:
    """Build a snapshot with realistic market data for regression testing."""
    returns = load_regression_returns()
    weights = load_regression_weights()
    factor_exposures = load_factor_exposures() if include_factors else None

    return PortfolioSnapshot(
        asset_ids=list(returns.columns),
        returns=returns,
        weights=weights,
        factor_exposures=factor_exposures,
        timestamp=pd.Timestamp("2024-12-31T00:00:00Z"),
    )


def build_stress_snapshot(include_factors: bool = False) -> PortfolioSnapshot:
    """Build a snapshot with extreme market events for stress testing."""
    returns = load_stress_returns()
    weights = load_stress_weights()

    # Create appropriate factor exposures for stress scenario
    if include_factors:
        factor_exposures = pd.DataFrame(
            {
                "MARKET": [1.2, 1.1, 1.3, 1.4, 1.5, 1.8, 1.6, 0.8, 0.7, 0.6],
                "VOLATILITY": [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.3, 0.5, 0.4, 0.3],
                "LIQUIDITY": [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.1, 0.9, 0.8, 0.7],
            },
            index=_STRESS_ASSETS,
        ).T
    else:
        factor_exposures = None

    return PortfolioSnapshot(
        asset_ids=list(returns.columns),
        returns=returns,
        weights=weights,
        factor_exposures=factor_exposures,
        timestamp=pd.Timestamp("2024-03-15T00:00:00Z"),
    )


def build_large_portfolio_snapshot() -> PortfolioSnapshot:
    """Build a large portfolio snapshot for performance testing."""
    returns = load_large_portfolio_returns()
    weights = load_large_portfolio_weights()

    return PortfolioSnapshot(
        asset_ids=list(returns.columns),
        returns=returns,
        weights=weights,
        factor_exposures=None,
        timestamp=pd.Timestamp("2024-09-30T00:00:00Z"),
    )


def build_performance_configuration() -> RiskConfiguration:
    """Build a configuration optimized for performance testing."""
    return build_demo_configuration(
        mc_paths=1000,  # Reduced for faster testing
        stress_scenarios=["quick_test"],
        reports=["covariance", "var"],  # Minimal reports
    )


__all__ = [
    "load_unit_returns",
    "load_unit_weights",
    "load_integration_returns",
    "load_integration_weights",
    "load_factor_exposures",
    "build_demo_configuration",
    "build_demo_snapshot",
    "load_regression_returns",
    "load_regression_weights",
    "load_stress_returns",
    "load_stress_weights",
    "load_large_portfolio_returns",
    "load_large_portfolio_weights",
    "build_regression_snapshot",
    "build_stress_snapshot",
    "build_large_portfolio_snapshot",
    "build_performance_configuration",
]

