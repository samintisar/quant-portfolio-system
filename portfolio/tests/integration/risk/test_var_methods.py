from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from scipy.stats import norm

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.src.risk.services.report_builder import RiskReportBuilder


def _portfolio_returns(snapshot: PortfolioSnapshot) -> np.ndarray:
    weights = snapshot.weights.reindex(snapshot.returns.columns).to_numpy()
    return snapshot.returns.to_numpy() @ weights


def _historical_var(portfolio_returns: np.ndarray, confidence: float) -> float:
    quantile = np.quantile(portfolio_returns, 1 - confidence)
    return max(0.0, -quantile)


def _parametric_var(portfolio_returns: np.ndarray, confidence: float) -> float:
    mean = float(np.mean(portfolio_returns))
    std = float(np.std(portfolio_returns, ddof=1))
    z = norm.ppf(1 - confidence)
    value = -(mean + z * std)
    return max(0.0, value)


def _monte_carlo_var(
    snapshot: PortfolioSnapshot,
    confidence: float,
    mc_paths: int,
    seed: int,
) -> float:
    weights = snapshot.weights.reindex(snapshot.returns.columns).to_numpy()
    mean_vector = snapshot.returns.to_numpy().mean(axis=0)
    cov_matrix = np.cov(snapshot.returns.to_numpy(), rowvar=False, ddof=1)
    rng = np.random.default_rng(seed)
    simulated = rng.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=mc_paths)
    portfolio_draws = simulated @ weights
    quantile = np.quantile(portfolio_draws, 1 - confidence)
    return max(0.0, -quantile)


def test_var_workflow_covers_multiple_methods(sample_snapshot: PortfolioSnapshot, config_builder) -> None:
    confidence = 0.95
    mc_paths = 4000
    seed = 321

    config: RiskConfiguration = config_builder(
        confidence_levels=[confidence],
        horizons=[1],
        reports=["var"],
        mc_paths=mc_paths,
        seed=seed,
        stress_scenarios=[],
        covariance_methods=["sample"],
        cvar_methods=["historical"],
    )

    builder = RiskReportBuilder()
    report = builder.generate_report(sample_snapshot, config)

    var_by_method = {metric.method: metric for metric in report.risk_metrics if metric.metric == "var"}
    assert set(var_by_method) == {"historical", "parametric", "monte_carlo"}

    portfolio_returns = _portfolio_returns(sample_snapshot)
    assert var_by_method["historical"].value == pytest.approx(
        _historical_var(portfolio_returns, confidence), rel=1e-6
    )
    assert var_by_method["parametric"].value == pytest.approx(
        _parametric_var(portfolio_returns, confidence), rel=1e-6
    )
    assert var_by_method["monte_carlo"].value == pytest.approx(
        _monte_carlo_var(sample_snapshot, confidence, mc_paths, seed), rel=5e-2
    )
