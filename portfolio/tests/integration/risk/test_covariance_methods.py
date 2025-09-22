from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.covariance import LedoitWolf

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.src.risk.services.report_builder import RiskReportBuilder


def _expected_sample_cov(snapshot: PortfolioSnapshot) -> np.ndarray:
    return np.cov(snapshot.returns.values, rowvar=False, ddof=1)


def _expected_ledoit_wolf_cov(snapshot: PortfolioSnapshot) -> np.ndarray:
    model = LedoitWolf().fit(snapshot.returns.values)
    return model.covariance_


def _expected_ewma_cov(snapshot: PortfolioSnapshot, decay: float) -> np.ndarray:
    centered = snapshot.returns.values - snapshot.returns.values.mean(axis=0)
    cov = np.zeros((centered.shape[1], centered.shape[1]))
    for row in centered:
        cov = decay * cov + (1 - decay) * np.outer(row, row)
    return cov


def test_covariance_workflow_produces_expected_matrices(
    sample_snapshot: PortfolioSnapshot,
    config_builder,
) -> None:
    config: RiskConfiguration = config_builder(reports=["covariance"], stress_scenarios=[])

    builder = RiskReportBuilder()
    report = builder.generate_report(sample_snapshot, config)

    methods = {result.method for result in report.covariance_results}
    assert methods == {"sample", "ledoit_wolf", "ewma"}

    sample_result = next(result for result in report.covariance_results if result.method == "sample")
    np.testing.assert_allclose(sample_result.matrix, _expected_sample_cov(sample_snapshot), rtol=1e-7, atol=1e-9)

    lw_result = next(result for result in report.covariance_results if result.method == "ledoit_wolf")
    np.testing.assert_allclose(lw_result.matrix, _expected_ledoit_wolf_cov(sample_snapshot), rtol=1e-6)

    ewma_result = next(result for result in report.covariance_results if result.method == "ewma")
    np.testing.assert_allclose(
        ewma_result.matrix,
        _expected_ewma_cov(sample_snapshot, config.decay_lambda),
        rtol=1e-6,
    )

    eigenvalues = np.linalg.eigvalsh(sample_result.matrix)
    assert np.all(eigenvalues >= -1e-8), "Sample covariance should be positive semi-definite"
