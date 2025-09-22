from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.src.risk.services.report_builder import RiskReportBuilder


EXPECTED_TYPES = {"factor_exposure", "var_timeseries", "cvar_distribution"}
EXPECTED_FORMATS = {"png", "svg", "html"}


def test_visualization_artifacts_are_generated(sample_snapshot: PortfolioSnapshot, config_builder) -> None:
    output_dir = Path("data/storage/reports/tests")
    config: RiskConfiguration = config_builder(
        reports=["visualizations"],
        stress_scenarios=[],
        covariance_methods=["sample"],
        var_methods=["historical"],
        cvar_methods=["historical"],
        reports_path=str(output_dir),
    )

    builder = RiskReportBuilder()
    report = builder.generate_report(sample_snapshot, config)

    assert report.visualizations, "Expected visualization artifacts to be returned"
    for artifact in report.visualizations:
        assert artifact.type in EXPECTED_TYPES
        assert artifact.format in EXPECTED_FORMATS
        assert str(output_dir) in str(artifact.path)
