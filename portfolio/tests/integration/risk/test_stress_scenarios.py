from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.src.risk.services.report_builder import RiskReportBuilder


def _scenario_catalog() -> dict:
    catalog_path = Path(__file__).resolve().parents[4] / "config" / "risk" / "stress_scenarios.json"
    return json.loads(catalog_path.read_text())


def test_stress_testing_generates_impacts(sample_snapshot: PortfolioSnapshot, config_builder) -> None:
    catalog = _scenario_catalog()
    scenario_ids = [scenario["scenario_id"] for scenario in catalog["scenarios"][:2]]

    config: RiskConfiguration = config_builder(
        reports=["stress"],
        stress_scenarios=scenario_ids,
        covariance_methods=["sample"],
        var_methods=["historical"],
        cvar_methods=["historical"],
    )

    builder = RiskReportBuilder()
    report = builder.generate_report(sample_snapshot, config)

    impact_by_id = {impact.scenario_id: impact for impact in report.stress_impacts}
    assert set(impact_by_id.keys()) == set(scenario_ids)

    for scenario in catalog["scenarios"]:
        if scenario["scenario_id"] not in scenario_ids:
            continue
        impact = impact_by_id[scenario["scenario_id"]]
        assert impact.expected_loss >= 0
        assert impact.worst_case_loss >= impact.expected_loss
        assert impact.scenario_id == scenario["scenario_id"]
