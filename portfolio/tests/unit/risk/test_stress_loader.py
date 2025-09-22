from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from portfolio.src.risk.services.scenario_catalog import ScenarioCatalog


@pytest.fixture(scope="module")
def catalog() -> ScenarioCatalog:
    catalog_path = Path(__file__).resolve().parents[4] / "config" / "risk" / "stress_scenarios.json"
    return ScenarioCatalog.from_path(catalog_path)


def test_scenario_catalog_loads_and_indexes(catalog: ScenarioCatalog) -> None:
    scenario = catalog.get("macro_recession")
    assert scenario.scenario_id == "macro_recession"
    assert scenario.shock_type == "macro"
    assert scenario.horizon == 10
    assert "equity_return" in scenario.parameters


def test_scenario_catalog_validates_ids(catalog: ScenarioCatalog) -> None:
    catalog.ensure_ids_exist(["macro_recession", "inflation_spike"])
    with pytest.raises(ValueError):
        catalog.ensure_ids_exist(["macro_recession", "unknown_scenario"])


def test_scenario_catalog_filters_by_tag(catalog: ScenarioCatalog) -> None:
    baseline = catalog.get_catalog("baseline")
    assert {scenario.scenario_id for scenario in baseline} == {"macro_recession", "inflation_spike"}

    with pytest.raises(KeyError):
        catalog.get_catalog("nonexistent")
