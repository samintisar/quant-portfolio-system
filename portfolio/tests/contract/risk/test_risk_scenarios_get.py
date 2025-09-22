import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import pytest
from fastapi.testclient import TestClient

from portfolio.src.api import risk_api


def _scenario_catalog() -> dict:
    catalog_path = Path(__file__).resolve().parents[4] / "config" / "risk" / "stress_scenarios.json"
    return json.loads(catalog_path.read_text())


@pytest.fixture()
def client() -> TestClient:
    if not hasattr(risk_api, "create_app"):
        pytest.fail("risk_api.create_app is not implemented yet")
    return TestClient(risk_api.create_app())


def test_get_risk_scenarios_returns_catalog(client: TestClient) -> None:
    expected_catalog = _scenario_catalog()

    response = client.get("/risk/scenarios")

    assert response.status_code == 200, response.text
    payload = response.json()

    assert "scenarios" in payload
    scenario_ids = {scenario["scenario_id"] for scenario in payload["scenarios"]}
    expected_ids = {scenario["scenario_id"] for scenario in expected_catalog["scenarios"]}
    assert scenario_ids == expected_ids

    for scenario in payload["scenarios"]:
        assert scenario["shock_type"] in {"factor", "asset", "macro"}
        assert scenario["horizon"] >= 1
