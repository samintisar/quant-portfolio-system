import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from portfolio.src.api import risk_api


@pytest.fixture()
def client() -> TestClient:
    if not hasattr(risk_api, "create_app"):
        pytest.fail("risk_api.create_app is not implemented yet")
    return TestClient(risk_api.create_app())


def _payload() -> dict:
    return {
        "portfolio_snapshot": {
            "asset_ids": ["EQ1", "EQ2"],
            "returns": [
                [0.012, 0.018],
                [-0.007, 0.004],
                [0.009, -0.003],
                [0.011, 0.02]
            ],
            "timestamps": [
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
                "2024-01-04T00:00:00Z",
                "2024-01-05T00:00:00Z"
            ],
            "weights": {"EQ1": 0.55, "EQ2": 0.45}
        },
        "configuration": {
            "confidence_levels": [0.95],
            "horizons": [1],
            "decay_lambda": 0.94,
            "mc_paths": 10000,
            "seed": 1234,
            "stress_scenarios": ["macro_recession"],
            "reports": ["covariance", "var", "cvar", "stress", "visualizations"],
            "covariance_methods": ["sample", "ledoit_wolf"],
            "var_methods": ["historical", "parametric"],
            "cvar_methods": ["historical"],
            "data_frequency": "daily"
        }
    }


def test_get_risk_report_returns_persisted_payload(client: TestClient) -> None:
    create_response = client.post("/risk/report", json=_payload())
    assert create_response.status_code == 202, create_response.text

    report_id = create_response.json()["report_id"]
    fetch_response = client.get(f"/risk/reports/{report_id}")

    assert fetch_response.status_code == 200, fetch_response.text
    body = fetch_response.json()
    assert body["report_id"] == report_id
    assert "risk_metrics" in body and isinstance(body["risk_metrics"], list)


def test_get_risk_report_returns_404_for_unknown_report(client: TestClient) -> None:
    missing_id = "non-existent-report-id"
    response = client.get(f"/risk/reports/{missing_id}")

    assert response.status_code == 404
    payload = response.json()
    assert payload.get("detail")
