import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from portfolio.src.api import risk_api



def _load_api_spec() -> dict:
    spec_path = Path(__file__).resolve().parents[4] / "specs" / "004-add-robust-risk" / "contracts" / "risk_metrics.openapi.json"
    return json.loads(spec_path.read_text())


def _build_payload() -> dict:
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
            "confidence_levels": [0.95, 0.99],
            "horizons": [1, 10],
            "decay_lambda": 0.94,
            "mc_paths": 10000,
            "seed": 1234,
            "stress_scenarios": ["macro_recession"],
            "reports": ["covariance", "var", "cvar", "stress", "visualizations"],
            "covariance_methods": ["sample", "ledoit_wolf", "ewma"],
            "var_methods": ["historical", "parametric", "monte_carlo"],
            "cvar_methods": ["historical", "monte_carlo"],
            "data_frequency": "daily"
        }
    }


@pytest.fixture()
def client() -> TestClient:
    if not hasattr(risk_api, "create_app"):
        pytest.fail("risk_api.create_app is not implemented yet")
    app = risk_api.create_app()
    return TestClient(app)


def test_post_risk_report_contract_returns_202_schema(client: TestClient) -> None:
    spec = _load_api_spec()
    payload = _build_payload()

    response = client.post("/risk/report", json=payload)

    assert response.status_code == 202, response.text

    body = response.json()
    required_keys = {"report_id", "generated_at", "covariance_results", "risk_metrics", "stress_impacts", "visualizations"}
    assert required_keys.issubset(body.keys())

    covariance_schema = spec["components"]["schemas"]["CovarianceResult"]
    for result in body.get("covariance_results", []):
        assert result["method"] in covariance_schema["properties"]["method"]["enum"]
        assert isinstance(result["matrix"], list)

    metric_schema = spec["components"]["schemas"]["RiskMetricEntry"]
    for metric in body.get("risk_metrics", []):
        assert metric["metric"] in metric_schema["properties"]["metric"]["enum"]
        assert metric["confidence"] in payload["configuration"]["confidence_levels"]
        assert metric["horizon"] in payload["configuration"]["horizons"]
        assert metric["value"] >= 0

    stress_schema = spec["components"]["schemas"]["StressImpact"]
    for impact in body.get("stress_impacts", []):
        assert impact["scenario_id"] in payload["configuration"]["stress_scenarios"]
        assert impact["expected_loss"] >= stress_schema["properties"]["expected_loss"]["minimum"]
        assert impact["worst_case_loss"] >= stress_schema["properties"]["worst_case_loss"]["minimum"]

    viz_schema = spec["components"]["schemas"]["VisualizationArtifact"]
    for artifact in body.get("visualizations", []):
        assert artifact["type"] in viz_schema["properties"]["type"]["enum"]
        assert artifact["format"] in viz_schema["properties"]["format"]["enum"]
        assert artifact["path"].startswith("data/storage/reports")
