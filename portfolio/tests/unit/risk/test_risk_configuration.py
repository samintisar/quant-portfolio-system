from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from portfolio.src.risk.models.configuration import RiskConfiguration


def _base_kwargs() -> dict:
    return {
        "confidence_levels": [0.95, 0.99],
        "horizons": [1, 10],
        "decay_lambda": 0.94,
        "mc_paths": 10000,
        "seed": 42,
        "stress_scenarios": ["macro_recession"],
        "data_frequency": "daily",
        "reports": ["covariance", "var", "cvar", "stress", "visualizations"],
        "covariance_methods": ["sample", "ledoit_wolf", "ewma"],
        "var_methods": ["historical", "parametric", "monte_carlo"],
        "cvar_methods": ["historical", "monte_carlo"],
        "reports_path": "data/storage/reports",
        "logging_config": "config/logging/risk_logging.yaml",
    }


def test_configuration_accepts_valid_values() -> None:
    config = RiskConfiguration(**_base_kwargs())
    assert config.confidence_levels == [0.95, 0.99]
    assert config.horizons == [1, 10]


def test_configuration_rejects_invalid_confidence_level() -> None:
    kwargs = _base_kwargs()
    kwargs["confidence_levels"] = [0.85]
    with pytest.raises(ValueError):
        RiskConfiguration(**kwargs)


def test_configuration_rejects_invalid_horizon() -> None:
    kwargs = _base_kwargs()
    kwargs["horizons"] = [0]
    with pytest.raises(ValueError):
        RiskConfiguration(**kwargs)


def test_configuration_requires_known_reports() -> None:
    kwargs = _base_kwargs()
    kwargs["reports"] = ["unknown"]
    with pytest.raises(ValueError):
        RiskConfiguration(**kwargs)


def test_configuration_requires_covariance_methods() -> None:
    kwargs = _base_kwargs()
    kwargs["covariance_methods"] = []
    with pytest.raises(ValueError):
        RiskConfiguration(**kwargs)
