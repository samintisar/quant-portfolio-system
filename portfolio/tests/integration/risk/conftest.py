from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pytest

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.tests.data.fixtures.risk_demo import (
    build_demo_configuration,
    load_integration_returns,
    load_integration_weights,
)


@pytest.fixture(scope="session")
def sample_returns_df() -> pd.DataFrame:
    return load_integration_returns().copy()


@pytest.fixture(scope="session")
def sample_weights() -> pd.Series:
    return load_integration_weights().copy()


@pytest.fixture()
def sample_snapshot(sample_returns_df: pd.DataFrame, sample_weights: pd.Series) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        asset_ids=list(sample_returns_df.columns),
        returns=sample_returns_df.copy(),
        weights=sample_weights.copy(),
        factor_exposures=None,
        timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
    )


@pytest.fixture()
def config_builder() -> callable:
    def _builder(**overrides) -> RiskConfiguration:
        return build_demo_configuration(**overrides)

    return _builder
