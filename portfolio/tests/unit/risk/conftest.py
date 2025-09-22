from __future__ import annotations

import pandas as pd
import pytest

from portfolio.tests.data.fixtures.risk_demo import load_unit_returns, load_unit_weights


@pytest.fixture(scope="session")
def unit_returns_df() -> pd.DataFrame:
    return load_unit_returns().copy()


@pytest.fixture(scope="session")
def unit_weights() -> pd.Series:
    return load_unit_weights().copy()
