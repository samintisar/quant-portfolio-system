from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.src.risk.models.configuration import RiskConfiguration
from portfolio.src.risk.models.snapshot import PortfolioSnapshot
from portfolio.src.risk.services import RiskReportBuilder


@pytest.mark.performance
@pytest.mark.slow
def test_monte_carlo_var_runtime_under_budget(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    observations = 2520
    assets = 500
    returns = rng.normal(loc=0.0005, scale=0.015, size=(observations, assets))
    columns = [f"ASSET_{idx:03d}" for idx in range(assets)]
    returns_df = pd.DataFrame(returns, columns=columns)

    weights = pd.Series(rng.random(assets), index=columns)
    weights /= weights.sum()

    snapshot = PortfolioSnapshot(
        asset_ids=columns,
        returns=returns_df,
        weights=weights,
        factor_exposures=None,
        timestamp=pd.Timestamp("2024-01-05T00:00:00Z"),
    )

    config = RiskConfiguration(
        confidence_levels=[0.95, 0.99],
        horizons=[1, 10],
        decay_lambda=0.94,
        mc_paths=10000,
        seed=2024,
        stress_scenarios=[],
        data_frequency="daily",
        reports=["var", "cvar"],
        covariance_methods=["sample"],
        var_methods=["historical", "parametric", "monte_carlo"],
        cvar_methods=["historical", "parametric", "monte_carlo"],
        reports_path=tmp_path,
        logging_config="config/logging/risk_logging.yaml",
    )

    builder = RiskReportBuilder()

    tracemalloc.start()
    start = time.perf_counter()
    report = builder.generate_report(snapshot, config)
    duration = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert duration < 5.0, f"Risk report generation exceeded runtime budget: {duration:.2f}s"
    assert peak < 1_000_000_000, f"Peak memory usage {peak / (1024 ** 2):.1f} MiB exceeded 1 GiB budget"
    assert any(metric.method == "monte_carlo" and metric.metric == "var" for metric in report.risk_metrics), (
        "Monte Carlo VaR should be present in risk metrics output"
    )
