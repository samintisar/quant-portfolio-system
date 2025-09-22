from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from portfolio.src.risk.models import (
    RiskConfiguration,
    RiskMetricEntry,
    RiskReport,
)
from portfolio.src.risk.services import ReportStore


def _parquet_engine_available() -> bool:
    try:
        import pyarrow  # type: ignore  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False


pytestmark = pytest.mark.skipif(
    not _parquet_engine_available(), reason="Parquet engine is required for ReportStore tests"
)


def _build_configuration(base_path: Path) -> RiskConfiguration:
    return RiskConfiguration(
        confidence_levels=[0.95],
        horizons=[1],
        decay_lambda=0.94,
        mc_paths=5000,
        seed=42,
        stress_scenarios=[],
        data_frequency="daily",
        reports=["var"],
        covariance_methods=["sample"],
        var_methods=["historical"],
        cvar_methods=["historical"],
        reports_path=base_path,
        logging_config=base_path / "logging.yaml",
    )


def _build_report(config: RiskConfiguration) -> RiskReport:
    metric = RiskMetricEntry(
        metric="var",
        confidence=0.95,
        horizon=1,
        value=0.01,
        method="historical",
        sample_size=100,
        metadata={"horizon_scale": 1.0},
    )

    return RiskReport(
        report_id="unit-test-report",
        configuration=config,
        covariance_results=[],
        risk_metrics=[metric],
        stress_impacts=[],
        visualizations=[],
        factor_exposure_summary=None,
        generated_at=datetime(2024, 1, 1, 0, 0, 0),
    )


def test_report_store_persists_artifacts_and_updates_index(tmp_path: Path) -> None:
    config = _build_configuration(tmp_path)
    report = _build_report(config)

    store = ReportStore(tmp_path)
    record = store.save(report)

    json_path = tmp_path / f"{report.report_id}.json"
    parquet_path = tmp_path / f"{report.report_id}_metrics.parquet"
    assert json_path.exists()
    assert parquet_path.exists()

    index_rows = store.load_index()
    assert len(index_rows) == 1
    assert index_rows[0]["report_id"] == report.report_id

    structured_records = store.list_records()
    assert structured_records[0].json_path == json_path
    assert structured_records[0].metrics_path == parquet_path

    reloaded = store.load(report.report_id)
    assert reloaded["report_id"] == report.report_id
    assert reloaded["risk_metrics"][0]["value"] == report.risk_metrics[0].value

    metrics = store.load_metrics(report.report_id)
    assert isinstance(metrics, pd.DataFrame)
    assert not metrics.empty


def test_get_record_returns_structured_metadata(tmp_path: Path) -> None:
    config = _build_configuration(tmp_path)
    report = _build_report(config)
    store = ReportStore(tmp_path)
    store.save(report)

    record = store.get_record(report.report_id)

    assert record.report_id == report.report_id
    assert record.generated_at == report.generated_at.isoformat()
    assert record.reports == ["var"]
