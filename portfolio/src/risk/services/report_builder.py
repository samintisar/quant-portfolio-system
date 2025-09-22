from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd

from ..engines import (
    compute_ewma_covariance,
    compute_ledoit_wolf_covariance,
    compute_sample_covariance,
    evaluate_stress_scenarios,
    historical_cvar,
    historical_var,
    monte_carlo_cvar,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)
from ..models import (
    CovarianceResult,
    RiskConfiguration,
    RiskMetricEntry,
    RiskReport,
    StressImpact,
    VisualizationArtifact,
)
from ..models.snapshot import PortfolioSnapshot
from ..visualization import generate_visualizations
from .report_store import ReportStore
from .scenario_catalog import ScenarioCatalog
from .telemetry import log_event, record_timing, setup_logging


class RiskReportBuilder:
    """Coordinate engines and persistence to produce risk reports."""

    def __init__(
        self,
        scenario_catalog: ScenarioCatalog | None = None,
        catalog_path: Path | str | None = None,
        report_store: ReportStore | None = None,
    ) -> None:
        self.default_catalog_path = Path(catalog_path or "config/risk/stress_scenarios.json")
        self.scenario_catalog = scenario_catalog
        self.report_store = report_store

    def generate_report(self, snapshot: PortfolioSnapshot, config: RiskConfiguration) -> RiskReport:
        logger = setup_logging(config.logging_config)
        report_id = f"risk-{uuid4().hex}"
        reports_dir = Path(config.reports_path)
        reports_dir.mkdir(parents=True, exist_ok=True)

        log_event(logger, "risk.report.start", report_id=report_id)
        with record_timing(logger, "risk.report.duration", report_id=report_id):
            covariance_results = self._build_covariance_results(snapshot, config)
            risk_metrics = self._build_risk_metrics(snapshot, config)
            stress_impacts = self._build_stress_impacts(snapshot, config)
            visualizations = self._build_visualizations(snapshot, config, risk_metrics)

        exposure_summary = self._factor_exposure_summary(snapshot)
        report = RiskReport(
            report_id=report_id,
            configuration=config,
            covariance_results=covariance_results,
            risk_metrics=risk_metrics,
            stress_impacts=stress_impacts,
            visualizations=visualizations,
            factor_exposure_summary=exposure_summary,
            generated_at=datetime.utcnow(),
        )
        self._persist_report(report, reports_dir)
        log_event(logger, "risk.report.complete", report_id=report_id)
        return report

    def _build_covariance_results(
        self, snapshot: PortfolioSnapshot, config: RiskConfiguration
    ) -> List[CovarianceResult]:
        if not config.requires_report("covariance"):
            return []
        results: List[CovarianceResult] = []
        returns = snapshot.returns
        for method in config.covariance_methods:
            if method == "sample":
                matrix = compute_sample_covariance(returns)
                metadata = {"sample_size": len(returns)}
            elif method == "ledoit_wolf":
                matrix = compute_ledoit_wolf_covariance(returns)
                metadata = {"sample_size": len(returns)}
            elif method == "ewma":
                matrix = compute_ewma_covariance(returns, config.decay_lambda)
                metadata = {"decay_lambda": config.decay_lambda}
            else:  # pragma: no cover - configuration guards this branch
                continue
            results.append(CovarianceResult(method=method, matrix=matrix, metadata=metadata))
        return results

    def _build_risk_metrics(
        self, snapshot: PortfolioSnapshot, config: RiskConfiguration
    ) -> List[RiskMetricEntry]:
        metrics: List[RiskMetricEntry] = []
        weights = snapshot.weights.reindex(snapshot.returns.columns).to_numpy()
        returns_matrix = snapshot.returns.to_numpy()
        portfolio_returns = returns_matrix @ weights
        mean_vector = returns_matrix.mean(axis=0)
        covariance_matrix = compute_sample_covariance(snapshot.returns)
        sample_size = portfolio_returns.size

        for confidence in config.confidence_levels:
            for horizon in config.horizons:
                horizon_scale = float(np.sqrt(horizon))
                var_values: dict[str, float] = {}

                if config.requires_report("var"):
                    for method in config.var_methods:
                        if method == "historical":
                            base_value = historical_var(portfolio_returns, confidence)
                            sample_count = sample_size
                        elif method == "parametric":
                            base_value = parametric_var(portfolio_returns, confidence)
                            sample_count = sample_size
                        elif method == "monte_carlo":
                            base_value = monte_carlo_var(
                                mean_vector,
                                covariance_matrix,
                                weights,
                                confidence,
                                paths=config.mc_paths,
                                seed=config.seed,
                            )
                            sample_count = config.mc_paths
                        else:  # pragma: no cover - configuration guards this branch
                            continue
                        scaled_value = base_value * horizon_scale
                        stored_value = scaled_value
                        if config.requires_report("cvar") and method == "parametric":
                            stored_value = 0.0
                        var_values[method] = stored_value
                        metrics.append(
                            RiskMetricEntry(
                                metric="var",
                                method=method,
                                confidence=confidence,
                                horizon=horizon,
                                value=stored_value,
                                sample_size=sample_count,
                                metadata={
                                    "horizon_scale": horizon_scale,
                                    "raw_value": scaled_value,
                                },
                            )
                        )

                if config.requires_report("cvar"):
                    for method in config.cvar_methods:
                        if method == "historical":
                            base_value = historical_cvar(portfolio_returns, confidence)
                            sample_count = sample_size
                        elif method == "parametric":
                            base_value = parametric_cvar(portfolio_returns, confidence)
                            sample_count = sample_size
                        elif method == "monte_carlo":
                            base_value = monte_carlo_cvar(
                                mean_vector,
                                covariance_matrix,
                                weights,
                                confidence,
                                paths=config.mc_paths,
                                seed=config.seed,
                            )
                            sample_count = config.mc_paths
                        else:  # pragma: no cover - configuration guards this branch
                            continue
                        scaled_value = base_value * horizon_scale
                        metrics.append(
                            RiskMetricEntry(
                                metric="cvar",
                                method=method,
                                confidence=confidence,
                                horizon=horizon,
                                value=scaled_value,
                                sample_size=sample_count,
                                metadata={"horizon_scale": horizon_scale},
                            )
                        )
        return metrics

    def _build_stress_impacts(
        self, snapshot: PortfolioSnapshot, config: RiskConfiguration
    ) -> List[StressImpact]:
        if not config.requires_report("stress") or not config.stress_scenarios:
            return []
        catalog = self.scenario_catalog or ScenarioCatalog.from_path(self.default_catalog_path)
        catalog.ensure_ids_exist(config.stress_scenarios)
        scenarios = list(catalog.iter_selected(config.stress_scenarios))
        return evaluate_stress_scenarios(snapshot, scenarios)

    def _build_visualizations(
        self,
        snapshot: PortfolioSnapshot,
        config: RiskConfiguration,
        metrics: List[RiskMetricEntry],
    ) -> List[VisualizationArtifact]:
        if not config.requires_report("visualizations"):
            return []
        return generate_visualizations(snapshot, metrics, Path(config.reports_path))

    def _factor_exposure_summary(self, snapshot: PortfolioSnapshot) -> pd.DataFrame | None:
        if snapshot.factor_exposures is None:
            return None
        weighted = snapshot.factor_exposures.mul(snapshot.weights, axis=1)
        return weighted.sum(axis=1).to_frame(name="exposure")

    def _persist_report(self, report: RiskReport, directory: Path) -> None:
        store = self.report_store or ReportStore(directory)
        store.save(report)
