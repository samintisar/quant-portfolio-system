from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..risk.models import PortfolioSnapshot
from ..risk.services import (
    ReportStore,
    RiskConfigurationLoader,
    RiskReportBuilder,
    ScenarioCatalog,
    log_event,
    record_timing,
    setup_logging,
)

CATALOG_PATH = Path("config/risk/stress_scenarios.json")
DEFAULT_REPORTS_PATH = Path("data/storage/reports")
DEFAULT_LOGGING_CONFIG = Path("config/logging/risk_logging.yaml")
REQUEST_ID_HEADER = "X-Request-ID"

LOGGER = setup_logging(DEFAULT_LOGGING_CONFIG)
REPORT_CACHE: Dict[str, Dict[str, Any]] = {}
REPORT_LOCATIONS: Dict[str, Path] = {}


def create_app() -> FastAPI:
    app = FastAPI(title="Risk Metrics API", version="1.0.0")

    @app.middleware("http")
    async def request_logger(request: Request, call_next):
        correlation_id = request.headers.get(REQUEST_ID_HEADER, uuid4().hex)
        request.state.correlation_id = correlation_id
        start = time.perf_counter()
        log_event(
            LOGGER,
            "risk.api.request.start",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            log_event(
                LOGGER,
                "risk.api.request.error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error=str(exc),
            )
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        log_event(
            LOGGER,
            "risk.api.request.complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )
        response.headers.setdefault(REQUEST_ID_HEADER, correlation_id)
        return response

    @app.post("/risk/report", status_code=202)
    def create_risk_report(request: Request, payload: Dict[str, Any]) -> JSONResponse:
        correlation_id = _current_correlation_id(request)
        snapshot_payload = payload.get("portfolio_snapshot")
        config_payload = payload.get("configuration")
        if snapshot_payload is None or config_payload is None:
            log_event(
                LOGGER,
                "risk.api.report.invalid_payload",
                correlation_id=correlation_id,
            )
            raise HTTPException(status_code=400, detail="Missing portfolio_snapshot or configuration")

        snapshot = _build_snapshot(snapshot_payload)
        config = RiskConfigurationLoader().build(config_payload)
        builder = RiskReportBuilder()
        with record_timing(LOGGER, "risk.api.report.generate", correlation_id=correlation_id):
            report = builder.generate_report(snapshot, config)
        report_dict = report.to_dict()
        REPORT_CACHE[report.report_id] = report_dict
        REPORT_LOCATIONS[report.report_id] = Path(config.reports_path)
        log_event(
            LOGGER,
            "risk.api.report.created",
            report_id=report.report_id,
            correlation_id=correlation_id,
            metrics=len(report_dict.get("risk_metrics", [])),
            scenarios=len(config.stress_scenarios),
        )
        return JSONResponse(content=report_dict, status_code=202)

    @app.get("/risk/reports/{report_id}")
    def get_risk_report(report_id: str, request: Request) -> JSONResponse:
        correlation_id = _current_correlation_id(request)
        if report_id in REPORT_CACHE:
            log_event(
                LOGGER,
                "risk.api.report.cache_hit",
                report_id=report_id,
                correlation_id=correlation_id,
            )
            return JSONResponse(content=REPORT_CACHE[report_id])

        store = _resolve_store(report_id)
        try:
            with record_timing(LOGGER, "risk.api.report.load", correlation_id=correlation_id, report_id=report_id):
                report_dict = store.load(report_id)
        except FileNotFoundError as exc:
            log_event(
                LOGGER,
                "risk.api.report.not_found",
                report_id=report_id,
                correlation_id=correlation_id,
            )
            raise HTTPException(status_code=404, detail="Report not found") from exc

        REPORT_CACHE[report_id] = report_dict
        log_event(
            LOGGER,
            "risk.api.report.returned",
            report_id=report_id,
            correlation_id=correlation_id,
        )
        return JSONResponse(content=report_dict)

    @app.get("/risk/scenarios")
    def list_risk_scenarios(request: Request) -> Dict[str, Any]:
        correlation_id = _current_correlation_id(request)
        catalog = ScenarioCatalog.from_path(CATALOG_PATH)
        scenarios = [
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "shock_type": scenario.shock_type,
                "parameters": scenario.parameters,
                "horizon": scenario.horizon,
            }
            for scenario in catalog.scenarios.values()
        ]
        log_event(
            LOGGER,
            "risk.api.scenarios.listed",
            correlation_id=correlation_id,
            scenario_count=len(scenarios),
        )
        return {"scenarios": scenarios}

    return app


def _build_snapshot(payload: Dict[str, Any]) -> PortfolioSnapshot:
    asset_ids: List[str] = payload.get("asset_ids", [])
    returns = payload.get("returns", [])
    timestamps = payload.get("timestamps", [])
    weights = payload.get("weights", {})
    factor_exposures = payload.get("factor_exposures")
    if not asset_ids or not returns or not timestamps or not weights:
        raise HTTPException(status_code=400, detail="Invalid portfolio snapshot payload")

    returns_df = pd.DataFrame(returns, columns=asset_ids)
    returns_df.index = pd.to_datetime(timestamps)
    weights_series = pd.Series(weights)
    factor_df = None
    if factor_exposures:
        factor_df = pd.DataFrame(factor_exposures)

    return PortfolioSnapshot(
        asset_ids=asset_ids,
        returns=returns_df,
        weights=weights_series,
        factor_exposures=factor_df,
        timestamp=returns_df.index.max() if not returns_df.index.empty else datetime.utcnow(),
    )


def _current_correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", uuid4().hex)


def _resolve_store(report_id: str) -> ReportStore:
    base_dir = REPORT_LOCATIONS.get(report_id, DEFAULT_REPORTS_PATH)
    return ReportStore(base_dir)
