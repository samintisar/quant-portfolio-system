# Risk Metrics Toolkit

Robust risk measurement tooling introduced in feature 004 delivers covariance analysis, Value-at-Risk (VaR), Conditional VaR (CVaR), stress testing, and visualization pipelines that share a single configuration surface. This document describes how to operate the service through both FastAPI and CLI entry points and how to interpret the persisted outputs.

## Architecture Overview

| Component | Location | Responsibility |
|-----------|----------|----------------|
| API | `portfolio/src/api/risk_api.py` | FastAPI app factory plus request logging/timing middleware |
| Services | `portfolio/src/risk/services/` | Configuration merge/validation, scenario catalog, data access, report builder, persistence |
| Engines | `portfolio/src/risk/engines/` | Covariance, VaR, CVaR, stress calculators |
| Visualization | `portfolio/src/risk/visualization/plots.py` | Factor exposure, VaR trend, CVaR distribution artefacts |
| Storage | `data/storage/reports/` | JSON reports, Parquet metrics, visualization assets, `index.{json,parquet}` |
| Config | `config/risk/` | `defaults.json` (95%/1-day, 99%/10-day presets) and `stress_scenarios.json` |

## FastAPI Quickstart

```bash
uvicorn portfolio.src.api.risk_api:create_app --reload
```

### POST `/risk/report`

- Payload requires a `portfolio_snapshot` and `configuration` object.
- Snapshot must supply `asset_ids`, aligned `returns` matrix, `timestamps`, and `weights`. Optional `factor_exposures` support exposure summaries.
- Configuration falls back to defaults when fields are omitted. Confidence levels and horizons are validated to `[0.9, 0.999]` and integers ≥1 respectively.
- Response (`202 Accepted`) contains the assembled `RiskReport` and writes persisted artefacts to `data/storage/reports/`.

### GET `/risk/reports/{report_id}`

- Returns the stored JSON report, rehydrating from disk when not served from the in-memory cache.
- Errors with `404` when the report is unknown; telemetry emits `risk.api.report.not_found`.

### GET `/risk/scenarios`

- Enumerates the validated scenario catalog derived from `config/risk/stress_scenarios.json`.

## CLI Quickstart

```bash
python scripts/run_risk_metrics.py \
  --snapshot-file data/storage/demo_snapshot.parquet \
  --config-file config/risk/defaults.json \
  --reports-path data/storage/reports
```

Key flags:

- `--confidence-level` / `--horizon`: override defaults (repeatable).
- `--var-method`, `--cvar-method`, `--covariance-method`: restrict calculations.
- `--stress-scenario`: subset or extend the catalog.
- `--mc-paths`, `--seed`: tune Monte Carlo variance.
- `--log-config`: specify alternate logging YAML.

See `examples/run_risk_metrics_demo.py` for a scripted run that wires demo data directly into the builder without touching disk.

## Configuration Reference

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `confidence_levels` | list[float] | `[0.95, 0.99]` | Must remain within `[0.9, 0.999]`. |
| `horizons` | list[int] | `[1, 10]` | Trading days; VaR/CVaR scale with `sqrt(horizon)`. |
| `decay_lambda` | float | `0.94` | EWMA smoothing factor; `(0,1)` bound enforced. |
| `mc_paths` | int | `10000` | Monte Carlo sample size; `[1000, 100000]`. |
| `seed` | int | `1234` | Optional RNG seed for reproducibility. |
| `stress_scenarios` | list[str] | `macro_recession`, `inflation_spike`, … | Must exist in the catalog. |
| `var_methods` | list[str] | `historical`, `parametric`, `monte_carlo` | Filter VaR calculators. |
| `cvar_methods` | list[str] | `historical`, `parametric`, `monte_carlo` | Filter CVaR calculators. |
| `reports` | list[str] | `covariance`, `var`, `cvar`, `stress`, `visualizations` | Enables/disables report sections. |
| `reports_path` | path | `data/storage/reports` | Target directory for persisted artefacts. |
| `logging_config` | path | `config/logging/risk_logging.yaml` | YAML config consumed by `setup_logging`. |

## Outputs and Persistence

After report generation, the `ReportStore` writes:

- `<report_id>.json`: serialized `RiskReport` payload returned by the API/CLI.
- `<report_id>_metrics.parquet`: tabular risk metrics suited for BI tooling.
- Visual assets (PNG/SVG/HTML) inside `data/storage/reports/<report_id>/` as created by the visualization pipeline.
- `index.json` / `index.parquet`: recency-sorted catalogue containing metadata per report (`report_id`, horizons, confidence levels, scenario IDs, and artefact paths). Use `ReportStore.list_records()` or `get_record()` for programmatic access.

## Performance & Observability

- Performance regression guard (`tests/performance/risk/test_risk_report_runtime.py`) ensures a 500-asset Monte Carlo VaR run completes in under 5 seconds and below 1 GiB peak memory.
- Middleware logs `risk.api.request.*` events with correlation IDs; `record_timing` emits phase-level durations (`risk.report.duration`, `risk.api.report.generate`, `risk.api.report.load`).
- Use the structured logs produced by `config/logging/risk_logging.yaml` for ingestion into central observability stacks.

## Validation Checklist

To validate changes touching the risk stack:

1. `python -m pytest portfolio/tests/unit/risk -q`
2. `python -m pytest portfolio/tests/integration/risk -q`
3. `python -m pytest portfolio/tests/contract/risk -q`
4. `python -m pytest tests/performance/risk -m "performance"`
5. Lint/typing via `flake8`, `mypy .`, and formatting checks (`black --check .`, `isort --check-only .`).

