# Portfolio Module

Portfolio houses the production-ready portfolio engines used across allocation, execution, and risk management.

## Layout

- `src/execution/`: execution policy adapters and order staging
- `src/optimization/`: allocation optimizers and constraint handlers
- `src/risk/`: robust risk measurement services (covariance, VaR/CVaR, stress, visualization)

Tests mirror the structure under `portfolio/tests/{unit, integration, contract}` and consume deterministic fixtures from `portfolio/tests/data/fixtures/`.

## Risk Metrics Toolkit

Phase 004 introduced a full risk metrics surface delivered through both API and CLI entry points.

### Core Capabilities

- **Covariance Estimation**: Sample, Ledoit-Wolf, and EWMA methods with positive semidefinite guarantees
- **Value-at-Risk (VaR)**: Historical, parametric, and Monte Carlo simulation methods
- **Conditional Value-at-Risk (CVaR)**: Expected shortfall calculations derived from VaR outputs
- **Stress Testing**: Macro scenarios, factor shocks, and custom scenario analysis
- **Visualization**: Factor exposure, VaR trends, and CVaR distribution plots

### API Interface

**FastAPI service**: `portfolio.src.api.risk_api.create_app()` exposes `/risk/report`, `/risk/reports/{id}`, and `/risk/scenarios`. Run locally with `uvicorn portfolio.src.api.risk_api:create_app --reload`.

**Key endpoints:**
- `POST /risk/report`: Generate comprehensive risk analysis
- `GET /risk/reports/{report_id}`: Retrieve completed reports
- `GET /risk/scenarios`: List available stress scenarios

### CLI Interface

**CLI workflow**: `scripts/run_risk_metrics.py` generates reports from stored returns/weights or inline overrides. See the quickstart in `examples/run_risk_metrics_demo.py`.

**Common usage patterns:**
```bash
# Using stored portfolio data
python scripts/run_risk_metrics.py \
  --portfolio-id test_portfolio \
  --confidence-level 0.95 \
  --horizon-days 1 \
  --covariance-method ledoit_wolf \
  --var-method historical \
  --include-stress-tests

# Using inline data
python scripts/run_risk_metrics.py \
  --returns-data "AAPL:0.01,0.02,-0.01;GOOGL:0.02,-0.01,0.03" \
  --weights-data "AAPL:0.6,GOOGL:0.4" \
  --confidence-level 0.99
```

### Configuration System

**Configuration**: defaults live in `config/risk/defaults.json`; stress scenarios in `config/risk/stress_scenarios.json`. Override values via API payloads, CLI flags, or by pointing to alternate config files.

**Key configuration parameters:**
- Confidence levels: 0.90-0.99 (default: 0.95)
- Time horizons: 1-252 days (default: 1)
- Covariance methods: sample, ledoit_wolf, ewma
- VaR methods: historical, parametric, monte_carlo
- Monte Carlo simulations: 1000-100000 (default: 10000)

### Data Persistence

**Persistence**: reports, metrics parquet, and visual artifacts are written under `data/storage/reports/` with metadata indexed in `index.json` and `index.parquet`.

**Output structure:**
- `<report_id>.json`: Complete risk report
- `<report_id>_metrics.parquet`: Tabular metrics for analysis
- `visualizations/`: Charts and plots (PNG/SVG/HTML)
- `index.json`/`index.parquet`: Report catalog metadata

### Performance Characteristics

**Benchmarks:**
- Portfolio size: Up to 500 assets
- VaR calculation: < 5 seconds for Monte Carlo simulation
- Memory usage: < 1 GB for large portfolios
- Report generation: < 10 seconds for full analysis

**Optimization features:**
- Parallel processing for large portfolios
- Cached covariance matrices
- Efficient Monte Carlo sampling
- Memory-conscious data structures

Refer to `docs/risk_metrics.md` for end-to-end usage examples, configuration tables, and performance guidance.

## Development Notes

- Ensure the repository root is on `PYTHONPATH` when running tests locally (handled automatically by `portfolio/tests/conftest.py`).
- Formatting: `black`, `isort`; linting: `flake8`; typing: `mypy .`.
- Targeted suites: `pytest portfolio/tests/unit/risk -q`, `pytest portfolio/tests/integration/risk -q`, `pytest portfolio/tests/contract/risk -q`.
