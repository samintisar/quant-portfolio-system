# Quickstart: Robust Risk Measurement Tools

## Prerequisites
1. Install dependencies: pip install -r docs/requirements.txt.
2. Prime storage directories: python scripts/setup_data_environment.py.
3. Ensure demo datasets exist in data/storage/demo_returns.parquet and data/storage/demo_weights.csv.

## Generate a Risk Report via CLI
```
python scripts/run_risk_metrics.py \
    --weights data/storage/demo_weights.csv \
    --returns data/storage/demo_returns.parquet \
    --confidence-levels 0.95 0.99 \
    --horizons 1 10 \
    --mc-paths 20000 \
    --scenarios macro_2008 custom_liquidity \
    --output-dir data/storage/reports/demo \
    --format json \
    --seed 42
```
- Output: data/storage/reports/demo/risk_report_<timestamp>.json plus PNG/SVG plots.
- Logs: logs/risk/run_risk_metrics.log (structured JSON lines).

## Start the Risk Metrics API
```
uvicorn portfolio.src.api.risk_api:app --reload --port 8081
```
- POST http://localhost:8081/risk/report with payload conforming to contracts/risk_metrics.openapi.json.
- GET http://localhost:8081/risk/scenarios returns scenario catalog.

## Validate Tests
```
pytest -m "risk and not slow"
pytest portfolio/tests/integration/risk/test_risk_workflow.py -k demo
mypy portfolio/src/risk
```
- Integration test seeds deterministic inputs to confirm VaR/CVaR thresholds.
- Hypothesis-based unit tests cover covariance stability and CVaR monotonicity.

## Update Visualizations
1. Generate factor exposure plot only:
```
python scripts/run_risk_metrics.py --weights data/storage/demo_weights.csv \
    --returns data/storage/demo_returns.parquet \
    --reports visualizations --output-dir data/storage/reports/factors
```
2. Inspect outputs under data/storage/reports/factors.

## Cleanup (Optional)
- Remove generated artifacts: Remove-Item data/storage/reports/demo -Recurse.
- Stop API server with Ctrl+C.
