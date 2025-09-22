# Risk Metrics CLI Contract

## Command
`python scripts/run_risk_metrics.py --weights PATH --returns PATH [options]`

## Required Arguments
- `--weights PATH`: CSV or Parquet file with columns `asset_id` and `weight`.
- `--returns PATH`: Parquet file containing time-indexed returns aligned with ingestion schema.

## Optional Arguments
- `--config PATH`: JSON file overriding defaults from `config/risk/defaults.json`.
- `--confidence-levels 0.95 0.99`: Space-separated floats overriding defaults.
- `--horizons 1 10`: Trading day horizons.
- `--mc-paths 10000`: Number of Monte Carlo simulations.
- `--scenarios stress_id ...`: Scenario identifiers to apply.
- `--output-dir PATH`: Directory for generated visualizations and serialized report (defaults to data/storage/reports).
- `--format json|parquet`: Serialization format for report payload (default json).
- `--seed INT`: Deterministic RNG seed.

## Behavior
1. Validate inputs against ingestion schema and ensure weights sum to 1 within tolerance 1e-6.
2. Load default configuration and overlay CLI overrides.
3. Compute covariance matrices, VaR, CVaR, and stress reports as requested.
4. Persist report to `risk_report_{timestamp}.json` (or parquet) and write plots to output directory.
5. Emit structured logs including configuration summary, data coverage, and performance metrics.

## Exit Codes
- `0`: Report generated successfully.
- `2`: Validation or schema mismatch.
- `3`: Computation failure (e.g., covariance not positive semi-definite).
- `4`: Output write failure.

## Logging
- INFO: configuration summary, runtime statistics.
- WARNING: data gaps, scenario overrides applied.
- ERROR: validation or computation errors (CLI exits non-zero).

## Examples
```
python scripts/run_risk_metrics.py --weights data/storage/demo_weights.csv \
    --returns data/storage/demo_returns.parquet \
    --confidence-levels 0.95 0.99 --horizons 1 10 --mc-paths 50000 \
    --scenarios macro_2008 liquidity_crunch --output-dir reports/demo
```
