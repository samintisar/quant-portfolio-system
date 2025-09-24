# Quant Portfolio System

A clear, minimal portfolio optimization project that demonstrates end‑to‑end workflows: data ingest (Yahoo Finance), mean‑variance optimization, simple risk/return metrics, and a lightweight FastAPI service. The goal is to keep it easy to run, explain, and maintain.

## Features
- Yahoo Finance data loader with simple caching to `data/`
- Mean‑variance optimization (long‑only, sum‑to‑1) via CVXPY
- Metrics: annualized return, volatility, Sharpe ratio, max drawdown
- Basic visualizations: equity curve, drawdown, efficient frontier (preview)
- FastAPI endpoints for optimize/analyze
- Walk‑forward backtesting (short windows for quick runs)
- Clean examples and a lab notebook

## Stack
- Python 3.11+, Pandas, NumPy
- yfinance (Yahoo Finance)
- CVXPY
- FastAPI (+ uvicorn)
- pytest
- Matplotlib

## Quick Start
1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Run the lab notebook (recommended first pass)
```bash
# Launch Jupyter (VS Code, Jupyter Lab, or classic notebooks)
# Open portfolio_optimization_lab.ipynb and run top-to-bottom
```

3) (Optional) Fetch and cache data
```bash
python scripts/fetch_market_data.py
```
This will populate `data/raw` and `data/processed` and create combined price CSVs for faster reruns.

## Project Structure
```
portfolio/                # Core logic
  api/                    # FastAPI app (portfolio.api.main:app)
  backtesting/            # Walk‑forward backtesting
  data/                   # Yahoo service and helpers
  optimizer/              # SimplePortfolioOptimizer (MVO, CVaR, BL)
  performance/            # Metrics + plots
examples/                 # Runnable examples and outputs
scripts/                  # Small, focused scripts (fetch, API, reports)
tests/                    # pytest suites
portfolio_optimization_lab.ipynb  # End‑to‑end lab notebook
config.yaml               # Minimal config (e.g., defaults)
```

## Notebook Guide
Open `portfolio_optimization_lab.ipynb` and run cells in order. You’ll see:
- Data Fetch & Caching: loads adjusted close prices and caches to `data/`
- Returns & Correlations: quick sanity checks
- Mean–Variance Optimization: weights + metrics; optional efficient frontier
- Parameter Sweep: risk model, weight caps, entropy
- Backtesting (short window): quick performance snapshot
- Outputs saved to `examples/figures/` and `examples/outputs/`

### Defaults used in the lab
- Risk model: Ledoit–Wolf shrinkage (toggle to `sample` or `oas`)
- Diversification: `weight_cap = 0.20`, `entropy_penalty = 0.03` (Effective Number of Holdings is printed)
- Walk‑forward: 1y train / 1q test, transaction costs 7.5 bps per rebalance (returns are net‑of‑cost)
- ML overlay: conservative tilt with a min‑signal threshold

## API Usage (Optional)
Start the API:
```bash
uvicorn portfolio.api.main:app --reload
```
Or use the script:
```bash
python scripts/run_api_server.py
```
Health check:
```bash
curl http://127.0.0.1:8000/health
```
Optimize (example payload):
```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "assets": [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOGL"}],
    "method": "mean_variance",
    "objective": "sharpe",
    "lookback_period": 252
  }'
```
Analyze:
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "assets": [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": "GOOGL"}],
    "weights": {"AAPL": 0.34, "MSFT": 0.33, "GOOGL": 0.33},
    "risk_free_rate": 0.02,
    "lookback_period": 252
  }'
```

## Examples
- `examples/data_pipeline_demo.py` — basic loader + cache
- `examples/walk_forward_demo.py` — quick walk‑forward snapshot
- `examples/ml_workflow_demo.py` — simple ML overlay illustration
- Figures and CSV outputs are written under `examples/figures/` and `examples/outputs/`

## Tests
Run all tests:
```bash
pytest -q
```
Tip: If you’re offline, prefer running tests that don’t require fresh downloads (data service caches under `data/`).

## Data & Configuration
- Primary data source: Yahoo Finance via `yfinance`
- Minimal preprocessing: forward/backward fills; simple validation
- Config via `config.yaml` (sane defaults); environment variables optional
- Handle missing/invalid symbols gracefully; continue with available data

## Troubleshooting
- SSL or rate‑limit issues on first fetch: rerun or use `scripts/fetch_market_data.py` to seed the cache.
- Optimization errors: ensure ≥2 symbols and non‑empty return series.
- Plots look coarse in notebooks: `%config InlineBackend.figure_format = 'retina'` is already set in the lab.
- API not reachable: confirm `uvicorn portfolio.api.main:app --reload` is running on port 8000.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
