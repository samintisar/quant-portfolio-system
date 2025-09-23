Project Demo Notebook Outline (portfolio_optimization_lab.ipynb)

- [ ] Full project demo overview (this notebook)
  - [ ] Explain scope: data pipeline, ML, optimization, backtesting, API, reporting
  - [ ] List core modules used and where code lives (`portfolio/`)

- [ ] Data pipeline demo
  - [ ] Use `YahooFinanceService.list_available_offline_data()` to show cache
  - [ ] `fetch_and_process_data(symbols, period)` and inspect a sample `quality_report`
  - [ ] `fetch_price_data(symbols, period)` consolidated prices for downstream steps

- [ ] ML workflow demo (single symbol)
  - [ ] Build features with `RandomForestPredictor.create_features(df)`
  - [ ] Prepare data, train, and capture metrics via `train()` and `validate_model()`
  - [ ] Plot top feature importance via `plot_feature_importance`

- [ ] Walk-forward backtesting demo
  - [ ] Configure and run `WalkForwardBacktester.run_backtest(symbols, start, end)`
  - [ ] Show strategy metrics and `generate_report()` summary
  - [ ] Plot equity/drawdown of primary strategy vs. SPY (if available)

- [ ] API quick demo (optional)
  - [ ] If API server is running, request `/optimize` and `/analyze` using `requests`
  - [ ] Otherwise, reference `examples/api_test_client.py` usage


- [ ] Create notebook scaffold and context
  - [ ] Add title and short description of goals (mean-variance, CVaR, BL)
  - [ ] Set working directory assumptions and paths for saving figures/outputs

- [ ] Setup parameters and imports
  - [ ] Imports: pandas, numpy, matplotlib.pyplot
  - [ ] From `portfolio.data.yahoo_service` import `YahooFinanceService`
  - [ ] From `portfolio.optimizer.optimizer` import `SimplePortfolioOptimizer`
  - [ ] From `portfolio.performance.calculator` import `SimplePerformanceCalculator`
  - [ ] From `portfolio.performance.visualization` import `plot_equity_curve`, `plot_drawdown_curve`
  - [ ] Define `symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]`, `period = "5y"`

- [ ] Fetch prices and quick data QA
  - [ ] Initialize `YahooFinanceService(use_offline_data=True, offline_data_dir="data")`
  - [ ] `prices = service.fetch_price_data(symbols, period)` (Adj Close)
  - [ ] Optional: per-symbol pipeline via `fetch_and_process_data` and review any `quality_report`
  - [ ] Preview data (dates, shape, head) and basic missing checks

- [ ] Compute returns
  - [ ] `asset_returns = prices.pct_change().dropna()`
  - [ ] Inspect summary stats (mean, std, corr)

- [ ] Mean-Variance optimization (baseline)
  - [ ] Init `opt = SimplePortfolioOptimizer()`
  - [ ] Run `opt.mean_variance_optimize(asset_returns, weight_cap=None)` and store `weights`
  - [ ] Compute `portfolio_returns = (asset_returns * pd.Series(weights)).sum(axis=1)`

- [ ] Performance metrics and report
  - [ ] Init `perf = SimplePerformanceCalculator()`
  - [ ] Optional benchmark: fetch SPY and compute `benchmark_returns`
  - [ ] `metrics = perf.calculate_metrics(portfolio_returns, benchmark_returns)` and print key fields

- [ ] Plots
  - [ ] Equity curve: `plot_equity_curve(portfolio_returns, benchmark_returns)`
  - [ ] Drawdown curve: `plot_drawdown_curve(portfolio_returns)`

- [ ] Efficient frontier preview
  - [ ] Use `opt.get_efficient_frontier(symbols, n_points=15)`
  - [ ] Scatter plot of (volatility, return), highlight baseline point

- [ ] Alternative allocations (optional)
  - [ ] CVaR: `opt.cvar_optimize(asset_returns, alpha=0.05, weight_cap=None)` → compare Sharpe
  - [ ] Black–Litterman: `opt.black_litterman_optimize(asset_returns, weight_cap=None)` → compare Sharpe

- [ ] Quick tuning knobs (simple grid)
  - [ ] Iterate over `risk_model in ["sample", "ledoit_wolf", "oas"]`
  - [ ] Iterate over `weight_cap in [None, 0.35, 0.25, 0.15]`
  - [ ] Iterate over `entropy_penalty in [0.0, 0.01, 0.05]` (via `mean_variance_optimize`)
  - [ ] Record best Sharpe/volatility and corresponding weights in a small table

- [ ] Save artifacts
  - [ ] Save weights to CSV in `examples/outputs/weights_{method}.csv`
  - [ ] Save figures to `examples/figures/`

- [ ] Notes and observations
  - [ ] Capture takeaways: stability vs. concentration, impact of `risk_model`, caps
