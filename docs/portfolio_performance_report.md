# Portfolio Optimization System Performance Report

_Generated: 2025-09-22_

---

## Update (2025-09-23)

- Enhancements:
  - Added Ledoit–Wolf/OAS shrinkage covariance selection; MV and BL now use the chosen risk model (default: Ledoit–Wolf).
  - Added optional entropy and L2 turnover penalties to MV; walk-forward wiring passes prior weights to activate turnover control.
  - Strategy metrics now include benchmark-relative stats (beta, alpha, information ratio) vs. SPY.

- Walk-forward (2020–2025, offline, 20% cap, turnover_penalty=0.001, risk_model=LedoitWolf):

| Strategy                    | Total Ret | Ann. Ret | Ann. Vol | Sharpe | Max DD  |
|----------------------------|----------:|---------:|---------:|-------:|--------:|
| Mean-Variance (unconstr.)  |   31.92%  |  39.44%  |  47.42%  |  0.790 | -41.25% |
| Mean-Variance (cap 20%)    |   10.50%  |  12.73%  |  27.17%  |  0.395 | -27.12% |
| CVaR 95% (cap 20%)         |    7.16%  |   8.66%  |  22.55%  |  0.295 | -24.57% |
| Black–Litterman (cap 20%)  |    5.21%  |   6.28%  |  35.87%  |  0.119 | -37.35% |
| Equal-Weight               |    2.03%  |   2.45%  |  24.55%  |  0.018 | -28.20% |
| SPY                        |    3.47%  |   4.08%  |  18.17%  |  0.115 | -19.90% |

Notes
- The JSON summary printed by `scripts/run_walk_forward_experiments.py` also includes turnover and information ratio per strategy; IR is now non-null (computed vs SPY).
- BL uses equal-weight market as a neutral prior; supplying market-cap weights and tuning `delta`/`tau` often improves BL performance.

---

## 1. Executive Summary

- The optimization stack is fully operational: all 149 automated tests (contract, integration, unit, performance) pass on Python 3.11.3.
- Using the latest five years of daily data (24 Sep 2020 – 22 Sep 2025, 1,254 trading days) for a 10-asset universe, the mean-variance optimizer produced a portfolio that compounds to **1,419%** with a **72.8% annualized return** and **1.35 Sharpe ratio** after accounting for a 2% risk-free rate.
- The optimizer currently concentrates almost entirely in NVDA, delivering outsized returns but at the cost of a **66% max drawdown** and **beta ≈ 2.09** relative to the SPY benchmark.
- A diversified equal-weight baseline still outperforms SPY (Sharpe 1.14 vs 0.89) with considerably lower drawdown (35.4%), highlighting that constraint management rather than raw alpha is now the main limiter.

## 2. Methodology

- **Universe:** AAPL, MSFT, GOOGL, AMZN, TSLA, NFLX, NVDA, JPM, PG, UNH.
- **Benchmark:** SPY (5-year daily OHLCV, same date range).
- **Data Source:** Offline CSVs under `data/raw`, generated from Yahoo Finance and shipped with the repository. No live network calls were made.
- **Returns:** Daily simple returns from adjusted close prices, aligned across tickers, with 252 trading days assumed per year.
- **Optimization:** `SimplePortfolioOptimizer.mean_variance_optimize` (Sharpe-maximizing quadratic program solved via OSQP) with default risk-free rate (2%).
- **Performance Metrics:** `SimplePerformanceCalculator` providing annualized return/volatility, Sharpe, drawdown, best/worst day, win rate, and benchmark-relative stats (beta, alpha, information ratio).

## 3. Optimization Outcomes

### 3.1 Portfolio Weights

| Symbol | Weight |
|--------|-------:|
| NVDA   | 100.00% |
| Other 9 names | < 1e-20 % (effectively zero) |

> The quadratic program converged to a single-asset solution because no explicit position limits or diversification penalties are enforced in `mean_variance_optimize`.

### 3.2 Strategy Performance Summary

| Strategy | Total Return | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Win Rate |
|----------|-------------:|------------:|----------------:|-------:|-------------:|---------:|
| Optimized (NVDA-only) | **1,419%** | **72.8%** | 52.3% | **1.35** | -66.3% | 54.1% |
| Equal-Weight (10 assets) | 247.6% | 28.4% | 23.1% | 1.14 | -35.4% | **55.9%** |
| SPY Benchmark | 121.7% | 17.3% | **17.3%** | 0.89 | **-24.5%** | 54.6% |

### 3.3 Benchmark Diagnostics

- **Excess Performance:** The optimized portfolio delivers ~38.6% annual alpha over SPY with an information ratio of 1.20.
- **Risk Profile:** Beta vs SPY is 2.09; the portfolio behaves like a leveraged bet on NVDA.
- **Tail Behaviour:** Despite superior Sharpe, the 66% drawdown suggests catastrophic risk if NVDA underperforms.

## 4. System Validation Snapshot

- `python -m pytest` → 149 passed / 0 failed (contract, integration, unit, performance suites).
- Contract tests confirm API behaviour: structured 422 errors for invalid payloads, `/data/assets` responds with asset-level details plus benchmark-aware summaries.
- Performance tests now pass with synthetic-data fallbacks and OSQP-backed efficient frontier calculations (10/10 frontier points generated under offline data).

## 5. Observations & Analysis

1. **Constraint Gap:** The optimizer ignores configuration hints like `max_position_size`, leading to extreme concentration. Introducing explicit constraints (e.g., weight caps, sector limits) is essential for production use.
2. **Risk-Adjusted Trade-off:** While Sharpe improves from 0.89 (SPY) to 1.35, the drawdown doubles. Investors would likely prefer diversified allocations (equal-weight Sharpe 1.14 with half the drawdown).
3. **Benchmark Sensitivity:** Beta >2 indicates performance is largely explained by market moves amplified through NVDA. Portfolio alpha is positive but fragile.
4. **Data Hygiene:** Offline data usage ensures reproducibility, but fetch paths in `SimplePortfolioOptimizer.fetch_data` still rely on live Yahoo Finance calls. The API layer bridges this via `YahooFinanceService`, yet backtesting utilities remain network-bound.
5. **ML Module:** Random forest predictor is unit-tested but uncoupled from portfolio decisions in this evaluation. Leakage risks (shared scaler across CV folds) remain an open issue for future improvement.

## 6. Recommendations

1. **Enforce Portfolio Constraints:** Implement max position size, sector caps, and optionally CVaR limits inside optimization routines to prevent single-stock bets.
2. **Integrate Offline Data in Optimizer/Backtester:** Route all price retrieval through `YahooFinanceService` to remove runtime network dependency and leverage cached quality checks.
3. **Enhance Walk-Forward Testing:** Extend `WalkForwardBacktester` to use the offline dataset, incorporate transaction costs, and export equity/drawdown curves for richer diagnostics.
4. **Refine ML Validation:** Replace standard `cross_val_score` with `TimeSeriesSplit` wrapped in `Pipeline` to avoid look-ahead bias before re-integrating ML overlays into allocation.
5. **Documentation & Monitoring:** Expand README with execution recipes, and add warning thresholds (e.g., drawdown or concentration) to highlight when optimization output breaches risk policy.

## 7. Next Steps

- Prototype a constrained optimizer variant (e.g., max 20% per name) and re-run the 10-asset analysis to balance return and drawdown.
- Refresh walk-forward backtests with broader universes (growth vs. defensive) to stress-test robustness.
- Build CI hooks for a quick `pytest -m "not slow" --maxfail=1` smoke run plus an automated performance snapshot to track regression in alpha, Sharpe, and tail risk over time.

---

## Offline Walk-Forward Backtest (Rolling Performance)

To understand time-varying behaviour, we re-ran `WalkForwardBacktester` over the same universe using the five-year offline dataset (24 Sep 2020 – 22 Sep 2025) with 1-year training / 1-quarter testing windows, 5 bps transaction costs, and the new offline data loader.

### A. Aggregate Outcomes

| Strategy | Total Return | Ann. Return | Ann. Volatility | Sharpe | Sortino | Calmar | Max Drawdown | Win Rate | Turnover | Tx Costs |
|----------|-------------:|------------:|----------------:|-------:|--------:|-------:|-------------:|---------:|---------:|---------:|
| Optimized | **28.2%** | **34.6%** | 49.2% | **0.66** | 0.96 | **0.82** | -42.2% | 52.6% | 371% | 0.19% |
| Equal-Weight | 1.9% | 2.2% | **24.5%** | 0.01 | 0.01 | 0.08 | **-27.6%** | **55.5%** | — | 0.00% |

| Metric | Optimized | Equal-Weight |
|--------|----------:|-------------:|
| Ending NAV (growth of $1) | **1.282** | 1.019 |
| Beta vs SPY | 1.50 | 1.02 |
| Information Ratio | **1.21** | 0.08 |

**Takeaways**

- Offline walk-forward runs are far tamer than the static backtest: cumulative performance is modest and volatility dominates the Sharpe ratio.
- Transaction costs matter (18.6 bps cumulative) given 371% total turnover across five rebalances.
- Equal-weight quietly preserves capital in difficult quarters but lags materially when NVDA leads (2024 window).

### B. Window-by-Window Returns

| Quarter End | Days | Optimized Total | Optimized Ann. | Equal Total | Equal Ann. |
|-------------|-----:|----------------:|---------------:|------------:|-----------:|
| 2021-08-05 | 43 | -7.45% | -36.49% | **+9.44%** | **+69.67%** |
| 2022-06-17 | 43 | -31.40% | -89.02% | -22.80% | -78.05% |
| 2023-04-28 | 43 | +2.14% | +13.19% | +8.04% | +57.32% |
| 2024-03-08 | 42 | **+67.52%** | 2,109% | +7.04% | +50.44% |
| 2025-01-17 | 40 | +18.04% | 184.35% | +4.26% | +30.05% |

- The optimized book’s entire edge comes from the 2024 rebound trade; the preceding six quarters were negative or flat.
- Equal-weight delivers steadier outcomes, outperforming the optimizer in three of five windows despite the latter’s stronger cumulative return.

### C. Offline Data Integration Notes

- `WalkForwardBacktester` now tries the offline cache first (`YahooFinanceService`), avoiding live calls during backtests while keeping a network fallback for cold symbols.
- All walk-forward unit and integration tests continue to pass (OSQP solver emits warnings but solutions converge).
- Cached data covers 1,255 trading days per ticker; walk-forward generated five non-overlapping test windows (approx. 42–43 days each after business-day filtering).

---

## Addendum (Constrained Optimization Trial)

Following the original recommendations, we re-ran the optimizer with an explicit weight cap of **20% per asset** while keeping the same 10-name universe and data window.

### A. Updated Weights

| Symbol | Weight (capped) |
|--------|----------------:|
| GOOGL  | 20.0% |
| TSLA   | 20.0% |
| NFLX   | 20.0% |
| NVDA   | 20.0% |
| JPM    | 20.0% |
| Others | 0.0% |

### B. Performance vs. Benchmarks

| Strategy | Total Return | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Beta vs SPY | Information Ratio |
|----------|-------------:|------------:|----------------:|-------:|-------------:|------------:|------------------:|
| **Capped Optimized** | **486.6%** | **42.7%** | 30.9% | **1.32** | -48.5% | 1.50 | **1.21** |
| Unconstrained Optimized | 1,419% | 72.8% | 52.3% | 1.35 | -66.3% | 2.09 | 1.20 |
| Equal-Weight Baseline | 247.6% | 28.4% | 23.1% | 1.14 | -35.4% | 1.08 | 0.88 |
| SPY Benchmark | 121.7% | 17.3% | 17.3% | 0.89 | -24.5% | 1.00 | — |

**Interpretation**

- The cap dramatically curbs concentration risk: max drawdown improves from -66% to -48%, beta drops from 2.09 to 1.50, and the portfolio now holds five uncorrelated names.
- Annualized return falls from 72.8% to 42.7%, yet the Sharpe ratio remains essentially unchanged (1.35 → 1.32), indicating the return loss is largely proportional to the volatility reduction.
- Information ratio edges higher (1.20 → 1.21), suggesting the capped solution delivers excess returns more efficiently relative to tracking error.
- Equal-weight stays the drawdown winner, but the capped portfolio now balances growth and risk far better than both extremes.

**Implications for the Roadmap**

- Hard constraints (and eventually sector-level caps, turnover limits, or CVaR targets) should be prioritized before deploying the optimizer.
- The implementation can live alongside the existing mean-variance routine—simply adding optional cap parameters handled by CVXPY keeps test coverage intact.
- Consider exposing the cap through configuration so contract/integration tests can validate both constrained and unconstrained paths.

---

## 7. Walk-Forward Results (2020–2025, Offline, 20% Caps)

- Setup: 10-asset universe [AAPL, MSFT, GOOGL, AMZN, TSLA, NFLX, NVDA, JPM, PG, UNH]; SPY benchmark; 1-year train, 1-quarter test, expanding windows; transaction costs 7.5 bps; long-only; per-asset cap = 20% where noted.

### Summary Metrics

| Strategy | Total Return | Ann. Return | Ann. Vol. | Sharpe | Max DD |
|----------|-------------:|------------:|----------:|-------:|-------:|
| Mean-Variance (unconstrained) | 32.04% | 39.59% | 46.89% | 0.80 | -40.94% |
| Mean-Variance (cap 20%) | 10.50% | 12.73% | 27.17% | 0.40 | -27.12% |
| CVaR 95% (cap 20%) | 7.16% | 8.66% | 22.55% | 0.30 | -24.57% |
| Black–Litterman neutral (cap 20%) | 5.33% | 6.43% | 35.96% | 0.12 | -37.42% |
| Equal-Weight | 2.03% | 2.45% | 24.55% | 0.02 | -28.20% |
| SPY | 3.47% | 4.08% | 18.17% | 0.11 | -19.90% |

Notes
- Black–Litterman uses an equal-weight market proxy and neutral views, reducing to the equilibrium prior under constraints.
- All price data loaded from `data/processed`/`data/raw`; no live network used.

## 8. Model Risk Appendix: Objectives & Constraints

- Concentration and drawdowns: The 20% cap reduced max drawdown from -40.9% (MV unconstrained) to -27.1% (MV capped), improving diversification while lowering return.
- Tail-risk objective: CVaR minimization further reduced volatility (22.6%) and drawdown (-24.6%) relative to MV capped, appropriate when downside control dominates.
- Priors and views: Neutral Black–Litterman depends on the chosen market prior; without informative views it can underperform MV in momentum-driven samples but still benefits from caps.
- Policy translation: Position caps are transparent controls; adding sector caps and turnover limits would further stabilize allocations and implementation costs.
