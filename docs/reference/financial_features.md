# Financial Features: Mathematical Formulations

This document provides comprehensive mathematical formulations for all financial features calculated in the quantitative trading system, including returns, volatility, and momentum indicators.

## Table of Contents

1. [Returns Calculations](#returns-calculations)
2. [Volatility Measures](#volatility-measures)
3. [Momentum Indicators](#momentum-indicators)
4. [Risk Metrics](#risk-metrics)
5. [Performance Metrics](#performance-metrics)

## Returns Calculations

### Simple Returns (Arithmetic Returns)

**Formula:**
```
r_simple,t = (P_t - P_t-1) / P_t-1
```

**Where:**
- `r_simple,t` = Simple return at time t
- `P_t` = Price at time t
- `P_t-1` = Price at time t-1

**Properties:**
- Range: (-1, ∞)
- Additive across securities but not across time
- Used for most financial applications

### Logarithmic Returns

**Formula:**
```
r_log,t = ln(P_t / P_t-1)
```

**Where:**
- `r_log,t` = Logarithmic return at time t
- `ln` = Natural logarithm

**Properties:**
- Range: (-∞, ∞)
- Additive across time: `r_log,t1→t3 = r_log,t1→t2 + r_log,t2→t3`
- Approximately equal to simple returns for small values: `r_log ≈ r_simple`
- Used in continuous compounding and time series analysis

### Percentage Returns

**Formula:**
```
r_pct,t = r_simple,t × 100
```

**Properties:**
- Range: (-100, ∞)
- Used for display and reporting purposes

### Multi-Period Returns

**Formula (Simple):**
```
r_multi,t→t+n = P_t+n / P_t - 1
```

**Formula (Logarithmic):**
```
r_log_multi,t→t+n = ln(P_t+n / P_t) = Σ(i=1 to n) r_log,t+i
```

### Cumulative Returns

**Formula (Simple):**
```
R_cumulative = (1 + r₁) × (1 + r₂) × ... × (1 + r_n) - 1
```

**Formula (Logarithmic):**
```
R_log_cumulative = exp(Σ(i=1 to n) r_log,i) - 1
```

### Annualized Returns

**Formula:**
```
r_annualized = (1 + R_total)^(1/T) - 1
```

**Where:**
- `R_total` = Total return over the period
- `T` = Number of years

### Excess Returns

**Formula:**
```
r_excess,t = r_asset,t - r_benchmark,t - r_f
```

**Where:**
- `r_asset,t` = Asset return at time t
- `r_benchmark,t` = Benchmark return at time t
- `r_f` = Risk-free rate

### Time-Weighted Returns (TWR)

**Formula:**
```
TWR = (1 + r₁) × (1 + r₂) × ... × (1 + r_n) - 1
```

### Money-Weighted Returns (MWR/IRR)

**Formula:**
```
Σ(i=0 to n) CF_i / (1 + IRR)^(t_i/T) = 0
```

**Where:**
- `CF_i` = Cash flow at time i
- `IRR` = Internal Rate of Return
- `t_i` = Time of cash flow i
- `T` = Total time period

## Volatility Measures

### Standard Deviation Volatility

**Formula:**
```
σ = √[Σ(i=1 to n) (r_i - μ_r)² / (n-1)]
```

**Where:**
- `σ` = Standard deviation
- `r_i` = Return at time i
- `μ_r` = Mean return
- `n` = Number of observations

### Annualized Volatility

**Formula:**
```
σ_annualized = σ_period × √N
```

**Where:**
- `σ_period` = Volatility per period
- `N` = Number of periods per year

### Parkinson Volatility

**Formula:**
```
σ_parkinson = √[(1 / (4 × ln(2))) × (1/n) × Σ(i=1 to n) ln(H_i / L_i)²] × √N
```

**Where:**
- `H_i` = High price at time i
- `L_i` = Low price at time i
- `n` = Number of observations

### Garman-Klass Volatility

**Formula:**
```
σ_gk = √[(1/2) × (1/n) × Σ(i=1 to n) ln(H_i / L_i)² -
        (2ln(2) - 1) × (1/n) × Σ(i=1 to n) ln(C_i / O_i)²] × √N
```

**Where:**
- `O_i` = Opening price at time i
- `C_i` = Closing price at time i

### Yang-Zhang Volatility

**Formula:**
```
σ_yz = √[σ_overnight² + k × σ_intraday² + (1 - k) × σ_close_to_close²] × √N
```

**Where:**
- `k = 0.34 / (1.34 + (window + 1) / (window - 1))`
- `σ_overnight² = (1/n) × Σ(i=1 to n) ln(O_i / C_i-1)²`
- `σ_intraday² = (1/n) × Σ(i=1 to n) ln(C_i / O_i)²`
- `σ_close_to_close² = (1/n) × Σ(i=1 to n) ln(C_i / C_i-1)²`

### Exponentially Weighted Moving Average (EWMA) Volatility

**Formula:**
```
σ²_t = λ × σ²_t-1 + (1 - λ) × r²_t-1
```

**Where:**
- `λ` = Decay factor (typically 0.94 for daily data)
- `r_t-1` = Return at time t-1

### GARCH(1,1) Volatility

**Formula:**
```
σ²_t = ω + α × r²_t-1 + β × σ²_t-1
```

**Where:**
- `ω` = Long-run variance parameter
- `α` = ARCH parameter (reaction to shocks)
- `β` = GARCH parameter (persistence of volatility)
- Constraint: `α + β < 1` for stationarity

### GJR-GARCH Volatility (Asymmetric GARCH)

**Formula:**
```
σ²_t = ω + α × r²_t-1 + γ × r²_t-1 × I(r_t-1 < 0) + β × σ²_t-1
```

**Where:**
- `I(r_t-1 < 0)` = Indicator function (1 if return negative, 0 otherwise)
- `γ` = Asymmetric coefficient

### Average True Range (ATR)

**Formula:**
```
TR_t = max(H_t - L_t, |H_t - C_t-1|, |L_t - C_t-1|)
ATR_t = (1/n) × Σ(i=t-n+1 to t) TR_i
```

### Realized Volatility

**Simple Method:**
```
RV = √[(1/n) × Σ(i=1 to n) r²_i]
```

**Hansen-Hodrick Method:**
```
RV_HH = √[r'Ωr]
```
where Ω is the Newey-West covariance matrix

### Volatility of Volatility (Vol-of-Vol)

**Formula:**
```
VoV = σ[ln(σ_t)]
```
Standard deviation of log-volatility changes

## Momentum Indicators

### Simple Momentum

**Formula:**
```
M_t = P_t - P_t-n
```

**Normalized (Percentage):**
```
M_pct,t = (P_t / P_t-n - 1) × 100
```

### Relative Strength Index (RSI)

**Formula:**
```
RS = Average Gain / Average Loss
RSI = 100 - [100 / (1 + RS)]
```

**Wilder's Smoothing:**
```
AvgGain_t = (AvgGain_t-1 × (n-1) + Gain_t) / n
AvgLoss_t = (AvgLoss_t-1 × (n-1) + Loss_t) / n
```

**Where:**
- `Gain_t = max(r_t, 0)`
- `Loss_t = max(-r_t, 0)`

### Rate of Change (ROC)

**Formula:**
```
ROC_t = [(P_t - P_t-n) / P_t-n] × 100
```

### Stochastic Oscillator

**%K:**
```
%K_t = [(C_t - L_L,n) / (H_H,n - L_L,n)] × 100
```

**%D:**
```
%D_t = SMA(%K_t, m)
```

**Where:**
- `C_t` = Close price at time t
- `L_L,n` = Lowest low over n periods
- `H_H,n` = Highest high over n periods
- `SMA` = Simple moving average
- `m` = Smoothing period for %D

### Moving Average Convergence Divergence (MACD)

**MACD Line:**
```
MACD_t = EMA(P_t, fast) - EMA(P_t, slow)
```

**Signal Line:**
```
Signal_t = EMA(MACD_t, signal_period)
```

**Histogram:**
```
Histogram_t = MACD_t - Signal_t
```

**Where:**
- `EMA(P, n) = P_t × α + (1-α) × EMA_t-1`
- `α = 2 / (n + 1)`

### Williams %R

**Formula:**
```
%R_t = -100 × [(H_H,n - C_t) / (H_H,n - L_L,n)]
```

### Commodity Channel Index (CCI)

**Formula:**
```
TP_t = (H_t + L_t + C_t) / 3
SMA_TP_t = (1/n) × Σ(i=t-n+1 to t) TP_i
MD_t = (1/n) × Σ(i=t-n+1 to t) |TP_i - SMA_TP_t|
CCI_t = (TP_t - SMA_TP_t) / (0.015 × MD_t)
```

### Money Flow Index (MFI)

**Formula:**
```
TP_t = (H_t + L_t + C_t) / 3
MF_t = TP_t × Volume_t
MFR = Positive_MF_Sum / Negative_MF_Sum
MFI_t = 100 - [100 / (1 + MFR)]
```

### Ultimate Oscillator

**Formula:**
```
BP_t = C_t - min(L_t, C_t-1)
TR_t = max(H_t, C_t-1) - min(L_t, C_t-1)
Avg7 = Σ(BP_t-6→t) / Σ(TR_t-6→t)
Avg14 = Σ(BP_t-13→t) / Σ(TR_t-13→t)
Avg28 = Σ(BP_t-27→t) / Σ(TR_t-27→t)
UO_t = 100 × (4×Avg7 + 2×Avg14 + Avg28) / 7
```

## Risk Metrics

### Value at Risk (VaR) - Historical Simulation

**Formula:**
```
VaR_α = -percentile(returns, α × 100)
```

### Conditional Value at Risk (CVaR)

**Formula:**
```
CVaR_α = -(1/α) × Σ(r_i ≤ VaR_α) r_i
```

### Beta (CAPM)

**Formula:**
```
β = Cov(r_asset, r_market) / Var(r_market)
```

### Alpha (Jensen's Alpha)

**Formula:**
```
α = r_asset - [r_f + β × (r_market - r_f)]
```

### Information Ratio

**Formula:**
```
IR = E[r_asset - r_benchmark] / σ[r_asset - r_benchmark]
```

### Treynor Ratio

**Formula:**
```
TR = E[r_asset - r_f] / β
```

### Sortino Ratio

**Formula:**
```
SR = E[r_asset - MAR] / σ_downside
```

**Where:**
- `MAR` = Minimum acceptable return
- `σ_downside` = Standard deviation of returns below MAR

### Maximum Drawdown

**Formula:**
```
DD_t = (PV_t - PV_peak,t) / PV_peak,t
Max_DD = min(DD_t)
```

**Where:**
- `PV_t` = Portfolio value at time t
- `PV_peak,t` = Peak portfolio value up to time t

### Calmar Ratio

**Formula:**
```
CR = r_annualized / |Max_DD|
```

## Performance Metrics

### Sharpe Ratio

**Formula:**
```
SR = E[r_asset - r_f] / σ_asset
```

### Tracking Error

**Formula:**
```
TE = σ[r_asset - r_benchmark]
```

### Up/Down Capture Ratio

**Up Capture:**
```
UpCapture = [Σ(r_asset > 0) r_asset] / [Σ(r_benchmark > 0) r_benchmark]
```

**Down Capture:**
```
DownCapture = [Σ(r_asset < 0) r_asset] / [Σ(r_benchmark < 0) r_benchmark]
```

### Profit Factor

**Formula:**
```
PF = Σ(r_profitable) / |Σ(r_unprofitable)|
```

### Win Rate

**Formula:**
```
WinRate = N_profitable / N_total
```

## Implementation Notes

### Annualization Factors

- **Daily:** 252 trading days
- **Weekly:** 52 weeks
- **Monthly:** 12 months
- **Quarterly:** 4 quarters

### Statistical Assumptions

1. **Returns:** Assumed to be stationary and ergodic
2. **Volatility:** Assumes volatility clustering (GARCH effects)
3. **Correlations:** Assumes time-varying correlations
4. **Normality:** Most metrics assume normal distribution; use robust alternatives for fat tails

### Data Requirements

- Minimum 30 observations for meaningful statistics
- 252+ observations for annualized metrics
- 1000+ observations for reliable volatility estimation
- OHLC data required for range-based volatility measures

### Numerical Considerations

- Use log returns for multi-period calculations
- Handle zero prices and dividends appropriately
- Account for missing data and survivorship bias
- Consider transaction costs and slippage in live trading

## References

- Hull, J. (2022). *Options, Futures and Other Derivatives*
- Tsay, R. (2010). *Analysis of Financial Time Series*
- Alexander, C. (2008). *Market Risk Analysis*
- Jorion, P. (2006). *Value at Risk*

---
*Document Version: 1.0.0*
*Last Updated: 2025-09-19*