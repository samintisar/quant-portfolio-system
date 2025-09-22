# Research Phase: Core Forecasting Models for Returns & Volatility

## Research Questions Addressed

### 1. Optimal Number of Regimes for HMM Models

**Enhancement**: While Gaussian HMMs with 3 regimes remain the core design, financial returns often exhibit **heavy tails and skewness**. Future iterations should test **Student-t HMM** or **mixture-of-Gaussian emissions** to better capture extreme market swings.

**Decision**: 3 regimes (Bull, Bear, High Volatility)
**Rationale**:
- Financial markets commonly exhibit 3 distinct regimes based on empirical research
- Bull markets: positive returns, moderate volatility
- Bear markets: negative returns, moderate to high volatility
- High volatility: extreme price swings regardless of direction
- More than 3 regimes leads to overfitting and reduced interpretability
- Fewer than 3 regimes misses important market state distinctions

**Alternatives Considered**:
- 2 regimes: Too simplistic, misses volatility distinctions
- 4 regimes: Risk of overfitting, harder to interpret and validate
- Dynamic regime detection: More complex, computational overhead

**Supporting Evidence**:
- Hamilton (1989) foundational work on regime switching used 2-3 states
- Ang and Bekaert (2002) found 3 regimes optimal for international equity markets
- Guidolin and Timmermann (2008) validated 3-regime models for asset allocation

### 2. Best Practices for Time Series Forecasting in Finance

**Enhancement**: Consider adding **Regime-Switching GARCH** models, which integrate regime dynamics with volatility clustering. These hybrid models can capture structural breaks more effectively than standalone ARIMA or GARCH.

**Decision**:
- ARIMA(p,d,q) with automated parameter selection using AIC/BIC
- GARCH(1,1) as baseline with extensions to EGARCH for asymmetric effects
- Rolling window estimation with 252 trading days (1 year)
- Out-of-sample validation with walk-forward testing

**Rationale**:
- ARIMA is well-established for financial time series, interpretable
- GARCH(1,1) captures volatility clustering effectively
- Rolling windows adapt to changing market conditions
- Walk-forward validation prevents look-ahead bias

**Best Practices**:
- Model diagnostics: Ljung-Box test for residual autocorrelation
- Volatility forecast evaluation: MSE, QLIKE loss functions
- Regime stability checks: Transition matrix analysis
- Statistical significance: p-values < 0.05 for all parameter estimates

### 3. Hidden Markov Model Implementation Approach

**Decision**:
- Gaussian HMM with continuous observation distributions
- Baum-Welch algorithm for parameter estimation
- Viterbi algorithm for regime decoding
- Multivariate features: returns, volatility, volume, economic indicators

**Rationale**:
- Gaussian distributions appropriate for normalized financial data
- Baum-Welch is standard EM algorithm for HMM training
- Viterbi provides most likely regime sequence
- Multiple features improve regime identification accuracy

**Validation Methods**:
- Regime persistence: Expected duration in each state
- Transition probability stability over time
- Regime correlation with known market events
- Out-of-sample regime prediction accuracy

### 4. Belief Networks for Economic Scenario Modeling

**Enhancement**: Data quality and reporting lags for economic indicators (e.g., GDP revisions, CPI delays) pose risks. To mitigate **look-ahead bias**, scenarios should explicitly account for data release lags and revision uncertainty. Additionally, stress-testing the Bayesian network with extreme shocks (e.g., sudden inflation spikes) is recommended.

**Decision**:
- Bayesian Network structure learning from economic data
- Conditional probability tables for scenario impacts
- Monte Carlo sampling for scenario generation
- Integration with regime models for conditional forecasts

**Rationale**:
- Bayesian networks handle uncertainty in economic relationships
- Conditional probabilities capture scenario dependencies
- Monte Carlo provides probabilistic scenario outcomes
- Integration enables regime-aware scenario analysis

**Key Economic Indicators**:
- GDP growth rates
- Inflation (CPI/PPI)
- Interest rates (yield curve)
- Employment data
- Consumer confidence
- Manufacturing indices

### 5. Statistical Validation Framework

**Enhancement**: The current **Sharpe ratio >1.5 target** may be unrealistic in live markets. Instead, use **relative benchmarks** (e.g., Sharpe uplift vs. passive strategy) as the primary measure of success. This ensures more robust evaluation under realistic market frictions.

**Decision**:
- Diebold-Mariano tests for forecast accuracy comparison
- Kolmogorov-Smirnov tests for distributional assumptions
- Bootstrap confidence intervals for forecast uncertainty
- Backtesting with realistic transaction costs

**Rationale**:
- Diebold-Mariano provides statistical significance of forecast improvements
- Distributional tests validate model assumptions
- Bootstrap methods account for forecast uncertainty
- Transaction costs ensure economic significance

**Performance Metrics**:
- Sharpe ratio (> 1.5 target)
- Maximum drawdown (< 15% limit)
- Information ratio vs benchmark
- Turnover and transaction cost analysis

### 6. Dependencies and Technology Stack

**Core Libraries**:
- `statsmodels`: ARIMA, statistical tests
- `arch`: GARCH family models
- `hmmlearn`: Hidden Markov Models
- `pgmpy`: Bayesian networks
- `scikit-learn`: Machine learning utilities
- `pandas`: Data manipulation
- `numpy`: Numerical computations

**Validation Libraries**:
- `pytest`: Unit testing
- `scipy`: Statistical functions
- `matplotlib/plotly`: Visualization
- `joblib`: Parallel processing

**Configuration Management**:
- JSON for model parameters
- YAML for pipeline configuration
- SQLite for metadata storage

### 7. Performance Requirements Validation

**Processing Speed**:
- Target: 10M data points in <30 seconds
- Approach: Vectorized operations, parallel processing
- Validation: Benchmark with synthetic datasets

**Memory Efficiency**:
- Target: <4GB memory usage
- Approach: Chunked processing, memory mapping
- Validation: Memory profiling during runs

**Scalability**:
- Linear scaling with dataset size
- Horizontal scaling for multiple assets
- Efficient data streaming for large datasets

## Resolved Unknowns

✅ **NEEDS CLARIFICATION: Optimal number of regimes for HMM models** → 3 regimes (Bull, Bear, High Volatility)

## Research Summary

**Additional Note**: Although classical models remain the foundation, benchmarking against **modern ML baselines** (e.g., XGBoost on lagged features, simple LSTM/Transformer architectures) would help validate whether traditional approaches remain competitive.

The research phase successfully resolved all key unknowns and established best practices for implementing core forecasting models:

1. **3-regime HMM** provides optimal balance between market state discrimination and overfitting
2. **ARIMA/GARCH** combination with proper validation ensures robust time series forecasting
3. **Bayesian networks** effectively model economic scenario dependencies
4. **Statistical validation framework** ensures model reliability and financial soundness
5. **Performance targets** are achievable with modern Python libraries and proper optimization

All constitutional requirements can be met with this approach, and the design is ready for Phase 1 implementation planning.

---
*Research completed: 2025-09-20 | Total unknowns resolved: 1*