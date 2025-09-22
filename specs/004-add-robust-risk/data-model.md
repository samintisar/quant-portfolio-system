# Data Model: Robust Risk Measurement Tools

## RiskConfiguration
- **Purpose**: Encapsulate parameters for risk calculations across covariance, VaR, CVaR, and stress testing.
- **Fields**:
  - confidence_levels: list[float] – e.g., [0.95, 0.99].
  - horizons: list[int] – trading days (1, 10).
  - decay_lambda: float – EWMA decay factor.
  - mc_paths: int – number of Monte Carlo simulations.
  - seed: int | None – reproducibility control.
  - stress_scenarios: list[str] – scenario identifiers to apply.
  - data_frequency: Literal["daily", "weekly", "monthly"] – aligns with ingestion cadence.
  - eports: list[str] – outputs requested ("covariance", "var", "cvar", "stress", "visualizations").
- **Validation**: Confidence levels between 0.9 and 0.999; horizons >= 1; mc_paths within [1_000, 100_000]; ensure scenarios exist in catalog when referenced.

## PortfolioSnapshot
- **Purpose**: Represent aligned portfolio returns and weights consumed by risk engines.
- **Fields**:
  - sset_ids: list[str] – identifiers consistent with ingestion outputs.
  - eturns: pandas.DataFrame – time-indexed returns (rows) by asset (columns).
  - weights: pandas.Series – current portfolio weights keyed by asset.
  - actor_exposures: pandas.DataFrame | None – optional factor loadings for exposure plots.
  - 	imestamp: datetime – snapshot extraction time.
- **Validation**: Returns index monotonic, no missing weights for assets, weights sum within tolerance of 1.0, align exposures columns with asset_ids.

## CovarianceResult
- **Purpose**: Store covariance matrices generated via different estimators.
- **Fields**:
  - method: Literal["sample", "ledoit_wolf", "ewma"].
  - matrix: numpy.ndarray – symmetric matrix sized len(asset_ids) x len(asset_ids).
  - metadata: dict[str, Any] – includes shrinkage value, decay_lambda, sample_size.
- **Validation**: Positive semi-definite check (eigenvalues >= 0 minus tolerance), metadata minimal fields present.

## RiskMetricEntry
- **Purpose**: Hold VaR/CVaR values per confidence level and horizon.
- **Fields**:
  - metric: Literal["var", "cvar"].
  - confidence: float.
  - horizon: int.
  - alue: float – expressed as loss (positive number representing magnitude).
  - method: Literal["historical", "parametric", "monte_carlo"].
  - sample_size: int – count of observations or simulated paths.
- **Validation**: Value >= 0; method-specific metadata recorded (e.g., distribution assumptions) in supplementary dict.

## StressScenario
- **Purpose**: Describe macro or custom shocks applied to the portfolio.
- **Fields**:
  - scenario_id: str.
  - 
ame: str.
  - description: str.
  - shock_type: Literal["factor", "asset", "macro"].
  - parameters: dict[str, Any] – e.g., factor deltas, correlation shifts.
  - horizon: int – days applied.
- **Validation**: scenario_id unique, parameters schema validated per shock_type, horizon > 0.

## StressImpact
- **Purpose**: Capture portfolio performance under a scenario.
- **Fields**:
  - scenario_id: str.
  - expected_loss: float.
  - worst_case_loss: float.
  - actor_contributions: dict[str, float].
  - 
otes: str | None – commentary.
- **Validation**: expected_loss and worst_case_loss >= 0; scenario_id matches StressScenario.

## RiskReport
- **Purpose**: Aggregate outputs from risk engines for API/CLI consumers.
- **Fields**:
  - configuration: RiskConfiguration.
  - covariance_results: list[CovarianceResult].
  - isk_metrics: list[RiskMetricEntry].
  - stress_impacts: list[StressImpact].
  - actor_exposure_summary: pandas.DataFrame – aggregated exposures by factor category.
  - generated_at: datetime.
- **Validation**: All requested reports populated; duplicates prevented by (metric, method, confidence, horizon) unique constraint.

## VisualizationArtifact
- **Purpose**: Represent rendered plots for distribution and exposure analysis.
- **Fields**:
  - rtifact_id: str.
  - 	ype: Literal["factor_exposure", "var_timeseries", "cvar_distribution", "stress_bar"].
  - path: Path – filesystem location under data/storage/reports/.
  - ormat: Literal["png", "svg", "html"].
  - generated_at: datetime.
- **Validation**: Path resolves under writable reports directory; format matches generator output.

## Relationships
- RiskReport references RiskConfiguration, PortfolioSnapshot, and derived artifacts.
- Each StressImpact ties back to a StressScenario defined in the config catalog.
- Visualization artifacts correspond to entries in RiskReport for traceability.
