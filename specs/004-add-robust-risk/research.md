# Phase 0 Research: Robust Risk Measurement Tools

## Covariance Estimation Strategy
- **Decision**: Use numpy for sample covariance, sklearn.covariance.LedoitWolf for shrinkage, and an in-house EWMA implementation with configurable decay.
- **Rationale**: Aligns with existing dependency stack, offers numerically stable shrinkage, and keeps EWMA transparent for tuning.
- **Alternatives Considered**: statsmodels covariance utilities (less direct control over shrinkage parameters); pandas.DataFrame.ewm().cov() (harder to expose reusable core functions).

## Monte Carlo VaR Sampling
- **Decision**: Simulate portfolio returns via multivariate normal draws using the selected covariance matrix and portfolio weights, with deterministic seeds for repeatability.
- **Rationale**: Provides fast sampling for up to 500 assets, integrates with numpy random generator, and respects covariance structure.
- **Alternatives Considered**: Bootstrapping historical returns (less responsive to covariance updates); copula-based sampling (more complex than needed for resume showcase).

## Stress Scenario Catalog
- **Decision**: Store macro shock templates in config/risk/stress_scenarios.json with keys for scenario metadata, factor shocks, and overrides, and allow users to supply custom JSON payloads.
- **Rationale**: JSON keeps scenarios human-editable, easily versioned, and consistent with existing config conventions.
- **Alternatives Considered**: YAML (added dependency); database-backed scenarios (overkill for offline workflows).

## Visualization Toolkit
- **Decision**: Implement matplotlib-based plotting utilities with optional Plotly export for interactive notebooks.
- **Rationale**: matplotlib already in requirements, works headless for CLI reports, and Plotly export covers interactive needs without forcing dependency at runtime.
- **Alternatives Considered**: seaborn (thin wrapper; still relies on matplotlib); bokeh (heavier dependency, not needed for MVP).

## Parameter Management
- **Decision**: Define RiskConfiguration dataclass parsing defaults from config/risk/defaults.json, supporting overrides via CLI flags and API payloads.
- **Rationale**: Centralized config ensures consistent defaults, simplifies validation, and keeps CLI/API parity.
- **Alternatives Considered**: Environment variables (harder to document and test); storing defaults inside code modules (less discoverable, harder to tweak).

## Data Ingestion Assumption
- **Decision**: Require callers to provide historical returns from validated datasets in data/storage/, with helper utilities to load canonical parquet files and validate schema.
- **Rationale**: Upholds Data Fidelity principle and guarantees consistent shape/type before risk computations.
- **Alternatives Considered**: On-demand fetching from external APIs (introduces latency and reproducibility issues); accepting arbitrary CSV without validation (risk of silent errors).
