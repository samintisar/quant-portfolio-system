# Tasks: Robust Risk Measurement Tools

**Input**: Design documents from `C:\Users\samin\Desktop\Github\quant-portfolio-system\specs\004-add-robust-risk\`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Phase 3.1: Setup
- [X] T001 Establish risk module scaffolding (`portfolio/src/risk/` packages, `portfolio/src/risk/engines/`, `portfolio/src/risk/visualization/`, `portfolio/src/api/__init__.py`, `portfolio/tests/{unit,contract,integration}/risk/__init__.py`).
- [X] T002 Create configuration artifacts (`config/risk/defaults.json`, `config/risk/stress_scenarios.json`) seeded with 95%/1-day and 99%/10-day presets plus sample macro shocks.
- [X] T003 [P] Add structured logging and storage scaffolding (`config/logging/risk_logging.yaml`, `logs/risk/.gitkeep`, `data/storage/reports/.gitkeep`).

## Phase 3.2: Tests First (TDD)
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [X] T004 [P] Contract test POST /risk/report in `portfolio/tests/contract/risk/test_risk_report_post.py` covering request and 202 response schema from `contracts/risk_metrics.openapi.json`.
- [X] T005 [P] Contract test GET /risk/reports/{report_id} in `portfolio/tests/contract/risk/test_risk_report_get.py` verifying 200/404 payloads.
- [X] T006 [P] Contract test GET /risk/scenarios in `portfolio/tests/contract/risk/test_risk_scenarios_get.py` asserting catalog schema.
- [X] T007 [P] CLI contract test in `portfolio/tests/contract/risk/test_risk_cli_contract.py` validating argument parsing from `contracts/risk_cli_contract.md`.
- [X] T008 [P] Integration test covariance workflow in `portfolio/tests/integration/risk/test_covariance_methods.py` asserting sample, Ledoit-Wolf, and EWMA outputs on demo data.
- [X] T009 [P] Integration test VaR selection in `portfolio/tests/integration/risk/test_var_methods.py` covering historical, parametric, and Monte Carlo flows with preset confidence levels.
- [X] T010 [P] Integration test CVaR computation in `portfolio/tests/integration/risk/test_cvar_metrics.py` asserting expected shortfall monotonicity.
- [X] T011 [P] Integration test stress testing in `portfolio/tests/integration/risk/test_stress_scenarios.py` applying macro and custom shocks.
- [X] T012 [P] Integration test visualization generation in `portfolio/tests/integration/risk/test_visualizations.py` confirming factor exposure, VaR trend, and CVaR distribution artifacts.
- [X] T013 [P] Unit tests for covariance engine in `portfolio/tests/unit/risk/test_covariance_engine.py` (positive semidefinite checks, EWMA decay bounds).
- [X] T014 [P] Unit tests for VaR/CVaR calculators in `portfolio/tests/unit/risk/test_var_cvar_engines.py` using deterministic seeds.
- [X] T015 [P] Unit tests for stress scenario loader in `portfolio/tests/unit/risk/test_stress_loader.py` covering JSON schema validation and overrides.
- [X] T016 [P] Unit tests for configuration/dataclass validation in `portfolio/tests/unit/risk/test_risk_configuration.py` (confidence level bounds, horizon limits, report selections).

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [X] T017 [P] Implement `RiskConfiguration` dataclass and validators in `portfolio/src/risk/models/configuration.py`.
- [X] T018 [P] Implement `PortfolioSnapshot` loader utilities in `portfolio/src/risk/models/snapshot.py`.
- [X] T019 [P] Implement `CovarianceResult` model in `portfolio/src/risk/models/covariance_result.py` with PSD checks.
- [X] T020 [P] Implement `RiskMetricEntry` model in `portfolio/src/risk/models/risk_metric.py` with unique key enforcement.
- [X] T021 [P] Implement `StressScenario` model in `portfolio/src/risk/models/stress_scenario.py` including parameter schema mapping.
- [X] T022 [P] Implement `StressImpact` model in `portfolio/src/risk/models/stress_impact.py` capturing contribution breakdowns.
- [X] T023 [P] Implement `RiskReport` aggregate container in `portfolio/src/risk/models/report.py` linking configuration, metrics, impacts, and artifacts.
- [X] T024 [P] Implement `VisualizationArtifact` model in `portfolio/src/risk/models/visualization.py` with filesystem validation helpers.
- [X] T025 Build covariance estimation engine in `portfolio/src/risk/engines/covariance.py` (sample, Ledoit-Wolf, EWMA) wired to numpy/sklearn.
- [X] T026 Develop VaR calculators in `portfolio/src/risk/engines/var.py` for historical, parametric, and Monte Carlo workflows.
- [X] T027 Implement CVaR computation module in `portfolio/src/risk/engines/cvar.py` deriving expected shortfall from VaR outputs.
- [X] T028 Implement stress testing engine in `portfolio/src/risk/engines/stress.py` applying factor/asset/macro shocks to snapshots.
- [X] T029 Create risk report assembler and persistence layer in `portfolio/src/risk/services/report_builder.py` (combines engines, writes outputs to `data/storage/reports`).
- [X] T030 Implement visualization helpers in `portfolio/src/risk/visualization/plots.py` generating factor exposure, VaR trend, and CVaR distribution charts.
- [X] T031 Implement scenario catalog loader in `portfolio/src/risk/services/scenario_catalog.py` reading `config/risk/stress_scenarios.json` with validation.
- [X] T032 Implement configuration service in `portfolio/src/risk/services/config_loader.py` merging defaults with request/CLI overrides and enforcing bounds.
- [X] T033 Add portfolio data access utilities in `portfolio/src/risk/services/data_access.py` to load validated returns/weights from `data/storage/`.
- [X] T034 Implement FastAPI application scaffold in `portfolio/src/api/risk_api.py` (app factory, dependency wiring, logging middleware).
- [X] T035 Implement POST `/risk/report` handler in `portfolio/src/api/risk_api.py` invoking report builder and returning 202 with report metadata.
- [X] T036 Implement GET `/risk/reports/{report_id}` handler in `portfolio/src/api/risk_api.py` retrieving persisted reports or 404.
- [X] T037 Implement GET `/risk/scenarios` handler in `portfolio/src/api/risk_api.py` exposing scenario catalog.
- [X] T038 Implement CLI interface in `scripts/run_risk_metrics.py` (argparse, config override, validation, logging setup).
- [X] T039 Wire CLI execution path to services in `scripts/run_risk_metrics.py` generating reports, writing artifacts, and emitting exit codes.
- [X] T040 Add structured logging & metrics instrumentation in `portfolio/src/risk/services/telemetry.py` and integrate with engines/API.
- [X] T041 Integrate risk workflows with quickstart fixtures by extending `examples/` with `examples/run_risk_metrics_demo.py` referencing CLI/API usage.

## Phase 3.4: Integration
- [X] T042 Connect report persistence to storage by adding serialization utilities in `portfolio/src/risk/services/report_store.py` (JSON/Parquet writers, retrieval, index maintenance).
- [X] T043 Add performance guard in `tests/performance/risk/test_risk_report_runtime.py` ensuring 500-asset Monte Carlo VaR completes <5s and <1 GB memory.
- [X] T044 Add observability hooks to API (`portfolio/src/api/risk_api.py`) for structured request/response logging and timing metrics.

## Phase 3.5: Polish
- [ ] T045 [P] Expand documentation in `portfolio/README.md` and create `docs/risk_metrics.md` describing API/CLI usage and configuration presets.
- [ ] T046 [P] Add regression dataset fixtures in `portfolio/tests/data/fixtures/risk_demo.py` for deterministic tests.
- [ ] T047 [P] Update `specs/004-add-robust-risk/plan.md` progress tracking and record validation evidence after implementation.
- [ ] T048 Run full validation suite (`pytest`, `flake8`, `mypy`, quickstart CLI/API smoke`) and capture artifacts referenced in PR description.

## Dependencies
- T001 blocks all subsequent work; T002 depends on T001 for config paths.
- Tests (T004傍016) must be completed and failing before implementation tasks (T017 onward) start.
- Model tasks (T017傍024) unblock engine/services (T025傍033).
- T025 precedes T026, which precedes T027; T029 depends on T025傍028 and T032傍033.
- API tasks (T034傍037) depend on services (T029傍033) and telemetry (T040) before final instrumentation (T044).
- CLI tasks (T038傍039) depend on configuration and report builder services (T029傍033).
- Integration tasks (T042傍044) require core implementation completion.
- Polish tasks (T045傍048) run only after all prior tasks pass.

## Parallel Execution Example
```
# Kick off contract and integration test authoring in parallel once setup is done:
Task.run("T004")  # POST /risk/report contract test
Task.run("T005")  # GET /risk/reports/{report_id} contract test
Task.run("T006")  # GET /risk/scenarios contract test
Task.run("T007")  # CLI contract test
Task.run("T008")  # Covariance integration scenario
Task.run("T009")  # VaR integration scenario
Task.run("T010")  # CVaR integration scenario
Task.run("T011")  # Stress testing integration scenario
Task.run("T012")  # Visualization integration scenario
Task.run("T013")  # Covariance engine unit tests
Task.run("T014")  # VaR/CVaR unit tests
Task.run("T015")  # Stress loader unit tests
Task.run("T016")  # Configuration validation unit tests
```

## Notes
- [P] tasks target distinct files/directories and can execute concurrently once prerequisites are satisfied.
- Maintain TDD discipline: ensure each newly written test fails against current code before implementation.
- Capture intermediate artifacts (OpenAPI diffs, CLI help text, performance metrics) to support final documentation and PR evidence.
- Commit after each task or small batch to keep history reviewable.
