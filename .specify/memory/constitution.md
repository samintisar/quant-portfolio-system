<!--
Sync Impact Report
Version change: 1.0.0 -> 1.1.0
Modified principles:
- Principle slot 1 -> I. Data Fidelity Is Mandatory
- Principle slot 2 -> II. Risk-First Portfolio Governance
- Principle slot 3 -> III. Test-Driven Statistical Validation
- Principle slot 4 -> IV. Reproducible CLI-Oriented Workflows
- Principle slot 5 -> V. Observability And Performance Discipline
Added sections:
- Operational Constraints
- Development Workflow & Quality Gates
Removed sections:
- None
Templates requiring updates:
- updated: .specify/templates/plan-template.md
- updated: CLAUDE.md
Follow-up TODOs:
- TODO(RATIFICATION_DATE): Document original adoption date from project history.
-->
# Quant Portfolio System Constitution

## Core Principles

### I. Data Fidelity Is Mandatory
All ingestion, preprocessing, and storage work MUST route through the modules under `data/src/` and
approved adapters. Every dataset powering strategies or portfolio engines MUST pass the automated
validation pipelines (`data/src/lib/validation.py`) and ship with quality reports saved to
`data/storage/`. Teams MUST run `python scripts/setup_data_environment.py` prior to altering storage
so required directories and schemas exist. Rationale: No allocation or forecast is trusted unless its
inputs are traceable, validated, and reproducible.

### II. Risk-First Portfolio Governance
Portfolio changes MUST demonstrate compliance with defined risk thresholds (Sharpe, max drawdown,
VaR/CVaR) before merge or deployment. Optimization configs in `portfolio/` and strategies in
`strategies/` MUST document risk assumptions and provide guardrail tests that fail when constraints
are breached. New models MUST expose scenario and stress results in review artifacts so reviewers can
reject unsafe proposals. Rationale: Capital protection outranks return seeking in production
workflows.

### III. Test-Driven Statistical Validation
Every change MUST start with failing tests that encode expected financial behaviour (`pytest`
workflows in `tests/`). Statistical assertions (coverage, factor significance, backtest expectations)
MUST live alongside unit, integration, and performance suites and run via `pytest -m "not slow"
before PR submission. Code is unmergeable until `python -m pytest`, `flake8`, and `mypy .` pass
without regression. Rationale: We only trust math that is reproducible and automatically checked.

### IV. Reproducible CLI-Oriented Workflows
All public capabilities MUST be invocable via CLI entry points or automation scripts (`data/src/cli`,
`scripts/`, `config/`). Configuration changes MUST be expressed through checked-in files under
`config/` with environment overrides documented, never via ad-hoc local edits. Workflows MUST
document invocation commands and expected artifacts in `examples/` or docs so another quant can
replay results on a fresh machine. Rationale: Reproducible pipelines keep research auditable and
shareable.

### V. Observability And Performance Discipline
Data pipelines, models, and strategies MUST emit structured logging and metrics sufficient to
diagnose failures and drift. Performance budgets (10M rows <30s, <4GB memory) MUST be enforced with
automated checks in `tests/performance/` and upheld before shipping compute-heavy changes in
`forecasting/`. Any GPU workloads MUST declare resource needs via `.env.gpu` and include monitoring
guidance. Rationale: Real-time operations demand transparent health signals and predictable
throughput.

## Operational Constraints

- Python 3.11+ is the only supported runtime; dependencies MUST flow through
  `docs/requirements.txt` and installations use `pip install -r docs/requirements.txt`.
- Repository structure MUST respect directory ownership; new assets belong in their domain-specific
  modules (`data/src/`, `portfolio/`, `strategies/`, `forecasting/`, `tests/`).
- Storage directories under `data/storage/` are runtime artifacts and MUST NOT be committed; data
  mutations require documenting provenance in PR descriptions.
- GPU workflows MUST isolate heavy compute inside `forecasting/` and declare configuration via
  `.env.gpu` templates.

## Development Workflow & Quality Gates

- Run `python scripts/setup_data_environment.py` before touching ingestion or backtest code.
- Format with `black` and `isort` (88-character width) before review; enforce `flake8` and `mypy .`.
- Pre-PR verification MUST include `pytest -m "not slow"`; release branches MUST pass `python -m
  pytest`.
- Backtests or data workflows MUST be validated by executing `python scripts/run_preprocessing.py
  --config config/pipeline_config.json` or the documented equivalent pipeline.
- Configuration changes MUST clone existing templates rather than editing committed defaults; record
  overrides and environment requirements in PR notes.
- Reviewers MUST confirm constitution gate checklists in plan and task templates before merge.

## Governance

Amendments require approval from the quantitative platform maintainers and documentation owners,
with risk lead sign-off when principles affect capital allocation. Every change MUST include a version
bump and update dependent templates (`.specify/templates/*.md`, agent guidance files, README
references). Versioning follows semantic rules: MAJOR for breaking removals or redefinition of
principles, MINOR for added principles or expanded duties, PATCH for clarifications. Submit amendment
PRs with rationale, testing summary, and rollback plan. Compliance reviews run quarterly; violations
trigger remediation tasks tracked in `tasks.md` and block deployments until resolved.

**Version**: 1.1.0 | **Ratified**: TODO(RATIFICATION_DATE): Document original adoption date from project
history. | **Last Amended**: 2025-09-21
