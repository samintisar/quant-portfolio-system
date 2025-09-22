# Repository Guidelines

Welcome to the quant-portfolio-system project. This guide highlights the essentials for shipping reliable quantitative tooling without disrupting production pipelines.

## Project Structure & Module Organization
- `data/src/` hosts ingestion feeds, preprocessing orchestration, and CLI adapters; runtime artifacts land in `data/storage/`.
- `portfolio/` contains allocation engines and optimization constraints; align new models with existing factor interfaces.
- `strategies/` implements trading logic and signal generation; reuse shared indicators from `data/src/lib/`.
- `forecasting/` adds GPU-enabled prediction flows; keep compute-heavy assets isolated here.
- `scripts/` and `examples/` hold reproducible workflows, while `tests/` mirrors domain folders (`unit/`, `integration/`, `statistical/`, `performance/`).
- `config/` stores pipeline JSON/YAML definitions and `.env.gpu` captures accelerator settings.

## Environment & Configuration Tips
- Target Python 3.11+; install toolchain with `pip install -r docs/requirements.txt`.
- Run `python scripts/setup_data_environment.py` before touching ingestion or backtests; it primes storage directories.
- Parameterize runs via files in `config/`; prefer copying templates rather than editing committed defaults.

## Build, Test, and Development Commands
- `python -m pytest` runs the full suite with coverage reports (`htmlcov/`, `coverage.xml`).
- `pytest -m "not slow"` is the pre-PR fast path; add `--maxfail=1` for quick triage.
- `black . && isort .` enforce formatting; both use an 88-character limit.
- `flake8` and `mypy .` gate linting and typing; silence warnings by fixing root causes, not by ignoring them.
- `python scripts/run_preprocessing.py --config config/pipeline_config.json` smoke-tests the end-to-end data flow.

## Coding Style & Naming Conventions
- Use 4-space indentation, typed function signatures, and descriptive module-level docstrings focused on financial context.
- Prefer snake_case for functions and variables, PascalCase for classes, UPPER_SNAKE_CASE for environment constants.
- Keep public APIs side-effect free and return domain dataclasses from `data/src/models` where practical.

## Testing Guidelines
- Place new tests alongside code mirrors (e.g., `tests/unit/data/test_preprocessing.py`).
- Name files `test_<feature>.py` and functions `test_<behavior>`; leverage `pytest` fixtures under `tests/conftest.py`.
- Tag long-running scenarios with `@pytest.mark.slow` or `@pytest.mark.performance` so CI filters stay effective.
- Failing coverage must be addressed before review; update expectations in `tests/performance/` rather than relaxing thresholds.

## Commit & Pull Request Guidelines
- Follow the imperative style seen in history (`Add GPU acceleration support...`); keep subject lines =72 characters.
- Reference tickets with `Refs #<id>` in the body and mention affected modules.
- Include a short testing summary (`pytest -m "not slow"`, `mypy .`) and attach coverage deltas when relevant.
- PR descriptions should outline risk, rollback strategy, and screenshots/logs for user-facing or performance-sensitive changes.
