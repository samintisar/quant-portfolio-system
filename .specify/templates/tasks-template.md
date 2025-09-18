# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Statistical validation tests for model outputs in tests/statistical/test_model_validation.py
- [ ] T005 [P] Data quality tests for input validation in tests/data/test_data_quality.py
- [ ] T006 [P] Backtesting framework tests in tests/backtesting/test_backtest_engine.py
- [ ] T007 [P] Risk metrics calculation tests in tests/risk/test_risk_metrics.py
- [ ] T008 [P] Portfolio optimization constraint tests in tests/optimization/test_constraints.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T009 [P] Financial data models in src/models/market_data.py
- [ ] T010 [P] Portfolio model in src/models/portfolio.py
- [ ] T011 [P] Data ingestion service in src/services/data_service.py
- [ ] T012 [P] Risk calculation service in src/services/risk_service.py
- [ ] T013 [P] Portfolio optimization engine in src/optimization/optimizer.py
- [ ] T014 [P] Backtesting engine in src/backtesting/engine.py
- [ ] T015 [P] CLI commands for data operations in src/cli/data_commands.py
- [ ] T016 [P] CLI commands for portfolio operations in src/cli/portfolio_commands.py

## Phase 3.4: Integration
- [ ] T017 Connect data service to Yahoo Finance API
- [ ] T018 Integrate optimization engine with portfolio model
- [ ] T019 Connect backtesting engine to historical data
- [ ] T020 Implement structured logging for model decisions
- [ ] T021 Add performance monitoring and alerts

## Phase 3.5: Polish & Validation
- [ ] T022 [P] Unit tests for mathematical functions in tests/unit/test_math_utils.py
- [ ] T023 [P] Performance benchmarking tests (execution time constraints)
- [ ] T024 [P] Update docs/mathematical_formulations.md
- [ ] T025 [P] Statistical significance tests for model outputs
- [ ] T026 Sharpe ratio validation against target (>1.5)
- [ ] T027 Maximum drawdown validation (<15%)
- [ ] T028 Run comprehensive backtesting validation

## Dependencies
- Tests (T004-T008) before implementation (T009-T016)
- T009, T010 block T011, T012, T013, T014
- T013, T014 block T018, T019
- T017, T018, T019 block T020, T021
- Implementation before validation (T022-T028)

## Parallel Example
```
# Launch T004-T008 together (all statistical tests):
Task: "Statistical validation tests for model outputs in tests/statistical/test_model_validation.py"
Task: "Data quality tests for input validation in tests/data/test_data_quality.py"
Task: "Backtesting framework tests in tests/backtesting/test_backtest_engine.py"
Task: "Risk metrics calculation tests in tests/risk/test_risk_metrics.py"
Task: "Portfolio optimization constraint tests in tests/optimization/test_constraints.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts

## Task Generation Rules
*Applied during main() execution*

1. **From Mathematical Models**:
   - Each model → statistical validation test [P]
   - Each algorithm → unit test for mathematical correctness [P]

2. **From Data Requirements**:
   - Each data source → data quality test [P]
   - Each entity → model creation task [P]
   - Data pipelines → integration test tasks

3. **From Risk Requirements**:
   - Each risk metric → calculation test [P]
   - Portfolio constraints → constraint validation test [P]
   - Performance targets → backtesting validation tasks

4. **From Financial Requirements**:
   - Each optimization strategy → backtest scenario
   - Performance benchmarks → statistical comparison tests

5. **Ordering**:
   - Setup → Statistical Tests → Models → Services → Optimization → Integration → Validation
   - Mathematical dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All mathematical models have statistical validation tests
- [ ] All data sources have quality validation tests
- [ ] All risk metrics have calculation tests
- [ ] All portfolio constraints have validation tests
- [ ] All tests come before implementation (TDD enforced)
- [ ] Statistical tests include significance thresholds
- [ ] Performance targets clearly defined (Sharpe >1.5, drawdown <15%)
- [ ] Parallel tasks truly independent (different mathematical domains)
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task
- [ ] Backtesting validation covers multiple market regimes