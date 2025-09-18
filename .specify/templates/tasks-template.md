# Tasks: [FEATURE NAME]

**Input**: Trading design documents from `/specs/[###-feature-name]/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No trading implementation plan found"
   → Extract: trading tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract trading entities → model tasks
   → contracts/: Each file → trading contract test task
   → research.md: Extract trading decisions → setup tasks
3. Generate tasks by category:
   → Setup: trading project init, dependencies, linting
   → Tests: trading contract tests, backtest tests
   → Core: trading models, strategies, CLI commands
   → Integration: data feeds, risk management, logging
   → Polish: unit tests, performance validation, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All trading contracts have tests?
   → All trading entities have models?
   → All trading strategies implemented?
9. Return: SUCCESS (trading tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single trading strategy**: `src/`, `tests/` at repository root
- **Multi-strategy platform**: `data/src/`, `strategies/src/`, `portfolio/src/`
- Paths shown below assume single trading strategy - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create trading project structure per implementation plan
- [ ] T002 Initialize [language] project with [trading framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools for quantitative code

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Trading contract test signal generation in tests/contract/test_signal_generation.py
- [ ] T005 [P] Trading contract test risk management in tests/contract/test_risk_management.py
- [ ] T006 [P] Backtest integration test strategy logic in tests/backtest/test_strategy_backtest.py
- [ ] T007 [P] Integration test data feed processing in tests/integration/test_data_feed.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T008 [P] Trading data model in src/data/models.py
- [ ] T009 [P] Trading strategy implementation in src/strategy/[strategy_name].py
- [ ] T010 [P] CLI --backtest-strategy in src/cli/trading_commands.py
- [ ] T011 Signal generation logic implementation
- [ ] T012 Risk management logic implementation
- [ ] T013 Data validation and preprocessing
- [ ] T014 Trading logging and monitoring

## Phase 3.4: Integration
- [ ] T015 Connect strategy to market data feeds
- [ ] T016 Risk management integration with strategy
- [ ] T017 Portfolio position tracking
- [ ] T018 Order execution simulation

## Phase 3.5: Polish
- [ ] T019 [P] Unit tests for trading logic in tests/unit/test_trading_logic.py
- [ ] T020 Performance validation (<1ms per data point)
- [ ] T021 [P] Update trading documentation in docs/trading_strategy.md
- [ ] T022 Code optimization and refactoring
- [ ] T023 Run backtest validation against historical data

## Dependencies
- Tests (T004-T007) before implementation (T008-T014)
- T008 blocks T009, T015
- T016 blocks T018
- Implementation before polish (T019-T023)

## Parallel Example
```
# Launch T004-T007 together:
Task: "Trading contract test signal generation in tests/contract/test_signal_generation.py"
Task: "Trading contract test risk management in tests/contract/test_risk_management.py"
Task: "Backtest integration test strategy logic in tests/backtest/test_strategy_backtest.py"
Task: "Integration test data feed processing in tests/integration/test_data_feed.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing
- Commit after each task
- Avoid: vague tasks, same file conflicts
- All trading strategies must include risk management

## Task Generation Rules
*Applied during main() execution*

1. **From Trading Contracts**:
   - Each trading contract file → trading contract test task [P]
   - Each trading signal → implementation task

2. **From Trading Data Model**:
   - Each trading entity → model creation task [P]
   - Relationships → strategy layer tasks

3. **From Trading Scenarios**:
   - Each trading scenario → backtest integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Data Models → Strategies → Risk Management → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All trading contracts have corresponding tests
- [ ] All trading entities have model tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task
- [ ] All trading strategies include risk management tasks