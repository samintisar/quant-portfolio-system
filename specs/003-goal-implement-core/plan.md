# Implementation Plan: Core Forecasting Models for Returns & Volatility

**Branch**: `003-goal-implement-core` | **Date**: 2025-09-20 | **Spec**: [specs/003-goal-implement-core/spec.md](specs/003-goal-implement-core/spec.md)
**Input**: Feature specification from `specs/003-goal-implement-core/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Implement core forecasting models for returns & volatility including ARIMA/GARCH time series models, Hidden Markov Models for regime detection, belief networks for scenario modeling, and signal validation mechanisms. The system must save signals and forecasts per asset with regime labels while maintaining statistical rigor and financial soundness.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: statsmodels, arch, hmmlearn, pgmpy, scikit-learn, pandas, numpy
**Storage**: Time series data files, JSON configuration, SQLite for metadata
**Testing**: pytest, statistical validation tests, Monte Carlo simulations
**Target Platform**: Linux/Windows server for batch processing
**Project Type**: single - quantitative research library
**Performance Goals**: 10M data points processed in <30 seconds, <4GB memory usage
**Constraints**: Statistical significance p<0.05, backtest out-of-sample validation, VaR < 2%
**Scale/Scope**: 500+ liquid instruments, 10+ years historical data, multiple asset classes

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Library-First Architecture Gate (Article I)
- [x] Feature decomposed into ≤3 mathematical libraries?
- [x] Each library has clear quantitative purpose (data/modeling/optimization/risk/backtesting)?
- [x] Libraries self-contained with minimal dependencies?
- [ ] Mathematical formulations documented?

### CLI & Reproducibility Gate (Article II)
- [ ] All models expose CLI interfaces?
- [ ] JSON configuration support for parameters?
- [ ] Seed control for reproducible results?
- [ ] Version-controlled configuration files?

### Test-First Gate (Article III)
- [ ] Statistical validation tests planned?
- [ ] Unit tests for mathematical correctness?
- [ ] Backtesting validation scenarios defined?
- [ ] Data pipeline integrity tests specified?

### Data Quality Gate (Article IV)
- [ ] Input validation strategy defined?
- [ ] Model output bounds checking planned?
- [ ] Missing value handling strategy specified?
- [ ] Performance monitoring approach defined?

### Risk Management Gate (Article V)
- [ ] Risk metrics calculation planned (VaR, drawdown)?
- [ ] Structured logging for model decisions?
- [ ] Alert systems for constraint violations?
- [ ] Performance attribution tracking specified?

### Financial Soundness Gate (Article VII)
- [ ] Benchmark validation against S&P 500 planned?
- [ ] Target Sharpe ratio > 1.5 achievable?
- [ ] Maximum drawdown < 15% constraint enforced?
- [ ] Starting with basic models before advanced techniques?

### Financial Constraints Compliance
- [ ] Single name concentration < 5% enforced?
- [ ] Sector concentration < 20% enforced?
- [ ] Leverage ratio < 1.5x for long-only strategies?
- [ ] VaR at 95% confidence < 2% daily portfolio value?
- [ ] Minimum 10 years historical data available?
- [ ] Coverage of 500+ liquid instruments planned?

## Project Structure

### Documentation (this feature)
```
specs/003-goal-implement-core/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: DEFAULT to Option 1 for quantitative research library

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - NEEDS CLARIFICATION: optimal number of regimes for HMM models
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Quantitative Trading Constitution v1.0.0 - See `.specify/memory/constitution.md`*