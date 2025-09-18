
# Trading Strategy Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Trading strategy specification from `/specs/[###-feature-name]/spec.md`

## Execution Flow (/plan command scope)
```
1. Load trading strategy spec from Input path
   → If not found: ERROR "No trading strategy spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Trading Strategy Type from context (momentum, mean-reversion, arbitrage, etc.)
   → Set Structure Decision based on strategy type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify trading approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve trading unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor trading design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
[Extract from trading strategy spec: primary trading requirement + technical approach from research]

## Technical Context
**Language/Version**: [e.g., Python 3.11 with pandas/numpy, R 4.3, or NEEDS CLARIFICATION]
**Primary Dependencies**: [e.g., pandas, numpy, scikit-learn, yfinance, TA-Lib or NEEDS CLARIFICATION]
**Data Sources**: [e.g., Yahoo Finance, Alpha Vantage, Bloomberg API or NEEDS CLARIFICATION]
**Backtesting**: [e.g., backtrader, zipline, custom framework or NEEDS CLARIFICATION]
**Risk Management**: [e.g., custom risk library, integrated with portfolio optimization or NEEDS CLARIFICATION]
**Target Environment**: [e.g., Linux server for trading, cloud deployment or NEEDS CLARIFICATION]
**Strategy Type**: [momentum/mean-reversion/arbitrage/statistical - determines structure]
**Performance Goals**: [e.g., Sharpe ratio > 1.5, max drawdown < 10%, win rate > 55% or NEEDS CLARIFICATION]
**Latency Requirements**: [e.g., <100ms signal generation, <10ms order execution or NEEDS CLARIFICATION]
**Data Requirements**: [e.g., 1-minute bars, daily close, tick data or NEEDS CLARIFICATION]

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Simplicity Gate (Article VII)
- [ ] Using ≤3 trading libraries (data, strategy, execution)?
- [ ] No future-proofing for hypothetical market conditions?
- [ ] Trading strategy is based on well-understood principles?

### Risk Management Gate (Article VI)
- [ ] Position limits clearly defined?
- [ ] Risk parameters specified (stop-loss, drawdown limits)?
- [ ] Performance targets measurable and realistic?

### Library-First Gate (Article I)
- [ ] Each trading component is self-contained and testable?
- [ ] No organizational-only trading libraries?
- [ ] CLI interface exposed for each trading component?

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single trading strategy (DEFAULT)
src/
├── data/           # Data ingestion and processing
├── strategy/       # Trading strategy implementation
├── execution/     # Order execution and risk management
└── cli/          # Command-line interface

tests/
├── backtest/     # Backtesting validation
├── integration/  # Integration tests
└── unit/         # Unit tests

# Option 2: Multi-strategy platform (when "portfolio" + "optimization" detected)
data/
├── src/
│   ├── feeds/      # Market data feeds
│   ├── processing/ # Data processing and normalization
│   └── storage/    # Data storage and retrieval
└── tests/

strategies/
├── src/
│   ├── momentum/   # Momentum-based strategies
│   ├── meanrev/    # Mean reversion strategies
│   └── arbitrage/  # Arbitrage strategies
└── tests/

portfolio/
├── src/
│   ├── optimization/ # Portfolio optimization
│   ├── risk/        # Risk management
│   └── execution/   # Order execution
└── tests/
```

**Structure Decision**: [DEFAULT to Option 1 unless Technical Context indicates multi-strategy platform]

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each trading dependency → best practices task
   - For each data integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {trading strategy context}"
   For each technology choice:
     Task: "Find best practices for {tech} in quantitative trading"
   For each trading pattern:
     Task: "Validate {trading pattern} effectiveness for {asset class}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen with quantitative justification]
   - Alternatives considered: [what else evaluated]
   - Backtest results: [historical validation if available]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract trading entities from strategy spec** → `data-model.md`:
   - Trading entity name, fields, relationships
   - Validation rules from trading requirements
   - State transitions for market regimes if applicable

2. **Generate trading strategy contracts** from functional requirements:
   - For each trading signal → strategy interface
   - For each risk rule → risk management contract
   - For each data feed → data contract
   - Output trading schemas to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per trading strategy component
   - Assert trading signal generation and risk management
   - Tests must fail (no implementation yet)

4. **Extract backtest scenarios** from trading stories:
   - Each trading scenario → backtest validation
   - Quickstart test = strategy validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude` for your AI assistant
   - If exists: Add only NEW trading tech from current plan
   - Preserve manual additions between markers
   - Update recent trading changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each trading contract → trading contract test task [P]
- Each trading entity → trading model creation task [P]
- Each trading scenario → backtest integration task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Data models before strategies before execution
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
| [e.g., 4th trading library] | [specific trading need] | [why 3 libraries insufficient] |
| [e.g., Complex risk model] | [specific risk requirement] | [why simple stop-loss insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [ ] Phase 0: Trading research complete (/plan command)
- [ ] Phase 1: Trading design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Backtesting and validation passed

**Gate Status**:
- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented
- [ ] Risk parameters validated

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
