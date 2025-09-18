---
description: Generate an actionable, dependency-ordered tasks.md for the trading strategy based on available design artifacts.
---

Given the context provided as an argument, do this:

1. Run `.specify/scripts/powershell/check-task-prerequisites.ps1 -Json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute.
2. Load and analyze available trading design documents:
   - Always read plan.md for trading tech stack and libraries
   - IF EXISTS: Read data-model.md for trading entities
   - IF EXISTS: Read contracts/ for trading strategy contracts
   - IF EXISTS: Read research.md for trading technical decisions
   - IF EXISTS: Read quickstart.md for trading test scenarios

   Note: Not all trading strategies have all documents. For example:
   - Simple trading strategies might not have complex contracts/
   - Data-focused strategies might not need separate data-model.md
   - Generate tasks based on what's available

3. Generate trading tasks following the template:
   - Use `.specify/templates/tasks-template.md` as the base
   - Replace example tasks with actual trading tasks based on:
     * **Setup tasks**: Trading project init, dependencies, linting
     * **Test tasks [P]**: One per trading contract, one per backtest scenario
     * **Core tasks**: One per trading entity, strategy, CLI command, signal
     * **Integration tasks**: Data feeds, risk management, logging
     * **Polish tasks [P]**: Unit tests, performance validation, trading docs

4. Task generation rules:
   - Each trading contract file → trading contract test task marked [P]
   - Each trading entity in data-model → trading model creation task marked [P]
   - Each trading signal → implementation task (not parallel if shared files)
   - Each trading scenario → backtest integration test marked [P]
   - Different files = can be parallel [P]
   - Same file = sequential (no [P])

5. Order tasks by dependencies:
   - Setup before everything
   - Tests before implementation (TDD)
   - Data models before trading strategies
   - Trading strategies before risk management
   - Core before integration
   - Everything before polish

6. Include parallel execution examples:
   - Group [P] tasks that can run together
   - Show actual Task agent commands

7. Create FEATURE_DIR/tasks.md with:
   - Correct trading strategy name from implementation plan
   - Numbered tasks (T001, T002, etc.)
   - Clear file paths for each trading task
   - Dependency notes
   - Parallel execution guidance

Context for task generation: $ARGUMENTS

The tasks.md should be immediately executable - each trading task must be specific enough that an LLM can complete it without additional context.

Example usage:
```
/tasks
```
