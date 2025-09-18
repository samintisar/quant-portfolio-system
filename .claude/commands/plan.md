---
description: Execute the trading strategy implementation planning workflow using the plan template to generate design artifacts.
---

Given the trading implementation details provided as an argument, do this:

1. Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from the repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. All future file paths must be absolute.
2. Read and analyze the trading strategy specification to understand:
   - The trading strategy requirements and scenarios
   - Functional and non-functional requirements for trading
   - Success criteria and acceptance criteria for the strategy
   - Any technical constraints or trading dependencies mentioned

3. Read the constitution at `.specify/memory/constitution.md` to understand quantitative trading constitutional requirements.

4. Execute the trading implementation plan template:
   - Load `.specify/templates/plan-template.md` (already copied to IMPL_PLAN path)
   - Set Input path to FEATURE_SPEC
   - Run the Execution Flow (main) function steps 1-9
   - The template is self-contained and executable
   - Follow error handling and gate checks as specified
   - Let the template guide artifact generation in $SPECS_DIR:
     * Phase 0 generates research.md with trading research
     * Phase 1 generates data-model.md, contracts/, quickstart.md
     * Phase 2 generates tasks.md
   - Incorporate user-provided details from arguments into Technical Context: $ARGUMENTS
   - Update Progress Tracking as you complete each phase

5. Verify execution completed:
   - Check Progress Tracking shows all phases complete
   - Ensure all required trading artifacts were generated
   - Confirm no ERROR states in execution
   - Validate risk parameters are properly defined

6. Report results with branch name, file paths, and generated trading artifacts.

Use absolute paths with the repository root for all file operations to avoid path issues.

Example usage:
```
/plan "Python 3.11 with pandas, numpy, scikit-learn. Use yfinance for data feeds and backtrader for backtesting. Target: Sharpe ratio > 1.2, max drawdown < 8%."
```
