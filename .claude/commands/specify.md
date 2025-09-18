---
description: Create or update the trading strategy specification from a natural language trading strategy description.
---

Given the trading strategy description provided as an argument, do this:

1. Run the script `.specify/scripts/powershell/create-new-feature.ps1 -Json "$ARGUMENTS"` from repo root and parse its JSON output for BRANCH_NAME and SPEC_FILE. All file paths must be absolute.
2. Load `.specify/templates/spec-template.md` to understand required sections for trading strategies.
3. Write the trading strategy specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the trading strategy description (arguments) while preserving section order and headings.
4. Report completion with branch name, spec file path, and readiness for the next phase.

Note: The script creates and checks out the new branch and initializes the spec file before writing.

Example usage:
```
/specify "Create a momentum trading strategy that uses moving average crossovers for entry signals and RSI for exit signals. The strategy should trade major forex pairs with a 1-hour timeframe and include risk management with 2% stop loss and 3% take profit levels."
```
