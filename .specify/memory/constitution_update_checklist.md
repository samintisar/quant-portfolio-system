# Constitution Update Checklist

When amending the constitution (`/memory/constitution.md`), ensure all dependent documents are updated to maintain consistency.

## Templates to Update

### When adding/modifying ANY article:
- [x] `/templates/plan-template.md` - Update Constitution Check section
- [x] `/templates/spec-template.md` - Update if requirements/scope affected
- [x] `/templates/tasks-template.md` - Update if new task types needed
- [x] `/.claude/commands/plan.md` - Update if planning process changes
- [x] `/.claude/commands/tasks.md` - Update if task generation affected
- [x] `/CLAUDE.md` - Update runtime development guidelines

### Article-specific updates:

#### Article I (Library-First Architecture):
- [x] Ensure templates emphasize mathematical library creation
- [x] Update CLI command examples for quantitative tools
- [x] Add mathematical formulation documentation requirements

#### Article II (CLI Interface & Reproducibility):
- [x] Update CLI flag requirements in templates
- [x] Add text I/O protocol reminders for reproducible research
- [x] Include seed control and configuration versioning

#### Article III (Test-First Development):
- [x] Update test order in all templates
- [x] Emphasize TDD requirements with statistical validation
- [x] Add statistical test approval gates

#### Article IV (Data Quality & Validation):
- [x] List data validation test triggers
- [x] Update test type priorities for financial data
- [x] Add pipeline integrity requirements

#### Article V (Risk Management & Observability):
- [x] Add structured logging requirements for model decisions
- [x] Include risk metrics monitoring
- [x] Update performance monitoring sections with VaR/drawdown

#### Article VI (Versioning & Model Governance):
- [x] Add model version increment reminders
- [x] Include parameter snapshot procedures
- [x] Update backward compatibility requirements

#### Article VII (Simplicity & Financial Soundness):
- [x] Update project count limits (max 3 libraries)
- [x] Add benchmark validation examples
- [x] Include financial soundness reminders

## Validation Steps

1. **Before committing constitution changes:**
   - [x] All templates reference new requirements
   - [x] Examples updated to match quantitative trading rules
   - [x] No contradictions between documents

2. **After updating templates:**
   - [x] Run through a sample implementation plan (ready for quantitative features)
   - [x] Verify all constitution requirements addressed
   - [x] Check that templates are self-contained (readable without constitution)

3. **Version tracking:**
   - [x] Update constitution version number (v1.0.0)
   - [x] Note version in template footers
   - [x] Add amendment to constitution history

## Common Misses

Watch for these often-forgotten updates:
- Command documentation (`/commands/*.md`)
- Checklist items in templates
- Example code/commands
- Domain-specific variations (web vs mobile vs CLI)
- Cross-references between documents

## Template Sync Status

Last sync check: 2025-09-17
- Constitution version: 1.0.0 (Quantitative Trading System)
- Templates aligned: ✅ (all quantitative trading requirements integrated)

---

*This checklist ensures the constitution's principles are consistently applied across all project documentation.*