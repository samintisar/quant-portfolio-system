# Agent Guidelines: Resume-Ready Quant Portfolio System

This guide keeps AI assistants aligned with the simplicity-first direction in `CLAUDE.md`. Treat every contribution as part of a resume portfolio project that should be easy to explain, demo, and maintain.

## Mission
- Ship a working, interview-ready portfolio optimization demo.
- Prefer clarity over abstraction; avoid enterprise patterns and deep hierarchies.
- Showcase core quantitative finance concepts without extra bells and whistles.

## Core Principles
- Start with the most direct implementation; only add complexity when absolutely necessary.
- Keep code readable and well-typed; optimize for comprehension during interviews.
- Focus on essentials: mean-variance optimization, Sharpe ratio, max drawdown, volatility, and clear data handling.
- Reuse existing modules and files instead of creating variants or layers of indirection.

## Minimal Tech Stack
- Python 3.11+, Pandas, NumPy, Yahoo Finance API.
- CVXPY for basic mean-variance optimization.
- FastAPI for lightweight endpoints.
- pytest for unit and integration coverage.
- Avoid bringing in additional packages unless they are strictly required for a core feature.

## Repository Shape
- Maintain a flat, simple structure. Keep logic in the existing `portfolio/`, `data/`, `api/`, `tests/`, `examples/`, and `scripts/` folders.
- Do not introduce new sub-packages or modules labeled "advanced", "enhanced", or "enterprise".
- Keep `requirements.txt` tidy and minimal; update it before installing anything new.

## Working Process for Agents
1. Deliver a basic, functional version first.
2. Confirm functionality with sample data (e.g., AAPL, GOOGL) before refining.
3. Add polish only when it clarifies the story for resumes or demos.
4. Document key decisions so they are easy to explain.

## Testing Expectations
- Cover core calculations and API flows with straightforward pytest suites.
- Tag longer scenarios appropriately, but avoid building elaborate test harnesses.
- Prioritize reliability of demo paths over exhaustive edge cases.

## Data & Configuration
- Use Yahoo Finance as the primary data source; keep preprocessing minimal.
- Handle missing or invalid symbols gracefully but simply.
- Rely on a small set of configuration files or environment variables that work out of the box.

## Deliverables Mindset
- Clean README notes, inline comments only where they aid explanation.
- Provide runnable examples in `examples/` that demonstrate optimization results.
- Aim for a working demo over theoretical completeness; recruiters value clarity and proof of competence.

---
*Latest alignment with CLAUDE.md: prioritize simplicity, demonstrable value, and minimalism.*
