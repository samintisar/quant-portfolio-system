<!--
Sync Impact Report
Version change: 1.2.0 -> 1.3.0
Modified principles:
- I. Data Integrity in Financial Markets -> I. Clean Data is Non-Negotiable
- II. Portfolio Risk Management Discipline -> II. Risk Management First
- III. Statistical Rigor and Model Validation -> III. Simple Models, Proven Results
- IV. Reproducible Quantitative Research -> IV. Reproducible by Design
- V. Performance Optimization and Monitoring -> V. Practical Performance
Added sections:
- Portfolio Essentials (simplified from Quantitative Finance Best Practices)
- ML Guidelines (simplified from ML Integration Standards)
- Keep It Simple (simplified from Operational Constraints)
Removed sections:
- Operational Constraints (overly complex)
- Development Workflow & Quality Gates (too process-heavy)
- Complex governance procedures
Templates requiring updates:
- ✅ updated: .specify/templates/plan-template.md
- ✅ updated: .specify/templates/tasks-template.md
- ✅ updated: .specify/templates/spec-template.md
Follow-up TODOs:
- None - focus on simplicity and resume-readiness
-->
# Quant Portfolio System Constitution

## Core Principles

### I. Clean Data is Non-Negotiable
All market data MUST be cleaned, validated, and handled for missing values before any analysis. Use simple, proven methods for data quality - no complex pipelines needed. Rationale: Bad data leads to bad decisions; keep data preprocessing straightforward and reliable.

### II. Risk Management First
Every portfolio optimization MUST include basic risk constraints: position limits, max drawdown protection, and volatility controls. Focus on core risk metrics (Sharpe ratio, max drawdown) rather than complex VaR calculations. Rationale: Demonstrates you understand risk management without overcomplicating the implementation.

### III. Simple Models, Proven Results
Use straightforward ML models (Random Forest, basic XGBoost) with simple validation. No ensemble methods or complex feature engineering needed. Focus on demonstrating clear model performance vs. benchmarks. Rationale: Shows you can apply ML effectively without overengineering.

### IV. Reproducible by Design
All analysis MUST be runnable through simple scripts with clear configurations. No complex CLI frameworks needed - just clean Python scripts that others can easily understand and run. Rationale: Demonstrates good coding practices and makes your project easy to review.

### V. Practical Performance
Optimize for readability and maintainability first. Performance targets should be reasonable for a demo project. Focus on clean, well-documented code rather than premature optimization. Rationale: Resume projects should showcase clean code, not complex engineering.

## Keep It Simple

- Use Python 3.11+ and manage dependencies via `docs/requirements.txt`
- Organize files in logical directories: `data/`, `portfolio/`, `strategies/`, `tests/`
- No complex build systems or deployment pipelines needed
- Focus on core functionality, not infrastructure

## Development Workflow

- Format code with `black` for consistency
- Write basic tests for core functionality
- Keep documentation simple and focused
- No complex CI/CD pipelines needed
- Prioritize working code over process

## Portfolio Essentials

- Focus on 3-4 core optimization methods (Mean-Variance, maybe Black-Litterman)
- Include basic backtesting with simple benchmarks
- Track key metrics: Sharpe ratio, max drawdown, returns
- No complex regime detection or ensemble methods needed

## ML Guidelines

- Use 1-2 simple ML models (Random Forest, basic XGBoost)
- Focus on feature importance and basic validation
- No complex ensemble or ablation studies
- Demonstrate clear value over traditional methods

## Governance

Keep it simple: if you need to change something, make sure it still demonstrates the core concepts clearly. Focus on what will impress interviewers - clean code, good documentation, and working functionality.

**Version**: 1.3.0 | **Ratified**: 2025-09-22 | **Last Amended**: 2025-09-22
