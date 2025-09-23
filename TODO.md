# Quant Portfolio System - Resume Project TODO

## Project Overview
**Goal**: Build a portfolio optimization system with ML enhancements for resume demonstration
**Focus**: Production-ready application with core features that impress recruiters
**Timeline**: 2-3 weeks (part-time)

---

## Spec-Driven Development Plan

### Phase 1: Portfolio Optimization (001)
**Spec**: `specs/001-portfolio-optimization/spec.md`
- Mean-Variance, Black-Litterman, CVaR optimization
- Basic risk constraints and performance metrics
- Simple validation approach

### Phase 2: ML Integration (002)
**Spec**: `specs/002-ml-integration/spec.md`
- Random Forest return prediction
- Basic feature engineering
- Simple validation and feature importance

### Phase 3: Data Pipeline (003)
**Spec**: `specs/003-data-pipeline/spec.md`
- Yahoo Finance data ingestion
- Cleaning, validation, normalization
- Simple quality reporting

### Phase 4: Backtesting (004)
**Spec**: `specs/004-backtesting/spec.md`
- Walk-forward validation
- Benchmark comparison
- Basic performance attribution

### Phase 5: API & Dashboard (005)
**Spec**: `specs/005-api-dashboard/spec.md`
- FastAPI endpoints
- Streamlit dashboard
- Simple deployment setup

---

## Spec Execution Status

### 001 - Portfolio Optimization
- [ ] Create `portfolio/optimization.py` - Core optimization methods
- [ ] Create `portfolio/metrics.py` - Performance calculation
- [ ] Create `portfolio/constraints.py` - Constraint handling
- [ ] Create `tests/test_optimization.py` - Basic optimization tests

### 002 - ML Integration
- [ ] Create `ml/predictor.py` - Return prediction model
- [ ] Create `ml/features.py` - Feature engineering
- [ ] Create `ml/validation.py` - Model validation
- [ ] Create `tests/test_ml.py` - ML model tests

### 003 - Data Pipeline
- [ ] Create `data/src/feeds/yahoo.py` - Yahoo Finance data ingestion
- [ ] Create `data/src/lib/cleaning.py` - Data cleaning utilities
- [ ] Create `data/src/lib/validation.py` - Data validation
- [ ] Create `data/src/lib/normalization.py` - Data normalization
- [ ] Create `tests/test_data.py` - Data processing tests

### 004 - Backtesting
- [ ] Create `backtesting/engine.py` - Backtesting core logic
- [ ] Create `backtesting/benchmarks.py` - Benchmark portfolios
- [ ] Create `backtesting/metrics.py` - Performance metrics
- [ ] Create `tests/test_backtesting.py` - Backtesting tests

### 005 - API & Dashboard
- [ ] Create `api/main.py` - FastAPI application
- [ ] Create `dashboard/app.py` - Streamlit dashboard
- [ ] Create `docker-compose.yml` - Docker setup
- [ ] Create `tests/test_api.py` - API tests

---

## Anti-Overengineering Checklist
- [ ] Each spec follows simplified constitution principles
- [ ] No complex ensemble methods or deep learning
- [ ] Focus on core concepts, not production features
- [ ] Code readability over performance optimization
- [ ] Simple validation approaches
- [ ] Resume-friendly feature demonstration

---

## Project Status: **SPEC-DRIVEN DEVELOPMENT**
**Current Focus**: Execute specs in order (001 ’ 002 ’ 003 ’ 004 ’ 005)
**Next Step**: Begin implementation of Spec 001 - Portfolio Optimization