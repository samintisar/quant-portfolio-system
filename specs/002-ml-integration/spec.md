# Spec: 002 - Machine Learning Integration

## Overview
Implement simple ML models for return prediction and covariance estimation that enhance portfolio optimization without overengineering.

## Requirements
- Return prediction using Random Forest
- Basic feature engineering (momentum, volatility, mean reversion)
- Covariance matrix enhancement
- Simple validation approach
- Feature importance analysis

## Implementation Plan
1. Use scikit-learn for Random Forest/XGBoost
2. Basic technical indicators as features
3. Simple train-test split validation
4. Clear feature importance reporting
5. Integration with optimization methods

## Success Criteria
- > 55% directional prediction accuracy
- Feature importance makes financial sense
- Model trains in < 30 seconds
- Clear improvement over benchmarks

## Anti-Overengineering Rules
- No deep learning models
- No complex ensemble methods
- No hyperparameter optimization
- No cross-validation on time series
- No feature selection algorithms

## Files to Create
- `ml/predictor.py` - Return prediction model
- `ml/features.py` - Feature engineering
- `ml/validation.py` - Model validation
- `tests/test_ml.py` - ML model tests

## Dependencies
- scikit-learn
- xgboost
- pandas
- numpy