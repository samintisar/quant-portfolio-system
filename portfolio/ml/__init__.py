"""
Machine Learning module for return prediction.
"""

from .predictor import RandomForestPredictor, XGBoostPredictor, EnsemblePredictor

__all__ = ['RandomForestPredictor', 'XGBoostPredictor', 'EnsemblePredictor']