"""
API endpoints module.

Contains all FastAPI endpoint definitions for the portfolio optimization system.
"""

from .optimize import router as optimize_router
from .analyze import router as analyze_router
from .assets import router as assets_router

__all__ = ['optimize_router', 'analyze_router', 'assets_router']