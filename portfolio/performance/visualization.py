"""Lightweight plotting utilities for portfolio performance visuals."""

from __future__ import annotations

from typing import Dict, Optional, Union

import matplotlib

matplotlib.use("Agg")  # Ensure headless rendering works in tests
import matplotlib.pyplot as plt
import pandas as pd

SeriesLike = Union[pd.Series, Dict[pd.Timestamp, float]]


def _to_series(data: SeriesLike) -> pd.Series:
    """Convert supported inputs to a pandas Series indexed by date."""
    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        series = pd.Series(data)

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    return series.sort_index()


def prepare_equity_curve(returns: SeriesLike) -> pd.Series:
    """Return cumulative growth of $1 based on simple returns."""
    series = _to_series(returns).dropna()
    if series.empty:
        return pd.Series(dtype=float)
    return (1 + series).cumprod()


def prepare_drawdown_curve(returns: SeriesLike) -> pd.Series:
    """Return running drawdown series using peak-to-trough logic."""
    equity = prepare_equity_curve(returns)
    if equity.empty:
        return equity
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    return drawdown


def plot_equity_curve(
    portfolio_returns: SeriesLike,
    benchmark_returns: Optional[SeriesLike] = None,
    title: str = "Equity Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot cumulative performance for portfolio (and optional benchmark)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    portfolio_curve = prepare_equity_curve(portfolio_returns)
    ax.plot(portfolio_curve.index, portfolio_curve.values, label="Strategy", linewidth=2)

    if benchmark_returns is not None:
        benchmark_curve = prepare_equity_curve(benchmark_returns)
        if not benchmark_curve.empty:
            ax.plot(benchmark_curve.index, benchmark_curve.values, label="Benchmark", linestyle="--")

    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax


def plot_drawdown_curve(
    returns: SeriesLike,
    title: str = "Drawdown Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot running drawdown for the portfolio."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    drawdown = prepare_drawdown_curve(returns)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(drawdown.min() * 1.1 if not drawdown.empty else -0.1, 0.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    return ax


def plot_feature_importance(
    feature_importance: Union[Dict[str, float], pd.Series],
    top_n: int = 15,
    title: str = "Feature Importance",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot feature importance scores as a horizontal bar chart."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    series = pd.Series(feature_importance).dropna().sort_values(ascending=True)
    if top_n > 0 and len(series) > top_n:
        series = series.tail(top_n)

    ax.barh(series.index, series.values, color="tab:blue", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    return ax


__all__ = [
    "prepare_equity_curve",
    "prepare_drawdown_curve",
    "plot_equity_curve",
    "plot_drawdown_curve",
    "plot_feature_importance",
]
