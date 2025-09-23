"""Tests for lightweight visualization helpers."""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio.performance.visualization import (
    prepare_equity_curve,
    prepare_drawdown_curve,
    plot_equity_curve,
    plot_drawdown_curve,
    plot_feature_importance,
)


def sample_returns():
    index = pd.date_range("2023-01-01", periods=6, freq="D")
    values = pd.Series([0.01, -0.005, 0.015, -0.02, 0.01, 0.005], index=index)
    return values


def test_prepare_equity_curve_growth():
    returns = sample_returns()
    curve = prepare_equity_curve(returns)
    assert len(curve) == len(returns)
    np.testing.assert_allclose(curve.iloc[0], 1 + returns.iloc[0])
    assert curve.iloc[-1] > 0.95  # portfolio ends near par


def test_prepare_drawdown_curve_never_positive():
    returns = sample_returns()
    drawdown = prepare_drawdown_curve(returns)
    assert len(drawdown) == len(returns)
    assert (drawdown <= 0 + 1e-9).all()
    assert drawdown.min() <= drawdown.max()


def test_plot_equity_curve_creates_lines():
    returns = sample_returns()
    benchmark = returns * 0.8
    ax = plot_equity_curve(returns, benchmark)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1
    plt.close(ax.figure)


def test_plot_drawdown_curve_creates_fill_between():
    returns = sample_returns()
    ax = plot_drawdown_curve(returns)
    assert isinstance(ax, plt.Axes)
    assert len(ax.collections) >= 1
    plt.close(ax.figure)


def test_plot_feature_importance_handles_dict():
    importance = {"feature_a": 0.2, "feature_b": 0.5, "feature_c": 0.3}
    ax = plot_feature_importance(importance)
    assert isinstance(ax, plt.Axes)
    assert len(ax.patches) == len(importance)
    plt.close(ax.figure)
