
"""One-command portfolio report generator with plots and text summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from portfolio.config import get_config
from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
from portfolio.performance.calculator import SimplePerformanceCalculator
from portfolio.performance.visualization import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_feature_importance,
)
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.ml import RandomForestPredictor


def _clean_weights(weights: Dict[str, float], columns: Sequence[str]) -> Dict[str, float]:
    """Align weights to available columns and ensure they sum to one."""
    series = pd.Series(weights or {}, dtype=float)
    series = series.reindex(columns).fillna(0.0)
    total = series.sum()
    if total <= 0:
        series = pd.Series(1.0 / len(columns), index=columns)
    else:
        series = series / total
    return {k: float(v) for k, v in series.items()}


def _serialize_numeric_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    """Convert numpy types to native Python floats for JSON."""
    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            cleaned[key] = value
    return cleaned


def run_report(
    symbols: Iterable[str],
    benchmark_symbol: Optional[str] = None,
    period: Optional[str] = None,
    output_dir: str | Path = "reports/latest",
    risk_free_rate: Optional[float] = None,
    include_feature_importance: bool = True,
    data_service: Optional[YahooFinanceService] = None,
    optimizer: Optional[SimplePortfolioOptimizer] = None,
    performance_calc: Optional[SimplePerformanceCalculator] = None,
    predictor: Optional[RandomForestPredictor] = None,
) -> Dict[str, object]:
    """Generate portfolio report artifacts and return summary metadata."""
    symbols = [s.upper() for s in symbols]
    if len(symbols) == 0:
        raise ValueError("At least one symbol is required")

    cfg = get_config()
    config_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    portfolio_cfg = config_dict.get("portfolio", {})
    default_period = portfolio_cfg.get("default_period", "3y")
    risk_free = risk_free_rate if risk_free_rate is not None else portfolio_cfg.get("risk_free_rate", 0.02)

    data_service = data_service or YahooFinanceService()
    optimizer = optimizer or SimplePortfolioOptimizer()
    performance_calc = performance_calc or SimplePerformanceCalculator(risk_free_rate=risk_free)
    if include_feature_importance:
        predictor = predictor or RandomForestPredictor()
    else:
        predictor = None
    period = period or default_period

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    price_data = data_service.fetch_price_data(symbols, period=period)
    if price_data.empty:
        raise ValueError("No price data available for requested symbols")

    price_data = price_data.sort_index().dropna(how="all")
    returns = price_data.pct_change().dropna()
    if returns.empty:
        raise ValueError("Unable to compute returns from price data")

    if len(returns.columns) == 1:
        weights = {returns.columns[0]: 1.0}
        optimization_result = {
            "expected_return": float(returns.mean().iloc[0] * performance_calc.trading_days_per_year),
            "expected_volatility": float(returns.std().iloc[0] * (performance_calc.trading_days_per_year ** 0.5)),
            "sharpe_ratio": 0.0,
        }
    else:
        optimization_result = optimizer.mean_variance_optimize(returns)
        weights = optimization_result.get("weights", {})

    weights = _clean_weights(weights, returns.columns)
    weights_series = pd.Series(weights)
    portfolio_returns = (returns * weights_series).sum(axis=1)

    benchmark_returns: Optional[pd.Series] = None
    if benchmark_symbol:
        benchmark_symbol = benchmark_symbol.upper()
        benchmark_prices = data_service.fetch_price_data([benchmark_symbol], period=period)
        if not benchmark_prices.empty:
            bench_series = benchmark_prices.iloc[:, 0].pct_change().dropna()
            aligned = pd.concat([portfolio_returns, bench_series], axis=1, join="inner")
            if not aligned.empty:
                portfolio_returns = aligned.iloc[:, 0]
                benchmark_returns = aligned.iloc[:, 1]

    metrics = performance_calc.calculate_metrics(portfolio_returns, benchmark_returns)
    report_text = performance_calc.generate_report(metrics)

    equity_path = output_path / "equity_curve.png"
    drawdown_path = output_path / "drawdown_curve.png"
    feature_path: Optional[Path] = None

    # Save equity curve plot
    eq_ax = plot_equity_curve(portfolio_returns, benchmark_returns)
    eq_ax.figure.tight_layout()
    eq_ax.figure.savefig(equity_path, dpi=150, bbox_inches="tight")
    plt.close(eq_ax.figure)

    # Save drawdown curve plot
    dd_ax = plot_drawdown_curve(portfolio_returns)
    dd_ax.figure.tight_layout()
    dd_ax.figure.savefig(drawdown_path, dpi=150, bbox_inches="tight")
    plt.close(dd_ax.figure)

    feature_importance: Dict[str, float] = {}
    if include_feature_importance and predictor is not None:
        base_symbol = symbols[0]
        try:
            history = data_service.fetch_historical_data(base_symbol, period=period)
            if not history.empty:
                features = predictor.create_features(history)
                if not features.empty:
                    X, y = predictor.prepare_features(features)
                    if len(X) >= 10:
                        ml_metrics = predictor.train(X, y)
                        feature_importance = ml_metrics.get("feature_importance", {}) or {}
        except Exception:
            feature_importance = {}

        if feature_importance:
            feature_path = output_path / "feature_importance.png"
            fi_ax = plot_feature_importance(feature_importance)
            fi_ax.figure.tight_layout()
            fi_ax.figure.savefig(feature_path, dpi=150, bbox_inches="tight")
            plt.close(fi_ax.figure)

    # Append weights and highlights to the text report
    report_lines = [report_text, "", "Weights:"]
    for symbol, weight in weights.items():
        report_lines.append(f"  {symbol}: {weight:.2%}")

    optimization_summary = {
        "expected_return": float(optimization_result.get("expected_return", 0.0)),
        "expected_volatility": float(optimization_result.get("expected_volatility", 0.0)),
        "sharpe_ratio": float(optimization_result.get("sharpe_ratio", 0.0)),
    }
    report_lines.append("")
    report_lines.append("Optimization Summary:")
    report_lines.append(f"  Expected Return: {optimization_summary['expected_return']:.2%}")
    report_lines.append(f"  Expected Volatility: {optimization_summary['expected_volatility']:.2%}")
    report_lines.append(f"  Implied Sharpe: {optimization_summary['sharpe_ratio']:.2f}")

    if feature_importance:
        report_lines.append("")
        report_lines.append("Top Features:")
        for feature, score in sorted(feature_importance.items(), key=lambda item: item[1])[-5:][::-1]:
            report_lines.append(f"  {feature}: {score:.4f}")

    report_path = output_path / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metrics_payload = {
        "symbols": symbols,
        "benchmark": benchmark_symbol,
        "weights": {k: float(v) for k, v in weights.items()},
        "metrics": _serialize_numeric_dict(metrics),
        "optimization": optimization_summary,
        "feature_importance": _serialize_numeric_dict(feature_importance),
        "paths": {
            "report": str(report_path),
            "equity_curve": str(equity_path),
            "drawdown_curve": str(drawdown_path),
        },
        "period": period,
    }
    if feature_path is not None:
        metrics_payload["paths"]["feature_importance"] = str(feature_path)

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metrics_payload["paths"]["metrics"] = str(metrics_path)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portfolio performance report with plots")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to include", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--benchmark", dest="benchmark", default="SPY", help="Benchmark symbol (default: SPY)")
    parser.add_argument("--period", dest="period", default=None, help="Lookback period (defaults to config)")
    parser.add_argument("--output", dest="output", default="reports/latest", help="Output directory for artifacts")
    parser.add_argument("--risk-free", dest="risk_free", type=float, default=None, help="Override risk-free rate")
    parser.add_argument("--no-features", dest="include_features", action="store_false", help="Skip feature importance plot")

    args = parser.parse_args()

    payload = run_report(
        symbols=args.symbols,
        benchmark_symbol=args.benchmark,
        period=args.period,
        output_dir=args.output,
        risk_free_rate=args.risk_free,
        include_feature_importance=args.include_features,
    )

    print("Report generated:")
    for key, value in payload["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
