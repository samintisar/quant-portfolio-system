"""
Run walk-forward experiments with realistic optimization settings on offline data.

Now supports simple knobs: risk_model, weight_cap, turnover_penalty, and method filter.
"""

import argparse
import json
from datetime import datetime
from typing import List

from portfolio.backtesting.walk_forward import BacktestConfig, run_walk_forward_backtest
from portfolio.config import get_config


def run(symbols: List[str], start_date: str, end_date: str,
        weight_cap: float, risk_model: str, turnover_penalty: float, method: str):
    cfg = BacktestConfig(
        train_years=1,
        test_quarters=1,
        include_ml_overlay=False,
        include_equal_weight_baseline=True,
        max_position_cap=weight_cap,
        risk_model=risk_model,
        turnover_penalty=turnover_penalty,
    )

    results = run_walk_forward_backtest(symbols, start_date, end_date, cfg)

    # Optionally filter to one method for reporting brevity
    if method and method.lower() != "all":
        keep = []
        if method == "mean_variance":
            keep = ["mv_unconstrained"]
        elif method == "mean_variance_capped":
            keep = ["mv_capped_20"]
        elif method == "cvar":
            keep = ["cvar_capped_20"]
        elif method == "black_litterman":
            keep = ["black_litterman_capped_20"]
        elif method == "equal_weight":
            keep = ["equal_weight"]
        elif method == "ml_overlay":
            keep = ["ml_overlay"]
        results = {k: v for k, v in results.items() if k in keep}

    # Build compact metrics dict
    out = {}
    for name, res in results.items():
        if name == "comparison":
            continue
        m = res.metrics or {}
        out[name] = {
            "total_return": round(float(m.get("total_return", 0.0)), 6),
            "annual_return": round(float(m.get("annual_return", 0.0)), 6),
            "annual_volatility": round(float(m.get("annual_volatility", 0.0)), 6),
            "sharpe_ratio": round(float(m.get("sharpe_ratio", 0.0)), 4),
            "sortino_ratio": round(float(m.get("sortino_ratio", 0.0)), 4),
            "max_drawdown": round(float(m.get("max_drawdown", 0.0)), 6),
            "turnover": round(float(getattr(res, 'turnover', 0.0)), 6),
            "information_ratio": round(float(m.get("information_ratio", 0.0)), 4) if "information_ratio" in m else None,
        }

    print(json.dumps({
        "universe": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "risk_model": risk_model,
        "weight_cap": weight_cap,
        "turnover_penalty": turnover_penalty,
        "method": method,
        "metrics": out,
    }, indent=2))


if __name__ == "__main__":
    cfg = get_config()
    defaults = cfg.to_dict() if hasattr(cfg, 'to_dict') else {}
    bt = defaults.get('backtest', {})
    opt = defaults.get('optimization', {})

    parser = argparse.ArgumentParser(description="Run walk-forward experiments")
    parser.add_argument('--start', default="2020-01-01")
    parser.add_argument('--end', default="2025-09-22")
    parser.add_argument('--symbols', nargs='*', default=[
        "AAPL","MSFT","GOOGL","AMZN","TSLA",
        "NFLX","NVDA","JPM","PG","UNH",
    ])
    parser.add_argument('--weight-cap', type=float, default=float(bt.get('max_position_cap', 0.20)))
    parser.add_argument('--risk-model', choices=['sample','ledoit_wolf','oas'], default=str(opt.get('risk_model', 'ledoit_wolf')))
    parser.add_argument('--turnover-penalty', type=float, default=float(opt.get('turnover_penalty', 0.0)))
    parser.add_argument('--method', default='all', help='Filter one method for reporting (or all)')
    args = parser.parse_args()

    run(args.symbols, args.start, args.end,
        args.weight_cap, args.risk_model, args.turnover_penalty, args.method)
