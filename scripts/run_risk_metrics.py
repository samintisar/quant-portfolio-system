from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from portfolio.src.risk.services import (
    RiskConfigurationLoader,
    RiskReportBuilder,
    load_portfolio_snapshot,
)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate portfolio risk metrics")
    parser.add_argument("--weights", required=True, help="Path to portfolio weights file")
    parser.add_argument("--returns", required=True, help="Path to returns file (parquet or csv)")
    parser.add_argument(
        "--confidence-levels",
        required=True,
        nargs="+",
        type=float,
        help="Confidence levels for VaR/CVaR computations",
    )
    parser.add_argument(
        "--horizons",
        required=True,
        nargs="+",
        type=int,
        help="Trading horizons in days",
    )
    parser.add_argument("--mc-paths", required=True, type=int, help="Monte Carlo simulation paths")
    parser.add_argument(
        "--scenarios",
        required=True,
        nargs="+",
        help="Stress scenario identifiers to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated risk reports",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["json", "parquet"],
        help="Output format for summary report",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Monte Carlo VaR")
    parser.add_argument("--profile", type=str, default=None, help="Base configuration profile")
    parser.add_argument(
        "--factor-exposures",
        type=str,
        default=None,
        help="Optional CSV with factor exposures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        snapshot = load_portfolio_snapshot(
            returns_path=args.returns,
            weights_path=args.weights,
            factor_exposures_path=args.factor_exposures,
        )
        loader = RiskConfigurationLoader()
        overrides = {
            "confidence_levels": list(args.confidence_levels),
            "horizons": list(args.horizons),
            "mc_paths": args.mc_paths,
            "seed": args.seed,
            "stress_scenarios": list(args.scenarios),
            "reports_path": args.output_dir,
            "reports": ["covariance", "var", "cvar", "stress", "visualizations"],
        }
        config = loader.build(overrides=overrides, profile=args.profile)
        builder = RiskReportBuilder()
        report = builder.generate_report(snapshot, config)
        report_dict = report.to_dict()

        output_dir = Path(config.reports_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            output_path = output_dir / f"{report.report_id}.json"
            output_path.write_text(json.dumps(report_dict, indent=2))
        else:
            output_path = output_dir / f"{report.report_id}.parquet"
            _write_parquet_summary(report_dict, output_path)

        print(f"Generated risk report {report.report_id} at {output_path}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error surface
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _write_parquet_summary(report: dict, output_path: Path) -> None:
    import pandas as pd

    metrics = report.get("risk_metrics", [])
    frame = pd.DataFrame(metrics)
    frame.to_parquet(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

