from __future__ import annotations

from pathlib import Path

from portfolio.src.risk.services import (
    RiskConfigurationLoader,
    RiskReportBuilder,
    load_portfolio_snapshot,
)


def main() -> None:
    returns_path = Path("data/storage/demo_returns.parquet")
    weights_path = Path("data/storage/demo_weights.csv")
    snapshot = load_portfolio_snapshot(returns_path, weights_path)

    loader = RiskConfigurationLoader()
    overrides = {
        "reports_path": "data/storage/reports/demo",
        "confidence_levels": [0.95, 0.99],
        "horizons": [1, 10],
        "stress_scenarios": ["macro_recession", "inflation_spike"],
    }
    config = loader.build(overrides=overrides)

    builder = RiskReportBuilder()
    report = builder.generate_report(snapshot, config)
    print(f"Risk report {report.report_id} generated in {config.reports_path}")


if __name__ == "__main__":
    main()

