import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest



def _load_parser():
    try:
        module = importlib.import_module("scripts.run_risk_metrics")
    except ModuleNotFoundError as exc:  # pragma: no cover - failing path under test
        pytest.fail("scripts.run_risk_metrics module is missing")
    if not hasattr(module, "build_cli_parser"):
        pytest.fail("scripts.run_risk_metrics.build_cli_parser is required by the CLI contract")
    return module.build_cli_parser()


def test_cli_contract_requires_core_arguments() -> None:
    parser = _load_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args([
        "--weights",
        "data/storage/demo_weights.csv",
        "--returns",
        "data/storage/demo_returns.parquet",
        "--confidence-levels",
        "0.95",
        "0.99",
        "--horizons",
        "1",
        "10",
        "--mc-paths",
        "10000",
        "--scenarios",
        "macro_recession",
        "inflation_spike",
        "--output-dir",
        "reports/demo",
        "--format",
        "json",
        "--seed",
        "123",
    ])

    assert Path(args.weights).name == "demo_weights.csv"
    assert Path(args.returns).suffix == ".parquet"
    assert args.output_dir == "reports/demo"
    assert args.scenarios == ["macro_recession", "inflation_spike"]
    assert args.format == "json"
    assert args.mc_paths == 10000


def test_cli_contract_format_choices() -> None:
    parser = _load_parser()
    format_action = parser._option_string_actions["--format"]
    assert set(format_action.choices) == {"json", "parquet"}
