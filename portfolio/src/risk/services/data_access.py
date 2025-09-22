from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.snapshot import PortfolioSnapshot


def load_returns(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame = frame.sort_index()
    return frame


def load_weights(path: Path | str) -> pd.Series:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        data = pd.read_parquet(path)
    else:
        data = pd.read_csv(path, index_col=0)
    if data.shape[1] == 0:
        raise ValueError("weights file must contain at least one column")
    series = data.iloc[:, 0].astype(float)
    series.name = series.name or "weight"
    return series


def load_portfolio_snapshot(
    returns_path: Path | str,
    weights_path: Path | str,
    factor_exposures_path: Optional[Path | str] = None,
) -> PortfolioSnapshot:
    returns = load_returns(returns_path)
    weights = load_weights(weights_path)
    exposures = None
    if factor_exposures_path:
        exposures = pd.read_csv(factor_exposures_path, index_col=0)
    return PortfolioSnapshot(
        asset_ids=list(returns.columns),
        returns=returns,
        weights=weights,
        factor_exposures=exposures,
        timestamp=returns.index.max() if not returns.empty else pd.Timestamp.utcnow(),
    )

