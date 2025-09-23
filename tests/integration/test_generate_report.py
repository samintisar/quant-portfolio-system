"""Integration tests for the automated report generator."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from scripts.generate_report import run_report
from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
from portfolio.performance.calculator import SimplePerformanceCalculator


class DummyDataService:
    """Stub YahooFinanceService that serves deterministic data."""

    def __init__(self, asset_prices: pd.DataFrame, benchmark_prices: pd.DataFrame, history: pd.DataFrame):
        self.asset_prices = asset_prices
        self.benchmark_prices = benchmark_prices
        self.history = history

    def fetch_price_data(self, symbols: Iterable[str], period: Optional[str] = None, force_online: bool = False) -> pd.DataFrame:
        symbols = [s.upper() for s in symbols]
        asset_cols = list(self.asset_prices.columns)
        bench_cols = list(self.benchmark_prices.columns)

        if len(symbols) > 1:
            if set(symbols) == set(asset_cols):
                return self.asset_prices.copy()
            return pd.DataFrame()

        symbol = symbols[0]
        if symbol in asset_cols:
            return self.asset_prices[[symbol]].copy()
        if symbol in bench_cols:
            return self.benchmark_prices.copy()
        return pd.DataFrame()

    def fetch_historical_data(self, symbol: str, period: Optional[str] = None, force_online: bool = False) -> pd.DataFrame:
        return self.history.copy()


class StubPredictor:
    """Minimal predictor stub that returns fixed feature importance."""

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data[['Adj Close']].copy()
        df['lag_1'] = df['Adj Close'].shift(1)
        df['target'] = df['Adj Close'].pct_change().shift(-1)
        return df.dropna()

    def prepare_features(self, df: pd.DataFrame):
        X = df[['Adj Close', 'lag_1']].values
        y = df['target'].values
        return X, y

    def train(self, X, y) -> Dict[str, float]:
        return {'feature_importance': {'Adj Close': 0.6, 'lag_1': 0.4}}


def build_sample_data() -> Dict[str, pd.DataFrame]:
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=80, freq='B')
    base = np.linspace(100, 120, len(dates))

    asset_prices = pd.DataFrame({
        'AAA': base + np.random.normal(0, 1, len(dates)).cumsum(),
        'BBB': base * 0.8 + np.random.normal(0, 1, len(dates)).cumsum(),
    }, index=dates)

    benchmark_prices = pd.DataFrame({'SPY': base * 0.9 + np.random.normal(0, 1, len(dates)).cumsum()}, index=dates)

    history = pd.DataFrame({
        'Open': asset_prices['AAA'].values,
        'High': asset_prices['AAA'].values + 1,
        'Low': asset_prices['AAA'].values - 1,
        'Close': asset_prices['AAA'].values,
        'Adj Close': asset_prices['AAA'].values,
        'Volume': np.random.randint(1_000_000, 2_000_000, len(dates)),
    }, index=dates)

    return {
        'asset_prices': asset_prices,
        'benchmark_prices': benchmark_prices,
        'history': history,
    }


def test_run_report_creates_artifacts(tmp_path: Path):
    data = build_sample_data()
    service = DummyDataService(**data)

    result = run_report(
        symbols=['AAA', 'BBB'],
        benchmark_symbol='SPY',
        period='1y',
        output_dir=tmp_path / 'report',
        include_feature_importance=True,
        data_service=service,
        optimizer=SimplePortfolioOptimizer(),
        performance_calc=SimplePerformanceCalculator(),
        predictor=StubPredictor(),
    )

    # Expect weights to sum to 1
    assert abs(sum(result['weights'].values()) - 1.0) < 1e-6

    # Metrics should include basic keys
    assert 'total_return' in result['metrics']
    assert 'max_drawdown' in result['metrics']

    paths = result['paths']
    assert Path(paths['report']).exists()
    assert Path(paths['equity_curve']).exists()
    assert Path(paths['drawdown_curve']).exists()
    assert Path(paths['metrics']).exists()
    assert 'feature_importance' in paths
    assert Path(paths['feature_importance']).exists()

    # Feature importance should be propagated
    assert result['feature_importance']['Adj Close'] > 0
    assert result['feature_importance']['lag_1'] > 0
