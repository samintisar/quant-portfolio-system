"""
Walk-forward backtesting system for quantitative portfolio optimization.

Implements expanding window validation with transaction costs and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from portfolio.optimizer.optimizer import SimplePortfolioOptimizer
from portfolio.performance.calculator import SimplePerformanceCalculator
from portfolio.data.yahoo_service import YahooFinanceService
from portfolio.ml import RandomForestPredictor

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting."""
    train_years: int = 1
    test_quarters: int = 1
    transaction_cost_bps: float = 7.5  # 7.5 bps per trade
    rebalance_frequency: str = 'quarterly'  # 'quarterly' or 'monthly'
    benchmark_symbol: str = 'SPY'
    include_equal_weight_baseline: bool = True
    include_ml_overlay: bool = True
    ml_tilt_alpha: float = 0.2  # strength of ML tilt over MVO weights
    max_position_cap: float = 0.18  # per-asset cap for constrained runs
    risk_model: str = 'ledoit_wolf'  # 'sample'|'ledoit_wolf'|'oas'
    turnover_penalty: float = 0.0
    entropy_penalty: float = 0.03  # diversification control for MVO
    cvar_alpha: float = 0.10  # tail probability for CVaR optimization

@dataclass
class BacktestResult:
    """Results from backtesting run."""
    dates: List[datetime]
    returns: pd.Series
    weights_history: pd.DataFrame
    metrics: Dict[str, float]
    transaction_costs: float
    turnover: float
    baseline_returns: Optional[pd.Series] = None
    ml_overlay_returns: Optional[pd.Series] = None

class WalkForwardBacktester:
    """Walk-forward backtesting system with transaction costs and performance metrics."""

    def __init__(self, config: BacktestConfig):
        """Initialize backtester with configuration."""
        self.config = config
        self.optimizer = SimplePortfolioOptimizer()
        self.performance_calc = SimplePerformanceCalculator()
        self.data_service = YahooFinanceService()
        self.train_days = config.train_years * 252  # Trading days per year
        self.test_days = config.test_quarters * 63  # ~63 trading days per quarter

        logger.info(f"Initialized WalkForwardBacktester with {config.train_years}y train, "
                   f"{config.test_quarters}q test, {config.transaction_cost_bps}bps costs")

        # Wire optimizer knobs (risk model, turnover penalty) from config
        try:
            self.optimizer.risk_model = self.config.risk_model
            self.optimizer.default_turnover_penalty = float(self.config.turnover_penalty)
        except Exception:
            pass

    def _load_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load price data using offline cache with graceful fallback."""
        collected = []

        for symbol in symbols:
            try:
                data = self.data_service.fetch_historical_data(symbol, period="5y")
                if not data.empty and "Adj Close" in data:
                    collected.append(data["Adj Close"].rename(symbol))
                    continue
            except Exception as offline_error:
                logger.debug(f"Offline load failed for {symbol}: {offline_error}")

        if collected:
            prices = pd.concat(collected, axis=1).dropna(how="all")
            if not prices.empty:
                return prices

        logger.info("Falling back to optimizer.fetch_data for price loading")
        return self.optimizer.fetch_data(symbols)

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, BacktestResult]:
        """
        Run walk-forward backtest.

        Args:
            symbols: List of asset symbols
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)

        Returns:
            Dictionary with backtest results for different strategies
        """
        try:
            # Fetch data for all symbols
            logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
            prices = self._load_price_data(symbols)

            # Filter to date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            prices = prices.loc[start_dt:end_dt]

            if prices.empty:
                raise ValueError("No data available for the specified date range")

            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(prices)

            logger.info(f"Generated {len(windows)} walk-forward windows")

            # Build SPY benchmark walk-forward first for benchmark-relative stats
            spy_walk_forward = None
            try:
                spy_prices = self._load_price_data([self.config.benchmark_symbol])
                spy_prices = spy_prices.loc[start_dt:end_dt]
                spy_returns = spy_prices.pct_change().dropna().iloc[:, 0]

                test_mask = pd.Series(False, index=spy_returns.index)
                for _, train_end, test_end in windows:
                    seg = spy_returns.loc[train_end + timedelta(days=1):test_end]
                    test_mask.loc[seg.index] = True
                spy_walk_forward = spy_returns.loc[test_mask[test_mask].index].reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Failed to prepare SPY benchmark series: {e}")

            # Run backtests for each strategy
            results = {}

            # Mean-variance optimization (unconstrained) and capped variant
            results['mv_unconstrained'] = self._run_strategy_backtest(
                symbols, prices, windows, 'mean_variance', benchmark_returns=spy_walk_forward
            )
            # Dynamic label reflecting configured cap (e.g., mv_capped_18 for 18%)
            cap_pct_label = int(round(self.config.max_position_cap * 100)) if self.config.max_position_cap is not None else 0
            mv_capped_label = f"mv_capped_{cap_pct_label}"
            results[mv_capped_label] = self._run_strategy_backtest(
                symbols, prices, windows, 'mean_variance_capped', benchmark_returns=spy_walk_forward
            )

            # CVaR and Black–Litterman strategies (with caps)
            results['cvar_capped_20'] = self._run_strategy_backtest(
                symbols, prices, windows, 'cvar', benchmark_returns=spy_walk_forward
            )
            results['black_litterman_capped_20'] = self._run_strategy_backtest(
                symbols, prices, windows, 'black_litterman', benchmark_returns=spy_walk_forward
            )

            # Equal weight baseline
            if self.config.include_equal_weight_baseline:
                results['equal_weight'] = self._run_strategy_backtest(
                    symbols, prices, windows, 'equal_weight'
                )

            # ML-weighted overlay (if enabled)
            if self.config.include_ml_overlay:
                results['ml_overlay'] = self._run_strategy_backtest(
                    symbols, prices, windows, 'ml_overlay'
                )

            # Maintain legacy alias for the primary optimized strategy
            primary_key = None
            # Prefer capped MVO (current configured cap), then others
            for candidate in (mv_capped_label, 'mv_unconstrained', 'cvar_capped_20', 'black_litterman_capped_20'):
                if candidate in results and results[candidate] is not None:
                    primary_key = candidate
                    break
            if primary_key and 'optimized' not in results:
                results['optimized'] = results[primary_key]

            # SPY benchmark walk-forward (buy and hold across test windows)
            try:
                spy_metrics = self._calculate_comprehensive_metrics(spy_walk_forward if spy_walk_forward is not None else pd.Series(dtype=float))
                results['spy'] = BacktestResult(
                    dates=[w[2] for w in windows],
                    returns=(spy_walk_forward if spy_walk_forward is not None else pd.Series(dtype=float)),
                    weights_history=pd.DataFrame({'SPY': [1.0] * len(windows)}, index=[w[2] for w in windows]),
                    metrics=spy_metrics,
                    transaction_costs=0.0,
                    turnover=0.0
                )
            except Exception as e:
                logger.warning(f"Failed to compute SPY benchmark: {e}")

            # Calculate comparative metrics (avoid double-counting optimized alias)
            comparison_inputs = dict(results)
            comparison_inputs.pop('comparison', None)
            if 'optimized' in comparison_inputs:
                for alias in (mv_capped_label, 'mv_unconstrained'):
                    if alias in comparison_inputs and comparison_inputs[alias] is comparison_inputs['optimized']:
                        comparison_inputs.pop(alias)
                        break
            results['comparison'] = self._calculate_comparison_metrics(comparison_inputs)

            logger.info("Backtest completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    def _generate_walk_forward_windows(self, prices: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Generate walk-forward train/test windows."""
        if prices.empty:
            return []

        windows = []
        start_date = prices.index[0]
        end_date = prices.index[-1]

        current_start = start_date

        while current_start + timedelta(days=self.train_days + self.test_days) <= end_date:
            train_end = current_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)

            # Ensure we have actual trading days
            train_end_actual = prices.index[prices.index <= train_end][-1]
            test_end_actual = prices.index[prices.index <= test_end][-1]

            windows.append((current_start, train_end_actual, test_end_actual))

            # Move to next window (expand by test period)
            current_start = test_end_actual + timedelta(days=1)

            # Stop if we don't have enough data for next window
            if current_start + timedelta(days=self.train_days + self.test_days) > end_date:
                break

        return windows

    def _run_strategy_backtest(self, symbols: List[str], prices: pd.DataFrame,
                             windows: List[Tuple], strategy: str,
                             benchmark_returns: Optional[pd.Series] = None) -> BacktestResult:
        """Run backtest for a specific strategy."""
        returns_list = []
        weights_history = []
        transaction_costs = 0
        previous_weights = None

        for train_start, train_end, test_end in windows:
            try:
                # Get training data
                train_prices = prices.loc[train_start:train_end]
                test_prices = prices.loc[train_end+timedelta(days=1):test_end]

                if test_prices.empty:
                    continue

                # Calculate optimal weights based on strategy
                if strategy == 'mean_variance':
                    weights = self._calculate_mean_variance_weights(train_prices, symbols, weight_cap=None, previous_weights=previous_weights)
                elif strategy == 'mean_variance_capped':
                    weights = self._calculate_mean_variance_weights(train_prices, symbols, weight_cap=self.config.max_position_cap, previous_weights=previous_weights)
                elif strategy == 'cvar':
                    weights = self._calculate_cvar_weights(train_prices, symbols, weight_cap=self.config.max_position_cap)
                elif strategy == 'black_litterman':
                    weights = self._calculate_black_litterman_weights(train_prices, symbols, weight_cap=self.config.max_position_cap)
                elif strategy == 'equal_weight':
                    weights = {symbol: 1.0/len(symbols) for symbol in symbols}
                elif strategy == 'ml_overlay':
                    weights = self._calculate_ml_overlay_weights(train_prices, symbols)
                else:
                    weights = {symbol: 1.0/len(symbols) for symbol in symbols}

                # Calculate transaction costs
                applied_turnover = 0.0
                if previous_weights is not None:
                    turnover = self._calculate_turnover(previous_weights, weights)
                    cost = turnover * self.config.transaction_cost_bps / 10000
                    transaction_costs += cost
                    applied_turnover = turnover

                # Store weights
                weights_history.append({
                    'date': test_end,
                    **weights
                })

                # Calculate test period returns
                test_returns = test_prices.pct_change().dropna()
                portfolio_returns = (test_returns * pd.Series(weights)).sum(axis=1)
                # Apply transaction cost at the start of the test segment (net-of-cost)
                if applied_turnover > 0 and not portfolio_returns.empty:
                    cost = applied_turnover * self.config.transaction_cost_bps / 10000
                    portfolio_returns.iloc[0] = portfolio_returns.iloc[0] - cost

                returns_list.extend(portfolio_returns.values)
                previous_weights = weights

            except Exception as e:
                logger.warning(f"Error in window {train_start} to {test_end}: {e}")
                continue

        # Create results
        returns_series = pd.Series(returns_list)
        weights_df = pd.DataFrame(weights_history).set_index('date')

        # Calculate metrics (benchmark-relative if provided)
        metrics = self._calculate_comprehensive_metrics(returns_series, benchmark_returns)

        # Compute diversification metrics (Effective Number of Holdings)
        try:
            if not weights_df.empty:
                def _enh(row: pd.Series) -> float:
                    import numpy as np
                    w = row.drop(labels=['date'], errors='ignore').astype(float)
                    w = w[w > 0]
                    if w.empty:
                        return 0.0
                    eps = 1e-12
                    return float(np.exp(-(w * np.log(w + eps)).sum()))
                enh_series = weights_df.apply(_enh, axis=1)
                metrics['avg_enh'] = float(enh_series.mean()) if not enh_series.empty else 0.0
                metrics['min_enh'] = float(enh_series.min()) if not enh_series.empty else 0.0
                metrics['max_enh'] = float(enh_series.max()) if not enh_series.empty else 0.0
        except Exception:
            pass

        # Calculate total turnover
        total_turnover = self._calculate_total_turnover(weights_history)

        return BacktestResult(
            dates=[w[2] for w in windows],
            returns=returns_series,
            weights_history=weights_df,
            metrics=metrics,
            transaction_costs=transaction_costs,
            turnover=total_turnover
        )

    def _calculate_mean_variance_weights(self, train_prices: pd.DataFrame, symbols: List[str], weight_cap: Optional[float] = None, previous_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate weights using mean-variance optimization."""
        try:
            returns = train_prices.pct_change().dropna()
            if returns.empty:
                return {symbol: 1.0/len(symbols) for symbol in symbols}

            prev_vec = None
            if previous_weights is not None:
                prev_vec = np.array([previous_weights.get(sym, 0.0) for sym in symbols])

            result = self.optimizer.mean_variance_optimize(
                returns,
                weight_cap=weight_cap,
                risk_model=self.config.risk_model,
                entropy_penalty=getattr(self.config, 'entropy_penalty', 0.0),
                turnover_penalty=self.config.turnover_penalty,
                previous_weights=prev_vec,
            )
            return result['weights']

        except Exception as e:
            logger.warning(f"Error in mean-variance optimization: {e}")
            return {symbol: 1.0/len(symbols) for symbol in symbols}

    def _calculate_ml_overlay_weights(self, train_prices: pd.DataFrame, symbols: List[str]) -> Dict[str, float]:
        """Calculate weights using an RF-based ML tilt over MVO weights.

        - Train RandomForest per symbol on the training window (next-day returns target)
        - Use the most recent predicted return as a cross-sectional signal
        - Z-score and tilt MVO weights multiplicatively: w_final ∝ w_mvo * exp(alpha * z)
        - Apply position cap and renormalize
        """
        try:
            # Baseline MVO weights on training window (capped)
            mvo_weights = self._calculate_mean_variance_weights(
                train_prices, symbols, weight_cap=self.config.max_position_cap, previous_weights=None
            )

            # Prepare signals per symbol
            alpha = float(getattr(self.config, 'ml_tilt_alpha', 0.2) or 0.2)
            signals: Dict[str, float] = {}

            train_start = train_prices.index.min()
            train_end = train_prices.index.max()

            for sym in symbols:
                try:
                    raw = self.data_service.fetch_historical_data(sym, period="5y")
                    if raw is None or raw.empty:
                        continue
                    # Slice to training window (ensure enough lookback already present)
                    df = raw.loc[:train_end]
                    df = df.loc[max(train_start, df.index.min()):train_end]
                    if df.empty:
                        continue

                    rf = RandomForestPredictor()
                    feat_df = rf.create_features(df)
                    if feat_df.empty:
                        continue
                    X, y = rf.prepare_features(feat_df)
                    if X.shape[0] < 50:
                        # Avoid unstable fits on tiny samples
                        continue
                    rf.train(X, y)
                    # Use the latest feature row for a next-day prediction
                    pred = float(rf.predict(X[-1].reshape(1, -1))[0])
                    signals[sym] = pred
                except Exception:
                    # Skip symbol on failure
                    continue

            if not signals:
                return mvo_weights

            # Z-score signals across universe (winsorize)
            sig_series = pd.Series({s: signals.get(s, 0.0) for s in symbols}, dtype=float)
            mean = float(sig_series.mean())
            std = float(sig_series.std(ddof=0))
            if std == 0 or np.isnan(std):
                z = pd.Series(0.0, index=sig_series.index)
            else:
                z = (sig_series - mean) / std
            z = z.clip(-2.0, 2.0)

            # Apply multiplicative exponential tilt with min-signal threshold
            w_mvo = pd.Series({s: mvo_weights.get(s, 0.0) for s in symbols}, dtype=float)
            strength = float(z.abs().mean())
            if strength < 0.10:
                # Too weak signal; avoid tilt
                tilted = w_mvo.copy()
            else:
                alpha_eff = alpha * min(1.0, strength)
                tilted = w_mvo * np.exp(alpha_eff * z)
            tilted = tilted.clip(lower=0.0)

            # Renormalize
            s = float(tilted.sum())
            if s == 0 or np.isnan(s):
                tilted = w_mvo.copy()
                s = float(max(tilted.sum(), 1e-12))
            tilted = tilted / s

            # Apply position cap and renormalize again
            cap = float(getattr(self.config, 'max_position_cap', 0.20) or 0.20)
            if cap is not None and cap > 0:
                tilted = tilted.clip(upper=cap)
                s2 = float(tilted.sum())
                if s2 == 0 or np.isnan(s2):
                    tilted = w_mvo.copy()
                    s2 = float(max(tilted.sum(), 1e-12))
                tilted = tilted / s2

            return {s: float(tilted.get(s, 0.0)) for s in symbols}

        except Exception as e:
            logger.warning(f"Error in ML overlay calculation: {e}")
            return {symbol: 1.0/len(symbols) for symbol in symbols}

    def _calculate_cvar_weights(self, train_prices: pd.DataFrame, symbols: List[str], weight_cap: Optional[float] = None) -> Dict[str, float]:
        """Calculate weights using CVaR minimization."""
        try:
            returns = train_prices.pct_change().dropna()
            if returns.empty:
                return {symbol: 1.0/len(symbols) for symbol in symbols}
            result = self.optimizer.cvar_optimize(returns, alpha=float(getattr(self.config, 'cvar_alpha', 0.10)), weight_cap=weight_cap)
            return result['weights']
        except Exception as e:
            logger.warning(f"Error in CVaR optimization: {e}")
            return {symbol: 1.0/len(symbols) for symbol in symbols}

    def _calculate_black_litterman_weights(self, train_prices: pd.DataFrame, symbols: List[str], weight_cap: Optional[float] = None) -> Dict[str, float]:
        """Calculate weights using Black–Litterman with neutral views (equal-weight market)."""
        try:
            returns = train_prices.pct_change().dropna()
            if returns.empty:
                return {symbol: 1.0/len(symbols) for symbol in symbols}
            market_w = np.ones(len(symbols)) / len(symbols)
            result = self.optimizer.black_litterman_optimize(returns, market_weights=market_w, weight_cap=weight_cap)
            return result['weights']
        except Exception as e:
            logger.warning(f"Error in Black–Litterman optimization: {e}")
            return {symbol: 1.0/len(symbols) for symbol in symbols}

    def _calculate_turnover(self, old_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover."""
        turnover = 0
        for symbol in old_weights:
            turnover += abs(new_weights.get(symbol, 0) - old_weights.get(symbol, 0))
        return turnover / 2

    def _calculate_total_turnover(self, weights_history: List[Dict]) -> float:
        """Calculate total turnover over backtest period."""
        if len(weights_history) < 2:
            return 0.0

        total_turnover = 0
        for i in range(1, len(weights_history)):
            old_weights = {k: v for k, v in weights_history[i-1].items() if k != 'date'}
            new_weights = {k: v for k, v in weights_history[i].items() if k != 'date'}
            total_turnover += self._calculate_turnover(old_weights, new_weights)
        return total_turnover

    def _calculate_comprehensive_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics including backtest-specific ones."""
        if returns.empty:
            return {}

        # Get basic metrics
        basic_metrics = self.performance_calc.calculate_metrics(returns, benchmark_returns=benchmark_returns)

        # Calculate additional metrics
        metrics = {}
        metrics.update(basic_metrics)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_volatility = downside_returns.std() * np.sqrt(252)
            if downside_volatility > 0:
                metrics['sortino_ratio'] = (metrics['annual_return'] - self.performance_calc.risk_free_rate) / downside_volatility
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = metrics.get('sharpe_ratio', 0)

        # Calmar ratio
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0

        # Win rate and loss rate
        metrics['win_rate'] = (returns > 0).mean()
        metrics['loss_rate'] = (returns < 0).mean()

        return metrics

    def _calculate_comparison_metrics(self, results: Dict[str, BacktestResult]) -> Dict[str, any]:
        """Calculate comparison metrics between strategies."""
        if len(results) < 2:
            return {}

        comparison = {}
        strategies = list(results.keys())

        # Calculate correlation between strategies
        returns_data = {}
        for strategy, result in results.items():
            if hasattr(result, 'returns') and not result.returns.empty:
                returns_data[strategy] = result.returns

        if len(returns_data) >= 2:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            comparison['correlation_matrix'] = correlation_matrix.to_dict()

            # Calculate relative performance
            if 'optimized' in returns_data and 'equal_weight' in returns_data:
                excess_returns = returns_data['optimized'] - returns_data['equal_weight']
                comparison['excess_returns_metrics'] = self.performance_calc.calculate_metrics(excess_returns)

        # Strategy rankings
        rankings = {}
        for metric in ['sharpe_ratio', 'sortino_ratio', 'annual_return', 'max_drawdown']:
            if metric == 'max_drawdown':
                # Lower is better for drawdown
                rankings[metric] = sorted(strategies,
                                        key=lambda s: abs(results[s].metrics.get(metric, 0)))
            else:
                # Higher is better for other metrics
                rankings[metric] = sorted(strategies,
                                        key=lambda s: results[s].metrics.get(metric, 0),
                                        reverse=True)

        comparison['rankings'] = rankings

        return comparison

    def generate_report(self, results: Dict[str, BacktestResult]) -> str:
        """Generate comprehensive backtest report."""
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD BACKTESTING REPORT")
        report.append("=" * 60)
        report.append("")

        # Configuration summary
        report.append("Configuration:")
        report.append(f"  Training Period: {self.config.train_years} year(s)")
        report.append(f"  Testing Period: {self.config.test_quarters} quarter(s)")
        report.append(f"  Transaction Costs: {self.config.transaction_cost_bps} bps")
        report.append(f"  Rebalance Frequency: {self.config.rebalance_frequency}")
        report.append("")

        # Strategy results
        for strategy, result in results.items():
            if strategy == 'comparison':
                continue

            report.append(f"--- {strategy.upper()} STRATEGY ---")
            metrics = result.metrics

            # Performance metrics
            report.append("Performance Metrics:")
            report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
            report.append(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
            report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            report.append("")

            # Trading metrics
            report.append("Trading Metrics:")
            report.append(f"  Total Transaction Costs: {result.transaction_costs:.4%}")
            report.append(f"  Annual Turnover: {result.turnover:.2%}")
            report.append(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
            report.append("")

        # Comparison analysis
        if 'comparison' in results:
            comp = results['comparison']
            if 'rankings' in comp:
                report.append("--- STRATEGY RANKINGS ---")
                for metric, ranking in comp['rankings'].items():
                    report.append(f"{metric}: {' > '.join(ranking)}")
                report.append("")

        return "\n".join(report)

def run_walk_forward_backtest(symbols: List[str], start_date: str, end_date: str,
                            config: Optional[BacktestConfig] = None) -> Dict[str, BacktestResult]:
    """
    Convenience function to run walk-forward backtest.

    Args:
        symbols: List of asset symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional backtest configuration

    Returns:
        Dictionary with backtest results
    """
    if config is None:
        config = BacktestConfig()

    backtester = WalkForwardBacktester(config)
    return backtester.run_backtest(symbols, start_date, end_date)
