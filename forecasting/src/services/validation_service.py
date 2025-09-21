"""
Core Validation and Evaluation Services for Financial Forecasting

This module provides comprehensive validation and evaluation services for financial forecasting models,
including forecast accuracy evaluation, benchmark comparison, statistical significance testing,
and financial performance metrics.

Services:
- ForecastEvaluator: Basic forecast accuracy evaluation
- BenchmarkComparator: Compare against benchmark strategies
- StatisticalSignificanceTester: Statistical significance testing
- FinancialPerformanceEvaluator: Financial performance metrics
- RegimeAwareEvaluator: Regime-aware forecast evaluation
- RobustnessTester: Robustness testing under different conditions
- UncertaintyQuantifier: Forecast uncertainty quantification
- MultiHorizonEvaluator: Multi-horizon forecast evaluation
- RelativePerformanceTester: Relative performance testing
- CrossValidationEvaluator: Cross-validation based evaluation
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, norm, pearsonr, spearmanr
import warnings
from enum import Enum
import math


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    MAE = "mae"                    # Mean Absolute Error
    MSE = "mse"                    # Mean Squared Error
    RMSE = "rmse"                  # Root Mean Squared Error
    MAPE = "mape"                  # Mean Absolute Percentage Error
    SMAPE = "smape"                # Symmetric Mean Absolute Percentage Error
    R2 = "r2"                      # R-squared
    Directional_Accuracy = "directional_accuracy"  # Directional accuracy
    Sharpe_Ratio = "sharpe_ratio"  # Sharpe ratio
    Information_Ratio = "information_ratio"  # Information ratio
    Max_Drawdown = "max_drawdown"  # Maximum drawdown
    Hit_Rate = "hit_rate"          # Hit rate (binary accuracy)


@dataclass
class ForecastEvaluationResult:
    """Result of forecast evaluation."""

    metric: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    is_significant: bool = False
    sample_size: int = 0
    evaluation_period: Optional[Tuple[datetime, datetime]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric': self.metric,
            'value': self.value,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'sample_size': self.sample_size,
            'evaluation_period': self.evaluation_period
        }


class ForecastEvaluator:
    """Basic forecast accuracy evaluation."""

    def __init__(self):
        self.results: List[ForecastEvaluationResult] = []

    def evaluate_forecasts(self, actual: pd.Series, forecast: pd.Series,
                          metrics: List[EvaluationMetric] = None) -> Dict[str, ForecastEvaluationResult]:
        """
        Evaluate forecast accuracy using multiple metrics.

        Args:
            actual: Actual values
            forecast: Forecasted values
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation results
        """
        if metrics is None:
            metrics = [EvaluationMetric.MAE, EvaluationMetric.RMSE, EvaluationMetric.MAPE]

        # Align series
        aligned_actual, aligned_forecast = actual.align(forecast, join='inner')

        if len(aligned_actual) == 0:
            raise ValueError("No overlapping data between actual and forecast")

        results = {}

        for metric in metrics:
            result = self._compute_metric(aligned_actual, aligned_forecast, metric)
            results[metric.value] = result
            self.results.append(result)

        return results

    def _compute_metric(self, actual: pd.Series, forecast: pd.Series,
                       metric: EvaluationMetric) -> ForecastEvaluationResult:
        """Compute a specific evaluation metric."""

        if metric == EvaluationMetric.MAE:
            value = np.mean(np.abs(actual - forecast))

        elif metric == EvaluationMetric.MSE:
            value = np.mean((actual - forecast) ** 2)

        elif metric == EvaluationMetric.RMSE:
            value = np.sqrt(np.mean((actual - forecast) ** 2))

        elif metric == EvaluationMetric.MAPE:
            # Avoid division by zero
            mask = actual != 0
            if not mask.any():
                value = np.inf
            else:
                value = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

        elif metric == EvaluationMetric.SMAPE:
            denominator = (np.abs(actual) + np.abs(forecast)) / 2
            mask = denominator != 0
            if not mask.any():
                value = np.inf
            else:
                value = np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask]) * 100

        elif metric == EvaluationMetric.R2:
            ss_res = np.sum((actual - forecast) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            value = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        elif metric == EvaluationMetric.Directional_Accuracy:
            # Correct direction prediction
            actual_changes = actual.diff().dropna()
            forecast_changes = forecast.diff().dropna()
            aligned_changes = actual_changes.align(forecast_changes, join='inner')

            if len(aligned_changes[0]) > 0:
                correct_direction = np.sign(aligned_changes[0]) == np.sign(aligned_changes[1])
                value = np.mean(correct_direction) * 100
            else:
                value = 0.0

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Calculate confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_values = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(actual), len(actual), replace=True)
            boot_actual = actual.iloc[indices]
            boot_forecast = forecast.iloc[indices]

            if metric == EvaluationMetric.MAE:
                boot_value = np.mean(np.abs(boot_actual - boot_forecast))
            elif metric == EvaluationMetric.RMSE:
                boot_value = np.sqrt(np.mean((boot_actual - boot_forecast) ** 2))
            elif metric == EvaluationMetric.MAPE:
                mask = boot_actual != 0
                if mask.any():
                    boot_value = np.mean(np.abs((boot_actual[mask] - boot_forecast[mask]) / boot_actual[mask])) * 100
                else:
                    boot_value = np.inf
            elif metric == EvaluationMetric.R2:
                ss_res = np.sum((boot_actual - boot_forecast) ** 2)
                ss_tot = np.sum((boot_actual - boot_actual.mean()) ** 2)
                boot_value = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            else:
                boot_value = value

            bootstrap_values.append(boot_value)

        # Calculate confidence interval
        alpha = 0.05
        lower_ci = np.percentile(bootstrap_values, alpha/2 * 100)
        upper_ci = np.percentile(bootstrap_values, (1 - alpha/2) * 100)

        return ForecastEvaluationResult(
            metric=metric.value,
            value=value,
            confidence_interval=(lower_ci, upper_ci),
            sample_size=len(actual),
            evaluation_period=(actual.index.min(), actual.index.max())
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations."""
        if not self.results:
            return {}

        metrics_summary = {}
        for result in self.results:
            if result.metric not in metrics_summary:
                metrics_summary[result.metric] = []
            metrics_summary[result.metric].append(result.value)

        summary = {}
        for metric, values in metrics_summary.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

        return summary


class BenchmarkComparator:
    """Compare against benchmark strategies."""

    def __init__(self):
        self.benchmark_results: Dict[str, Dict[str, float]] = {}

    def compare_against_benchmarks(self, actual: pd.Series, forecast: pd.Series,
                                 benchmark_forecasts: Dict[str, pd.Series] = None,
                                 benchmark_strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare forecast against benchmark strategies.

        Args:
            actual: Actual values
            forecast: Your forecast
            benchmark_forecasts: Dictionary of benchmark forecasts
            benchmark_strategies: List of benchmark strategies to generate

        Returns:
            Comparison results
        """
        if benchmark_strategies is None:
            benchmark_strategies = ['naive', 'moving_average', 'exponential_smoothing']

        # Generate benchmark forecasts if not provided
        if benchmark_forecasts is None:
            benchmark_forecasts = {}

            for strategy in benchmark_strategies:
                if strategy == 'naive':
                    # Naive forecast (last value)
                    benchmark_forecasts[strategy] = actual.shift(1)

                elif strategy == 'moving_average':
                    # Moving average
                    benchmark_forecasts[strategy] = actual.rolling(5).mean()

                elif strategy == 'exponential_smoothing':
                    # Exponential smoothing
                    benchmark_forecasts[strategy] = actual.ewm(alpha=0.3).mean()

                elif strategy == 'random_walk':
                    # Random walk with drift
                    drift = (actual.iloc[-1] - actual.iloc[0]) / len(actual)
                    benchmark_forecasts[strategy] = actual.shift(1) + drift

        # Evaluate all forecasts
        evaluator = ForecastEvaluator()

        # Evaluate your forecast
        your_results = evaluator.evaluate_forecasts(actual, forecast)

        # Evaluate benchmarks
        benchmark_results = {}
        for name, bench_forecast in benchmark_forecasts.items():
            try:
                bench_eval = evaluator.evaluate_forecasts(actual, bench_forecast)
                benchmark_results[name] = {metric.value: result.value for metric, result in bench_eval.items()}
            except:
                # Skip if evaluation fails
                continue

        # Calculate relative performance
        relative_performance = {}
        for metric in your_results:
            your_value = your_results[metric].value

            for bench_name, bench_metrics in benchmark_results.items():
                if metric in bench_metrics:
                    bench_value = bench_metrics[metric]

                    # For error metrics, lower is better
                    if metric in ['mae', 'mse', 'rmse', 'mape', 'smape']:
                        improvement = (bench_value - your_value) / bench_value * 100
                    else:
                        # For accuracy metrics, higher is better
                        improvement = (your_value - bench_value) / bench_value * 100

                    key = f"{metric}_vs_{bench_name}"
                    relative_performance[key] = improvement

        # Statistical significance testing
        significance_tests = self._test_statistical_significance(actual, forecast, benchmark_forecasts)

        results = {
            'your_forecast': {result.metric: result.to_dict() for result in your_results.values()},
            'benchmarks': benchmark_results,
            'relative_performance': relative_performance,
            'significance_tests': significance_tests,
            'summary': {
                'metrics_computed': list(your_results.keys()),
                'benchmarks_tested': list(benchmark_results.keys()),
                'evaluation_period': (actual.index.min(), actual.index.max()),
                'sample_size': len(actual)
            }
        }

        self.benchmark_results = results
        return results

    def _test_statistical_significance(self, actual: pd.Series, forecast: pd.Series,
                                     benchmark_forecasts: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Test statistical significance of forecast improvements."""
        significance_results = {}

        # Calculate forecast errors
        your_errors = actual - forecast

        for bench_name, bench_forecast in benchmark_forecasts.items():
            aligned_actual, aligned_bench = actual.align(bench_forecast, join='inner')
            if len(aligned_actual) == 0:
                continue

            bench_errors = aligned_actual - aligned_bench

            # Align errors
            aligned_your, aligned_bench = your_errors.align(bench_errors, join='inner')

            if len(aligned_your) > 10:  # Need sufficient samples
                # Paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(aligned_your**2, aligned_bench**2)

                    # Diebold-Mariano test for forecast accuracy
                    dm_stat, dm_p_value = self._diebold_mariano_test(aligned_your, aligned_bench)

                    significance_results[f'your_vs_{bench_name}'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'dm_statistic': dm_stat,
                        'dm_p_value': dm_p_value,
                        'is_significant': p_value < 0.05,
                        'is_dm_significant': dm_p_value < 0.05
                    }
                except:
                    pass

        return significance_results

    def _diebold_mariano_test(self, errors1: pd.Series, errors2: pd.Series) -> Tuple[float, float]:
        """Diebold-Mariano test for forecast accuracy comparison."""
        # Loss differential
        diff = errors1**2 - errors2**2

        # Mean loss differential
        mean_diff = diff.mean()

        # Standard error (assuming no autocorrelation)
        se = diff.std() / np.sqrt(len(diff))

        # Test statistic
        dm_stat = mean_diff / se if se > 0 else 0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value


class StatisticalSignificanceTester:
    """Statistical significance testing for forecasts."""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, float]] = {}

    def test_forecast_significance(self, actual: pd.Series, forecast: pd.Series,
                                 null_forecast: pd.Series = None,
                                 test_type: str = 'dm') -> Dict[str, Any]:
        """
        Test statistical significance of forecast accuracy.

        Args:
            actual: Actual values
            forecast: Your forecast
            null_forecast: Null hypothesis forecast
            test_type: Type of test ('dm', 'white', 'clark_west')

        Returns:
            Test results
        """
        if null_forecast is None:
            # Use naive forecast as null
            null_forecast = actual.shift(1)

        # Align series
        aligned_actual, aligned_forecast = actual.align(forecast, join='inner')
        aligned_actual, aligned_null = aligned_actual.align(null_forecast, join='inner')

        if len(aligned_actual) < 10:
            return {'error': 'Insufficient data for significance testing'}

        # Calculate errors
        forecast_errors = aligned_actual - aligned_forecast
        null_errors = aligned_actual - aligned_null

        results = {}

        if test_type in ['dm', 'all']:
            # Diebold-Mariano test
            dm_stat, dm_p_value = self._diebold_mariano_test(forecast_errors, null_errors)
            results['diebold_mariano'] = {
                'statistic': dm_stat,
                'p_value': dm_p_value,
                'is_significant': dm_p_value < 0.05,
                'conclusion': 'Reject null' if dm_p_value < 0.05 else 'Fail to reject null'
            }

        if test_type in ['white', 'all']:
            # White's reality check
            white_results = self._white_reality_check(forecast_errors, null_errors)
            results['white_reality_check'] = white_results

        if test_type in ['clark_west', 'all']:
            # Clark-West test
            cw_results = self._clark_west_test(forecast_errors, null_errors)
            results['clark_west'] = cw_results

        # Additional tests
        results['additional_tests'] = self._additional_tests(forecast_errors, null_errors)

        self.test_results = results
        return results

    def _diebold_mariano_test(self, errors1: pd.Series, errors2: pd.Series) -> Tuple[float, float]:
        """Diebold-Mariano test implementation."""
        # Loss differential series
        diff = errors1**2 - errors2**2

        # Mean and variance
        mean_diff = diff.mean()
        var_diff = diff.var()

        # Standard error
        se = np.sqrt(var_diff / len(diff)) if var_diff > 0 else 1e-10

        # Test statistic
        dm_stat = mean_diff / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value

    def _white_reality_check(self, errors1: pd.Series, errors2: pd.Series) -> Dict[str, float]:
        """White's reality check for multiple comparisons."""
        # Simplified implementation
        diff = errors1**2 - errors2**2
        mean_diff = diff.mean()
        se = diff.std() / np.sqrt(len(diff))

        # Bootstrap p-value
        n_bootstrap = 1000
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            boot_diff = np.random.choice(diff, len(diff), replace=True)
            boot_stat = boot_diff.mean() / (boot_diff.std() / np.sqrt(len(boot_diff)))
            bootstrap_stats.append(boot_stat)

        # Calculate p-value
        actual_stat = mean_diff / se if se > 0 else 0
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(actual_stat))

        return {
            'statistic': actual_stat,
            'p_value': p_value,
            'bootstrap_samples': n_bootstrap,
            'is_significant': p_value < 0.05
        }

    def _clark_west_test(self, errors1: pd.Series, errors2: pd.Series) -> Dict[str, float]:
        """Clark-West test for nested models."""
        # Loss differential with adjustment
        diff = errors1**2 - errors2**2 + 2 * errors2 * (errors2 - errors1)

        mean_diff = diff.mean()
        se = diff.std() / np.sqrt(len(diff))

        cw_stat = mean_diff / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(cw_stat)))

        return {
            'statistic': cw_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }

    def _additional_tests(self, errors1: pd.Series, errors2: pd.Series) -> Dict[str, Any]:
        """Additional statistical tests."""
        results = {}

        # Correlation test
        if len(errors1) > 3:
            corr, p_value = pearsonr(errors1, errors2)
            results['correlation'] = {
                'correlation': corr,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }

        # Variance ratio test
        var1, var2 = errors1.var(), errors2.var()
        if var2 > 0:
            f_stat = var1 / var2
            df1, df2 = len(errors1) - 1, len(errors2) - 1
            p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

            results['variance_ratio'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }

        return results


class FinancialPerformanceEvaluator:
    """Financial performance metrics for forecasts."""

    def __init__(self):
        self.performance_results: Dict[str, float] = {}

    def evaluate_financial_performance(self, actual: pd.Series, forecast: pd.Series,
                                     trading_costs: float = 0.001,
                                     risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Evaluate financial performance of trading strategy based on forecasts.

        Args:
            actual: Actual price/return series
            forecast: Forecast returns
            trading_costs: Trading costs as fraction
            risk_free_rate: Annual risk-free rate

        Returns:
            Financial performance metrics
        """
        # Generate trading signals (simplified)
        signals = np.sign(forecast)

        # Calculate returns
        if len(actual) != len(signals):
            # Align series
            aligned_actual, aligned_signals = actual.align(signals, join='inner')
        else:
            aligned_actual, aligned_signals = actual, signals

        # Strategy returns
        strategy_returns = aligned_signals.shift(1) * aligned_actual.pct_change().dropna()

        # Apply trading costs
        trades = np.diff(np.sign(aligned_signals))
        costs = np.abs(trades) * trading_costs
        strategy_returns = strategy_returns - costs[:-1] if len(costs) > 0 else strategy_returns

        # Calculate performance metrics
        results = {}

        # Return metrics
        results['total_return'] = (1 + strategy_returns).prod() - 1
        results['annualized_return'] = results['total_return'] * (252 / len(strategy_returns))
        results['volatility'] = strategy_returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        if results['volatility'] > 0:
            results['sharpe_ratio'] = (results['annualized_return'] - risk_free_rate) / results['volatility']
        else:
            results['sharpe_ratio'] = 0

        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        results['max_drawdown'] = drawdown.min()

        # Calmar ratio
        if results['max_drawdown'] != 0:
            results['calmar_ratio'] = results['annualized_return'] / abs(results['max_drawdown'])
        else:
            results['calmar_ratio'] = np.inf

        # Win rate
        results['win_rate'] = np.mean(strategy_returns > 0) * 100

        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        results['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else np.inf

        # Information ratio (vs. buy and hold)
        buy_hold_returns = aligned_actual.pct_change().dropna()
        aligned_strategy, aligned_bh = strategy_returns.align(buy_hold_returns, join='inner')

        if len(aligned_strategy) > 0 and aligned_bh.std() > 0:
            excess_returns = aligned_strategy - aligned_bh
            results['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            results['information_ratio'] = 0

        # Sortino ratio (downside risk)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            results['sortino_ratio'] = (results['annualized_return'] - risk_free_rate) / downside_std
        else:
            results['sortino_ratio'] = np.inf

        # Additional metrics
        results['trading_metrics'] = {
            'total_trades': int(np.sum(np.abs(np.diff(np.sign(aligned_signals))))),
            'winning_trades': int(np.sum(strategy_returns > 0)),
            'losing_trades': int(np.sum(strategy_returns < 0)),
            'avg_win': strategy_returns[strategy_returns > 0].mean() if np.sum(strategy_returns > 0) > 0 else 0,
            'avg_loss': strategy_returns[strategy_returns < 0].mean() if np.sum(strategy_returns < 0) > 0 else 0,
            'avg_trade': strategy_returns.mean()
        }

        self.performance_results = results
        return results


class RegimeAwareEvaluator:
    """Regime-aware forecast evaluation."""

    def __init__(self):
        self.regime_results: Dict[str, Dict[str, float]] = {}

    def evaluate_by_regime(self, actual: pd.Series, forecast: pd.Series,
                          regimes: pd.Series) -> Dict[str, Any]:
        """
        Evaluate forecast performance by market regime.

        Args:
            actual: Actual values
            forecast: Forecast values
            regimes: Regime labels aligned with actual/forecast

        Returns:
            Regime-specific evaluation results
        """
        # Align all series
        aligned_actual, aligned_forecast = actual.align(forecast, join='inner')
        aligned_actual, aligned_regimes = aligned_actual.align(regimes, join='inner')

        if len(aligned_actual) == 0:
            return {'error': 'No overlapping data between series'}

        evaluator = ForecastEvaluator()

        regime_results = {}
        unique_regimes = aligned_regimes.unique()

        for regime in unique_regimes:
            regime_mask = aligned_regimes == regime
            regime_actual = aligned_actual[regime_mask]
            regime_forecast = aligned_forecast[regime_mask]

            if len(regime_actual) > 5:  # Need sufficient data
                regime_eval = evaluator.evaluate_forecasts(regime_actual, regime_forecast)
                regime_results[str(regime)] = {
                    metric.value: result.to_dict() for metric, result in regime_eval.items()
                }
                regime_results[str(regime)]['sample_size'] = len(regime_actual)

        # Compare across regimes
        regime_comparison = self._compare_regimes(regime_results)

        # Overall performance with regime context
        overall_eval = evaluator.evaluate_forecasts(aligned_actual, aligned_forecast)

        results = {
            'regime_results': regime_results,
            'regime_comparison': regime_comparison,
            'overall_performance': {metric.value: result.to_dict() for metric, result in overall_eval.items()},
            'summary': {
                'regimes_analyzed': list(unique_regimes),
                'total_observations': len(aligned_actual),
                'evaluation_period': (aligned_actual.index.min(), aligned_actual.index.max())
            }
        }

        self.regime_results = results
        return results

    def _compare_regimes(self, regime_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance across regimes."""
        comparison = {}

        # Extract RMSE for comparison (if available)
        rmse_by_regime = {}
        for regime, results in regime_results.items():
            if 'rmse' in results:
                rmse_by_regime[regime] = results['rmse']['value']

        if rmse_by_regime:
            comparison['rmse_comparison'] = rmse_by_regime
            comparison['best_regime'] = min(rmse_by_regime, key=rmse_by_regime.get)
            comparison['worst_regime'] = max(rmse_by_regime, key=rmse_by_regime.get)

        # Similar analysis for other metrics
        for metric in ['mae', 'mape', 'r2']:
            metric_by_regime = {}
            for regime, results in regime_results.items():
                if metric in results:
                    metric_by_regime[regime] = results[metric]['value']

            if metric_by_regime:
                comparison[f'{metric}_comparison'] = metric_by_regime

        return comparison


# Additional service classes (simplified implementations for brevity)
class RobustnessTester:
    """Robustness testing under different conditions."""

    def __init__(self):
        self.robustness_results: Dict[str, Any] = {}

    def test_robustness(self, actual: pd.Series, forecast: pd.Series,
                      test_scenarios: List[str] = None) -> Dict[str, Any]:
        """Test forecast robustness under different scenarios."""
        if test_scenarios is None:
            test_scenarios = ['noise_injection', 'missing_data', 'outliers', 'regime_shifts']

        results = {}
        evaluator = ForecastEvaluator()

        baseline_results = evaluator.evaluate_forecasts(actual, forecast)

        for scenario in test_scenarios:
            scenario_results = self._apply_scenario(actual, forecast, scenario)
            results[scenario] = scenario_results

        # Calculate robustness scores
        robustness_scores = self._calculate_robustness_scores(baseline_results, results)

        return {
            'baseline': {metric.value: result.to_dict() for metric, result in baseline_results.items()},
            'scenario_results': results,
            'robustness_scores': robustness_scores,
            'overall_robustness': np.mean(list(robustness_scores.values())) if robustness_scores else 0
        }

    def _apply_scenario(self, actual: pd.Series, forecast: pd.Series, scenario: str) -> Dict[str, Any]:
        """Apply a specific scenario to test robustness."""
        evaluator = ForecastEvaluator()

        if scenario == 'noise_injection':
            # Add Gaussian noise
            noisy_actual = actual + np.random.normal(0, actual.std() * 0.1, len(actual))
            results = evaluator.evaluate_forecasts(noisy_actual, forecast)

        elif scenario == 'missing_data':
            # Randomly remove 10% of data
            mask = np.random.random(len(actual)) > 0.1
            results = evaluator.evaluate_forecasts(actual[mask], forecast[mask])

        elif scenario == 'outliers':
            # Add outliers
            outlier_actual = actual.copy()
            outlier_indices = np.random.choice(len(actual), int(len(actual) * 0.05), replace=False)
            outlier_actual.iloc[outlier_indices] *= 3  # 3x outliers
            results = evaluator.evaluate_forecasts(outlier_actual, forecast)

        elif scenario == 'regime_shifts':
            # Simulate regime shift (change volatility)
            shift_point = len(actual) // 2
            shifted_actual = actual.copy()
            shifted_actual.iloc[shift_point:] *= 2  # Double volatility in second half
            results = evaluator.evaluate_forecasts(shifted_actual, forecast)

        else:
            results = {}

        return {metric.value: result.to_dict() for metric, result in results.items()}

    def _calculate_robustness_scores(self, baseline: Dict, scenario_results: Dict) -> Dict[str, float]:
        """Calculate robustness scores based on performance degradation."""
        scores = {}

        for scenario, results in scenario_results.items():
            degradation_scores = []

            for metric in baseline:
                if metric in results:
                    baseline_val = baseline[metric].value
                    scenario_val = results[metric].value

                    # Calculate relative degradation
                    if baseline_val != 0:
                        degradation = abs(scenario_val - baseline_val) / abs(baseline_val)
                        degradation_scores.append(degradation)

            if degradation_scores:
                # Lower degradation = higher robustness
                scores[scenario] = 1 / (1 + np.mean(degradation_scores))
            else:
                scores[scenario] = 0.0

        return scores


class UncertaintyQuantifier:
    """Forecast uncertainty quantification."""

    def __init__(self):
        self.uncertainty_results: Dict[str, Any] = {}

    def quantify_uncertainty(self, actual: pd.Series, forecast: pd.Series,
                           forecast_intervals: pd.DataFrame = None) -> Dict[str, Any]:
        """Quantify forecast uncertainty."""
        results = {}

        # Basic uncertainty metrics
        errors = actual - forecast
        results['error_statistics'] = {
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'mae': np.abs(errors).mean(),
            'rmse': np.sqrt((errors ** 2).mean())
        }

        # Coverage analysis if intervals provided
        if forecast_intervals is not None:
            coverage_results = self._analyze_coverage(actual, forecast_intervals)
            results['coverage_analysis'] = coverage_results

        # Prediction interval quality
        results['interval_quality'] = self._assess_interval_quality(errors, forecast_intervals)

        # Uncertainty decomposition
        results['uncertainty_decomposition'] = self._decompose_uncertainty(errors)

        self.uncertainty_results = results
        return results

    def _analyze_coverage(self, actual: pd.Series, intervals: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction interval coverage."""
        coverage_results = {}

        for col in intervals.columns:
            if 'lower' in col.lower() and 'upper' in col.lower():
                # Extract confidence level from column name (simplified)
                if '95' in col:
                    cl = 0.95
                elif '99' in col:
                    cl = 0.99
                else:
                    cl = 0.90

                lower_col = col
                upper_col = col.replace('lower', 'upper')

                if upper_col in intervals.columns:
                    # Check coverage
                    covered = (actual >= intervals[lower_col]) & (actual <= intervals[upper_col])
                    coverage_rate = covered.mean()

                    coverage_results[f'coverage_{int(cl*100)}'] = {
                        'actual_coverage': coverage_rate,
                        'nominal_coverage': cl,
                        'coverage_deficit': cl - coverage_rate,
                        'is_adequate': coverage_rate >= cl * 0.9  # 90% of nominal
                    }

        return coverage_results

    def _assess_interval_quality(self, errors: pd.Series, intervals: pd.DataFrame = None) -> Dict[str, float]:
        """Assess prediction interval quality."""
        quality_metrics = {}

        # Interval width (if available)
        if intervals is not None:
            for col in intervals.columns:
                if 'lower' in col.lower() and 'upper' in col.lower():
                    upper_col = col.replace('lower', 'upper')
                    if upper_col in intervals.columns:
                        width = (intervals[upper_col] - intervals[col]).mean()
                        quality_metrics[f'avg_interval_width'] = width

        # Error distribution characteristics
        quality_metrics['error_skewness'] = errors.skew()
        quality_metrics['error_kurtosis'] = errors.kurtosis()

        # Normality test
        if len(errors) > 8:
            stat, p_value = stats.jarque_bera(errors)
            quality_metrics['normality_p_value'] = p_value
            quality_metrics['is_normal'] = p_value > 0.05

        return quality_metrics

    def _decompose_uncertainty(self, errors: pd.Series) -> Dict[str, float]:
        """Decompose forecast uncertainty into components."""
        decomposition = {}

        # Total uncertainty (variance)
        decomposition['total_uncertainty'] = errors.var()

        # Bias component
        decomposition['bias_component'] = errors.mean() ** 2

        # Variance component
        decomposition['variance_component'] = errors.var()

        # Decomposition ratio
        total = decomposition['total_uncertainty']
        if total > 0:
            decomposition['bias_ratio'] = decomposition['bias_component'] / total
            decomposition['variance_ratio'] = decomposition['variance_component'] / total

        return decomposition


# Additional service classes can be implemented following similar patterns
# MultiHorizonEvaluator, RelativePerformanceTester, CrossValidationEvaluator
# These would follow the same structure with specialized evaluation methods