"""
Benchmark comparison and analysis for portfolio performance.

Implements comprehensive benchmark analysis including attribution, tracking error,
and relative performance metrics.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

from portfolio.logging_config import get_logger, ValidationError
from portfolio.config import get_config
from portfolio.models.performance import PortfolioPerformance

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks for comparison."""
    MARKET_INDEX = "market_index"
    PEER_GROUP = "peer_group"
    CUSTOM_BENCHMARK = "custom_benchmark"
    RISK_FREE = "risk_free"


class BenchmarkAnalyzer:
    """
    Comprehensive benchmark analysis and comparison tools.

    Provides detailed comparison between portfolio and benchmark performance
    with attribution analysis and tracking metrics.
    """

    def __init__(self):
        """Initialize the benchmark analyzer."""
        self.config = get_config()
        self.trading_days_per_year = self.config.performance.trading_days_per_year if hasattr(self.config, 'performance') else 252

        # Common benchmark symbols
        self.common_benchmarks = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'MSCI World': 'ACWI',
            'Bloomberg Aggregate': 'AGG',
            '10-Year Treasury': '^TNX'
        }

        logger.info(f"Initialized BenchmarkAnalyzer with {len(self.common_benchmarks)} common benchmarks")

    def compare_with_benchmark(self,
                               portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               benchmark_name: str = "Benchmark") -> Dict[str, Any]:
        """
        Compare portfolio performance against benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            benchmark_name: Name of the benchmark

        Returns:
            Dictionary with comparison results
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            logger.warning("Empty returns series provided")
            return {}

        try:
            # Align series to common dates
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                logger.warning("No overlapping dates between portfolio and benchmark")
                return {}

            port_returns, bench_returns = aligned_returns

            results = {
                'benchmark_name': benchmark_name,
                'analysis_period': {
                    'start_date': port_returns.index[0].strftime('%Y-%m-%d'),
                    'end_date': port_returns.index[-1].strftime('%Y-%m-%d'),
                    'trading_days': len(port_returns)
                },
                'return_analysis': self._analyze_returns(port_returns, bench_returns),
                'risk_analysis': self._analyze_risk(port_returns, bench_returns),
                'attribution_analysis': self._analyze_attribution(port_returns, bench_returns),
                'regime_analysis': self._analyze_regimes(port_returns, bench_returns)
            }

            return results

        except Exception as e:
            logger.error(f"Error comparing with benchmark: {e}")
            return {}

    def calculate_tracking_metrics(self,
                                 portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tracking error and related metrics.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Dictionary with tracking metrics
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        try:
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                return {}

            port_returns, bench_returns = aligned_returns

            # Calculate excess returns
            excess_returns = port_returns - bench_returns

            # Tracking error (annualized)
            tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)

            # Annualized excess return
            annual_excess_return = excess_returns.mean() * self.trading_days_per_year

            # Information ratio
            information_ratio = annual_excess_return / tracking_error if tracking_error > 0 else 0.0

            # Tracking error volatility
            tracking_vol = excess_returns.std()

            # Maximum tracking error
            max_tracking_error = excess_returns.abs().max()

            # Up/down tracking error
            up_tracking_error = excess_returns[excess_returns > 0].std() if (excess_returns > 0).any() else 0.0
            down_tracking_error = excess_returns[excess_returns < 0].std() if (excess_returns < 0).any() else 0.0

            return {
                'tracking_error': float(tracking_error),
                'annualized_excess_return': float(annual_excess_return),
                'information_ratio': float(information_ratio),
                'tracking_volatility': float(tracking_vol),
                'max_tracking_error': float(max_tracking_error),
                'up_tracking_error': float(up_tracking_error),
                'down_tracking_error': float(down_tracking_error),
                'tracking_error_days': len(excess_returns)
            }

        except Exception as e:
            logger.error(f"Error calculating tracking metrics: {e}")
            return {}

    def calculate_upside_downside_capture(self,
                                        portfolio_returns: pd.Series,
                                        benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate upside and downside capture ratios.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Dictionary with capture ratios
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        try:
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                return {}

            port_returns, bench_returns = aligned_returns

            # Upside capture (positive benchmark periods)
            pos_bench_mask = bench_returns > 0
            if pos_bench_mask.sum() > 0:
                port_pos_return = port_returns[pos_bench_mask].mean()
                bench_pos_return = bench_returns[pos_bench_mask].mean()
                upside_capture = (port_pos_return / bench_pos_return) * 100 if bench_pos_return != 0 else 0.0
            else:
                upside_capture = 0.0

            # Downside capture (negative benchmark periods)
            neg_bench_mask = bench_returns < 0
            if neg_bench_mask.sum() > 0:
                port_neg_return = port_returns[neg_bench_mask].mean()
                bench_neg_return = bench_returns[neg_bench_mask].mean()
                downside_capture = (port_neg_return / bench_neg_return) * 100 if bench_neg_return != 0 else 0.0
            else:
                downside_capture = 0.0

            # Overall capture ratio
            if bench_returns.mean() != 0:
                overall_capture = (port_returns.mean() / bench_returns.mean()) * 100
            else:
                overall_capture = 0.0

            return {
                'upside_capture': float(upside_capture),
                'downside_capture': float(downside_capture),
                'overall_capture': float(overall_capture),
                'positive_periods': int(pos_bench_mask.sum()),
                'negative_periods': int(neg_bench_mask.sum())
            }

        except Exception as e:
            logger.error(f"Error calculating capture ratios: {e}")
            return {}

    def calculate_beta_analysis(self,
                              portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series,
                              window: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate beta and related metrics with optional rolling analysis.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            window: Rolling window size (optional)

        Returns:
            Dictionary with beta analysis
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        try:
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                return {}

            port_returns, bench_returns = aligned_returns

            results = {}

            # Static beta calculation
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            benchmark_variance = np.var(bench_returns)

            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
                alpha = port_returns.mean() - beta * bench_returns.mean()
            else:
                beta = 0.0
                alpha = 0.0

            results.update({
                'beta': float(beta),
                'alpha': float(alpha),
                'correlation': float(np.corrcoef(port_returns, bench_returns)[0, 1]),
                'r_squared': float(np.corrcoef(port_returns, bench_returns)[0, 1] ** 2)
            })

            # Rolling beta analysis (if window specified)
            if window is not None and len(port_returns) >= window:
                rolling_betas = []
                rolling_alphas = []
                rolling_correlations = []

                for i in range(window - 1, len(port_returns)):
                    port_window = port_returns.iloc[i - window + 1:i + 1]
                    bench_window = bench_returns.iloc[i - window + 1:i + 1]

                    if len(port_window) == window:
                        cov = np.cov(port_window, bench_window)[0, 1]
                        var = np.var(bench_window)
                        rolling_beta = cov / var if var > 0 else 0.0
                        rolling_alpha = port_window.mean() - rolling_beta * bench_window.mean()
                        rolling_corr = np.corrcoef(port_window, bench_window)[0, 1]

                        rolling_betas.append(rolling_beta)
                        rolling_alphas.append(rolling_alpha)
                        rolling_correlations.append(rolling_corr)

                if rolling_betas:
                    results.update({
                        f'rolling_beta_{window}d_mean': float(np.mean(rolling_betas)),
                        f'rolling_beta_{window}d_std': float(np.std(rolling_betas)),
                        f'rolling_beta_{window}d_min': float(np.min(rolling_betas)),
                        f'rolling_beta_{window}d_max': float(np.max(rolling_betas)),
                        f'rolling_alpha_{window}d_mean': float(np.mean(rolling_alphas)),
                        f'rolling_correlation_{window}d_mean': float(np.mean(rolling_correlations))
                    })

            return results

        except Exception as e:
            logger.error(f"Error calculating beta analysis: {e}")
            return {}

    def calculate_relative_value_at_risk(self,
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series,
                                       confidence_level: float = 0.95) -> Dict[str, float]:
        """
        calculate relative Value at Risk (VaR).

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            confidence_level: Confidence level for VaR

        Returns:
            Dictionary with relative VaR metrics
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        try:
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                return {}

            port_returns, bench_returns = aligned_returns

            # Calculate excess returns
            excess_returns = port_returns - bench_returns

            # Calculate VaR for excess returns
            var_percentile = 100 * (1 - confidence_level)
            relative_var = np.percentile(excess_returns, var_percentile)

            # Calculate relative CVaR
            tail_returns = excess_returns[excess_returns <= relative_var]
            relative_cvar = tail_returns.mean() if len(tail_returns) > 0 else relative_var

            return {
                'relative_var': float(relative_var),
                'relative_cvar': float(relative_cvar),
                'confidence_level': confidence_level,
                'tail_observations': len(tail_returns)
            }

        except Exception as e:
            logger.error(f"Error calculating relative VaR: {e}")
            return {}

    def generate_performance_attribution(self,
                                       portfolio_returns: pd.Series,
                                       benchmark_returns: pd.Series,
                                       weights_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate performance attribution analysis.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            weights_history: Historical portfolio weights (optional)

        Returns:
            Dictionary with attribution results
        """
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}

        try:
            aligned_returns = self._align_series(portfolio_returns, benchmark_returns)
            if len(aligned_returns[0]) == 0:
                return {}

            port_returns, bench_returns = aligned_returns

            # Basic attribution
            excess_returns = port_returns - bench_returns
            total_excess_return = excess_returns.sum()

            # Attribution by period
            positive_contribution = excess_returns[excess_returns > 0].sum()
            negative_contribution = excess_returns[excess_returns < 0].sum()

            results = {
                'total_excess_return': float(total_excess_return),
                'positive_contribution': float(positive_contribution),
                'negative_contribution': float(negative_contribution),
                'attribution_ratio': float(positive_contribution / abs(negative_contribution)) if negative_contribution != 0 else float('inf'),
                'win_rate': float((excess_returns > 0).mean()),
                'avg_positive_excess': float(excess_returns[excess_returns > 0].mean()) if (excess_returns > 0).any() else 0.0,
                'avg_negative_excess': float(excess_returns[excess_returns < 0].mean()) if (excess_returns < 0).any() else 0.0
            }

            # Monthly attribution if enough data
            if len(port_returns) >= 21:  # Approx 1 month of trading days
                monthly_port = port_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_bench = bench_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_excess = monthly_port - monthly_bench

                results['monthly_attribution'] = {
                    'avg_monthly_excess': float(monthly_excess.mean()),
                    'positive_months': int((monthly_excess > 0).sum()),
                    'total_months': len(monthly_excess),
                    'best_month': float(monthly_excess.max()),
                    'worst_month': float(monthly_excess.min())
                }

            return results

        except Exception as e:
            logger.error(f"Error generating performance attribution: {e}")
            return {}

    def _analyze_returns(self, port_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, Any]:
        """Analyze return characteristics."""
        # Annualized returns
        port_annual = port_returns.mean() * self.trading_days_per_year
        bench_annual = bench_returns.mean() * self.trading_days_per_year
        excess_annual = port_annual - bench_annual

        # Cumulative returns
        port_cumulative = (1 + port_returns).prod() - 1
        bench_cumulative = (1 + bench_returns).prod() - 1
        excess_cumulative = port_cumulative - bench_cumulative

        return {
            'annualized_return': {
                'portfolio': float(port_annual),
                'benchmark': float(bench_annual),
                'excess': float(excess_annual)
            },
            'cumulative_return': {
                'portfolio': float(port_cumulative),
                'benchmark': float(bench_cumulative),
                'excess': float(excess_cumulative)
            }
        }

    def _analyze_risk(self, port_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, Any]:
        """Analyze risk characteristics."""
        # Volatilities
        port_vol = port_returns.std() * np.sqrt(self.trading_days_per_year)
        bench_vol = bench_returns.std() * np.sqrt(self.trading_days_per_year)

        # Beta and alpha
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        alpha = port_returns.mean() * self.trading_days_per_year - beta * bench_returns.mean() * self.trading_days_per_year

        # Sharpe ratios (assuming risk-free rate = 0 for simplicity)
        port_sharpe = port_annual / port_vol if port_vol > 0 else 0.0
        bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0.0

        return {
            'volatility': {
                'portfolio': float(port_vol),
                'benchmark': float(bench_vol),
                'ratio': float(port_vol / bench_vol) if bench_vol > 0 else 0.0
            },
            'beta_alpha': {
                'beta': float(beta),
                'alpha': float(alpha)
            },
            'sharpe_ratio': {
                'portfolio': float(port_sharpe),
                'benchmark': float(bench_sharpe),
                'difference': float(port_sharpe - bench_sharpe)
            }
        }

    def _analyze_attribution(self, port_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance attribution."""
        excess_returns = port_returns - bench_returns

        # Period-by-period attribution
        outperformance_days = (excess_returns > 0).sum()
        underperformance_days = (excess_returns < 0).sum()

        return {
            'outperformance_days': int(outperformance_days),
            'underperformance_days': int(underperformance_days),
            'outperformance_rate': float(outperformance_days / len(excess_returns)),
            'avg_outperformance': float(excess_returns[excess_returns > 0].mean()) if (excess_returns > 0).any() else 0.0,
            'avg_underperformance': float(excess_returns[excess_returns < 0].mean()) if (excess_returns < 0).any() else 0.0
        }

    def _analyze_regimes(self, port_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        # Define regimes based on benchmark performance
        bench_cumulative = (1 + bench_returns).cumprod()
        bench_peak = bench_cumulative.expanding().max()
        drawdown = (bench_cumulative - bench_peak) / bench_peak

        # Define regimes
        bull_market = drawdown > -0.05  # Less than 5% drawdown
        bear_market = drawdown < -0.15  # More than 15% drawdown
        correction = (drawdown <= -0.05) & (drawdown >= -0.15)  # 5-15% drawdown

        excess_returns = port_returns - bench_returns

        regime_performance = {}
        for regime_name, regime_mask in [
            ('Bull Market', bull_market),
            ('Bear Market', bear_market),
            ('Correction', correction)
        ]:
            if regime_mask.sum() > 0:
                regime_excess = excess_returns[regime_mask]
                regime_performance[regime_name] = {
                    'days': int(regime_mask.sum()),
                    'avg_excess_return': float(regime_excess.mean()),
                    'excess_volatility': float(regime_excess.std()),
                    'win_rate': float((regime_excess > 0).mean())
                }

        return regime_performance

    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two time series to common dates."""
        aligned = pd.concat([series1, series2], axis=1, join='inner')
        if len(aligned.columns) >= 2:
            return aligned.iloc[:, 0], aligned.iloc[:, 1]
        else:
            return series1, series2

    def generate_benchmark_report(self,
                                 portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 benchmark_name: str = "Benchmark") -> Dict[str, Any]:
        """
        Generate comprehensive benchmark comparison report.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            benchmark_name: Name of the benchmark

        Returns:
            Dictionary with formatted benchmark report
        """
        comparison = self.compare_with_benchmark(portfolio_returns, benchmark_returns, benchmark_name)
        if not comparison:
            return {'error': 'Unable to generate comparison'}

        try:
            report = {
                'summary': {
                    'benchmark_name': benchmark_name,
                    'analysis_period': f"{comparison['analysis_period']['start_date']} to {comparison['analysis_period']['end_date']}",
                    'trading_days': comparison['analysis_period']['trading_days']
                },
                'performance_comparison': {
                    'portfolio_return': f"{comparison['return_analysis']['annualized_return']['portfolio']:.2%}",
                    'benchmark_return': f"{comparison['return_analysis']['annualized_return']['benchmark']:.2%}",
                    'excess_return': f"{comparison['return_analysis']['annualized_return']['excess']:.2%}",
                    'cumulative_excess': f"{comparison['return_analysis']['cumulative_return']['excess']:.2%}"
                },
                'risk_comparison': {
                    'portfolio_volatility': f"{comparison['risk_analysis']['volatility']['portfolio']:.2%}",
                    'benchmark_volatility': f"{comparison['risk_analysis']['volatility']['benchmark']:.2%}",
                    'volatility_ratio': f"{comparison['risk_analysis']['volatility']['ratio']:.2f}",
                    'beta': f"{comparison['risk_analysis']['beta_alpha']['beta']:.2f}",
                    'alpha': f"{comparison['risk_analysis']['beta_alpha']['alpha']:.2%}"
                },
                'attribution_summary': {
                    'outperformance_days': comparison['attribution_analysis']['outperformance_days'],
                    'underperformance_days': comparison['attribution_analysis']['underperformance_days'],
                    'outperformance_rate': f"{comparison['attribution_analysis']['outperformance_rate']:.1%}",
                    'avg_outperformance': f"{comparison['attribution_analysis']['avg_outperformance']:.2%}",
                    'avg_underperformance': f"{comparison['attribution_analysis']['avg_underperformance']:.2%}"
                },
                'regime_analysis': comparison['regime_analysis']
            }

            # Add additional metrics
            tracking_metrics = self.calculate_tracking_metrics(portfolio_returns, benchmark_returns)
            if tracking_metrics:
                report['tracking_metrics'] = {
                    'tracking_error': f"{tracking_metrics['tracking_error']:.2%}",
                    'information_ratio': f"{tracking_metrics['information_ratio']:.2f}"
                }

            capture_ratios = self.calculate_upside_downside_capture(portfolio_returns, benchmark_returns)
            if capture_ratios:
                report['capture_ratios'] = {
                    'upside_capture': f"{capture_ratios['upside_capture']:.1f}%",
                    'downside_capture': f"{capture_ratios['downside_capture']:.1f}%",
                    'overall_capture': f"{capture_ratios['overall_capture']:.1f}%"
                }

            return report

        except Exception as e:
            logger.error(f"Error generating benchmark report: {e}")
            return {'error': str(e)}

    def __str__(self) -> str:
        """String representation."""
        return f"BenchmarkAnalyzer(benchmarks={list(self.common_benchmarks.keys())})"