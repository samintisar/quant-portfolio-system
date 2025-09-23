"""
Simple performance metrics calculator with essential metrics only.

Focuses on the most important portfolio performance metrics without overengineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SimplePerformanceCalculator:
    """Simple performance calculator with essential metrics only."""

    def __init__(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252):
        """Initialize with basic parameters."""
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate daily returns from price series."""
        return prices.pct_change().dropna()

    def calculate_portfolio_returns(self, price_data: pd.DataFrame,
                                  weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from asset prices and weights."""
        returns = price_data.pct_change().dropna()
        return (returns * pd.Series(weights)).sum(axis=1)

    def calculate_metrics(self, returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate essential performance metrics.

        Args:
            returns: Portfolio return series
            benchmark_returns: Optional benchmark return series

        Returns:
            Dictionary with essential performance metrics
        """
        if returns.empty:
            return {}

        try:
            metrics = {}

            # Basic return metrics
            metrics['total_return'] = (1 + returns).prod() - 1
            metrics['annual_return'] = self._annualize_return(returns)
            metrics['annual_volatility'] = returns.std() * np.sqrt(self.trading_days_per_year)

            # Risk-adjusted metrics
            if metrics['annual_volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / metrics['annual_volatility']
            else:
                metrics['sharpe_ratio'] = 0

            # Drawdown metrics
            metrics['max_drawdown'] = self._calculate_max_drawdown(returns)

            # Basic statistics
            metrics['best_day'] = returns.max()
            metrics['worst_day'] = returns.min()
            metrics['win_rate'] = (returns > 0).mean()

            # Benchmark-relative metrics (if provided)
            if benchmark_returns is not None and not benchmark_returns.empty:
                metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize total return."""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.trading_days_per_year
        if years > 0:
            return (1 + total_return) ** (1 / years) - 1
        return total_return

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_benchmark_metrics(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark-relative metrics."""
        # Align series
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        if len(aligned_returns) < 2:
            return {}

        port_aligned = aligned_returns.iloc[:, 0]
        bench_aligned = aligned_returns.iloc[:, 1]

        metrics = {}

        # Beta
        try:
            covariance = np.cov(port_aligned, bench_aligned)[0, 1]
            benchmark_variance = np.var(bench_aligned)
            if benchmark_variance > 0:
                metrics['beta'] = covariance / benchmark_variance
            else:
                metrics['beta'] = 0
        except:
            metrics['beta'] = 0

        # Alpha
        try:
            port_annual = self._annualize_return(port_aligned)
            bench_annual = self._annualize_return(bench_aligned)
            beta = metrics.get('beta', 0)
            metrics['alpha'] = port_annual - (self.risk_free_rate + beta * (bench_annual - self.risk_free_rate))
        except:
            metrics['alpha'] = 0

        # Information ratio
        try:
            excess_returns = port_aligned - bench_aligned
            tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)
            if tracking_error > 0:
                annual_excess = excess_returns.mean() * self.trading_days_per_year
                metrics['information_ratio'] = annual_excess / tracking_error
            else:
                metrics['information_ratio'] = 0
        except:
            metrics['information_ratio'] = 0

        return metrics

    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate a simple text report from metrics."""
        if not metrics:
            return "No metrics available"

        report = []
        report.append("=== Portfolio Performance Report ===")
        report.append("")

        # Return metrics
        report.append("Return Metrics:")
        report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
        report.append(f"  Best Day: {metrics.get('best_day', 0):.2%}")
        report.append(f"  Worst Day: {metrics.get('worst_day', 0):.2%}")
        report.append("")

        # Risk metrics
        report.append("Risk Metrics:")
        report.append(f"  Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
        report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
        report.append("")

        # Risk-adjusted metrics
        report.append("Risk-Adjusted Metrics:")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append("")

        # Benchmark metrics (if available)
        if 'beta' in metrics:
            report.append("Benchmark-Relative Metrics:")
            report.append(f"  Beta: {metrics.get('beta', 0):.2f}")
            report.append(f"  Alpha: {metrics.get('alpha', 0):.2%}")
            report.append(f"  Information Ratio: {metrics.get('information_ratio', 0):.2f}")

        return "\n".join(report)

    def __str__(self):
        return f"SimplePerformanceCalculator(risk_free_rate={self.risk_free_rate})"