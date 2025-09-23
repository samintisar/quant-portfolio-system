"""
Performance metrics calculator for portfolio analysis.

Implements comprehensive performance calculations for portfolio optimization results.
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


class ReturnPeriod(Enum):
    """Return calculation periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class PerformanceCalculator:
    """
    Comprehensive performance metrics calculator for portfolios.

    Provides calculation of standard financial performance metrics
    with proper annualization and statistical validation.
    """

    def __init__(self):
        """Initialize the performance calculator."""
        self.config = get_config()
        self.trading_days_per_year = self.config.performance.trading_days_per_year if hasattr(self.config, 'performance') else 252
        self.risk_free_rate = self.config.performance.risk_free_rate if hasattr(self.config, 'performance') else 0.02

        logger.info(f"Initialized PerformanceCalculator with {self.trading_days_per_year} trading days/year")

    def calculate_returns(self, prices: pd.Series,
                         return_period: ReturnPeriod = ReturnPeriod.DAILY,
                         method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series with datetime index
            return_period: Return calculation period
            method: Return calculation method ('simple' or 'log')

        Returns:
            Return series
        """
        if len(prices) < 2:
            logger.warning("Insufficient price data for return calculation")
            return pd.Series(dtype=float)

        try:
            if return_period == ReturnPeriod.DAILY:
                if method == 'simple':
                    returns = prices.pct_change().dropna()
                elif method == 'log':
                    returns = np.log(prices / prices.shift(1)).dropna()
                else:
                    raise ValidationError(f"Unsupported return method: {method}")
            else:
                # Resample to specified period
                if method == 'simple':
                    period_prices = prices.resample(self._get_resample_freq(return_period)).last()
                    returns = period_prices.pct_change().dropna()
                elif method == 'log':
                    period_prices = prices.resample(self._get_resample_freq(return_period)).last()
                    returns = np.log(period_prices / period_prices.shift(1)).dropna()

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(dtype=float)

    def calculate_portfolio_returns(self,
                                   price_data: pd.DataFrame,
                                   weights: Dict[str, float],
                                   return_period: ReturnPeriod = ReturnPeriod.DAILY) -> pd.Series:
        """
        Calculate portfolio returns from multiple assets.

        Args:
            price_data: DataFrame with asset prices (columns = assets)
            weights: Dictionary of asset weights
            return_period: Return calculation period

        Returns:
            Portfolio return series
        """
        if price_data.empty:
            logger.warning("Empty price data provided")
            return pd.Series(dtype=float)

        # Validate weights
        available_assets = [col for col in price_data.columns if col in weights]
        if not available_assets:
            logger.warning("No valid assets in weights")
            return pd.Series(dtype=float)

        # Normalize weights for available assets
        total_weight = sum(weights[asset] for asset in available_assets)
        if total_weight == 0:
            logger.warning("Total weight is zero")
            return pd.Series(dtype=float)

        normalized_weights = {asset: weights[asset] / total_weight for asset in available_assets}

        try:
            # Calculate individual asset returns
            asset_returns = {}
            for asset in available_assets:
                returns = self.calculate_returns(price_data[asset], return_period)
                if not returns.empty:
                    asset_returns[asset] = returns

            if not asset_returns:
                logger.warning("No valid returns calculated for any asset")
                return pd.Series(dtype=float)

            # Create returns DataFrame
            returns_df = pd.DataFrame(asset_returns)

            # Calculate weighted portfolio returns
            portfolio_returns = sum(returns_df[asset] * normalized_weights[asset]
                                   for asset in available_assets)

            return portfolio_returns.dropna()

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)

    def calculate_performance_metrics(self,
                                    returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series] = None,
                                    risk_free_rate: Optional[float] = None) -> PortfolioPerformance:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: Portfolio return series
            benchmark_returns: Optional benchmark return series
            risk_free_rate: Risk-free rate (defaults to config value)

        Returns:
            PortfolioPerformance object with all metrics
        """
        if returns.empty:
            logger.warning("Empty returns series provided")
            return PortfolioPerformance()

        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        try:
            performance = PortfolioPerformance()

            # Basic metrics
            performance.annual_return = self._calculate_annual_return(returns)
            performance.annual_volatility = self._calculate_annual_volatility(returns)
            performance.sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
            performance.max_drawdown = self._calculate_max_drawdown(returns)
            performance.calmar_ratio = self._calculate_calmar_ratio(returns, risk_free_rate)

            # Advanced metrics
            performance.sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
            performance.omega_ratio = self._calculate_omega_ratio(returns, risk_free_rate)

            # Benchmark-relative metrics
            if benchmark_returns is not None and not benchmark_returns.empty:
                # Calculate benchmark return
                performance.benchmark_return = self._calculate_annual_return(benchmark_returns)

                # Align series
                aligned_returns = self._align_series(returns, benchmark_returns)
                if len(aligned_returns) > 1:
                    port_aligned, bench_aligned = aligned_returns
                    performance.beta = self._calculate_beta(port_aligned, bench_aligned)
                    performance.alpha = self._calculate_alpha(port_aligned, bench_aligned, risk_free_rate)
                    performance.information_ratio = self._calculate_information_ratio(port_aligned, bench_aligned)
                    performance.upside_capture = self._calculate_upside_capture(port_aligned, bench_aligned)
                    performance.downside_capture = self._calculate_downside_capture(port_aligned, bench_aligned)

            # Additional statistics
            performance.best_day = returns.max() if len(returns) > 0 else 0.0
            performance.worst_day = returns.min() if len(returns) > 0 else 0.0
            performance.days_up = (returns > 0).sum() if len(returns) > 0 else 0
            performance.days_down = (returns < 0).sum() if len(returns) > 0 else 0
            performance.total_days = len(returns)

            if performance.total_days > 0:
                performance.win_rate = performance.days_up / performance.total_days

            return performance

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PortfolioPerformance()

    def calculate_rolling_metrics(self,
                                 returns: pd.Series,
                                 window: int = 252,
                                 benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Portfolio return series
            window: Rolling window size (default: 252 trading days = 1 year)
            benchmark_returns: Optional benchmark returns

        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window:
            logger.warning(f"Insufficient data for rolling window of {window}")
            return pd.DataFrame()

        try:
            rolling_data = {}

            # Rolling return
            rolling_cumulative = (1 + returns).rolling(window=window).apply(lambda x: x.prod(), raw=True) - 1
            rolling_data['rolling_return'] = rolling_cumulative

            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(self.trading_days_per_year)
            rolling_data['rolling_volatility'] = rolling_vol

            # Rolling Sharpe ratio
            rolling_mean = returns.rolling(window=window).mean() * self.trading_days_per_year
            rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_vol
            rolling_data['rolling_sharpe'] = rolling_sharpe

            # Rolling max drawdown
            rolling_peak = (1 + returns).rolling(window=window, min_periods=1).max()
            rolling_drawdown = (1 + returns) / rolling_peak - 1
            rolling_max_dd = rolling_drawdown.rolling(window=window, min_periods=1).min()
            rolling_data['rolling_max_drawdown'] = rolling_max_dd

            # Rolling beta and alpha (if benchmark provided)
            if benchmark_returns is not None:
                aligned_returns = self._align_series(returns, benchmark_returns)
                if len(aligned_returns[0]) >= window:
                    port_aligned, bench_aligned = aligned_returns

                    def rolling_beta(port_ret, bench_ret, window_size):
                        if len(port_ret) < window_size:
                            return np.nan
                        cov = np.cov(port_ret[-window_size:], bench_ret[-window_size:])[0, 1]
                        var = np.var(bench_ret[-window_size:])
                        return cov / var if var > 0 else np.nan

                    rolling_betas = []
                    for i in range(window, len(port_aligned) + 1):
                        beta = rolling_beta(port_aligned[:i], bench_aligned[:i], window)
                        rolling_betas.append(beta)

                    rolling_data['rolling_beta'] = [np.nan] * (window - 1) + rolling_betas

            return pd.DataFrame(rolling_data, index=returns.index)

        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        # Calculate total return
        total_return = (1 + returns).prod() - 1

        # Annualize based on data frequency
        years = len(returns) / self.trading_days_per_year
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = total_return

        return float(annual_return)

    def _calculate_annual_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) <= 1:
            return 0.0

        volatility = returns.std()
        annual_volatility = volatility * np.sqrt(self.trading_days_per_year)

        return float(annual_volatility)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = self._calculate_annual_return(returns)
        annual_volatility = self._calculate_annual_volatility(returns)

        if annual_volatility > 0:
            excess_return = annual_return - risk_free_rate
            sharpe_ratio = excess_return / annual_volatility
        else:
            sharpe_ratio = 0.0

        return float(sharpe_ratio)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()
        return float(max_drawdown)

    def _calculate_calmar_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = self._calculate_annual_return(returns)
        max_drawdown = self._calculate_max_drawdown(returns)

        if abs(max_drawdown) > 0.001:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0 if annual_return <= 0 else float('inf')

        return float(calmar_ratio) if calmar_ratio != float('inf') else 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = self._calculate_annual_return(returns)

        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(self.trading_days_per_year)
        else:
            downside_deviation = 0.0

        excess_return = annual_return - risk_free_rate

        if downside_deviation > 0:
            sortino_ratio = excess_return / downside_deviation
        else:
            sortino_ratio = 0.0 if excess_return <= 0 else float('inf')

        return float(sortino_ratio) if sortino_ratio != float('inf') else 0.0

    def _calculate_omega_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = risk_free_rate / self.trading_days_per_year

        # Excess returns over risk-free rate
        excess_returns = returns - daily_rf

        # Separate gains and losses
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())

        if losses > 0:
            omega_ratio = gains / losses
        else:
            omega_ratio = float('inf') if gains > 0 else 1.0

        return float(omega_ratio) if omega_ratio != float('inf') else 0.0

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta against benchmark."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 0.0

        return float(beta)

    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate alpha against benchmark."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Calculate annualized returns
        port_return = self._calculate_annual_return(portfolio_returns)
        bench_return = self._calculate_annual_return(benchmark_returns)

        # Calculate beta
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)

        # Calculate alpha
        alpha = port_return - (risk_free_rate + beta * (bench_return - risk_free_rate))

        return float(alpha)

    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Excess returns
        excess_returns = portfolio_returns - benchmark_returns

        # Tracking error (annualized)
        tracking_error = excess_returns.std() * np.sqrt(self.trading_days_per_year)

        # Annualized excess return
        annual_excess_return = excess_returns.mean() * self.trading_days_per_year

        if tracking_error > 0:
            information_ratio = annual_excess_return / tracking_error
        else:
            information_ratio = 0.0

        return float(information_ratio)

    def _calculate_upside_capture(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate upside capture ratio."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Filter positive benchmark periods
        pos_bench_mask = benchmark_returns > 0
        if pos_bench_mask.sum() == 0:
            return 0.0

        port_positive = portfolio_returns[pos_bench_mask].mean()
        bench_positive = benchmark_returns[pos_bench_mask].mean()

        if bench_positive > 0:
            upside_capture = (port_positive / bench_positive) * 100
        else:
            upside_capture = 0.0

        return float(upside_capture)

    def _calculate_downside_capture(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate downside capture ratio."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Filter negative benchmark periods
        neg_bench_mask = benchmark_returns < 0
        if neg_bench_mask.sum() == 0:
            return 0.0

        port_negative = portfolio_returns[neg_bench_mask].mean()
        bench_negative = benchmark_returns[neg_bench_mask].mean()

        if bench_negative < 0:
            downside_capture = (port_negative / bench_negative) * 100
        else:
            downside_capture = 0.0

        return float(downside_capture)

    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two time series to common dates."""
        aligned = pd.concat([series1, series2], axis=1, join='inner')
        if len(aligned.columns) >= 2:
            return aligned.iloc[:, 0], aligned.iloc[:, 1]
        else:
            return series1, series2

    def _get_resample_freq(self, return_period: ReturnPeriod) -> str:
        """Get pandas resample frequency string."""
        freq_map = {
            ReturnPeriod.WEEKLY: 'W',
            ReturnPeriod.MONTHLY: 'M',
            ReturnPeriod.QUARTERLY: 'Q',
            ReturnPeriod.ANNUAL: 'Y'
        }
        return freq_map.get(return_period, 'D')

    def generate_performance_report(self,
                                  performance: PortfolioPerformance,
                                  period_days: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            performance: PortfolioPerformance object
            period_days: Number of days in the analysis period

        Returns:
            Dictionary with formatted performance report
        """
        report = {
            'summary': {
                'annual_return': f"{performance.annual_return:.2%}",
                'annual_volatility': f"{performance.annual_volatility:.2%}",
                'sharpe_ratio': f"{performance.sharpe_ratio:.2f}",
                'max_drawdown': f"{performance.max_drawdown:.2%}",
                'calmar_ratio': f"{performance.calmar_ratio:.2f}",
                'sortino_ratio': f"{performance.sortino_ratio:.2f}",
                'win_rate': f"{performance.win_rate:.1%}" if performance.win_rate else "N/A"
            },
            'risk_metrics': {
                'best_day': f"{performance.best_day:.2%}",
                'worst_day': f"{performance.worst_day:.2%}",
                'days_up': performance.days_up,
                'days_down': performance.days_down,
                'total_days': performance.total_days
            },
            'benchmark_metrics': {}
        }

        # Add benchmark metrics if available
        if performance.beta is not None:
            report['benchmark_metrics'] = {
                'beta': f"{performance.beta:.2f}",
                'alpha': f"{performance.alpha:.2%}",
                'information_ratio': f"{performance.information_ratio:.2f}",
                'upside_capture': f"{performance.upside_capture:.1f}%",
                'downside_capture': f"{performance.downside_capture:.1f}%"
            }

        # Add period analysis
        if period_days:
            years = period_days / self.trading_days_per_year
            report['period_analysis'] = {
                'period_days': period_days,
                'period_years': f"{years:.2f}",
                'annualized': True if years >= 1 else False
            }

        return report

    def __str__(self) -> str:
        """String representation."""
        return f"PerformanceCalculator(trading_days={self.trading_days_per_year}, risk_free_rate={self.risk_free_rate:.2%})"