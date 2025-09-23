"""
Return calculation utilities for portfolio optimization system.

Handles various return calculations and financial metrics.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

from portfolio.logging_config import get_logger, ValidationError
from portfolio.config import get_config

logger = get_logger(__name__)


class ReturnType(Enum):
    """Types of return calculations."""
    SIMPLE = "simple"        # Simple returns: (P_t - P_{t-1}) / P_{t-1}
    LOG = "log"              # Log returns: ln(P_t / P_{t-1})
    ARITHMETIC = "arithmetic" # Arithmetic mean return
    GEOMETRIC = "geometric"   # Geometric mean return


class ReturnCalculator:
    """
    Return calculation utilities for financial analysis.

    Simple implementation with comprehensive return calculations.
    """

    def __init__(self):
        """Initialize the return calculator."""
        self.config = get_config()
        self.annualization_factor = self.config.performance.annualization_factor

        logger.info(f"Initialized ReturnCalculator with annualization_factor: {self.annualization_factor}")

    def calculate_returns(self, prices: pd.Series,
                         return_type: ReturnType = ReturnType.SIMPLE,
                         dropna: bool = True) -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            return_type: Type of return calculation
            dropna: Whether to drop NaN values

        Returns:
            Return series
        """
        if len(prices) < 2:
            logger.warning("Insufficient data for return calculation")
            return pd.Series(dtype=float)

        try:
            if return_type == ReturnType.SIMPLE:
                returns = prices.pct_change()
            elif return_type == ReturnType.LOG:
                returns = np.log(prices / prices.shift(1))
            else:
                raise ValueError(f"Unsupported return_type: {return_type}")

            if dropna:
                returns = returns.dropna()

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.Series(dtype=float)

    def calculate_portfolio_returns(self, price_data: pd.DataFrame,
                                 weights: Dict[str, float],
                                 return_type: ReturnType = ReturnType.SIMPLE) -> pd.Series:
        """
        Calculate portfolio returns from multiple assets.

        Args:
            price_data: DataFrame with asset prices (columns = assets)
            weights: Dictionary of asset weights
            return_type: Type of return calculation

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
                if asset in price_data.columns:
                    returns = self.calculate_returns(price_data[asset], return_type)
                    asset_returns[asset] = returns

            # Create returns DataFrame
            returns_df = pd.DataFrame(asset_returns)

            # Calculate weighted portfolio returns
            portfolio_returns = sum(returns_df[asset] * normalized_weights[asset]
                                   for asset in available_assets)

            return portfolio_returns.dropna()

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return pd.Series(dtype=float)

    def annualize_returns(self, returns: pd.Series,
                         periods_per_year: Optional[int] = None) -> float:
        """
        Annualize return series.

        Args:
            returns: Return series
            periods_per_year: Number of periods per year (default from config)

        Returns:
            Annualized return
        """
        if len(returns) == 0:
            return 0.0

        if periods_per_year is None:
            periods_per_year = self.annualization_factor

        try:
            # Calculate cumulative return
            cumulative_return = np.prod(1 + returns) - 1

            # Annualize
            years = len(returns) / periods_per_year
            if years > 0:
                annualized_return = (1 + cumulative_return) ** (1 / years) - 1
            else:
                annualized_return = cumulative_return

            return annualized_return

        except Exception as e:
            logger.error(f"Error annualizing returns: {e}")
            return 0.0

    def calculate_volatility(self, returns: pd.Series,
                           periods_per_year: Optional[int] = None,
                           annualize: bool = True) -> float:
        """
        Calculate volatility of return series.

        Args:
            returns: Return series
            periods_per_year: Number of periods per year
            annualize: Whether to annualize volatility

        Returns:
            Volatility
        """
        if len(returns) <= 1:
            return 0.0

        if periods_per_year is None:
            periods_per_year = self.annualization_factor

        try:
            volatility = returns.std()

            if annualize:
                volatility = volatility * np.sqrt(periods_per_year)

            return float(volatility)

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, returns: pd.Series,
                              risk_free_rate: float = 0.02,
                              periods_per_year: Optional[int] = None) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0

        try:
            # Annualize returns and volatility
            annual_return = self.annualize_returns(returns, periods_per_year)
            annual_volatility = self.calculate_volatility(returns, periods_per_year, annualize=True)

            if annual_volatility > 0:
                excess_return = annual_return - risk_free_rate
                sharpe_ratio = excess_return / annual_volatility
            else:
                sharpe_ratio = 0.0

            return float(sharpe_ratio)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, pd.DatetimeIndex]:
        """
        Calculate maximum drawdown and its dates.

        Args:
            returns: Return series

        Returns:
            Tuple of (max_drawdown, drawdown_dates)
        """
        if len(returns) == 0:
            return 0.0, pd.DatetimeIndex([])

        try:
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max

            max_dd = drawdown.min()
            max_dd_dates = drawdown[drawdown == max_dd].index

            return float(max_dd), max_dd_dates

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, pd.DatetimeIndex([])

    def calculate_beta(self, portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series) -> float:
        """
        Calculate beta against benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Beta value
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        try:
            # Align series
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            if len(aligned_data) < 2:
                return 0.0

            port_returns = aligned_data.iloc[:, 0]
            bench_returns = aligned_data.iloc[:, 1]

            # Calculate covariance and variance
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            bench_variance = np.var(bench_returns)

            if bench_variance > 0:
                beta = covariance / bench_variance
            else:
                beta = 0.0

            return float(beta)

        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0

    def calculate_alpha(self, portfolio_returns: pd.Series,
                       benchmark_returns: pd.Series,
                       risk_free_rate: float = 0.02,
                       periods_per_year: Optional[int] = None) -> float:
        """
        Calculate alpha against benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Alpha value
        """
        try:
            # Calculate annualized returns
            port_return = self.annualize_returns(portfolio_returns, periods_per_year)
            bench_return = self.annualize_returns(benchmark_returns, periods_per_year)

            # Calculate beta
            beta = self.calculate_beta(portfolio_returns, benchmark_returns)

            # Calculate alpha
            alpha = port_return - (risk_free_rate + beta * (bench_return - risk_free_rate))

            return float(alpha)

        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0.0

    def calculate_information_ratio(self, portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  periods_per_year: Optional[int] = None) -> float:
        """
        Calculate information ratio.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods_per_year: Number of periods per year

        Returns:
            Information ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        try:
            # Align series
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            if len(aligned_data) < 2:
                return 0.0

            port_returns = aligned_data.iloc[:, 0]
            bench_returns = aligned_data.iloc[:, 1]

            # Calculate excess returns
            excess_returns = port_returns - bench_returns

            # Calculate tracking error (annualized)
            tracking_error = self.calculate_volatility(excess_returns, periods_per_year, annualize=True)

            if tracking_error > 0:
                # Annualized excess return
                annual_excess_return = self.annualize_returns(excess_returns, periods_per_year)
                information_ratio = annual_excess_return / tracking_error
            else:
                information_ratio = 0.0

            return float(information_ratio)

        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series,
                               risk_free_rate: float = 0.02,
                               periods_per_year: Optional[int] = None,
                               mar: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            mar: Minimum acceptable return (default: risk_free_rate)

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        if mar is None:
            mar = risk_free_rate / periods_per_year if periods_per_year else risk_free_rate

        try:
            # Annualized return
            annual_return = self.annualize_returns(returns, periods_per_year)

            # Downside deviation
            downside_returns = returns[returns < mar]
            if len(downside_returns) > 0:
                downside_deviation = np.sqrt(np.mean((downside_returns - mar) ** 2))
                if periods_per_year:
                    downside_deviation *= np.sqrt(periods_per_year)
            else:
                downside_deviation = 0.0

            # Sortino ratio
            excess_return = annual_return - risk_free_rate
            if downside_deviation > 0:
                sortino_ratio = excess_return / downside_deviation
            else:
                sortino_ratio = 0.0 if excess_return <= 0 else float('inf')

            return float(sortino_ratio) if sortino_ratio != float('inf') else 0.0

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_calmar_ratio(self, returns: pd.Series,
                             periods_per_year: Optional[int] = None) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Return series
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        try:
            # Annualized return
            annual_return = self.annualize_returns(returns, periods_per_year)

            # Maximum drawdown
            max_drawdown, _ = self.calculate_max_drawdown(returns)

            # Calmar ratio
            if abs(max_drawdown) > 0.001:  # Avoid division by very small numbers
                calmar_ratio = annual_return / abs(max_drawdown)
            else:
                calmar_ratio = 0.0 if annual_return <= 0 else float('inf')

            return float(calmar_ratio) if calmar_ratio != float('inf') else 0.0

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def calculate_correlation_matrix(self, returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.

        Args:
            returns_dict: Dictionary of asset return series

        Returns:
            Correlation matrix
        """
        if not returns_dict:
            return pd.DataFrame()

        try:
            # Create DataFrame from returns
            returns_df = pd.DataFrame(returns_dict)

            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def calculate_covariance_matrix(self, returns_dict: Dict[str, pd.Series],
                                  annualize: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix for multiple assets.

        Args:
            returns_dict: Dictionary of asset return series
            annualize: Whether to annualize covariance

        Returns:
            Covariance matrix
        """
        if not returns_dict:
            return pd.DataFrame()

        try:
            # Create DataFrame from returns
            returns_df = pd.DataFrame(returns_dict)

            # Calculate covariance matrix
            covariance_matrix = returns_df.cov()

            # Annualize if requested
            if annualize:
                covariance_matrix *= self.annualization_factor

            return covariance_matrix

        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            return pd.DataFrame()

    def get_performance_summary(self, returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02,
                              periods_per_year: Optional[int] = None) -> Dict[str, float]:
        """
        Get comprehensive performance summary.

        Args:
            returns: Return series
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Dictionary with performance metrics
        """
        if len(returns) == 0:
            return {}

        try:
            summary = {
                'annual_return': self.annualize_returns(returns, periods_per_year),
                'annual_volatility': self.calculate_volatility(returns, periods_per_year, annualize=True),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
                'max_drawdown': self.calculate_max_drawdown(returns)[0],
                'sortino_ratio': self.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
                'calmar_ratio': self.calculate_calmar_ratio(returns, periods_per_year)
            }

            # Add benchmark-relative metrics if available
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                summary.update({
                    'beta': self.calculate_beta(returns, benchmark_returns),
                    'alpha': self.calculate_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year),
                    'information_ratio': self.calculate_information_ratio(returns, benchmark_returns, periods_per_year)
                })

            return summary

        except Exception as e:
            logger.error(f"Error calculating performance summary: {e}")
            return {}

    def __str__(self) -> str:
        """String representation."""
        return f"ReturnCalculator(annualization_factor={self.annualization_factor})"