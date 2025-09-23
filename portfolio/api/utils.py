"""
API utilities and helper functions.

Provides common utilities for API endpoints including validation,
formatting, and data processing.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from decimal import Decimal, InvalidOperation

from portfolio.logging_config import get_logger

logger = get_logger(__name__)


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol.

    Args:
        symbol: Stock symbol to validate

    Returns:
        Normalized symbol (uppercase, stripped)

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")

    normalized = symbol.strip().upper()

    if not normalized.isalnum():
        raise ValueError("Symbol must contain only alphanumeric characters")

    if len(normalized) < 1 or len(normalized) > 10:
        raise ValueError("Symbol must be between 1 and 10 characters")

    return normalized


def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Validate portfolio weights.

    Args:
        weights: Dictionary of symbol -> weight mappings

    Returns:
        Validated weights dictionary

    Raises:
        ValueError: If weights are invalid
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty")

    if not isinstance(weights, dict):
        raise ValueError("Weights must be a dictionary")

    # Check weight values
    total_weight = 0.0
    for symbol, weight in weights.items():
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Weight for {symbol} must be numeric")
        if weight < 0 or weight > 1:
            raise ValueError(f"Weight for {symbol} must be between 0 and 1")
        total_weight += weight

    # Check if weights sum to 1 (with small tolerance)
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Weights must sum to 1.0 (current sum: {total_weight:.3f})")

    return weights


def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> tuple:
    """
    Validate date range strings.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        Tuple of validated datetime objects

    Raises:
        ValueError: If dates are invalid
    """
    try:
        start_dt = None
        end_dt = None

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            if start_dt < datetime(2000, 1, 1):
                raise ValueError("Start date must be after 2000-01-01")
            if start_dt > datetime.now():
                raise ValueError("Start date cannot be in the future")

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if end_dt > datetime.now():
                raise ValueError("End date cannot be in the future")

        if start_dt and end_dt and start_dt >= end_dt:
            raise ValueError("Start date must be before end date")

        return start_dt, end_dt

    except ValueError as e:
        if "does not match format" in str(e):
            raise ValueError("Dates must be in YYYY-MM-DD format")
        raise


def format_financial_number(value: Union[float, int, Decimal, None],
                          decimals: int = 4,
                          percentage: bool = False) -> Optional[str]:
    """
    Format financial numbers for display.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        percentage: Whether to format as percentage

    Returns:
        Formatted string or None if value is None
    """
    if value is None:
        return None

    try:
        if isinstance(value, str):
            value = Decimal(value)
        elif isinstance(value, (int, float)):
            value = Decimal(str(value))

        if percentage:
            formatted = f"{float(value) * 100:.{decimals}%"
        else:
            formatted = f"{float(value):.{decimals}f}"

        return formatted

    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"Could not format value: {value}")
        return str(value)


def calculate_performance_metrics(returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns series
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    if returns.empty:
        return {}

    metrics = {}

    try:
        # Basic return metrics
        metrics['total_return'] = float((1 + returns).prod() - 1)
        metrics['annual_return'] = float(returns.mean() * 252)
        metrics['volatility'] = float(returns.std() * (252 ** 0.5))

        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            excess_return = metrics['annual_return'] - risk_free_rate
            metrics['sharpe_ratio'] = excess_return / metrics['volatility']
            metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate)

        # Drawdown metrics
        metrics['max_drawdown'] = float(calculate_max_drawdown(returns))
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])

        # Value at Risk
        metrics['var_95'] = float(np.percentile(returns, 5))
        metrics['var_99'] = float(np.percentile(returns, 1))
        metrics['cvar_95'] = float(returns[returns <= metrics['var_95']].mean())
        metrics['cvar_99'] = float(returns[returns <= metrics['var_99']].mean())

        # Benchmark-relative metrics
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_return = (1 + benchmark_returns).prod() - 1
            portfolio_return = metrics['total_return']

            if benchmark_return != 0:
                metrics['information_ratio'] = (portfolio_return - benchmark_return) / returns.std()

            # Beta calculation
            if benchmark_returns.std() != 0:
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                metrics['beta'] = covariance / benchmark_variance

                # Alpha calculation
                metrics['alpha'] = portfolio_return - (risk_free_rate + metrics['beta'] * (benchmark_return - risk_free_rate))

                # Up/downside capture
                if benchmark_return > 0:
                    metrics['upside_capture'] = (portfolio_return / benchmark_return) * 100
                else:
                    metrics['downside_capture'] = (portfolio_return / benchmark_return) * 100

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")

    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std() * (252 ** 0.5)
    annual_return = returns.mean() * 252

    if downside_std == 0:
        return 0.0

    return (annual_return - risk_free_rate) / downside_std


def sanitize_data_for_json(data: Any) -> Any:
    """
    Sanitize data for JSON serialization.

    Args:
        data: Data to sanitize

    Returns:
        JSON-serializable data
    """
    if data is None:
        return None

    if isinstance(data, (int, float, str, bool)):
        return data

    if isinstance(data, dict):
        return {key: sanitize_data_for_json(value) for key, value in data.items()}

    if isinstance(data, (list, tuple)):
        return [sanitize_data_for_json(item) for item in data]

    if isinstance(data, pd.DataFrame):
        return data.to_dict('records')

    if isinstance(data, pd.Series):
        return data.to_dict()

    if isinstance(data, (np.integer, np.floating)):
        return float(data)

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, datetime):
        return data.isoformat()

    if isinstance(data, Decimal):
        return float(data)

    # Convert unknown types to string
    return str(data)


def create_api_response(success: bool = True,
                       data: Optional[Any] = None,
                       message: Optional[str] = None,
                       error_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create standardized API response.

    Args:
        success: Whether the operation was successful
        data: Response data
        message: Response message
        error_details: Error details if not successful

    Returns:
        Standardized response dictionary
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }

    if data is not None:
        response["data"] = sanitize_data_for_json(data)

    if message:
        response["message"] = message

    if not success and error_details:
        response["error_details"] = error_details

    return response


def parse_period_string(period: str) -> str:
    """
    Parse and validate period string for Yahoo Finance API.

    Args:
        period: Period string (e.g., "1d", "1mo", "1y")

    Returns:
        Validated period string

    Raises:
        ValueError: If period is invalid
    """
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    if not isinstance(period, str):
        raise ValueError("Period must be a string")

    period = period.lower().strip()

    if period not in valid_periods:
        raise ValueError(f"Invalid period. Must be one of: {', '.join(valid_periods)}")

    return period


def calculate_correlation_matrix(returns_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_data: DataFrame with asset returns

    Returns:
        Dictionary of correlation matrices
    """
    if returns_data.empty:
        return {}

    try:
        correlation_matrix = returns_data.corr()

        # Convert to dictionary format
        result = {}
        for col1 in correlation_matrix.columns:
            result[col1] = {}
            for col2 in correlation_matrix.columns:
                result[col1][col2] = float(correlation_matrix.loc[col1, col2])

        return result

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return {}


def generate_random_portfolio_weights(num_assets: int, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Generate random portfolio weights that sum to 1.

    Args:
        num_assets: Number of assets
        seed: Random seed for reproducibility

    Returns:
        Dictionary of random weights
    """
    if seed is not None:
        np.random.seed(seed)

    if num_assets <= 0:
        raise ValueError("Number of assets must be positive")

    # Generate random numbers and normalize
    random_weights = np.random.random(num_assets)
    normalized_weights = random_weights / random_weights.sum()

    # Create dictionary with generic asset names
    weights_dict = {}
    for i, weight in enumerate(normalized_weights):
        weights_dict[f"asset_{i+1}"] = float(weight)

    return weights_dict