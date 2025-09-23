"""
Simple logging configuration for portfolio optimization system.
"""

import logging
import os
from typing import Optional


class PortfolioError(Exception):
    """Base exception for portfolio optimization errors."""
    pass


class DataError(PortfolioError):
    """Exception for data-related errors."""
    pass


class OptimizationError(PortfolioError):
    """Exception for optimization-related errors."""
    pass


class ValidationError(PortfolioError):
    """Exception for validation errors."""
    pass


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """Set up basic logging configuration."""
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Set up root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler()]
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Initialize logging
setup_logging()
data_logger = get_logger("portfolio.data")
optimization_logger = get_logger("portfolio.optimization")