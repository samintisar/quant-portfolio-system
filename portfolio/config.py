"""
Simple configuration system for portfolio optimization.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    """Configuration for data fetching and processing."""
    default_period: str = field(default="5y")
    min_data_points: int = field(default=252)
    max_missing_pct: float = field(default=0.05)
    enable_caching: bool = field(default=True)
    cache_dir: str = field(default="./data/cache")

@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization methods."""
    risk_free_rate: float = field(default=0.02)
    max_iterations: int = field(default=1000)
    tolerance: float = field(default=1e-8)
    max_position_size: float = field(default=0.05)
    max_sector_concentration: float = field(default=0.20)
    max_drawdown: float = field(default=0.15)
    max_volatility: float = field(default=0.25)
    cvar_alpha: float = field(default=0.95)

@dataclass
class PerformanceConfig:
    """Configuration for performance metrics and calculations."""
    default_benchmark: str = field(default="SPY")
    annualization_factor: int = field(default=252)
    trading_days_per_year: int = field(default=252)
    risk_free_rate: float = field(default=0.02)
    confidence_level: float = field(default=0.95)
    return_decimals: int = field(default=4)
    percentage_decimals: int = field(default=2)

@dataclass
class SystemConfig:
    """Main system configuration combining all sub-configurations."""
    debug: bool = field(default=False)
    log_level: str = field(default="INFO")
    max_assets: int = field(default=50)
    min_assets: int = field(default=5)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create configuration from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_assets=int(os.getenv("MAX_ASSETS", "50")),
            min_assets=int(os.getenv("MIN_ASSETS", "5")),
            data=DataConfig(
                default_period=os.getenv("DEFAULT_DATA_PERIOD", "5y"),
                min_data_points=int(os.getenv("MIN_DATA_POINTS", "252")),
                max_missing_pct=float(os.getenv("MAX_MISSING_PCT", "0.05")),
                enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
                cache_dir=os.getenv("CACHE_DIR", "./data/cache")
            ),
            optimization=OptimizationConfig(
                risk_free_rate=float(os.getenv("RISK_FREE_RATE", "0.02")),
                max_iterations=int(os.getenv("MAX_ITERATIONS", "1000")),
                tolerance=float(os.getenv("TOLERANCE", "1e-8")),
                max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.05")),
                max_sector_concentration=float(os.getenv("MAX_SECTOR_CONCENTRATION", "0.20")),
                max_drawdown=float(os.getenv("MAX_DRAWDOWN", "0.15")),
                max_volatility=float(os.getenv("MAX_VOLATILITY", "0.25")),
                cvar_alpha=float(os.getenv("CVAR_ALPHA", "0.95"))
            ),
            performance=PerformanceConfig(
                default_benchmark=os.getenv("DEFAULT_BENCHMARK", "SPY"),
                annualization_factor=int(os.getenv("ANNUALIZATION_FACTOR", "252")),
                trading_days_per_year=int(os.getenv("TRADING_DAYS_PER_YEAR", "252")),
                risk_free_rate=float(os.getenv("RISK_FREE_RATE", "0.02")),
                confidence_level=float(os.getenv("CONFIDENCE_LEVEL", "0.95")),
                return_decimals=int(os.getenv("RETURN_DECIMALS", "4")),
                percentage_decimals=int(os.getenv("PERCENTAGE_DECIMALS", "2"))
            )
        )

# Global configuration instance
config = SystemConfig.from_env()

def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config