"""
Economic indicators data integration with revision and lag tracking.

Implements comprehensive economic data management including:
- Multiple economic data source integration
- Data revision tracking and analysis
- Lag management and compensation
- Real-time data monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum


class DataStatus(Enum):
    """Status of economic data."""
    PRELIMINARY = "preliminary"
    REVISED = "revised"
    FINAL = "final"
    UNKNOWN = "unknown"


@dataclass
class EconomicIndicator:
    """Economic indicator metadata."""
    name: str
    code: str
    frequency: str  # daily, weekly, monthly, quarterly, annual
    typical_lag: int  # days
    revision_pattern: str  # low, medium, high
    unit: str
    source: str


class EconomicDataRevision:
    """Track data revisions over time."""

    def __init__(self):
        self.revision_history = {}
        self.logger = logging.getLogger(__name__)

    def record_revision(self,
                      indicator: str,
                      date: str,
                      old_value: float,
                      new_value: float,
                      revision_type: str,
                      notes: str = ""):
        """Record a data revision."""
        if indicator not in self.revision_history:
            self.revision_history[indicator] = []

        revision_record = {
            'indicator': indicator,
            'date': date,
            'old_value': old_value,
            'new_value': new_value,
            'revision_type': revision_type,
            'revision_magnitude': abs(new_value - old_value),
            'revision_percentage': abs((new_value - old_value) / old_value * 100) if old_value != 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        }

        self.revision_history[indicator].append(revision_record)
        self.logger.info(f"Recorded revision for {indicator} on {date}: {old_value} -> {new_value}")

    def get_revision_history(self, indicator: str) -> List[Dict]:
        """Get revision history for an indicator."""
        return self.revision_history.get(indicator, [])

    def get_revision_statistics(self, indicator: str) -> Dict[str, Any]:
        """Calculate revision statistics for an indicator."""
        history = self.get_revision_history(indicator)
        if not history:
            return {"message": "No revision history available"}

        revisions = [r['revision_magnitude'] for r in history]
        percentage_revisions = [r['revision_percentage'] for r in history]

        return {
            'total_revisions': len(history),
            'avg_revision_magnitude': np.mean(revisions),
            'max_revision_magnitude': max(revisions),
            'avg_revision_percentage': np.mean(percentage_revisions),
            'max_revision_percentage': max(percentage_revisions),
            'revision_frequency': len(history) / len(set(r['date'] for r in history)) if history else 0,
            'recent_revisions': sorted(history, key=lambda x: x['timestamp'], reverse=True)[:5]
        }

    def detect_anomaly_revisions(self, threshold_std: float = 3.0) -> List[Dict]:
        """Detect unusually large revisions."""
        anomalies = []

        for indicator, history in self.revision_history.items():
            if len(history) < 10:  # Need sufficient history
                continue

            revisions = [r['revision_magnitude'] for r in history]
            mean_rev = np.mean(revisions)
            std_rev = np.std(revisions)

            for revision in history:
                if revision['revision_magnitude'] > mean_rev + threshold_std * std_rev:
                    anomalies.append({
                        'indicator': indicator,
                        'revision': revision,
                        'z_score': (revision['revision_magnitude'] - mean_rev) / std_rev
                    })

        return anomalies


class DataLagManager:
    """Manage and compensate for data lags."""

    def __init__(self):
        self.lag_profiles = {
            'GDP_GROWTH': {'typical_lag': 45, 'revision_window': 180},
            'INFLATION_CPI': {'typical_lag': 15, 'revision_window': 30},
            'UNEMPLOYMENT_RATE': {'typical_lag': 7, 'revision_window': 21},
            'FED_FUNDS_RATE': {'typical_lag': 0, 'revision_window': 0},
            'TREASURY_YIELD_10Y': {'typical_lag': 0, 'revision_window': 0},
            'RETAIL_SALES': {'typical_lag': 15, 'revision_window': 30},
            'DURABLE_GOODS': {'typical_lag': 20, 'revision_window': 45},
            'CONSUMER_SENTIMENT': {'typical_lag': 2, 'revision_window': 7}
        }

    def get_lag_adjustment(self, indicator: str, as_of_date: str) -> Dict[str, Any]:
        """
        Calculate lag adjustment for an indicator as of a specific date.

        Returns:
            Dictionary containing lag adjustment information
        """
        if indicator not in self.lag_profiles:
            return {"error": f"No lag profile for indicator: {indicator}"}

        profile = self.lag_profiles[indicator]
        as_of = datetime.strptime(as_of_date, "%Y-%m-%d")

        # Calculate latest available date considering lag
        latest_available = as_of - timedelta(days=profile['typical_lag'])

        # Calculate when data might be revised
        revision_deadline = as_of + timedelta(days=profile['revision_window'])

        return {
            'indicator': indicator,
            'as_of_date': as_of_date,
            'typical_lag_days': profile['typical_lag'],
            'latest_available_date': latest_available.strftime("%Y-%m-%d"),
            'revision_window_days': profile['revision_window'],
            'revision_deadline': revision_deadline.strftime("%Y-%m-%d"),
            'data_status': self._get_data_status(as_of, latest_available)
        }

    def _get_data_status(self, as_of_date: datetime, available_date: datetime) -> str:
        """Determine data status based on lag."""
        days_since_available = (as_of_date - available_date).days

        if days_since_available <= 7:
            return "recent"
        elif days_since_available <= 30:
            return "current"
        elif days_since_available <= 90:
            return "aging"
        else:
            return "stale"

    def adjust_for_lag(self,
                      indicator: str,
                      data: pd.DataFrame,
                      as_of_date: str,
                      method: str = "forward_fill") -> pd.DataFrame:
        """
        Adjust data for lag effects.

        Args:
            indicator: Economic indicator name
            data: DataFrame with economic data
            as_of_date: As-of date for analysis
            method: Adjustment method (forward_fill, linear_extrapolation, seasonal_adjust)

        Returns:
            Adjusted DataFrame
        """
        lag_info = self.get_lag_adjustment(indicator, as_of_date)

        if method == "forward_fill":
            # Forward fill to current date
            data_adjusted = data.copy()
            data_adjusted = data_adjusted.reindex(
                pd.date_range(start=data.index.min(), end=as_of_date, freq='D'),
                method='ffill'
            )
        elif method == "linear_extrapolation":
            # Linear extrapolation for recent missing periods
            data_adjusted = data.copy()
            data_adjusted = data_adjusted.reindex(
                pd.date_range(start=data.index.min(), end=as_of_date, freq='D')
            )
            data_adjusted = data_adjusted.interpolate(method='linear')
        else:
            data_adjusted = data.copy()

        return data_adjusted


class EconomicDataIntegrationService:
    """Main service for economic data integration with revision tracking."""

    def __init__(self):
        self.indicators = self._initialize_indicators()
        self.revision_tracker = EconomicDataRevision()
        self.lag_manager = DataLagManager()
        self.logger = logging.getLogger(__name__)

    def _initialize_indicators(self) -> Dict[str, EconomicIndicator]:
        """Initialize supported economic indicators."""
        return {
            'GDP_GROWTH': EconomicIndicator(
                'GDP Growth Rate', 'GDP', 'quarterly', 45, 'high', '%', 'BEA'
            ),
            'INFLATION_CPI': EconomicIndicator(
                'Consumer Price Inflation', 'CPI', 'monthly', 15, 'medium', '%', 'BLS'
            ),
            'UNEMPLOYMENT_RATE': EconomicIndicator(
                'Unemployment Rate', 'UNEMP', 'monthly', 7, 'low', '%', 'BLS'
            ),
            'FED_FUNDS_RATE': EconomicIndicator(
                'Federal Funds Rate', 'FFR', 'daily', 0, 'none', '%', 'FED'
            ),
            'TREASURY_YIELD_10Y': EconomicIndicator(
                '10-Year Treasury Yield', '10Y', 'daily', 0, 'none', '%', 'TREASURY'
            ),
            'RETAIL_SALES': EconomicIndicator(
                'Retail Sales', 'RETAIL', 'monthly', 15, 'medium', '%', 'CENSUS'
            ),
            'DURABLE_GOODS': EconomicIndicator(
                'Durable Goods Orders', 'DURABLE', 'monthly', 20, 'medium', '%', 'CENSUS'
            ),
            'CONSUMER_SENTIMENT': EconomicIndicator(
                'Consumer Sentiment Index', 'SENTIMENT', 'monthly', 2, 'low', 'index', 'UMICH'
            )
        }

    async def fetch_economic_data(self,
                                indicators: List[str],
                                start_date: str,
                                end_date: str,
                                as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch economic data with revision tracking and lag adjustment.

        Args:
            indicators: List of economic indicators to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            as_of_date: As-of date for analysis (defaults to current date)

        Returns:
            Dictionary containing economic data and metadata
        """
        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")

        try:
            self.logger.info(f"Fetching economic data for {len(indicators)} indicators")

            results = {}

            for indicator in indicators:
                if indicator not in self.indicators:
                    results[indicator] = {"error": f"Unknown indicator: {indicator}"}
                    continue

                # Fetch raw data (mock implementation)
                raw_data = await self._fetch_raw_indicator_data(
                    indicator, start_date, end_date
                )

                if raw_data is None:
                    results[indicator] = {"error": f"Failed to fetch data for {indicator}"}
                    continue

                # Apply lag adjustments
                adjusted_data = self.lag_manager.adjust_for_lag(
                    indicator, raw_data, as_of_date
                )

                # Get lag information
                lag_info = self.lag_manager.get_lag_adjustment(indicator, as_of_date)

                # Get revision statistics
                revision_stats = self.revision_tracker.get_revision_statistics(indicator)

                results[indicator] = {
                    'indicator_info': self.indicators[indicator].__dict__,
                    'raw_data': raw_data.to_dict('records'),
                    'adjusted_data': adjusted_data.to_dict('records'),
                    'lag_adjustment': lag_info,
                    'revision_statistics': revision_stats,
                    'data_quality': self._assess_data_quality(raw_data),
                    'fetch_metadata': {
                        'start_date': start_date,
                        'end_date': end_date,
                        'as_of_date': as_of_date,
                        'fetch_time': datetime.now().isoformat()
                    }
                }

            return {
                'indicators': indicators,
                'data': results,
                'summary': {
                    'successful_fetches': len([r for r in results.values() if 'error' not in r]),
                    'failed_fetches': len([r for r in results.values() if 'error' in r]),
                    'total_indicators': len(indicators),
                    'as_of_date': as_of_date
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch economic data: {e}")
            return {"error": str(e)}

    async def _fetch_raw_indicator_data(self,
                                       indicator: str,
                                       start_date: str,
                                       end_date: str) -> Optional[pd.DataFrame]:
        """Fetch raw indicator data (mock implementation)."""
        try:
            # In real implementation, this would connect to FRED, BEA, BLS, etc.
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate mock data based on indicator type
            if 'GDP' in indicator:
                values = np.random.normal(2.0, 0.5, len(dates))
            elif 'INFLATION' in indicator:
                values = np.random.normal(2.5, 0.3, len(dates))
            elif 'UNEMP' in indicator:
                values = np.random.normal(4.0, 0.5, len(dates))
            elif 'RATE' in indicator:
                values = np.random.normal(2.5, 0.2, len(dates))
            else:
                values = np.random.normal(100, 10, len(dates))

            data = pd.DataFrame({
                'Date': dates,
                'Value': values
            })
            data.set_index('Date', inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch raw data for {indicator}: {e}")
            return None

    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality for economic data."""
        if len(data) == 0:
            return {"quality": "poor", "issues": ["No data"]}

        issues = []
        quality_score = 100

        # Check for missing data
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(data)) * 100
            issues.append(f"Missing data: {missing_percentage:.1f}%")
            quality_score -= missing_percentage

        # Check for outliers
        if 'Value' in data.columns:
            Q1 = data['Value'].quantile(0.25)
            Q3 = data['Value'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data['Value'] < (Q1 - 1.5 * IQR)) |
                       (data['Value'] > (Q3 + 1.5 * IQR))).sum()

            if outliers > 0:
                outlier_percentage = (outliers / len(data)) * 100
                issues.append(f"Outliers: {outlier_percentage:.1f}%")
                quality_score -= outlier_percentage * 2

        # Determine quality level
        if quality_score >= 90:
            quality_level = "excellent"
        elif quality_score >= 70:
            quality_level = "good"
        elif quality_score >= 50:
            quality_level = "fair"
        else:
            quality_level = "poor"

        return {
            "quality": quality_level,
            "quality_score": quality_score,
            "issues": issues,
            "records": len(data)
        }

    async def update_revision_data(self,
                                  indicator: str,
                                  date: str,
                                  old_value: float,
                                  new_value: float,
                                  revision_type: str = "update") -> Dict[str, Any]:
        """Update revision tracking for an indicator."""
        self.revision_tracker.record_revision(
            indicator, date, old_value, new_value, revision_type
        )

        return {
            "status": "success",
            "indicator": indicator,
            "date": date,
            "old_value": old_value,
            "new_value": new_value,
            "revision_type": revision_type,
            "timestamp": datetime.now().isoformat()
        }

    async def get_lag_adjustments(self,
                                 indicators: List[str],
                                 as_of_date: str) -> Dict[str, Any]:
        """Get lag adjustments for multiple indicators."""
        adjustments = {}

        for indicator in indicators:
            adjustments[indicator] = self.lag_manager.get_lag_adjustment(
                indicator, as_of_date
            )

        return {
            "indicators": indicators,
            "as_of_date": as_of_date,
            "lag_adjustments": adjustments,
            "timestamp": datetime.now().isoformat()
        }

    async def detect_revision_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in revision patterns."""
        anomalies = self.revision_tracker.detect_anomaly_revisions()

        return {
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "detection_time": datetime.now().isoformat(),
            "threshold_used": 3.0
        }

    async def get_economic_dashboard(self,
                                   as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive economic data dashboard."""
        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")

        # Get all indicators
        indicators = list(self.indicators.keys())

        # Get recent data (last 90 days)
        end_date = as_of_date
        start_date = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

        # Fetch data
        data_result = await self.fetch_economic_data(
            indicators, start_date, end_date, as_of_date
        )

        # Get revision anomalies
        anomalies = await self.detect_revision_anomalies()

        # Get lag adjustments
        lag_adjustments = await self.get_lag_adjustments(indicators, as_of_date)

        return {
            "dashboard_date": as_of_date,
            "data_summary": data_result.get('summary', {}),
            "indicators": data_result.get('data', {}),
            "revision_anomalies": anomalies,
            "lag_adjustments": lag_adjustments,
            "system_health": {
                "active_indicators": len(indicators),
                "revision_tracking": "active",
                "lag_management": "active"
            }
        }