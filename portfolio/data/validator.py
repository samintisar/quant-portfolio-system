"""
Data validation and cleaning utilities for portfolio optimization system.

Handles data quality checks, validation, and preprocessing.
Simple, clean implementation avoiding overengineering for resume projects.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from portfolio.logging_config import get_logger, ValidationError
from portfolio.config import get_config

logger = get_logger(__name__)


class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"      # Complete data, no issues
    GOOD = "good"              # Minor issues, acceptable
    FAIR = "fair"              # Some issues, usable with caution
    POOR = "poor"              # Significant issues, not recommended
    UNUSABLE = "unusable"      # Too many issues, cannot use


class DataValidator:
    """
    Data validation and cleaning for financial data.

    Simple implementation with comprehensive validation rules.
    """

    def __init__(self):
        """Initialize the data validator."""
        self.config = get_config()
        self.quality_thresholds = {
            'max_missing_pct': self.config.data.max_missing_pct,
            'min_data_points': self.config.data.min_data_points,
            'max_outlier_pct': 0.05  # 5% outliers threshold
        }

        logger.info("Initialized DataValidator")

    def validate_asset_data(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate asset data for quality and completeness.

        Args:
            data: DataFrame with price data
            symbol: Asset symbol for logging

        Returns:
            Validation report dictionary
        """
        if data.empty:
            return {
                'symbol': symbol,
                'is_valid': False,
                'quality_level': DataQualityLevel.UNUSABLE,
                'issues': ['Empty dataset'],
                'recommendations': ['No data available'],
                'data_points': 0,
                'missing_percentage': 100.0
            }

        validation_report = {
            'symbol': symbol,
            'data_points': len(data),
            'date_range': {
                'start': data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else None,
                'end': data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else None
            },
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'quality_metrics': {}
        }

        # Check required columns
        self._validate_columns(data, validation_report)

        # Check data completeness
        self._validate_completeness(data, validation_report)

        # Check data quality
        self._validate_data_quality(data, validation_report)

        # Check for outliers
        self._validate_outliers(data, validation_report)

        # Check date consistency
        self._validate_dates(data, validation_report)

        # Determine overall quality
        self._assess_overall_quality(validation_report)

        return validation_report

    def _validate_columns(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate required columns exist."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            report['issues'].append(f"Missing required columns: {missing_columns}")
            report['recommendations'].append("Ensure all OHLCV columns are present")

        # Check column data types
        for col in data.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    report['issues'].append(f"Column {col} should be numeric")
                    report['recommendations'].append(f"Convert {col} to numeric type")

    def _validate_completeness(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate data completeness."""
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 100

        report['quality_metrics']['missing_percentage'] = missing_percentage

        if missing_percentage > 0:
            report['warnings'].append(f"Missing data: {missing_percentage:.1f}%")

        if missing_percentage > self.quality_thresholds['max_missing_pct'] * 100:
            report['issues'].append(f"Too much missing data: {missing_percentage:.1f}%")
            report['recommendations'].append("Consider using a different data source or time period")

    def _validate_data_quality(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate data quality and consistency."""
        if data.empty:
            return

        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                negative_prices = (data[col] < 0).sum()
                if negative_prices > 0:
                    report['issues'].append(f"Negative {col} prices: {negative_prices} occurrences")
                    report['recommendations'].append(f"Remove or correct negative {col} values")

        # Check for zero prices
        for col in price_columns:
            if col in data.columns:
                zero_prices = (data[col] == 0).sum()
                if zero_prices > 0:
                    report['warnings'].append(f"Zero {col} prices: {zero_prices} occurrences")

        # Check volume consistency
        if 'volume' in data.columns:
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                report['issues'].append(f"Negative volume: {negative_volume} occurrences")

        # Check OHLC consistency
        self._validate_ohlc_consistency(data, report)

    def _validate_ohlc_consistency(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate OHLC price consistency."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return

        # High should be >= Low
        invalid_hl = (data['high'] < data['low']).sum()
        if invalid_hl > 0:
            report['issues'].append(f"High < Low in {invalid_hl} rows")
            report['recommendations'].append("Correct or remove invalid OHLC data")

        # High should be >= Open and Close
        invalid_high = ((data['high'] < data['open']) | (data['high'] < data['close'])).sum()
        if invalid_high > 0:
            report['warnings'].append(f"High < Open/Close in {invalid_high} rows")

        # Low should be <= Open and Close
        invalid_low = ((data['low'] > data['open']) | (data['low'] > data['close'])).sum()
        if invalid_low > 0:
            report['warnings'].append(f"Low > Open/Close in {invalid_low} rows")

    def _validate_outliers(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate for extreme outliers."""
        if len(data) < 10:
            return  # Not enough data for outlier detection

        # Check for extreme price changes
        if 'close' in data.columns:
            returns = data['close'].pct_change().abs()
            extreme_changes = returns[returns > 0.5]  # >50% daily change

            if len(extreme_changes) > 0:
                outlier_pct = (len(extreme_changes) / len(data)) * 100
                report['quality_metrics']['extreme_changes_pct'] = outlier_pct

                if outlier_pct > self.quality_thresholds['max_outlier_pct'] * 100:
                    report['warnings'].append(f"Extreme price changes: {outlier_pct:.1f}% of data")
                    report['recommendations'].append("Review and possibly remove extreme outliers")

        # Check for price outliers using IQR method
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                outliers = self._detect_outliers_iqr(data[col])
                outlier_pct = (len(outliers) / len(data)) * 100

                if outlier_pct > 5:  # More than 5% outliers
                    report['warnings'].append(f"Potential outliers in {col}: {outlier_pct:.1f}% of data")

    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        if len(series) < 4:
            return pd.Series([False] * len(series), index=series.index)

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return (series < lower_bound) | (series > upper_bound)

    def _validate_dates(self, data: pd.DataFrame, report: Dict[str, Any]) -> None:
        """Validate date consistency and coverage."""
        if not isinstance(data.index, pd.DatetimeIndex):
            report['issues'].append("Index is not datetime")
            report['recommendations'].append("Convert index to datetime")
            return

        # Check for duplicate dates
        duplicate_dates = data.index.duplicated().sum()
        if duplicate_dates > 0:
            report['warnings'].append(f"Duplicate dates: {duplicate_dates} occurrences")
            report['recommendations'].append("Remove duplicate dates")

        # Check date range
        if len(data) > 1:
            date_span = data.index[-1] - data.index[0]
            expected_days = (date_span.days + 1) * 0.7  # Rough estimate considering weekends

            if len(data) < expected_days * 0.5:  # Less than 50% expected data points
                report['warnings'].append("Sparse data - many dates missing")

        # Check data frequency
        if len(data) > 2:
            days_diff = (data.index[1:] - data.index[:-1])
            median_diff = days_diff.median()

            if median_diff == timedelta(days=1):
                expected_freq = "Daily"
            elif median_diff == timedelta(days=7):
                expected_freq = "Weekly"
            elif median_diff >= timedelta(days=28) and median_diff <= timedelta(days=31):
                expected_freq = "Monthly"
            else:
                expected_freq = "Irregular"

            report['quality_metrics']['frequency'] = expected_freq

    def _assess_overall_quality(self, report: Dict[str, Any]) -> None:
        """Assess overall data quality level."""
        issues_count = len(report['issues'])
        warnings_count = len(report['warnings'])
        missing_pct = report['quality_metrics'].get('missing_percentage', 0)
        data_points = report['data_points']

        # Determine quality level
        if issues_count > 0 or missing_pct > 50 or data_points < self.quality_thresholds['min_data_points']:
            report['quality_level'] = DataQualityLevel.UNUSABLE
            report['is_valid'] = False
        elif issues_count == 0 and warnings_count <= 2 and missing_pct <= 5:
            report['quality_level'] = DataQualityLevel.EXCELLENT
            report['is_valid'] = True
        elif issues_count == 0 and warnings_count <= 5 and missing_pct <= 10:
            report['quality_level'] = DataQualityLevel.GOOD
            report['is_valid'] = True
        elif issues_count == 0 and missing_pct <= 20:
            report['quality_level'] = DataQualityLevel.FAIR
            report['is_valid'] = True
        else:
            report['quality_level'] = DataQualityLevel.POOR
            report['is_valid'] = False

    def clean_data(self, data: pd.DataFrame,
                   remove_outliers: bool = True,
                   fill_missing: str = 'forward') -> pd.DataFrame:
        """
        Clean and preprocess data.

        Args:
            data: DataFrame to clean
            remove_outliers: Whether to remove extreme outliers
            fill_missing: Method to fill missing data ('forward', 'backward', 'drop', 'none')

        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data

        cleaned_data = data.copy()

        try:
            # Remove duplicate dates
            cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='first')]

            # Sort by date
            cleaned_data = cleaned_data.sort_index()

            # Handle missing data
            cleaned_data = self._handle_missing_data(cleaned_data, fill_missing)

            # Remove invalid rows
            cleaned_data = self._remove_invalid_rows(cleaned_data)

            # Remove outliers if requested
            if remove_outliers:
                cleaned_data = self._remove_outliers(cleaned_data)

            # Forward fill remaining missing values
            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')

            logger.info(f"Cleaned data: {len(data)} -> {len(cleaned_data)} rows")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data

    def _handle_missing_data(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle missing data based on specified method."""
        if method == 'forward':
            return data.fillna(method='ffill')
        elif method == 'backward':
            return data.fillna(method='bfill')
        elif method == 'drop':
            return data.dropna()
        elif method == 'none':
            return data
        else:
            logger.warning(f"Unknown missing data method: {method}")
            return data

    def _remove_invalid_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid data."""
        if data.empty:
            return data

        original_len = len(data)

        # Remove negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] >= 0]

        # Remove negative volume
        if 'volume' in data.columns:
            data = data[data['volume'] >= 0]

        # Fix OHLC inconsistencies
        if all(col in data.columns for col in ['high', 'low']):
            # Ensure high >= low
            data = data[data['high'] >= data['low']]

        if all(col in data.columns for col in ['high', 'open', 'close']):
            # Ensure high >= max(open, close)
            data = data[data['high'] >= data[['open', 'close']].max(axis=1)]

        if all(col in data.columns for col in ['low', 'open', 'close']):
            # Ensure low <= min(open, close)
            data = data[data['low'] <= data[['open', 'close']].min(axis=1)]

        removed_rows = original_len - len(data)
        if removed_rows > 0:
            logger.debug(f"Removed {removed_rows} invalid rows")

        return data

    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove extreme outliers using z-score method."""
        if len(data) < 10:
            return data

        original_len = len(data)
        price_columns = ['open', 'high', 'low', 'close']

        for col in price_columns:
            if col in data.columns:
                # Calculate z-scores
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())

                # Remove extreme outliers (z > threshold)
                data = data[z_scores <= threshold]

        removed_rows = original_len - len(data)
        if removed_rows > 0:
            logger.debug(f"Removed {removed_rows} outlier rows")

        return data

    def validate_multiple_assets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple assets' data.

        Args:
            data_dict: Dictionary of symbol -> DataFrame

        Returns:
            Dictionary of validation reports
        """
        reports = {}

        for symbol, data in data_dict.items():
            reports[symbol] = self.validate_asset_data(data, symbol)

        # Add summary
        valid_count = sum(1 for report in reports.values() if report.get('is_valid', False))
        total_count = len(reports)

        logger.info(f"Validated {total_count} assets: {valid_count} valid, {total_count - valid_count} invalid")

        return reports

    def get_data_quality_summary(self, reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of data quality across all assets.

        Args:
            reports: Dictionary of validation reports

        Returns:
            Quality summary
        """
        if not reports:
            return {}

        quality_counts = {}
        total_issues = 0
        total_warnings = 0
        total_data_points = 0

        for report in reports.values():
            quality_level = report.get('quality_level', DataQualityLevel.UNUSABLE)
            quality_counts[quality_level.value] = quality_counts.get(quality_level.value, 0) + 1

            total_issues += len(report.get('issues', []))
            total_warnings += len(report.get('warnings', []))
            total_data_points += report.get('data_points', 0)

        summary = {
            'total_assets': len(reports),
            'quality_distribution': quality_counts,
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'average_data_points': total_data_points / len(reports) if reports else 0,
            'validation_timestamp': datetime.now().isoformat()
        }

        return summary

    def __str__(self) -> str:
        """String representation."""
        return f"DataValidator(thresholds={self.quality_thresholds})"