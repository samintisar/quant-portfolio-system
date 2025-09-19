"""
Quality Metrics Calculation Service

Calculates and analyzes data quality metrics for financial datasets,
including completeness, consistency, accuracy, and timeliness measures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..models.quality_metrics import QualityMetrics, QualityReport, MetricType, MetricStatus
from ..models.processing_log import ProcessingLog, ProcessingSession


class QualityService:
    """Service for calculating and managing data quality metrics."""

    def __init__(self, max_workers: int = 4):
        """Initialize the quality service.

        Args:
            max_workers: Maximum number of workers for parallel processing
        """
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.metrics_cache = {}

    def calculate_completeness(self, df: pd.DataFrame, dataset_id: str,
                              metric_id: Optional[str] = None) -> QualityMetrics:
        """Calculate completeness metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"completeness_{dataset_id}_{int(time.time())}"

        # Calculate overall completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

        # Calculate column-wise completeness
        column_completeness = {}
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            col_total = len(df)
            col_completeness[col] = 1.0 - (col_missing / col_total) if col_total > 0 else 0.0

        # Calculate row-wise completeness
        row_completeness = 1.0 - (df.isnull().sum(axis=1) / df.shape[1])

        metadata = {
            'column_completeness': column_completeness,
            'row_completeness_mean': row_completeness.mean(),
            'row_completeness_std': row_completeness.std(),
            'missing_cells': int(missing_cells),
            'total_cells': int(total_cells)
        }

        return QualityMetrics.create_completeness_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            completeness_ratio=completeness_ratio,
            threshold=0.95,
            metadata=metadata
        )

    def calculate_consistency(self, df: pd.DataFrame, dataset_id: str,
                             metric_id: Optional[str] = None) -> QualityMetrics:
        """Calculate consistency metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"consistency_{dataset_id}_{int(time.time())}"

        consistency_issues = 0
        total_checks = 0
        consistency_details = {}

        # Check OHLC consistency
        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_columns):
            total_checks += 4

            # High >= Low
            invalid_hl = (df['high'] < df['low']).sum()
            consistency_issues += invalid_hl
            consistency_details['high_low_inconsistencies'] = int(invalid_hl)

            # High >= Open, High >= Close
            invalid_ho = (df['high'] < df['open']).sum()
            invalid_hc = (df['high'] < df['close']).sum()
            consistency_issues += invalid_ho + invalid_hc
            consistency_details['high_open_inconsistencies'] = int(invalid_ho)
            consistency_details['high_close_inconsistencies'] = int(invalid_hc)

            # Low <= Open, Low <= Close
            invalid_lo = (df['low'] > df['open']).sum()
            invalid_lc = (df['low'] > df['close']).sum()
            consistency_issues += invalid_lo + invalid_lc
            consistency_details['low_open_inconsistencies'] = int(invalid_lo)
            consistency_details['low_close_inconsistencies'] = int(invalid_lc)

        # Check volume consistency (if volume column exists)
        if 'volume' in df.columns:
            total_checks += 1
            negative_volume = (df['volume'] < 0).sum()
            consistency_issues += negative_volume
            consistency_details['negative_volume'] = int(negative_volume)

        # Check timestamp consistency (if timestamp column exists)
        if 'timestamp' in df.columns:
            total_checks += 2
            # Check for duplicate timestamps
            duplicates = df['timestamp'].duplicated().sum()
            consistency_issues += duplicates
            consistency_details['duplicate_timestamps'] = int(duplicates)

            # Check chronological order
            if not df['timestamp'].is_monotonic_increasing:
                consistency_issues += 1
                consistency_details['not_chronological'] = True

        # Calculate consistency score
        consistency_score = 1.0 - (consistency_issues / max(total_checks * len(df), 1))

        metadata = {
            'total_issues': int(consistency_issues),
            'total_checks': total_checks,
            'details': consistency_details,
            'check_types': ['OHLC', 'volume', 'timestamp']
        }

        return QualityMetrics.create_consistency_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            consistency_score=consistency_score,
            threshold=0.90,
            metadata=metadata
        )

    def calculate_accuracy(self, df: pd.DataFrame, dataset_id: str,
                         metric_id: Optional[str] = None) -> QualityMetrics:
        """Calculate accuracy metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"accuracy_{dataset_id}_{int(time.time())}"

        accuracy_issues = 0
        total_checks = 0
        accuracy_details = {}

        # Check price accuracy
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                total_checks += 1
                col_data = df[col].dropna()

                if len(col_data) > 0:
                    # Check for negative prices
                    negative_prices = (col_data < 0).sum()
                    accuracy_issues += negative_prices
                    accuracy_details[f'{col}_negative_prices'] = int(negative_prices)

                    # Check for zero prices (suspicious for most financial data)
                    zero_prices = (col_data == 0).sum()
                    accuracy_issues += zero_prices * 0.5  # Partial penalty
                    accuracy_details[f'{col}_zero_prices'] = int(zero_prices)

                    # Check for extreme values (potential data errors)
                    q99 = col_data.quantile(0.99)
                    extreme_values = (col_data > q99 * 10).sum()
                    accuracy_issues += extreme_values
                    accuracy_details[f'{col}_extreme_values'] = int(extreme_values)

        # Check volume accuracy
        if 'volume' in df.columns:
            total_checks += 1
            volume_data = df['volume'].dropna()

            if len(volume_data) > 0:
                # Check for negative volume
                negative_volume = (volume_data < 0).sum()
                accuracy_issues += negative_volume
                accuracy_details['volume_negative'] = int(negative_volume)

                # Check for extreme volume
                q99_volume = volume_data.quantile(0.99)
                extreme_volume = (volume_data > q99_volume * 100).sum()
                accuracy_issues += extreme_volume
                accuracy_details['volume_extreme'] = int(extreme_volume)

        # Calculate accuracy score
        max_possible_issues = total_checks * len(df)
        accuracy_score = 1.0 - (accuracy_issues / max(max_possible_issues, 1))

        metadata = {
            'total_issues': int(accuracy_issues),
            'total_checks': total_checks,
            'details': accuracy_details,
            'check_types': ['price_accuracy', 'volume_accuracy']
        }

        return QualityMetrics.create_accuracy_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            accuracy_score=accuracy_score,
            threshold=0.85,
            metadata=metadata
        )

    def calculate_timeliness(self, df: pd.DataFrame, dataset_id: str,
                           metric_id: Optional[str] = None,
                           current_time: Optional[datetime] = None) -> QualityMetrics:
        """Calculate timeliness metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier
            current_time: Current time reference (defaults to now)

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"timeliness_{dataset_id}_{int(time.time())}"

        if current_time is None:
            current_time = datetime.now()

        latency_seconds = 0.0
        timeliness_details = {}

        if 'timestamp' in df.columns:
            # Convert timestamp column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate data latency
            latest_timestamp = df['timestamp'].max()
            # Ensure both are datetime objects for subtraction
            if isinstance(latest_timestamp, pd.Timestamp):
                latest_timestamp = latest_timestamp.to_pydatetime()
            latency = current_time - latest_timestamp
            latency_seconds = latency.total_seconds()

            timeliness_details = {
                'latest_timestamp': latest_timestamp.isoformat(),
                'current_timestamp': current_time.isoformat(),
                'latency_days': latency.days,
                'latency_hours': latency_seconds / 3600,
                'data_frequency': self._estimate_data_frequency(df['timestamp'])
            }

        metadata = {
            'latency_seconds': latency_seconds,
            'details': timeliness_details,
            'current_time': current_time.isoformat()
        }

        return QualityMetrics.create_timeliness_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            latency_seconds=latency_seconds,
            threshold=300.0,  # 5 minutes
            metadata=metadata
        )

    def calculate_uniqueness(self, df: pd.DataFrame, dataset_id: str,
                            metric_id: Optional[str] = None) -> QualityMetrics:
        """Calculate uniqueness metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"uniqueness_{dataset_id}_{int(time.time())}"

        uniqueness_details = {}
        total_duplicates = 0

        # Check for completely duplicate rows
        duplicate_rows = df.duplicated().sum()
        total_duplicates += duplicate_rows
        uniqueness_details['duplicate_rows'] = int(duplicate_rows)

        # Check for duplicate timestamps (if timestamp column exists)
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            total_duplicates += duplicate_timestamps
            uniqueness_details['duplicate_timestamps'] = int(duplicate_timestamps)

        # Check for duplicate combinations of key columns
        key_columns = ['timestamp', 'symbol'] if 'symbol' in df.columns else ['timestamp']
        if all(col in df.columns for col in key_columns):
            duplicate_keys = df.duplicated(subset=key_columns).sum()
            total_duplicates += duplicate_keys
            uniqueness_details[f'duplicate_{"_".join(key_columns)}'] = int(duplicate_keys)

        # Calculate uniqueness score
        total_records = len(df)
        uniqueness_ratio = 1.0 - (total_duplicates / max(total_records, 1))

        metadata = {
            'total_duplicates': int(total_duplicates),
            'total_records': int(total_records),
            'details': uniqueness_details
        }

        return QualityMetrics.create_uniqueness_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            uniqueness_ratio=uniqueness_ratio,
            threshold=0.95,
            metadata=metadata
        )

    def calculate_validity(self, df: pd.DataFrame, dataset_id: str,
                          metric_id: Optional[str] = None) -> QualityMetrics:
        """Calculate validity metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            metric_id: Optional metric identifier

        Returns:
            QualityMetrics object
        """
        if metric_id is None:
            metric_id = f"validity_{dataset_id}_{int(time.time())}"

        validity_issues = 0
        total_checks = 0
        validity_details = {}

        # Check data types
        for col in df.columns:
            total_checks += 1
            col_data = df[col]

            # Check for mixed data types in columns
            if col_data.apply(type).nunique() > 1:
                validity_issues += 1
                validity_details[f'{col}_mixed_types'] = True

        # Check for specific validity rules
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            total_checks += 1
            col_data = df[col].dropna()

            if len(col_data) > 0:
                # Check for infinite values
                infinite_values = np.isinf(col_data).sum()
                validity_issues += infinite_values
                validity_details[f'{col}_infinite_values'] = int(infinite_values)

        # Check timestamp validity
        if 'timestamp' in df.columns:
            total_checks += 1
            try:
                pd.to_datetime(df['timestamp'])
            except:
                validity_issues += 1
                validity_details['timestamp_invalid'] = True

        # Calculate validity score
        validity_score = 1.0 - (validity_issues / max(total_checks, 1))

        metadata = {
            'total_issues': int(validity_issues),
            'total_checks': total_checks,
            'details': validity_details
        }

        return QualityMetrics.create_validity_metric(
            metric_id=metric_id,
            dataset_id=dataset_id,
            validity_ratio=validity_score,
            threshold=0.90,
            metadata=metadata
        )

    def calculate_all_metrics(self, df: pd.DataFrame, dataset_id: str,
                             current_time: Optional[datetime] = None) -> QualityReport:
        """Calculate all quality metrics for the dataset.

        Args:
            df: Input DataFrame
            dataset_id: Dataset identifier
            current_time: Current time reference

        Returns:
            QualityReport object
        """
        metrics = []

        # Calculate metrics in parallel for better performance
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.calculate_completeness, df, dataset_id): 'completeness',
                executor.submit(self.calculate_consistency, df, dataset_id): 'consistency',
                executor.submit(self.calculate_accuracy, df, dataset_id): 'accuracy',
                executor.submit(self.calculate_timeliness, df, dataset_id, None, current_time): 'timeliness',
                executor.submit(self.calculate_uniqueness, df, dataset_id): 'uniqueness',
                executor.submit(self.calculate_validity, df, dataset_id): 'validity'
            }

            for future in as_completed(futures):
                try:
                    metric = future.result()
                    metrics.append(metric)
                except Exception as e:
                    metric_type = futures[future]
                    self.logger.error(f"Failed to calculate {metric_type} metric: {e}")

        # Calculate overall score
        if metrics:
            overall_score = np.mean([metric.value for metric in metrics])
        else:
            overall_score = 0.0

        # Cache metrics
        self.metrics_cache[dataset_id] = metrics

        return QualityReport(
            dataset_id=dataset_id,
            overall_score=overall_score,
            metrics=metrics
        )

    def get_quality_trends(self, dataset_id: str, metrics_history: List[QualityMetrics],
                          metric_type: str) -> Dict[str, Any]:
        """Analyze quality trends over time.

        Args:
            dataset_id: Dataset identifier
            metrics_history: List of historical metrics
            metric_type: Type of metric to analyze

        Returns:
            Trend analysis results
        """
        # Filter metrics by type
        filtered_metrics = [m for m in metrics_history if m.metric_type == metric_type]

        if len(filtered_metrics) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        # Extract values and timestamps
        values = [m.value for m in filtered_metrics]
        timestamps = [m.timestamp for m in filtered_metrics]

        # Calculate trend
        trend_direction = "stable"
        if len(values) >= 3:
            recent_trend = values[-1] - values[-3]
            if recent_trend > 0.05:
                trend_direction = "improving"
            elif recent_trend < -0.05:
                trend_direction = "declining"

        # Calculate statistics
        trend_analysis = {
            'dataset_id': dataset_id,
            'metric_type': metric_type,
            'current_value': values[-1],
            'previous_value': values[-2] if len(values) > 1 else None,
            'change': values[-1] - values[-2] if len(values) > 1 else 0,
            'change_percentage': ((values[-1] - values[-2]) / values[-2] * 100) if len(values) > 1 and values[-2] != 0 else 0,
            'trend_direction': trend_direction,
            'min_value': min(values),
            'max_value': max(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'data_points': len(values),
            'period_start': min(timestamps).isoformat(),
            'period_end': max(timestamps).isoformat()
        }

        return trend_analysis

    def _estimate_data_frequency(self, timestamp_series: pd.Series) -> str:
        """Estimate the frequency of time series data.

        Args:
            timestamp_series: Series of timestamps

        Returns:
            Estimated frequency string
        """
        if len(timestamp_series) < 2:
            return "unknown"

        # Sort timestamps
        sorted_timestamps = timestamp_series.sort_values()

        # Calculate most common time difference
        time_diffs = sorted_timestamps.diff().dropna()

        if len(time_diffs) == 0:
            return "unknown"

        # Convert to seconds
        time_diffs_seconds = time_diffs.dt.total_seconds()

        # Find the most common difference
        mode_diff = time_diffs_seconds.mode().iloc[0]

        # Map to frequency strings
        if mode_diff < 60:
            return "1m"
        elif mode_diff < 300:
            return "5m"
        elif mode_diff < 900:
            return "15m"
        elif mode_diff < 1800:
            return "30m"
        elif mode_diff < 3600:
            return "1h"
        elif mode_diff < 14400:
            return "4h"
        elif mode_diff < 86400:
            return "1D"
        elif mode_diff < 604800:
            return "1W"
        else:
            return "1M"

    def get_quality_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Get quality summary for a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Quality summary dictionary
        """
        if dataset_id not in self.metrics_cache:
            return {'error': 'No quality metrics found for dataset'}

        metrics = self.metrics_cache[dataset_id]

        summary = {
            'dataset_id': dataset_id,
            'metrics_count': len(metrics),
            'metrics_by_type': {},
            'overall_status': 'healthy',
            'issues': []
        }

        # Group metrics by type
        for metric in metrics:
            if metric.metric_type not in summary['metrics_by_type']:
                summary['metrics_by_type'][metric.metric_type] = []

            summary['metrics_by_type'][metric.metric_type].append({
                'value': metric.value,
                'status': metric.status,
                'timestamp': metric.timestamp.isoformat()
            })

        # Determine overall status
        failing_metrics = [m for m in metrics if m.is_failing()]
        warning_metrics = [m for m in metrics if m.is_warning()]

        if failing_metrics:
            summary['overall_status'] = 'failing'
            summary['issues'] = [f"{m.metric_type}: {m.value:.3f} (threshold: {m.threshold:.3f})" for m in failing_metrics]
        elif warning_metrics:
            summary['overall_status'] = 'warning'
            summary['issues'] = [f"{m.metric_type}: {m.value:.3f} (threshold: {m.threshold:.3f})" for m in warning_metrics]

        return summary

    def clear_cache(self, dataset_id: Optional[str] = None):
        """Clear quality metrics cache.

        Args:
            dataset_id: Specific dataset to clear, or None for all
        """
        if dataset_id:
            self.metrics_cache.pop(dataset_id, None)
        else:
            self.metrics_cache.clear()

    def export_quality_report(self, report: QualityReport, output_path: str,
                             format: str = 'json') -> str:
        """Export quality report to file.

        Args:
            report: QualityReport to export
            output_path: Output file path
            format: Export format ('json', 'csv')

        Returns:
            Path to exported file
        """
        if format.lower() == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Convert metrics to DataFrame
            metrics_data = []
            for metric in report.metrics:
                metrics_data.append({
                    'metric_id': metric.metric_id,
                    'dataset_id': metric.dataset_id,
                    'metric_type': metric.metric_type,
                    'value': metric.value,
                    'threshold': metric.threshold,
                    'status': metric.status,
                    'timestamp': metric.timestamp.isoformat()
                })

            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return output_path