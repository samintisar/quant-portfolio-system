"""
Data quality monitoring with real-time anomaly detection.

Implements comprehensive data quality monitoring system including:
- Real-time data quality assessment
- Anomaly detection with multiple algorithms
- Quality trend analysis
- Automated alerting and reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from collections import deque
import statistics

# Statistical libraries
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class QualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies."""
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"


@dataclass
class QualityMetric:
    """Individual quality metric."""
    name: str
    value: float
    threshold: float
    is_critical: bool
    description: str


@dataclass
class AnomalyEvent:
    """Anomaly detection event."""
    timestamp: str
    metric_name: str
    value: float
    threshold: float
    severity: str
    anomaly_type: AnomalyType
    description: str
    context: Dict[str, Any]


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def detect(self, data: pd.DataFrame, metrics: List[str]) -> List[AnomalyEvent]:
        """Detect anomalies in data."""
        pass


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detection using z-score and IQR methods."""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.logger = logging.getLogger(__name__)

    def detect(self, data: pd.DataFrame, metrics: List[str]) -> List[AnomalyEvent]:
        """Detect statistical anomalies."""
        anomalies = []

        for metric in metrics:
            if metric not in data.columns:
                continue

            values = data[metric].dropna()
            if len(values) < 10:  # Need sufficient data
                continue

            # Z-score detection
            z_scores = np.abs(stats.zscore(values))
            z_anomalies = values[z_scores > self.z_threshold]

            # IQR detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            iqr_anomalies = values[(values < lower_bound) | (values > upper_bound)]

            # Combine anomalies
            all_anomaly_indices = set(z_anomalies.index) | set(iqr_anomalies.index)

            for idx in all_anomaly_indices:
                value = data.loc[idx, metric]
                z_score = z_scores[values.index.get_loc(idx)] if idx in values.index else 0

                anomalies.append(AnomalyEvent(
                    timestamp=data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else idx,
                    metric_name=metric,
                    value=value,
                    threshold=self.z_threshold,
                    severity="high" if z_score > 4 else "medium",
                    anomaly_type=AnomalyType.STATISTICAL,
                    description=f"Statistical anomaly in {metric}: z-score = {z_score:.2f}",
                    context={"z_score": z_score, "method": "statistical"}
                ))

        return anomalies


class TemporalAnomalyDetector(AnomalyDetector):
    """Temporal anomaly detection for time series data."""

    def __init__(self, window_size: int = 30, threshold_std: float = 2.5):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.logger = logging.getLogger(__name__)

    def detect(self, data: pd.DataFrame, metrics: List[str]) -> List[AnomalyEvent]:
        """Detect temporal anomalies using rolling statistics."""
        anomalies = []

        for metric in metrics:
            if metric not in data.columns:
                continue

            values = data[metric].dropna()
            if len(values) < self.window_size * 2:
                continue

            # Calculate rolling statistics
            rolling_mean = values.rolling(window=self.window_size).mean()
            rolling_std = values.rolling(window=self.window_size).std()

            # Detect anomalies where values deviate from rolling mean
            z_scores = (values - rolling_mean) / rolling_std
            temporal_anomalies = values[np.abs(z_scores) > self.threshold_std]

            for idx, value in temporal_anomalies.items():
                z_score = z_scores.loc[idx]

                anomalies.append(AnomalyEvent(
                    timestamp=data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else idx,
                    metric_name=metric,
                    value=value,
                    threshold=self.threshold_std,
                    severity="high" if abs(z_score) > 4 else "medium",
                    anomaly_type=AnomalyType.TEMPORAL,
                    description=f"Temporal anomaly in {metric}: deviation = {z_score:.2f}σ",
                    context={
                        "z_score": z_score,
                        "rolling_mean": rolling_mean.loc[idx],
                        "rolling_std": rolling_std.loc[idx],
                        "method": "temporal"
                    }
                ))

        return anomalies


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detection."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def detect(self, data: pd.DataFrame, metrics: List[str]) -> List[AnomalyEvent]:
        """Detect anomalies using Isolation Forest."""
        anomalies = []

        # Prepare data
        available_metrics = [m for m in metrics if m in data.columns]
        if not available_metrics:
            return anomalies

        X = data[available_metrics].dropna()
        if len(X) < 10:
            return anomalies

        # Fit and predict
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.detector.fit(X_scaled)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)

        anomaly_scores = self.detector.decision_function(X_scaled)
        predictions = self.detector.predict(X_scaled)

        # Identify anomalies
        anomaly_indices = X.index[predictions == -1]

        for idx in anomaly_indices:
            anomaly_score = anomaly_scores[X.index.get_loc(idx)]
            severity = "high" if anomaly_score < -0.2 else "medium"

            # Find the most anomalous metric
            values = data.loc[idx, available_metrics]
            if hasattr(values, 'idxmax'):
                most_anomalous_metric = values.abs().idxmax()
            else:
                most_anomalous_metric = available_metrics[0]

            anomalies.append(AnomalyEvent(
                timestamp=data.loc[idx, 'timestamp'] if 'timestamp' in data.columns else idx,
                metric_name=most_anomalous_metric,
                value=data.loc[idx, most_anomalous_metric],
                threshold=0.0,
                severity=severity,
                anomaly_type=AnomalyType.STATISTICAL,
                description=f"Isolation Forest anomaly in {most_anomalous_metric}: score = {anomaly_score:.3f}",
                context={
                    "anomaly_score": anomaly_score,
                    "all_scores": {metric: data.loc[idx, metric] for metric in available_metrics},
                    "method": "isolation_forest"
                }
            ))

        return anomalies


class DataQualityMonitor:
    """Main data quality monitoring service."""

    def __init__(self,
                 alert_threshold: float = 0.7,
                 history_size: int = 1000):
        self.alert_threshold = alert_threshold
        self.history_size = history_size
        self.detectors = [
            StatisticalAnomalyDetector(),
            TemporalAnomalyDetector(),
            IsolationForestDetector()
        ]
        self.quality_history = deque(maxlen=history_size)
        self.anomaly_history = deque(maxlen=history_size)
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.quality_thresholds = {
            'completeness': 0.95,  # 95% complete data
            'consistency': 0.90,  # 90% consistent
            'accuracy': 0.85,    # 85% accurate
            'timeliness': 0.80,  # 80% timely
            'validity': 0.90    # 90% valid
        }

    async def assess_data_quality(self,
                                data: pd.DataFrame,
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess overall data quality comprehensively.

        Returns:
            Dictionary containing quality metrics and analysis
        """
        try:
            assessment_time = datetime.now().isoformat()

            # Calculate individual quality metrics
            metrics = self._calculate_quality_metrics(data)

            # Detect anomalies
            anomalies = []
            for detector in self.detectors:
                detector_anomalies = detector.detect(data, list(metrics.keys()))
                anomalies.extend(detector_anomalies)

            # Calculate overall quality score
            overall_score = self._calculate_overall_score(metrics)

            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)

            # Store in history
            quality_record = {
                'timestamp': assessment_time,
                'metrics': metrics,
                'overall_score': overall_score,
                'quality_level': quality_level.value,
                'anomaly_count': len(anomalies)
            }
            self.quality_history.append(quality_record)

            # Store anomalies
            for anomaly in anomalies:
                self.anomaly_history.append(anomaly)

            # Generate alerts if needed
            alerts = self._generate_alerts(overall_score, anomalies, metrics)

            return {
                'assessment_time': assessment_time,
                'metrics': metrics,
                'overall_score': overall_score,
                'quality_level': quality_level.value,
                'anomalies': [self._anomaly_to_dict(a) for a in anomalies],
                'alerts': alerts,
                'trend_analysis': self._analyze_trends(),
                'metadata': metadata or {}
            }

        except Exception as e:
            self.logger.error(f"Failed to assess data quality: {e}")
            return {
                'assessment_time': datetime.now().isoformat(),
                'error': str(e),
                'overall_score': 0.0,
                'quality_level': QualityLevel.CRITICAL.value
            }

    def _calculate_quality_metrics(self, data: pd.DataFrame) -> Dict[str, QualityMetric]:
        """Calculate individual quality metrics."""
        metrics = {}

        # Completeness - missing data percentage
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
        metrics['completeness'] = QualityMetric(
            name="completeness",
            value=completeness,
            threshold=self.quality_thresholds['completeness'],
            is_critical=True,
            description="Percentage of non-missing data"
        )

        # Consistency - duplicate detection
        duplicate_rows = data.duplicated().sum()
        consistency = 1.0 - (duplicate_rows / len(data)) if len(data) > 0 else 1.0
        metrics['consistency'] = QualityMetric(
            name="consistency",
            value=consistency,
            threshold=self.quality_thresholds['consistency'],
            is_critical=True,
            description="Percentage of unique records"
        )

        # Timeliness - data freshness
        if hasattr(data.index, 'max'):
            latest_date = data.index.max()
            days_old = (datetime.now() - latest_date).days if isinstance(latest_date, datetime) else 0
            timeliness = max(0.0, 1.0 - days_old / 30.0)  # Decays over 30 days
        else:
            timeliness = 1.0
        metrics['timeliness'] = QualityMetric(
            name="timeliness",
            value=timeliness,
            threshold=self.quality_thresholds['timeliness'],
            is_critical=False,
            description="Data recency score"
        )

        # Validity - data type and range validation
        validity_score = self._calculate_validity_score(data)
        metrics['validity'] = QualityMetric(
            name="validity",
            value=validity_score,
            threshold=self.quality_thresholds['validity'],
            is_critical=True,
            description="Data type and range validation score"
        )

        # Accuracy - statistical consistency
        accuracy_score = self._calculate_accuracy_score(data)
        metrics['accuracy'] = QualityMetric(
            name="accuracy",
            value=accuracy_score,
            threshold=self.quality_thresholds['accuracy'],
            is_critical=True,
            description="Statistical consistency score"
        )

        return metrics

    def _calculate_validity_score(self, data: pd.DataFrame) -> float:
        """Calculate data validity score."""
        valid_records = 0
        total_records = len(data)

        if total_records == 0:
            return 0.0

        # Check numeric columns for reasonable ranges
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Remove infinite values
            finite_mask = np.isfinite(data[col])
            # Check for extreme outliers (more than 10 standard deviations)
            if finite_mask.any():
                z_scores = np.abs(stats.zscore(data[col][finite_mask]))
                valid_records += (z_scores <= 10).sum()

        return valid_records / (total_records * len(numeric_columns)) if numeric_columns else 1.0

    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate statistical accuracy score."""
        accuracy_components = []

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            values = data[col].dropna()
            if len(values) < 2:
                continue

            # Check for reasonable variance
            cv = values.std() / values.mean() if values.mean() != 0 else 0
            if 0.001 <= cv <= 1000:  # Reasonable coefficient of variation
                accuracy_components.append(1.0)
            else:
                accuracy_components.append(0.5)

            # Check for temporal consistency
            if len(values) > 10:
                changes = values.diff().dropna()
                if len(changes) > 0:
                    change_cv = changes.std() / changes.abs().mean() if changes.abs().mean() != 0 else 0
                    if change_cv <= 5.0:  # Reasonable change variability
                        accuracy_components.append(1.0)
                    else:
                        accuracy_components.append(0.7)

        return np.mean(accuracy_components) if accuracy_components else 1.0

    def _calculate_overall_score(self, metrics: Dict[str, QualityMetric]) -> float:
        """Calculate overall quality score."""
        weighted_scores = []
        total_weight = 0

        for metric in metrics.values():
            weight = 2.0 if metric.is_critical else 1.0
            score = metric.value / metric.threshold  # Normalized by threshold
            weighted_scores.append(score * weight)
            total_weight += weight

        return min(1.0, sum(weighted_scores) / total_weight) if total_weight > 0 else 0.0

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.FAIR
        elif score >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _generate_alerts(self,
                        overall_score: float,
                        anomalies: List[AnomalyEvent],
                        metrics: Dict[str, QualityMetric]) -> List[str]:
        """Generate quality alerts."""
        alerts = []

        # Overall quality alerts
        if overall_score < self.alert_threshold:
            alerts.append(f"Low overall quality score: {overall_score:.3f}")

        # Metric-specific alerts
        for metric in metrics.values():
            if metric.value < metric.threshold and metric.is_critical:
                alerts.append(f"Critical metric below threshold: {metric.name} = {metric.value:.3f}")

        # Anomaly alerts
        if anomalies:
            severity_counts = {}
            for anomaly in anomalies:
                severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1

            for severity, count in severity_counts.items():
                alerts.append(f"{count} {severity} severity anomalies detected")

        return alerts

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        if len(self.quality_history) < 10:
            return {"message": "Insufficient history for trend analysis"}

        recent_scores = [record['overall_score'] for record in list(self.quality_history)[-30:]]
        older_scores = [record['overall_score'] for record in list(self.quality_history)[-60:-30]]

        trend_analysis = {
            'current_score': recent_scores[-1] if recent_scores else 0,
            'trend_direction': 'stable',
            'trend_strength': 0.0,
            'volatility': np.std(recent_scores) if recent_scores else 0
        }

        if len(recent_scores) >= 10 and len(older_scores) >= 10:
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)

            if recent_avg > older_avg + 0.05:
                trend_analysis['trend_direction'] = 'improving'
            elif recent_avg < older_avg - 0.05:
                trend_analysis['trend_direction'] = 'declining'
            else:
                trend_analysis['trend_direction'] = 'stable'

            trend_analysis['trend_strength'] = abs(recent_avg - older_avg)

        return trend_analysis

    def _anomaly_to_dict(self, anomaly: AnomalyEvent) -> Dict[str, Any]:
        """Convert anomaly event to dictionary."""
        return {
            'timestamp': anomaly.timestamp,
            'metric_name': anomaly.metric_name,
            'value': anomaly.value,
            'threshold': anomaly.threshold,
            'severity': anomaly.severity,
            'anomaly_type': anomaly.anomaly_type.value,
            'description': anomaly.description,
            'context': anomaly.context
        }

    async def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive quality monitoring dashboard."""
        current_metrics = list(self.quality_history)[-1] if self.quality_history else None

        recent_anomalies = [self._anomaly_to_dict(a) for a in list(self.anomaly_history)[-100:]]

        return {
            'dashboard_time': datetime.now().isoformat(),
            'current_quality': current_metrics,
            'trend_analysis': self._analyze_trends(),
            'recent_anomalies': recent_anomalies,
            'anomaly_statistics': self._get_anomaly_statistics(),
            'system_health': {
                'monitoring_active': True,
                'detectors_active': len(self.detectors),
                'history_size': len(self.quality_history),
                'last_assessment': current_metrics['timestamp'] if current_metrics else None
            }
        }

    def _get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly statistics."""
        if not self.anomaly_history:
            return {"message": "No anomalies detected yet"}

        recent_anomalies = list(self.anomaly_history)[-1000:]  # Last 1000 anomalies

        type_counts = {}
        severity_counts = {}
        metric_counts = {}

        for anomaly in recent_anomalies:
            type_counts[anomaly.anomaly_type.value] = type_counts.get(anomaly.anomaly_type.value, 0) + 1
            severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
            metric_counts[anomaly.metric_name] = metric_counts.get(anomaly.metric_name, 0) + 1

        return {
            'total_anomalies': len(recent_anomalies),
            'by_type': type_counts,
            'by_severity': severity_counts,
            'by_metric': dict(sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'detection_methods': list(set(a.context.get('method', 'unknown') for a in recent_anomalies))
        }