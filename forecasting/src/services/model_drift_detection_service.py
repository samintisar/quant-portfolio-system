"""
Model drift detection and automated retraining triggers.

Implements comprehensive model drift detection system including:
- Real-time model performance monitoring
- Statistical drift detection algorithms
- Concept drift identification
- Automated retraining triggers
- Model health assessment and reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import json
from collections import deque, defaultdict
import joblib
from pathlib import Path
import time

# Statistical libraries
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


class DriftType(Enum):
    """Types of model drift."""
    PERFORMANCE_DRIFT = "performance_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetrainingTrigger(Enum):
    """Retraining trigger types."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PERFORMANCE_BASED = "performance_based"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    model_id: str
    detection_timestamp: str
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    confidence: float
    affected_features: List[str]
    performance_metrics: Dict[str, float]
    baseline_comparison: Dict[str, float]
    recommendations: List[str]
    retraining_triggered: bool
    retraining_reason: Optional[str]


@dataclass
class ModelHealthStatus:
    """Comprehensive model health status."""
    model_id: str
    health_score: float
    overall_status: str
    drift_status: Dict[str, DriftSeverity]
    performance_status: Dict[str, float]
    last_updated: str
    issues_detected: List[str]
    recommended_actions: List[str]


@dataclass
class RetrainingEvent:
    """Model retraining event."""
    model_id: str
    retraining_timestamp: str
    trigger_type: RetrainingTrigger
    trigger_reason: str
    retraining_successful: bool
    performance_improvement: Optional[float]
    training_duration_seconds: float
    new_model_version: str


class DriftDetector(ABC):
    """Abstract base class for drift detectors."""

    @abstractmethod
    def detect_drift(self,
                    reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    model_id: str) -> DriftDetectionResult:
        """Detect drift in model data."""
        pass


class PerformanceDriftDetector(DriftDetector):
    """Detect performance-based drift using model metrics."""

    def __init__(self, performance_threshold: float = 0.1, significance_level: float = 0.05):
        self.performance_threshold = performance_threshold
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)

    def detect_drift(self,
                    reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    model_id: str) -> DriftDetectionResult:
        """Detect performance drift in model metrics."""
        try:
            # Calculate performance metrics for both datasets
            ref_metrics = self._calculate_performance_metrics(reference_data)
            current_metrics = self._calculate_performance_metrics(current_data)

            if not ref_metrics or not current_metrics:
                return DriftDetectionResult(
                    model_id=model_id,
                    detection_timestamp=datetime.now().isoformat(),
                    drift_type=DriftType.PERFORMANCE_DRIFT,
                    severity=DriftSeverity.NONE,
                    drift_score=0.0,
                    confidence=0.0,
                    affected_features=[],
                    performance_metrics={},
                    baseline_comparison={},
                    recommendations=["Insufficient data for drift detection"],
                    retraining_triggered=False,
                    retraining_reason=None
                )

            # Calculate drift score
            drift_score, severity = self._calculate_drift_score(ref_metrics, current_metrics)

            # Statistical significance test
            confidence = self._test_significance(ref_metrics, current_metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(severity, drift_score)

            # Determine if retraining should be triggered
            retraining_triggered = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            retraining_reason = f"Performance drift detected with severity {severity.value}" if retraining_triggered else None

            return DriftDetectionResult(
                model_id=model_id,
                detection_timestamp=datetime.now().isoformat(),
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=severity,
                drift_score=drift_score,
                confidence=confidence,
                affected_features=list(current_metrics.keys()),
                performance_metrics=current_metrics,
                baseline_comparison=ref_metrics,
                recommendations=recommendations,
                retraining_triggered=retraining_triggered,
                retraining_reason=retraining_reason
            )

        except Exception as e:
            self.logger.error(f"Performance drift detection failed for model {model_id}: {e}")
            return self._create_error_result(model_id, DriftType.PERFORMANCE_DRIFT, str(e))

    def _calculate_performance_metrics(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate performance metrics from data."""
        try:
            # This would typically use actual model predictions
            # For now, simulate metrics based on data characteristics
            if len(data) < 10:
                return None

            metrics = {}
            if 'predictions' in data.columns and 'actual' in data.columns:
                metrics['accuracy'] = accuracy_score(data['actual'], data['predictions'])
                metrics['precision'] = precision_score(data['actual'], data['predictions'], average='weighted')
                metrics['recall'] = recall_score(data['actual'], data['predictions'], average='weighted')
                metrics['f1'] = f1_score(data['actual'], data['predictions'], average='weighted')
            else:
                # Simulate metrics based on data variance
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data_variance = data[numeric_cols].var().mean()
                    metrics['simulated_accuracy'] = max(0.5, min(0.95, 1.0 - data_variance * 0.1))
                    metrics['simulated_error_rate'] = 1.0 - metrics['simulated_accuracy']

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return None

    def _calculate_drift_score(self,
                              ref_metrics: Dict[str, float],
                              current_metrics: Dict[str, float]) -> Tuple[float, DriftSeverity]:
        """Calculate drift score and severity."""
        drift_scores = []

        for metric_name in ref_metrics:
            if metric_name in current_metrics:
                ref_value = ref_metrics[metric_name]
                current_value = current_metrics[metric_name]

                if ref_value != 0:
                    relative_change = abs(current_value - ref_value) / abs(ref_value)
                    drift_scores.append(relative_change)

        if not drift_scores:
            return 0.0, DriftSeverity.NONE

        avg_drift = np.mean(drift_scores)

        # Determine severity
        if avg_drift > 0.3:
            severity = DriftSeverity.CRITICAL
        elif avg_drift > 0.2:
            severity = DriftSeverity.HIGH
        elif avg_drift > 0.1:
            severity = DriftSeverity.MEDIUM
        elif avg_drift > 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE

        return avg_drift, severity

    def _test_significance(self,
                          ref_metrics: Dict[str, float],
                          current_metrics: Dict[str, float]) -> float:
        """Test statistical significance of drift."""
        # For simplicity, use drift score as confidence
        # In real implementation, this would use proper statistical tests
        drift_score = np.mean([
            abs(current_metrics.get(k, 0) - v) / abs(v) if v != 0 else 0
            for k, v in ref_metrics.items()
            if k in current_metrics
        ])

        return min(1.0, drift_score * 3)  # Convert to confidence score

    def _generate_recommendations(self, severity: DriftSeverity, drift_score: float) -> List[str]:
        """Generate recommendations based on drift severity."""
        recommendations = ["Continue monitoring model performance"]

        if severity == DriftSeverity.LOW:
            recommendations.extend([
                "Review recent data quality",
                "Check for seasonal patterns"
            ])
        elif severity == DriftSeverity.MEDIUM:
            recommendations.extend([
                "Consider incremental model update",
                "Review feature engineering"
            ])
        elif severity == DriftSeverity.HIGH:
            recommendations.extend([
                "Schedule retraining session",
                "Analyze data distribution changes"
            ])
        elif severity == DriftSeverity.CRITICAL:
            recommendations.extend([
                "Immediate retraining required",
                "Investigate root cause of performance degradation",
                "Consider emergency model deployment"
            ])

        return recommendations

    def _create_error_result(self, model_id: str, drift_type: DriftType, error: str) -> DriftDetectionResult:
        """Create error result for drift detection."""
        return DriftDetectionResult(
            model_id=model_id,
            detection_timestamp=datetime.now().isoformat(),
            drift_type=drift_type,
            severity=DriftSeverity.NONE,
            drift_score=0.0,
            confidence=0.0,
            affected_features=[],
            performance_metrics={},
            baseline_comparison={},
            recommendations=[f"Drift detection failed: {error}"],
            retraining_triggered=False,
            retraining_reason=None
        )


class DataDriftDetector(DriftDetector):
    """Detect data distribution drift using statistical tests."""

    def __init__(self, significance_level: float = 0.05, min_sample_size: int = 30):
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        self.logger = logging.getLogger(__name__)

    def detect_drift(self,
                    reference_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    model_id: str) -> DriftDetectionResult:
        """Detect data distribution drift."""
        try:
            numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return self._create_error_result(model_id, DriftType.DATA_DRIFT, "No numeric features found")

            drift_results = {}
            affected_features = []

            for col in numeric_cols:
                ref_values = reference_data[col].dropna()
                curr_values = current_data[col].dropna()

                if len(ref_values) < self.min_sample_size or len(curr_values) < self.min_sample_size:
                    continue

                # Kolmogorov-Smirnov test for distribution drift
                ks_stat, ks_p_value = stats.ks_2samp(ref_values, curr_values)

                # Jensen-Shannon divergence
                ref_hist, _ = np.histogram(ref_values, bins=20, density=True)
                curr_hist, _ = np.histogram(curr_values, bins=20, density=True)
                js_divergence = jensenshannon(ref_hist, curr_hist)

                drift_detected = ks_p_value < self.significance_level or js_divergence > 0.1

                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'js_divergence': js_divergence,
                    'drift_detected': drift_detected
                }

                if drift_detected:
                    affected_features.append(col)

            # Calculate overall drift score
            overall_drift_score = np.mean([r['js_divergence'] for r in drift_results.values()])

            # Determine severity
            if overall_drift_score > 0.3:
                severity = DriftSeverity.CRITICAL
            elif overall_drift_score > 0.2:
                severity = DriftSeverity.HIGH
            elif overall_drift_score > 0.1:
                severity = DriftSeverity.MEDIUM
            elif overall_drift_score > 0.05:
                severity = DriftSeverity.LOW
            else:
                severity = DriftSeverity.NONE

            # Confidence based on number of significant drifts
            significant_drifts = sum(1 for r in drift_results.values() if r['drift_detected'])
            confidence = significant_drifts / len(drift_results) if drift_results else 0

            # Generate recommendations
            recommendations = self._generate_data_drift_recommendations(severity, affected_features)

            # Determine retraining
            retraining_triggered = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            retraining_reason = f"Data drift detected in {len(affected_features)} features" if retraining_triggered else None

            return DriftDetectionResult(
                model_id=model_id,
                detection_timestamp=datetime.now().isoformat(),
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_score=overall_drift_score,
                confidence=confidence,
                affected_features=affected_features,
                performance_metrics=drift_results,
                baseline_comparison={},
                recommendations=recommendations,
                retraining_triggered=retraining_triggered,
                retraining_reason=retraining_reason
            )

        except Exception as e:
            self.logger.error(f"Data drift detection failed for model {model_id}: {e}")
            return self._create_error_result(model_id, DriftType.DATA_DRIFT, str(e))

    def _generate_data_drift_recommendations(self, severity: DriftSeverity, affected_features: List[str]) -> List[str]:
        """Generate recommendations for data drift."""
        recommendations = ["Monitor data quality and distribution"]

        if affected_features:
            recommendations.append(f"Drift detected in features: {', '.join(affected_features[:5])}")

        if severity == DriftSeverity.LOW:
            recommendations.extend([
                "Review data collection processes",
                "Check for seasonal variations"
            ])
        elif severity == DriftSeverity.MEDIUM:
            recommendations.extend([
                "Consider data preprocessing adjustments",
                "Update feature engineering pipeline"
            ])
        elif severity == DriftSeverity.HIGH:
            recommendations.extend([
                "Schedule model retraining",
                "Investigate data source changes"
            ])
        elif severity == DriftSeverity.CRITICAL:
            recommendations.extend([
                "Immediate retraining required",
                "Audit data collection and processing pipeline",
                "Consider temporary model deactivation"
            ])

        return recommendations

    def _create_error_result(self, model_id: str, drift_type: DriftType, error: str) -> DriftDetectionResult:
        """Create error result."""
        return DriftDetectionResult(
            model_id=model_id,
            detection_timestamp=datetime.now().isoformat(),
            drift_type=drift_type,
            severity=DriftSeverity.NONE,
            drift_score=0.0,
            confidence=0.0,
            affected_features=[],
            performance_metrics={},
            baseline_comparison={},
            recommendations=[f"Data drift detection failed: {error}"],
            retraining_triggered=False,
            retraining_reason=None
        )


class ModelDriftDetectionService:
    """Main model drift detection and retraining service."""

    def __init__(self,
                 detection_interval: float = 3600.0,  # 1 hour
                 retraining_config: Optional[Dict[str, Any]] = None):
        self.detection_interval = detection_interval
        self.retraining_config = retraining_config or self._default_retraining_config()
        self.logger = logging.getLogger(__name__)

        # Detectors
        self.detectors = {
            'performance': PerformanceDriftDetector(),
            'data': DataDriftDetector()
        }

        # Model tracking
        self.model_baseline_data = {}
        self.model_performance_history = defaultdict(list)
        self.drift_history = deque(maxlen=10000)
        self.retraining_events = defaultdict(list)

        # Service state
        self.detection_active = False
        self.last_detection_time = None

    def _default_retraining_config(self) -> Dict[str, Any]:
        """Default retraining configuration."""
        return {
            'auto_retrain': True,
            'min_drift_severity': DriftSeverity.HIGH,
            'performance_threshold': 0.15,
            'max_retraining_per_day': 3,
            'retraining_timeout_seconds': 3600,
            'backup_model_required': True
        }

    async def start_drift_detection(self):
        """Start continuous drift detection."""
        if self.detection_active:
            return

        self.detection_active = True
        self.logger.info("Starting model drift detection service")

        while self.detection_active:
            try:
                await self._detection_cycle()
                await asyncio.sleep(self.detection_interval)
            except Exception as e:
                self.logger.error(f"Drift detection cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying

    def stop_drift_detection(self):
        """Stop drift detection."""
        self.detection_active = False
        self.logger.info("Model drift detection service stopped")

    async def _detection_cycle(self):
        """Execute one drift detection cycle."""
        self.last_detection_time = datetime.now()

        # This would iterate through all registered models
        # For now, this is a placeholder for actual implementation
        pass

    def register_model(self,
                      model_id: str,
                      baseline_data: pd.DataFrame,
                      model_metadata: Optional[Dict[str, Any]] = None):
        """Register a model for drift detection."""
        try:
            self.model_baseline_data[model_id] = {
                'data': baseline_data,
                'metadata': model_metadata or {},
                'registration_time': datetime.now().isoformat(),
                'last_retraining': None
            }

            self.logger.info(f"Model {model_id} registered for drift detection")

        except Exception as e:
            self.logger.error(f"Failed to register model {model_id}: {e}")

    async def detect_model_drift(self,
                                 model_id: str,
                                 current_data: pd.DataFrame) -> List[DriftDetectionResult]:
        """
        Detect drift for a specific model.

        Args:
            model_id: Model identifier
            current_data: Current data for comparison

        Returns:
            List of drift detection results
        """
        try:
            if model_id not in self.model_baseline_data:
                raise ValueError(f"Model {model_id} not registered")

            baseline_info = self.model_baseline_data[model_id]
            baseline_data = baseline_info['data']

            drift_results = []

            # Run all detectors
            for detector_name, detector in self.detectors.items():
                try:
                    result = detector.detect_drift(baseline_data, current_data, model_id)
                    drift_results.append(result)

                    # Store in history
                    self.drift_history.append(result)

                    # Check if retraining should be triggered
                    if result.retraining_triggered and self.retraining_config['auto_retrain']:
                        await self._trigger_retraining(model_id, result)

                except Exception as e:
                    self.logger.error(f"{detector_name} detector failed for model {model_id}: {e}")

            # Update model performance history
            await self._update_performance_history(model_id, drift_results)

            return drift_results

        except Exception as e:
            self.logger.error(f"Drift detection failed for model {model_id}: {e}")
            return []

    async def _trigger_retraining(self, model_id: str, drift_result: DriftDetectionResult):
        """Trigger model retraining."""
        try:
            # Check retraining limits
            today = datetime.now().date()
            today_retrainings = [
                event for event in self.retraining_events[model_id]
                if datetime.fromisoformat(event.retraining_timestamp).date() == today
            ]

            if len(today_retrainings) >= self.retraining_config['max_retraining_per_day']:
                self.logger.warning(f"Maximum retraining limit reached for model {model_id} today")
                return

            # Perform retraining (placeholder implementation)
            retraining_result = await self._perform_retraining(model_id, drift_result)

            if retraining_result.retraining_successful:
                self.logger.info(f"Model {model_id} retraining completed successfully")
            else:
                self.logger.warning(f"Model {model_id} retraining failed")

        except Exception as e:
            self.logger.error(f"Failed to trigger retraining for model {model_id}: {e}")

    async def _perform_retraining(self,
                                 model_id: str,
                                 drift_result: DriftDetectionResult) -> RetrainingEvent:
        """Perform actual model retraining."""
        start_time = time.time()

        try:
            # In real implementation, this would:
            # 1. Collect recent training data
            # 2. Retrain the model
            # 3. Validate the new model
            # 4. Deploy the new model

            # Placeholder implementation
            await asyncio.sleep(1.0)  # Simulate training time

            end_time = time.time()
            training_duration = end_time - start_time

            # Generate new model version
            old_version = self.model_baseline_data[model_id]['metadata'].get('version', '1.0.0')
            new_version = self._increment_version(old_version)

            # Create retraining event
            retraining_event = RetrainingEvent(
                model_id=model_id,
                retraining_timestamp=datetime.now().isoformat(),
                trigger_type=RetrainingTrigger.DRIFT_DETECTED,
                trigger_reason=drift_result.retraining_reason or "Unknown",
                retraining_successful=True,
                performance_improvement=0.1,  # Placeholder improvement
                training_duration_seconds=training_duration,
                new_model_version=new_version
            )

            # Store retraining event
            self.retraining_events[model_id].append(retraining_event)

            # Update model baseline
            self.model_baseline_data[model_id]['metadata']['version'] = new_version
            self.model_baseline_data[model_id]['last_retraining'] = retraining_event.retraining_timestamp

            return retraining_event

        except Exception as e:
            self.logger.error(f"Model retraining failed for {model_id}: {e}")

            return RetrainingEvent(
                model_id=model_id,
                retraining_timestamp=datetime.now().isoformat(),
                trigger_type=RetrainingTrigger.DRIFT_DETECTED,
                trigger_reason=drift_result.retraining_reason or "Unknown",
                retraining_successful=False,
                performance_improvement=None,
                training_duration_seconds=time.time() - start_time,
                new_model_version="failed"
            )

    def _increment_version(self, version: str) -> str:
        """Increment model version number."""
        try:
            parts = version.split('.')
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.0"

    async def _update_performance_history(self, model_id: str, drift_results: List[DriftDetectionResult]):
        """Update model performance history."""
        for result in drift_results:
            if result.performance_metrics:
                performance_record = {
                    'timestamp': result.detection_timestamp,
                    'metrics': result.performance_metrics,
                    'drift_score': result.drift_score,
                    'severity': result.severity.value
                }
                self.model_performance_history[model_id].append(performance_record)

    async def get_model_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive model health dashboard."""
        try:
            # Model health status for all registered models
            model_health = {}
            for model_id in self.model_baseline_data.keys():
                health_status = await self.assess_model_health(model_id)
                model_health[model_id] = asdict(health_status)

            # Recent drift events
            recent_drifts = [
                asdict(drift) for drift in list(self.drift_history)[-50:]
            ]

            # Retraining statistics
            retraining_stats = self._calculate_retraining_statistics()

            # Overall system health
            system_health = self._assess_system_health()

            return {
                'dashboard_time': datetime.now().isoformat(),
                'model_health': model_health,
                'recent_drift_events': recent_drifts,
                'retraining_statistics': retraining_stats,
                'system_health': system_health,
                'service_status': {
                    'detection_active': self.detection_active,
                    'registered_models': len(self.model_baseline_data),
                    'total_drifts_detected': len(self.drift_history),
                    'last_detection_time': self.last_detection_time.isoformat() if self.last_detection_time else None
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate health dashboard: {e}")
            return {'error': str(e)}

    async def assess_model_health(self, model_id: str) -> ModelHealthStatus:
        """Assess health of a specific model."""
        try:
            if model_id not in self.model_performance_history:
                return ModelHealthStatus(
                    model_id=model_id,
                    health_score=0.0,
                    overall_status="unknown",
                    drift_status={},
                    performance_status={},
                    last_updated=datetime.now().isoformat(),
                    issues_detected=["Model not registered or no performance history"],
                    recommended_actions=["Register model for monitoring"]
                )

            # Get recent performance data
            recent_history = self.model_performance_history[model_id][-30:]  # Last 30 records
            if not recent_history:
                return ModelHealthStatus(
                    model_id=model_id,
                    health_score=0.5,
                    overall_status="insufficient_data",
                    drift_status={},
                    performance_status={},
                    last_updated=datetime.now().isoformat(),
                    issues_detected=["Insufficient performance data"],
                    recommended_actions=["Monitor model performance"]
                )

            # Calculate health score
            health_metrics = self._calculate_health_metrics(recent_history)

            # Determine overall status
            if health_metrics['health_score'] >= 0.8:
                overall_status = "healthy"
            elif health_metrics['health_score'] >= 0.6:
                overall_status = "degraded"
            elif health_metrics['health_score'] >= 0.4:
                overall_status = "unhealthy"
            else:
                overall_status = "critical"

            # Identify issues and recommendations
            issues_detected = []
            recommended_actions = []

            if health_metrics['recent_drift_score'] > 0.2:
                issues_detected.append("High drift detected")
                recommended_actions.append("Consider model retraining")

            if health_metrics['performance_trend'] < -0.1:
                issues_detected.append("Declining performance")
                recommended_actions.append("Investigate performance degradation")

            if health_metrics['stability_score'] < 0.7:
                issues_detected.append("Unstable performance")
                recommended_actions.append("Review data quality and model stability")

            return ModelHealthStatus(
                model_id=model_id,
                health_score=health_metrics['health_score'],
                overall_status=overall_status,
                drift_status=health_metrics['drift_status'],
                performance_status=health_metrics['performance_status'],
                last_updated=datetime.now().isoformat(),
                issues_detected=issues_detected,
                recommended_actions=recommended_actions
            )

        except Exception as e:
            self.logger.error(f"Failed to assess model health for {model_id}: {e}")
            return ModelHealthStatus(
                model_id=model_id,
                health_score=0.0,
                overall_status="error",
                drift_status={},
                performance_status={},
                last_updated=datetime.now().isoformat(),
                issues_detected=[f"Health assessment failed: {str(e)}"],
                recommended_actions=["Check monitoring system"]
            )

    def _calculate_health_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate health metrics from performance history."""
        if not history:
            return {'health_score': 0.0}

        # Recent drift scores
        recent_drifts = [h['drift_score'] for h in history[-10:]]
        avg_drift_score = np.mean(recent_drifts) if recent_drifts else 0.0

        # Performance trend
        if len(history) >= 10:
            early_performance = np.mean([
                list(h['metrics'].values())[0] if h['metrics'] else 0.5
                for h in history[-20:-10]
            ])
            recent_performance = np.mean([
                list(h['metrics'].values())[0] if h['metrics'] else 0.5
                for h in history[-10:]
            ])
            performance_trend = (recent_performance - early_performance) / early_performance if early_performance > 0 else 0.0
        else:
            performance_trend = 0.0

        # Stability score (inverse of variance)
        if len(recent_drifts) > 1:
            stability_score = 1.0 - min(1.0, np.std(recent_drifts) * 5)
        else:
            stability_score = 1.0

        # Overall health score
        health_score = (
            (1.0 - avg_drift_score) * 0.4 +  # Drift impact
            max(0.0, 1.0 + performance_trend) * 0.3 +  # Performance trend
            stability_score * 0.3  # Stability
        )

        # Drift status summary
        drift_status = {}
        for record in history[-5:]:
            severity = record.get('severity', 'none')
            drift_status[severity] = drift_status.get(severity, 0) + 1

        # Performance status
        latest_metrics = history[-1].get('metrics', {})
        performance_status = {k: v for k, v in latest_metrics.items()}

        return {
            'health_score': health_score,
            'recent_drift_score': avg_drift_score,
            'performance_trend': performance_trend,
            'stability_score': stability_score,
            'drift_status': drift_status,
            'performance_status': performance_status
        }

    def _calculate_retraining_statistics(self) -> Dict[str, Any]:
        """Calculate retraining statistics."""
        total_retrainings = sum(len(events) for events in self.retraining_events.values())
        successful_retrainings = sum(
            len([e for e in events if e.retraining_successful])
            for events in self.retraining_events.values()
        )

        recent_retrainings = [
            event for events in self.retraining_events.values()
            for event in events
            if datetime.fromisoformat(event.retraining_timestamp) > datetime.now() - timedelta(days=7)
        ]

        return {
            'total_retrainings': total_retrainings,
            'successful_retrainings': successful_retrainings,
            'success_rate': successful_retrainings / total_retrainings if total_retrainings > 0 else 0.0,
            'retrainings_last_week': len(recent_retrainings),
            'average_training_duration': np.mean([
                e.training_duration_seconds for e in recent_retrainings
            ]) if recent_retrainings else 0.0
        }

    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        registered_models = len(self.model_baseline_data)
        healthy_models = 0

        for model_id in self.model_baseline_data:
            try:
                health_status = asyncio.run(self.assess_model_health(model_id))
                if health_status.health_score >= 0.6:
                    healthy_models += 1
            except:
                pass

        return {
            'registered_models': registered_models,
            'healthy_models': healthy_models,
            'system_health_score': healthy_models / registered_models if registered_models > 0 else 0.0,
            'drift_detection_active': self.detection_active,
            'auto_retraining_enabled': self.retraining_config['auto_retrain']
        }