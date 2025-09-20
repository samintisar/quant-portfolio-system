"""
Advanced performance monitoring with model convergence tracking.

Implements comprehensive performance monitoring system including:
- Real-time model convergence tracking
- Performance metrics collection and analysis
- Automated performance alerts
- Convergence detection algorithms
- Model health monitoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import asyncio
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import json
from collections import deque, defaultdict
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Scientific computing
from scipy import stats
from scipy.optimize import minimize_scalar


class ConvergenceStatus(Enum):
    """Model convergence status."""
    CONVERGED = "converged"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    STALLED = "stalled"
    UNKNOWN = "unknown"


class PerformanceAlert(Enum):
    """Performance alert types."""
    HIGH_MEMORY = "high_memory"
    SLOW_CONVERGENCE = "slow_convergence"
    DIVERGENCE_DETECTED = "divergence_detected"
    POOR_PERFORMANCE = "poor_performance"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_DRIFT = "model_drift"


@dataclass
class ConvergenceMetrics:
    """Model convergence tracking metrics."""
    iteration: int
    loss_value: float
    gradient_norm: float
    parameter_change: float
    learning_rate: float
    timestamp: str
    convergence_score: float
    status: ConvergenceStatus


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: Optional[float]
    disk_io_mb_s: float
    network_io_mb_s: float
    process_count: int
    load_average: Optional[float]


@dataclass
class ModelPerformanceMetrics:
    """Model-specific performance metrics."""
    model_id: str
    model_type: str
    training_loss: float
    validation_loss: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    convergence_metrics: ConvergenceMetrics
    timestamp: str


@dataclass
class PerformanceAlert:
    """Performance alert event."""
    alert_type: PerformanceAlert
    severity: str
    message: str
    timestamp: str
    affected_model: Optional[str]
    metrics: Dict[str, Any]
    recommendations: List[str]


class ConvergenceDetector:
    """Detect model convergence using various algorithms."""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 window_size: int = 5):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)

    def detect_convergence(self,
                          loss_history: List[float],
                          current_iteration: int) -> Tuple[ConvergenceStatus, float]:
        """
        Detect convergence based on loss history.

        Returns:
            Tuple of (convergence_status, convergence_score)
        """
        if len(loss_history) < self.window_size:
            return ConvergenceStatus.UNKNOWN, 0.0

        # Calculate recent loss statistics
        recent_losses = loss_history[-self.window_size:]
        current_loss = loss_history[-1]
        best_loss = min(loss_history)

        # Check for convergence conditions
        convergence_score = self._calculate_convergence_score(loss_history)

        # Convergence criteria
        if convergence_score > 0.95:
            return ConvergenceStatus.CONVERGED, convergence_score

        # Check for divergence
        if current_loss > best_loss * 2.0 and len(loss_history) > 20:
            return ConvergenceStatus.DIVERGING, convergence_score

        # Check for oscillation
        if self._detect_oscillation(recent_losses):
            return ConvergenceStatus.OSCILLATING, convergence_score

        # Check for stalled training
        if self._detect_stall(loss_history[-self.patience:]):
            return ConvergenceStatus.STALLED, convergence_score

        return ConvergenceStatus.UNKNOWN, convergence_score

    def _calculate_convergence_score(self, loss_history: List[float]) -> float:
        """Calculate convergence score between 0 and 1."""
        if len(loss_history) < 2:
            return 0.0

        # Recent loss stability
        recent_losses = loss_history[-self.window_size:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        stability_score = 1.0 - min(1.0, loss_std / (abs(loss_mean) + 1e-8))

        # Improvement rate
        if len(loss_history) >= 10:
            recent_improvement = (loss_history[-10] - loss_history[-1]) / (abs(loss_history[-10]) + 1e-8)
            improvement_score = max(0.0, 1.0 - abs(recent_improvement) * 10)
        else:
            improvement_score = 0.0

        # Overall convergence score
        return (stability_score + improvement_score) / 2.0

    def _detect_oscillation(self, losses: List[float]) -> bool:
        """Detect if losses are oscillating."""
        if len(losses) < 4:
            return False

        # Count sign changes in gradients
        gradients = np.diff(losses)
        sign_changes = np.sum(np.diff(np.sign(gradients)) != 0)
        oscillation_ratio = sign_changes / (len(gradients) - 1)

        return oscillation_ratio > 0.5

    def _detect_stall(self, recent_losses: List[float]) -> bool:
        """Detect if training has stalled."""
        if len(recent_losses) < self.patience:
            return False

        # Check if improvement is below threshold
        improvement = abs(recent_losses[0] - recent_losses[-1])
        return improvement < self.min_delta


class ResourceMonitor:
    """Monitor system resource usage."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while self.monitoring:
            try:
                # Get current resource usage
                metrics = self._collect_resource_metrics()

                # Store metrics (in real implementation, this would go to a time-series database)
                self._store_metrics(metrics)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(5.0)  # Wait before retrying

    def _collect_resource_metrics(self) -> PerformanceMetrics:
        """Collect current resource metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=psutil.cpu_percent(interval=0.1),
                memory_usage_mb=memory_info.rss / 1024 / 1024,
                gpu_usage_percent=self._get_gpu_usage(),
                disk_io_mb_s=self._get_disk_io(),
                network_io_mb_s=self._get_network_io(),
                process_count=len(psutil.pids()),
                load_average=self._get_load_average()
            )
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                gpu_usage_percent=None,
                disk_io_mb_s=0.0,
                network_io_mb_s=0.0,
                process_count=0,
                load_average=None
            )

    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available."""
        try:
            # Try to get GPU usage (nvidia-ml-py or similar)
            # For now, return None
            return None
        except:
            return None

    def _get_disk_io(self) -> float:
        """Get disk I/O rate."""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024
            return 0.0
        except:
            return 0.0

    def _get_network_io(self) -> float:
        """Get network I/O rate."""
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                return (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024
            return 0.0
        except:
            return 0.0

    def _get_load_average(self) -> Optional[float]:
        """Get system load average."""
        try:
            return psutil.getloadavg()[0]  # 1-minute load average
        except:
            return None

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics (placeholder for database storage)."""
        pass


class PerformanceMonitoringService:
    """Main performance monitoring service."""

    def __init__(self,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 monitoring_interval: float = 1.0):
        self.alert_thresholds = alert_thresholds or {
            'memory_usage_mb': 4096.0,
            'cpu_usage_percent': 80.0,
            'convergence_iterations': 1000,
            'stall_iterations': 50
        }

        self.convergence_detector = ConvergenceDetector()
        self.resource_monitor = ResourceMonitor(monitoring_interval)
        self.logger = logging.getLogger(__name__)

        # Performance history
        self.convergence_history = defaultdict(list)
        self.performance_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)

        # Model tracking
        self.active_models = {}
        self.model_metrics = defaultdict(list)

        # Start monitoring
        self.resource_monitor.start_monitoring()

    async def track_model_convergence(self,
                                    model_id: str,
                                    loss_history: List[float],
                                    current_iteration: int,
                                    learning_rate: float,
                                    gradient_norm: Optional[float] = None,
                                    parameters: Optional[Dict[str, np.ndarray]] = None) -> ConvergenceMetrics:
        """
        Track model convergence and generate metrics.

        Args:
            model_id: Unique model identifier
            loss_history: Historical loss values
            current_iteration: Current training iteration
            learning_rate: Current learning rate
            gradient_norm: Current gradient norm
            parameters: Current model parameters

        Returns:
            ConvergenceMetrics with analysis results
        """
        try:
            # Detect convergence status
            status, convergence_score = self.convergence_detector.detect_convergence(
                loss_history, current_iteration
            )

            # Calculate parameter change if parameters provided
            parameter_change = 0.0
            if parameters and model_id in self.active_models:
                old_params = self.active_models[model_id]['parameters']
                parameter_change = self._calculate_parameter_change(old_params, parameters)

            # Create convergence metrics
            convergence_metrics = ConvergenceMetrics(
                iteration=current_iteration,
                loss_value=loss_history[-1] if loss_history else 0.0,
                gradient_norm=gradient_norm or 0.0,
                parameter_change=parameter_change,
                learning_rate=learning_rate,
                timestamp=datetime.now().isoformat(),
                convergence_score=convergence_score,
                status=status
            )

            # Store metrics
            self.convergence_history[model_id].append(convergence_metrics)
            self.active_models[model_id] = {
                'parameters': parameters,
                'last_update': datetime.now().isoformat(),
                'status': status
            }

            # Check for alerts
            await self._check_convergence_alerts(model_id, convergence_metrics)

            return convergence_metrics

        except Exception as e:
            self.logger.error(f"Convergence tracking failed for model {model_id}: {e}")
            return ConvergenceMetrics(
                iteration=current_iteration,
                loss_value=0.0,
                gradient_norm=0.0,
                parameter_change=0.0,
                learning_rate=learning_rate,
                timestamp=datetime.now().isoformat(),
                convergence_score=0.0,
                status=ConvergenceStatus.UNKNOWN
            )

    async def track_model_performance(self,
                                    model_id: str,
                                    model_type: str,
                                    training_loss: float,
                                    validation_loss: float,
                                    accuracy: Optional[float] = None,
                                    precision: Optional[float] = None,
                                    recall: Optional[float] = None,
                                    f1_score: Optional[float] = None) -> ModelPerformanceMetrics:
        """
        Track model performance metrics.

        Args:
            model_id: Unique model identifier
            model_type: Type of model
            training_loss: Training loss value
            validation_loss: Validation loss value
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1_score: Model F1 score

        Returns:
            ModelPerformanceMetrics with analysis results
        """
        try:
            # Get latest convergence metrics
            convergence_metrics = None
            if model_id in self.convergence_history and self.convergence_history[model_id]:
                convergence_metrics = self.convergence_history[model_id][-1]

            # Create performance metrics
            performance_metrics = ModelPerformanceMetrics(
                model_id=model_id,
                model_type=model_type,
                training_loss=training_loss,
                validation_loss=validation_loss,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                convergence_metrics=convergence_metrics,
                timestamp=datetime.now().isoformat()
            )

            # Store metrics
            self.model_metrics[model_id].append(performance_metrics)

            # Check for performance alerts
            await self._check_performance_alerts(performance_metrics)

            return performance_metrics

        except Exception as e:
            self.logger.error(f"Performance tracking failed for model {model_id}: {e}")
            raise

    async def _check_convergence_alerts(self, model_id: str, metrics: ConvergenceMetrics):
        """Check for convergence-related alerts."""
        alerts = []

        # Slow convergence alert
        if (metrics.iteration > self.alert_thresholds['convergence_iterations'] and
            metrics.convergence_score < 0.8):
            alerts.append(PerformanceAlert(
                alert_type=PerformanceAlert.SLOW_CONVERGENCE,
                severity="warning",
                message=f"Model {model_id} converging slowly after {metrics.iteration} iterations",
                timestamp=datetime.now().isoformat(),
                affected_model=model_id,
                metrics={
                    'iteration': metrics.iteration,
                    'convergence_score': metrics.convergence_score,
                    'loss_value': metrics.loss_value
                },
                recommendations=[
                    "Consider increasing learning rate",
                    "Check data quality and preprocessing",
                    "Review model architecture complexity"
                ]
            ))

        # Divergence alert
        if metrics.status == ConvergenceStatus.DIVERGING:
            alerts.append(PerformanceAlert(
                alert_type=PerformanceAlert.DIVERGENCE_DETECTED,
                severity="critical",
                message=f"Model {model_id} is diverging",
                timestamp=datetime.now().isoformat(),
                affected_model=model_id,
                metrics={
                    'iteration': metrics.iteration,
                    'loss_value': metrics.loss_value,
                    'convergence_score': metrics.convergence_score
                },
                recommendations=[
                    "Reduce learning rate immediately",
                    "Check for numerical instability",
                    "Consider reinitializing model"
                ]
            ))

        # Stalled training alert
        if metrics.status == ConvergenceStatus.STALLED:
            alerts.append(PerformanceAlert(
                alert_type=PerformanceAlert.POOR_PERFORMANCE,
                severity="warning",
                message=f"Model {model_id} training has stalled",
                timestamp=datetime.now().isoformat(),
                affected_model=model_id,
                metrics={
                    'iteration': metrics.iteration,
                    'loss_value': metrics.loss_value,
                    'parameter_change': metrics.parameter_change
                },
                recommendations=[
                    "Adjust learning rate schedule",
                    "Try different optimization algorithm",
                    "Consider early stopping"
                ]
            ))

        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            self.logger.warning(f"Performance alert: {alert.message}")

    async def _check_performance_alerts(self, metrics: ModelPerformanceMetrics):
        """Check for performance-related alerts."""
        alerts = []

        # Overfitting detection
        if (metrics.training_loss < metrics.validation_loss * 0.5 and
            metrics.accuracy is not None and metrics.accuracy > 0.95):
            alerts.append(PerformanceAlert(
                alert_type=PerformanceAlert.MODEL_DRIFT,
                severity="warning",
                message=f"Model {metrics.model_id} may be overfitting",
                timestamp=datetime.now().isoformat(),
                affected_model=metrics.model_id,
                metrics={
                    'training_loss': metrics.training_loss,
                    'validation_loss': metrics.validation_loss,
                    'accuracy': metrics.accuracy
                },
                recommendations=[
                    "Add regularization",
                    "Increase dropout",
                    "Use more training data",
                    "Implement early stopping"
                ]
            ))

        # Poor performance alert
        if (metrics.accuracy is not None and metrics.accuracy < 0.6 and
            metrics.training_loss > 1.0):
            alerts.append(PerformanceAlert(
                alert_type=PerformanceAlert.POOR_PERFORMANCE,
                severity="warning",
                message=f"Model {metrics.model_id} has poor performance",
                timestamp=datetime.now().isoformat(),
                affected_model=metrics.model_id,
                metrics={
                    'accuracy': metrics.accuracy,
                    'training_loss': metrics.training_loss,
                    'validation_loss': metrics.validation_loss
                },
                recommendations=[
                    "Review feature engineering",
                    "Check data quality",
                    "Consider different model architecture",
                    "Hyperparameter tuning needed"
                ]
            ))

        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
            self.logger.warning(f"Performance alert: {alert.message}")

    def _calculate_parameter_change(self,
                                  old_params: Dict[str, np.ndarray],
                                  new_params: Dict[str, np.ndarray]) -> float:
        """Calculate parameter change between iterations."""
        total_change = 0.0
        total_params = 0

        for key in old_params:
            if key in new_params:
                old_param = old_params[key]
                new_param = new_params[key]

                if old_param.shape == new_param.shape:
                    param_change = np.mean(np.abs(new_param - old_param))
                    total_change += param_change
                    total_params += 1

        return total_change / total_params if total_params > 0 else 0.0

    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance monitoring dashboard."""
        try:
            # Current system metrics
            current_metrics = self.resource_monitor._collect_resource_metrics()

            # Model performance summary
            model_summary = {}
            for model_id, metrics_list in self.model_metrics.items():
                if metrics_list:
                    latest_metrics = metrics_list[-1]
                    model_summary[model_id] = {
                        'model_type': latest_metrics.model_type,
                        'training_loss': latest_metrics.training_loss,
                        'validation_loss': latest_metrics.validation_loss,
                        'accuracy': latest_metrics.accuracy,
                        'last_update': latest_metrics.timestamp,
                        'convergence_status': latest_metrics.convergence_metrics.status.value if latest_metrics.convergence_metrics else 'unknown'
                    }

            # Recent alerts
            recent_alerts = [
                asdict(alert) for alert in list(self.alert_history)[-10:]
            ]

            # Convergence trends
            convergence_trends = {}
            for model_id, convergence_list in self.convergence_history.items():
                if convergence_list:
                    recent_convergence = convergence_list[-20:]  # Last 20 iterations
                    convergence_trends[model_id] = {
                        'current_score': recent_convergence[-1].convergence_score,
                        'trend_direction': 'improving' if recent_convergence[-1].convergence_score > recent_convergence[0].convergence_score else 'declining',
                        'iterations_tracked': len(recent_convergence)
                    }

            return {
                'dashboard_time': datetime.now().isoformat(),
                'system_metrics': asdict(current_metrics),
                'model_summary': model_summary,
                'recent_alerts': recent_alerts,
                'convergence_trends': convergence_trends,
                'monitoring_status': {
                    'active_models': len(self.active_models),
                    'total_models_tracked': len(self.model_metrics),
                    'total_alerts': len(self.alert_history),
                    'monitoring_active': self.resource_monitor.monitoring
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate performance dashboard: {e}")
            return {'error': str(e)}

    async def get_model_analysis(self, model_id: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific model."""
        try:
            if model_id not in self.model_metrics:
                return {'error': f'Model {model_id} not found'}

            metrics_history = self.model_metrics[model_id]
            convergence_history = self.convergence_history.get(model_id, [])

            if not metrics_history:
                return {'error': f'No metrics available for model {model_id}'}

            latest_metrics = metrics_history[-1]

            # Calculate performance trends
            if len(metrics_history) > 1:
                loss_trend = np.mean([m.training_loss for m in metrics_history[-10:]])
                loss_improvement = metrics_history[-10].training_loss - latest_metrics.training_loss
            else:
                loss_trend = latest_metrics.training_loss
                loss_improvement = 0.0

            # Convergence analysis
            convergence_analysis = {}
            if convergence_history:
                recent_convergence = convergence_history[-20:]
                convergence_analysis = {
                    'current_status': recent_convergence[-1].status.value,
                    'convergence_score': recent_convergence[-1].convergence_score,
                    'stability_score': np.std([c.convergence_score for c in recent_convergence]),
                    'average_parameter_change': np.mean([c.parameter_change for c in recent_convergence]),
                    'training_iterations': len(convergence_history)
                }

            return {
                'model_id': model_id,
                'model_type': latest_metrics.model_type,
                'current_performance': {
                    'training_loss': latest_metrics.training_loss,
                    'validation_loss': latest_metrics.validation_loss,
                    'accuracy': latest_metrics.accuracy,
                    'f1_score': latest_metrics.f1_score
                },
                'performance_trends': {
                    'loss_trend': loss_trend,
                    'loss_improvement': loss_improvement,
                    'metrics_tracked': len(metrics_history)
                },
                'convergence_analysis': convergence_analysis,
                'recommendations': self._generate_model_recommendations(latest_metrics, convergence_analysis),
                'last_update': latest_metrics.timestamp
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze model {model_id}: {e}")
            return {'error': str(e)}

    def _generate_model_recommendations(self,
                                      metrics: ModelPerformanceMetrics,
                                      convergence_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for model improvement."""
        recommendations = []

        # Based on performance metrics
        if metrics.validation_loss > metrics.training_loss * 1.5:
            recommendations.append("Consider adding regularization to reduce overfitting")

        if metrics.accuracy and metrics.accuracy < 0.7:
            recommendations.append("Model performance is below threshold - consider feature engineering or architecture changes")

        # Based on convergence analysis
        if convergence_analysis:
            status = convergence_analysis.get('current_status')
            if status == 'diverging':
                recommendations.append("Model is diverging - reduce learning rate or check data quality")
            elif status == 'stalled':
                recommendations.append("Training has stalled - adjust learning rate or optimization algorithm")
            elif convergence_analysis.get('convergence_score', 0) < 0.8:
                recommendations.append("Convergence is slow - consider adaptive learning rate or different optimizer")

        return recommendations

    def shutdown(self):
        """Shutdown monitoring service."""
        self.resource_monitor.stop_monitoring()
        self.logger.info("Performance monitoring service shutdown")