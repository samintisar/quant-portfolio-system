"""
Structured logging for all model decisions with regime context.

Implements comprehensive logging system including:
- Structured JSON logging for model decisions
- Regime-aware logging context
- Performance and debugging information
- Automated log rotation and archiving
- Real-time log monitoring and alerting
"""

import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback
from contextlib import asynccontextmanager


class LogLevel(Enum):
    """Log levels with severity."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelDecisionType(Enum):
    """Types of model decisions."""
    FORECAST_GENERATED = "forecast_generated"
    MODEL_TRAINED = "model_trained"
    REGIME_DETECTED = "regime_detected"
    PARAMETER_UPDATED = "parameter_updated"
    VALIDATION_PERFORMED = "validation_performed"
    ANOMALY_DETECTED = "anomaly_detected"
    SCENARIO_ANALYZED = "scenario_analyzed"
    SIGNAL_VALIDATED = "signal_validated"
    ML_BENCHMARKED = "ml_benchmarked"


@dataclass
class RegimeContext:
    """Context information for market regime."""
    regime_id: Optional[int] = None
    regime_type: Optional[str] = None
    regime_probability: Optional[float] = None
    regime_confidence: Optional[float] = None
    regime_duration: Optional[int] = None
    transition_probabilities: Optional[Dict[str, float]] = None


@dataclass
class ModelContext:
    """Context information for model operations."""
    model_type: str
    model_version: str
    model_id: str
    training_parameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    data_metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceContext:
    """Performance monitoring context."""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size: Optional[int] = None
    throughput_per_second: Optional[float] = None


@dataclass
class StructuredLogEntry:
    """Structured log entry with comprehensive context."""
    timestamp: str
    level: LogLevel
    message: str
    operation_type: ModelDecisionType
    model_context: ModelContext
    regime_context: Optional[RegimeContext] = None
    performance_context: Optional[PerformanceContext] = None
    additional_context: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = getattr(record, 'structured_entry', {})

        # Add standard logging fields
        log_entry.update({
            'level': record.levelname,
            'logger_name': record.name,
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'thread_id': record.thread,
            'process_id': record.process
        })

        # Add exception info if present
        if record.exc_info:
            log_entry['error_details'] = {
                'exception_type': record.exc_info[0].__name__,
                'exception_message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_entry, default=str)


class RegimeAwareLogger:
    """Main structured logging service with regime context."""

    def __init__(self,
                 log_dir: str = "logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_console: bool = True,
                 max_file_size_mb: int = 100,
                 backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = log_level
        self.enable_console = enable_console
        self.logger = logging.getLogger("forecasting_system")

        # Configure logging
        self._setup_logging(max_file_size_mb, backup_count)

        # Performance tracking
        self.performance_stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'logs_by_operation': {},
            'average_execution_time': 0
        }

    def _setup_logging(self, max_file_size_mb: int, backup_count: int):
        """Setup logging configuration."""
        self.logger.setLevel(getattr(logging, self.log_level.value))

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler with rotation
        log_file = self.log_dir / "forecasting_system.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Error file handler
        error_log_file = self.log_dir / "forecasting_errors.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)

        # Performance log handler
        perf_log_file = self.log_dir / "forecasting_performance.log"
        perf_handler = TimedRotatingFileHandler(
            perf_log_file,
            when='midnight',
            interval=1,
            backupCount=30
        )
        perf_handler.setFormatter(JSONFormatter())
        self.performance_logger = logging.getLogger("forecasting_performance")
        self.performance_logger.addHandler(perf_handler)
        self.performance_logger.setLevel(logging.INFO)

        # Console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(console_handler)

    @asynccontextmanager
    async def log_operation(self,
                          operation_type: ModelDecisionType,
                          model_context: ModelContext,
                          regime_context: Optional[RegimeContext] = None,
                          session_id: Optional[str] = None,
                          trace_id: Optional[str] = None):
        """Context manager for logging operations with timing."""
        start_time = datetime.now()
        start_memory = self._get_memory_usage()

        try:
            yield
        except Exception as e:
            # Log error
            await self.log_error(
                message=f"Operation failed: {str(e)}",
                operation_type=operation_type,
                model_context=model_context,
                regime_context=regime_context,
                error_details={
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc()
                },
                session_id=session_id,
                trace_id=trace_id
            )
            raise
        else:
            # Log successful completion
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            end_memory = self._get_memory_usage()

            performance_context = PerformanceContext(
                execution_time_ms=execution_time_ms,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=self._get_cpu_usage(),
                throughput_per_second=1000.0 / execution_time_ms if execution_time_ms > 0 else None
            )

            await self.log_operation_completed(
                message=f"Operation completed successfully",
                operation_type=operation_type,
                model_context=model_context,
                regime_context=regime_context,
                performance_context=performance_context,
                session_id=session_id,
                trace_id=trace_id
            )

    async def log_forecast_generated(self,
                                  forecast_data: Dict[str, Any],
                                  model_context: ModelContext,
                                  regime_context: Optional[RegimeContext] = None,
                                  performance_context: Optional[PerformanceContext] = None):
        """Log forecast generation with comprehensive context."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message="Forecast generated",
            operation_type=ModelDecisionType.FORECAST_GENERATED,
            model_context=model_context,
            regime_context=regime_context,
            performance_context=performance_context,
            additional_context={
                'forecast_horizon': forecast_data.get('horizon'),
                'forecast_symbols': forecast_data.get('symbols', []),
                'forecast_confidence': forecast_data.get('confidence_level'),
                'model_accuracy': forecast_data.get('accuracy_metrics')
            }
        )

        await self._log_structured_entry(log_entry)

    async def log_model_trained(self,
                               training_results: Dict[str, Any],
                               model_context: ModelContext,
                               regime_context: Optional[RegimeContext] = None,
                               performance_context: Optional[PerformanceContext] = None):
        """Log model training with performance metrics."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message="Model training completed",
            operation_type=ModelDecisionType.MODEL_TRAINED,
            model_context=model_context,
            regime_context=regime_context,
            performance_context=performance_context,
            additional_context={
                'training_samples': training_results.get('training_samples'),
                'validation_samples': training_results.get('validation_samples'),
                'training_accuracy': training_results.get('training_accuracy'),
                'validation_accuracy': training_results.get('validation_accuracy'),
                'convergence_iterations': training_results.get('iterations'),
                'hyperparameters': training_results.get('hyperparameters')
            }
        )

        await self._log_structured_entry(log_entry)

    async def log_regime_detected(self,
                                regime_data: Dict[str, Any],
                                model_context: ModelContext,
                                performance_context: Optional[PerformanceContext] = None):
        """Log regime detection results."""
        regime_context = RegimeContext(
            regime_id=regime_data.get('regime_id'),
            regime_type=regime_data.get('regime_type'),
            regime_probability=regime_data.get('probability'),
            regime_confidence=regime_data.get('confidence'),
            regime_duration=regime_data.get('duration'),
            transition_probabilities=regime_data.get('transition_matrix')
        )

        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message="Market regime detected",
            operation_type=ModelDecisionType.REGIME_DETECTED,
            model_context=model_context,
            regime_context=regime_context,
            performance_context=performance_context,
            additional_context={
                'detection_method': regime_data.get('method'),
                'data_period': regime_data.get('data_period'),
                'emission_model': regime_data.get('emission_model')
            }
        )

        await self._log_structured_entry(log_entry)

    async def log_validation_performed(self,
                                     validation_results: Dict[str, Any],
                                     model_context: ModelContext,
                                     regime_context: Optional[RegimeContext] = None,
                                     performance_context: Optional[PerformanceContext] = None):
        """Log model validation results."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message="Model validation performed",
            operation_type=ModelDecisionType.VALIDATION_PERFORMED,
            model_context=model_context,
            regime_context=regime_context,
            performance_context=performance_context,
            additional_context={
                'validation_metrics': validation_results.get('metrics'),
                'statistical_significance': validation_results.get('significance'),
                'benchmark_comparison': validation_results.get('benchmark_comparison'),
                'passed_validation': validation_results.get('passed', True)
            }
        )

        await self._log_structured_entry(log_entry)

    async def log_anomaly_detected(self,
                                 anomaly_data: Dict[str, Any],
                                 model_context: ModelContext,
                                 regime_context: Optional[RegimeContext] = None):
        """Log anomaly detection events."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.WARNING,
            message="Anomaly detected",
            operation_type=ModelDecisionType.ANOMALY_DETECTED,
            model_context=model_context,
            regime_context=regime_context,
            additional_context={
                'anomaly_type': anomaly_data.get('type'),
                'anomaly_score': anomaly_data.get('score'),
                'anomaly_severity': anomaly_data.get('severity'),
                'affected_metrics': anomaly_data.get('metrics'),
                'detection_method': anomaly_data.get('method')
            }
        )

        await self._log_structured_entry(log_entry)

    async def log_operation_completed(self,
                                    message: str,
                                    operation_type: ModelDecisionType,
                                    model_context: ModelContext,
                                    regime_context: Optional[RegimeContext] = None,
                                    performance_context: Optional[PerformanceContext] = None,
                                    session_id: Optional[str] = None,
                                    trace_id: Optional[str] = None):
        """Log successful operation completion."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.INFO,
            message=message,
            operation_type=operation_type,
            model_context=model_context,
            regime_context=regime_context,
            performance_context=performance_context,
            session_id=session_id,
            trace_id=trace_id
        )

        await self._log_structured_entry(log_entry)

    async def log_error(self,
                       message: str,
                       operation_type: ModelDecisionType,
                       model_context: ModelContext,
                       regime_context: Optional[RegimeContext] = None,
                       error_details: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None,
                       trace_id: Optional[str] = None):
        """Log error with detailed context."""
        log_entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=LogLevel.ERROR,
            message=message,
            operation_type=operation_type,
            model_context=model_context,
            regime_context=regime_context,
            error_details=error_details,
            session_id=session_id,
            trace_id=trace_id
        )

        await self._log_structured_entry(log_entry)

    async def _log_structured_entry(self, log_entry: StructuredLogEntry):
        """Log structured entry to all appropriate handlers."""
        try:
            # Update performance statistics
            self._update_performance_stats(log_entry)

            # Convert to dictionary for logging
            log_dict = asdict(log_entry)

            # Log to main logger
            log_level = getattr(logging, log_entry.level.value)
            record = logging.LogRecord(
                name=self.logger.name,
                level=log_level,
                pathname="",
                lineno=0,
                msg="",
                args=(),
                exc_info=None
            )
            record.structured_entry = log_dict

            self.logger.handle(record)

            # Log performance metrics if available
            if log_entry.performance_context:
                perf_log_dict = {
                    'timestamp': log_entry.timestamp,
                    'operation_type': log_entry.operation_type.value,
                    'model_type': log_entry.model_context.model_type,
                    'execution_time_ms': log_entry.performance_context.execution_time_ms,
                    'memory_usage_mb': log_entry.performance_context.memory_usage_mb,
                    'cpu_usage_percent': log_entry.performance_context.cpu_usage_percent,
                    'regime_id': log_entry.regime_context.regime_id if log_entry.regime_context else None
                }

                perf_record = logging.LogRecord(
                    name=self.performance_logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg="",
                    args=(),
                    exc_info=None
                )
                perf_record.structured_entry = perf_log_dict
                self.performance_logger.handle(perf_record)

        except Exception as e:
            # Fallback to simple logging if structured logging fails
            self.logger.error(f"Structured logging failed: {e}")

    def _update_performance_stats(self, log_entry: StructuredLogEntry):
        """Update performance statistics."""
        self.performance_stats['total_logs'] += 1

        # Update level statistics
        level_key = log_entry.level.value
        self.performance_stats['logs_by_level'][level_key] = \
            self.performance_stats['logs_by_level'].get(level_key, 0) + 1

        # Update operation statistics
        operation_key = log_entry.operation_type.value
        self.performance_stats['logs_by_operation'][operation_key] = \
            self.performance_stats['logs_by_operation'].get(operation_key, 0) + 1

        # Update average execution time
        if log_entry.performance_context:
            current_avg = self.performance_stats['average_execution_time']
            new_time = log_entry.performance_context.execution_time_ms
            total_logs = self.performance_stats['total_logs']

            self.performance_stats['average_execution_time'] = \
                (current_avg * (total_logs - 1) + new_time) / total_logs

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        return {
            'statistics': self.performance_stats,
            'log_configuration': {
                'log_level': self.log_level.value,
                'log_directory': str(self.log_dir),
                'console_logging': self.enable_console
            },
            'recent_activity': {
                'total_logs_last_hour': self._count_recent_logs(hours=1),
                'error_rate_last_hour': self._calculate_error_rate(hours=1),
                'average_execution_time': self.performance_stats['average_execution_time']
            }
        }

    def _count_recent_logs(self, hours: int = 1) -> int:
        """Count logs from recent hours."""
        # This would parse log files for recent entries
        # For now, return estimate based on current stats
        return int(self.performance_stats['total_logs'] / 24 * hours)

    def _calculate_error_rate(self, hours: int = 1) -> float:
        """Calculate error rate for recent hours."""
        total_logs = self._count_recent_logs(hours)
        error_logs = self.performance_stats['logs_by_level'].get('ERROR', 0)
        return error_logs / total_logs if total_logs > 0 else 0.0

    async def query_logs(self,
                       operation_type: Optional[ModelDecisionType] = None,
                       level: Optional[LogLevel] = None,
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs with filtering criteria."""
        # This would implement log querying from files
        # For now, return empty list
        return []

    async def export_logs(self,
                         format: str = "json",
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None) -> str:
        """Export logs to file."""
        # This would implement log export functionality
        return "Export functionality to be implemented"


# Global logger instance
_structured_logger: Optional[RegimeAwareLogger] = None


def get_structured_logger() -> RegimeAwareLogger:
    """Get global structured logger instance."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = RegimeAwareLogger()
    return _structured_logger


def set_structured_logger(logger: RegimeAwareLogger):
    """Set global structured logger instance."""
    global _structured_logger
    _structured_logger = logger