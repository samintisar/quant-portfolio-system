"""
Performance Monitoring Service for Preprocessing Operations

Tracks execution time, memory usage, and system metrics for
all preprocessing operations with real-time monitoring.
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: float
    rows_processed: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation': self.operation,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_start': self.memory_start,
            'memory_end': self.memory_end,
            'memory_peak': self.memory_peak,
            'cpu_percent': self.cpu_percent,
            'rows_processed': self.rows_processed,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of current system performance."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_available: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    active_threads: int
    open_files: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used': self.memory_used,
            'memory_available': self.memory_available,
            'disk_io_read': self.disk_io_read,
            'disk_io_write': self.disk_io_write,
            'network_io_sent': self.network_io_sent,
            'network_io_recv': self.network_io_recv,
            'active_threads': self.active_threads,
            'open_files': self.open_files
        }


class PerformanceMonitor:
    """Performance monitoring service for preprocessing operations."""

    def __init__(self, max_history: int = 1000, snapshot_interval: int = 1):
        """Initialize the performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in history
            snapshot_interval: Interval in seconds for system snapshots
        """
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        self.snapshot_interval = snapshot_interval

        # Performance history
        self.metrics_history: List[PerformanceMetric] = []
        self.system_snapshots: deque = deque(maxlen=max_history)

        # Current monitoring state
        self.current_operation: Optional[str] = None
        self.operation_start_time: Optional[float] = None
        self.operation_start_memory: Optional[float] = None
        self.operation_peak_memory: Optional[float] = None

        # System monitoring
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.system_stats = defaultdict(list)

        # Performance thresholds
        self.thresholds = {
            'memory_warning': 0.8,  # 80% memory usage
            'memory_critical': 0.9,  # 90% memory usage
            'cpu_warning': 0.8,  # 80% CPU usage
            'duration_warning': 30.0,  # 30 seconds
            'duration_critical': 60.0,  # 60 seconds
        }

        # Initialize system monitoring
        self._initialize_system_monitoring()

    def _initialize_system_monitoring(self):
        """Initialize system monitoring thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def _monitor_system(self):
        """Monitor system resources in background thread."""
        process = psutil.Process()
        last_disk_io = psutil.disk_io_counters()
        last_network_io = psutil.net_io_counters()

        while self.monitoring:
            try:
                # Get current system stats
                snapshot = self._capture_system_snapshot()
                self.system_snapshots.append(snapshot)

                # Update running averages
                self.system_stats['cpu'].append(snapshot.cpu_percent)
                self.system_stats['memory'].append(snapshot.memory_percent)

                # Calculate IO rates
                if last_disk_io:
                    current_disk_io = psutil.disk_io_counters()
                    if current_disk_io and last_disk_io:
                        disk_read_rate = (current_disk_io.read_bytes - last_disk_io.read_bytes) / self.snapshot_interval
                        disk_write_rate = (current_disk_io.write_bytes - last_disk_io.write_bytes) / self.snapshot_interval
                        self.system_stats['disk_read_rate'].append(disk_read_rate)
                        self.system_stats['disk_write_rate'].append(disk_write_rate)
                    last_disk_io = current_disk_io

                if last_network_io:
                    current_network_io = psutil.net_io_counters()
                    if current_network_io and last_network_io:
                        network_sent_rate = (current_network_io.bytes_sent - last_network_io.bytes_sent) / self.snapshot_interval
                        network_recv_rate = (current_network_io.bytes_recv - last_network_io.bytes_recv) / self.snapshot_interval
                        self.system_stats['network_sent_rate'].append(network_sent_rate)
                        self.system_stats['network_recv_rate'].append(network_recv_rate)
                    last_network_io = current_network_io

                time.sleep(self.snapshot_interval)

            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.snapshot_interval)

    def _capture_system_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot."""
        process = psutil.Process()
        memory_info = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()

        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_used=process.memory_info().rss,
            memory_available=memory_info.available,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_io_sent=network_io.bytes_sent if network_io else 0,
            network_io_recv=network_io.bytes_recv if network_io else 0,
            active_threads=process.num_threads(),
            open_files=len(process.open_files())
        )

    def start_operation(self, operation: str, rows_processed: int = 0) -> str:
        """Start monitoring an operation.

        Args:
            operation: Operation name
            rows_processed: Number of rows being processed

        Returns:
            Operation ID
        """
        self.current_operation = operation
        self.operation_start_time = time.time()
        self.operation_start_memory = psutil.Process().memory_info().rss
        self.operation_peak_memory = self.operation_start_memory

        operation_id = f"{operation}_{int(time.time() * 1000)}"
        self.logger.info(f"Starting performance monitoring for operation: {operation} (ID: {operation_id})")

        return operation_id

    def end_operation(self, operation_id: str, success: bool = True,
                     error_message: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """End monitoring an operation.

        Args:
            operation_id: Operation identifier
            success: Whether operation was successful
            error_message: Error message if operation failed
            metadata: Additional operation metadata

        Returns:
            Performance metric
        """
        if not self.current_operation or not self.operation_start_time:
            raise ValueError("No active operation to end")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        duration = end_time - self.operation_start_time

        # Calculate CPU usage during operation
        process = psutil.Process()
        cpu_percent = process.cpu_percent()

        # Create performance metric
        metric = PerformanceMetric(
            operation=self.current_operation,
            start_time=self.operation_start_time,
            end_time=end_time,
            duration=duration,
            memory_start=self.operation_start_memory,
            memory_end=end_memory,
            memory_peak=self.operation_peak_memory,
            cpu_percent=cpu_percent,
            rows_processed=metadata.get('rows_processed', 0) if metadata else 0,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        # Add to history
        self.metrics_history.append(metric)

        # Trim history if needed
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

        # Check thresholds and log warnings
        self._check_thresholds(metric)

        # Reset operation state
        self.current_operation = None
        self.operation_start_time = None
        self.operation_start_memory = None
        self.operation_peak_memory = None

        self.logger.info(f"Operation '{metric.operation}' completed in {duration:.2f}s, "
                       f"memory used: {(metric.memory_end - metric.memory_start) / 1024 / 1024:.2f}MB")

        return metric

    def update_peak_memory(self):
        """Update peak memory for current operation."""
        if self.current_operation:
            current_memory = psutil.Process().memory_info().rss
            if current_memory > self.operation_peak_memory:
                self.operation_peak_memory = current_memory

    def _check_thresholds(self, metric: PerformanceMetric):
        """Check performance thresholds and log warnings."""
        memory_increase = (metric.memory_end - metric.memory_start) / psutil.virtual_memory().total

        if memory_increase > self.thresholds['memory_critical']:
            self.logger.critical(f"CRITICAL: Memory increase of {memory_increase:.1%} during {metric.operation}")
        elif memory_increase > self.thresholds['memory_warning']:
            self.logger.warning(f"WARNING: Memory increase of {memory_increase:.1%} during {metric.operation}")

        if metric.cpu_percent > self.thresholds['cpu_warning'] * 100:
            self.logger.warning(f"WARNING: High CPU usage ({metric.cpu_percent:.1f}%) during {metric.operation}")

        if metric.duration > self.thresholds['duration_critical']:
            self.logger.critical(f"CRITICAL: Operation {metric.operation} took {metric.duration:.1f}s")
        elif metric.duration > self.thresholds['duration_warning']:
            self.logger.warning(f"WARNING: Operation {metric.operation} took {metric.duration:.1f}s")

    def get_operation_metrics(self, operation: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metrics for a specific operation.

        Args:
            operation: Operation name
            limit: Maximum number of metrics to return

        Returns:
            List of performance metrics
        """
        metrics = [m for m in self.metrics_history if m.operation == operation]
        if limit:
            metrics = metrics[-limit:]
        return metrics

    def get_recent_metrics(self, minutes: int = 60) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes.

        Args:
            minutes: Number of minutes to look back

        Returns:
            List of recent performance metrics
        """
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.start_time > cutoff_time]

    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations.

        Args:
            operation: Specific operation to summarize, or None for all operations

        Returns:
            Performance summary dictionary
        """
        metrics = self.metrics_history
        if operation:
            metrics = [m for m in metrics if m.operation == operation]

        if not metrics:
            return {'error': 'No metrics available'}

        # Calculate summary statistics
        total_operations = len(metrics)
        successful_operations = sum(1 for m in metrics if m.success)
        total_duration = sum(m.duration for m in metrics)
        total_memory_increase = sum(m.memory_end - m.memory_start for m in metrics)
        total_rows_processed = sum(m.rows_processed for m in metrics)

        avg_duration = total_duration / total_operations
        avg_memory_increase = total_memory_increase / total_operations
        avg_rows_per_second = total_rows_processed / total_duration if total_duration > 0 else 0

        # Calculate success rate
        success_rate = successful_operations / total_operations

        # Get operation breakdown
        operation_stats = {}
        for m in metrics:
            if m.operation not in operation_stats:
                operation_stats[m.operation] = {
                    'count': 0,
                    'success_count': 0,
                    'total_duration': 0.0,
                    'total_memory': 0.0,
                    'total_rows': 0
                }

            operation_stats[m.operation]['count'] += 1
            operation_stats[m.operation]['success_count'] += 1 if m.success else 0
            operation_stats[m.operation]['total_duration'] += m.duration
            operation_stats[m.operation]['total_memory'] += m.memory_end - m.memory_start
            operation_stats[m.operation]['total_rows'] += m.rows_processed

        return {
            'summary': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': success_rate,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'total_memory_increase': total_memory_increase,
                'average_memory_increase': avg_memory_increase,
                'total_rows_processed': total_rows_processed,
                'average_rows_per_second': avg_rows_per_second
            },
            'operation_breakdown': operation_stats,
            'system_stats': {
                'avg_cpu_percent': sum(self.system_stats['cpu']) / len(self.system_stats['cpu']) if self.system_stats['cpu'] else 0,
                'avg_memory_percent': sum(self.system_stats['memory']) / len(self.system_stats['memory']) if self.system_stats['memory'] else 0,
                'current_snapshots': len(self.system_snapshots)
            }
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.system_snapshots:
            return {'status': 'no_data', 'message': 'No system snapshots available'}

        latest_snapshot = self.system_snapshots[-1]

        # Determine health status
        status = 'healthy'
        issues = []

        if latest_snapshot.memory_percent > self.thresholds['memory_critical'] * 100:
            status = 'critical'
            issues.append(f'Critical memory usage: {latest_snapshot.memory_percent:.1f}%')
        elif latest_snapshot.memory_percent > self.thresholds['memory_warning'] * 100:
            status = 'warning'
            issues.append(f'High memory usage: {latest_snapshot.memory_percent:.1f}%')

        if latest_snapshot.cpu_percent > self.thresholds['cpu_warning'] * 100:
            if status != 'critical':
                status = 'warning'
            issues.append(f'High CPU usage: {latest_snapshot.cpu_percent:.1f}%')

        return {
            'status': status,
            'issues': issues,
            'current_snapshot': latest_snapshot.to_dict(),
            'metrics_history_size': len(self.metrics_history),
            'active_operation': self.current_operation
        }

    def export_metrics(self, output_path: str, operation: Optional[str] = None,
                      format: str = 'json') -> str:
        """Export performance metrics to file.

        Args:
            output_path: Path to save metrics
            operation: Specific operation to export, or None for all
            format: Export format ('json' or 'csv')

        Returns:
            Path where metrics were saved
        """
        metrics = self.metrics_history
        if operation:
            metrics = [m for m in metrics if m.operation == operation]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            data = {
                'metrics': [m.to_dict() for m in metrics],
                'system_stats': dict(self.system_stats),
                'exported_at': datetime.now().isoformat()
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'csv':
            # Convert to DataFrame and save as CSV
            df_data = []
            for m in metrics:
                row = m.to_dict()
                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported {len(metrics)} metrics to {output_path}")
        return str(output_path)

    def clear_metrics(self):
        """Clear all performance metrics."""
        self.metrics_history.clear()
        self.system_snapshots.clear()
        self.system_stats.clear()
        self.logger.info("Performance metrics cleared")

    def set_thresholds(self, **thresholds):
        """Update performance thresholds.

        Args:
            **thresholds: Threshold values to update
        """
        self.thresholds.update(thresholds)
        self.logger.info(f"Updated performance thresholds: {thresholds}")

    def __del__(self):
        """Cleanup on destruction."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


# Context manager for performance monitoring
class PerformanceContext:
    """Context manager for automatic performance monitoring."""

    def __init__(self, operation: str, rows_processed: int = 0,
                 monitor: Optional[PerformanceMonitor] = None):
        """Initialize performance context.

        Args:
            operation: Operation name
            rows_processed: Number of rows being processed
            monitor: Performance monitor instance
        """
        self.operation = operation
        self.rows_processed = rows_processed
        self.monitor = monitor or performance_monitor
        self.operation_id = None
        self.metadata = {'rows_processed': rows_processed}

    def __enter__(self):
        """Enter context - start monitoring."""
        self.operation_id = self.monitor.start_operation(self.operation, self.rows_processed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - end monitoring."""
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        self.monitor.end_operation(
            self.operation_id,
            success=success,
            error_message=error_message,
            metadata=self.metadata
        )

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the operation."""
        self.metadata[key] = value

    def update_peak_memory(self):
        """Update peak memory for current operation."""
        self.monitor.update_peak_memory()


# Decorator for performance monitoring
def monitor_performance(operation_name: Optional[str] = None):
    """Decorator for automatic performance monitoring.

    Args:
        operation_name: Operation name, or None to use function name
    """
    def decorator(func):
        name = operation_name or func.__name__

        def wrapper(*args, **kwargs):
            with PerformanceContext(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator