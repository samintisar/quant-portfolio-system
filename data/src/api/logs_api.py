"""
Logs API class for preprocessing log management.

This module provides API endpoints for managing preprocessing logs,
including retrieval, filtering, and export functionality.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import sqlite3
import os


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Log entry data structure."""
    id: str
    timestamp: str
    level: LogLevel
    message: str
    dataset_id: str
    operation: str
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


@dataclass
class LogFilter:
    """Log filter criteria."""
    dataset_id: Optional[str] = None
    level: Optional[LogLevel] = None
    operation: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    user_id: Optional[str] = None
    message_contains: Optional[str] = None
    limit: Optional[int] = 100


class LogsAPI:
    """API class for preprocessing log management."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize LogsAPI with database path."""
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', '..', 'storage', 'preprocessing_logs.db')
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for log storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    metadata TEXT,
                    user_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_id ON logs(dataset_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON logs(level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_operation ON logs(operation)')

            conn.commit()

    def get_logs_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get logs endpoint with filtering capabilities.

        Args:
            request_data: Dictionary containing filter criteria

        Returns:
            Dictionary with logs and metadata
        """
        try:
            # Extract dataset_id from path (would normally come from URL path)
            dataset_id = request_data.get('dataset_id')

            if not dataset_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: dataset_id',
                    'code': 400
                }

            # Validate dataset_id format
            if not isinstance(dataset_id, str) or len(dataset_id.strip()) == 0:
                return {
                    'status': 'error',
                    'message': 'Invalid dataset_id format',
                    'code': 400
                }

            # Build filter from request data
            log_filter = LogFilter(
                dataset_id=dataset_id,
                level=LogLevel(request_data.get('level')) if request_data.get('level') else None,
                operation=request_data.get('operation'),
                start_time=request_data.get('start_time'),
                end_time=request_data.get('end_time'),
                user_id=request_data.get('user_id'),
                message_contains=request_data.get('message_contains'),
                limit=request_data.get('limit', 100)
            )

            # Query logs from database
            logs = self._query_logs(log_filter)

            # Apply additional filtering
            if log_filter.message_contains:
                logs = [log for log in logs if log_filter.message_contains.lower() in log.message.lower()]

            # Limit results
            if log_filter.limit:
                logs = logs[:log_filter.limit]

            return {
                'status': 'success',
                'data': {
                    'logs': [asdict(log) for log in logs],
                    'total_count': len(logs),
                    'dataset_id': dataset_id
                },
                'metadata': {
                    'filter_applied': asdict(log_filter),
                    'query_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to retrieve logs: {str(e)}',
                'code': 500
            }

    def _query_logs(self, log_filter: LogFilter) -> List[LogEntry]:
        """Query logs from database based on filter criteria."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM logs WHERE dataset_id = ?"
            params = [log_filter.dataset_id]

            # Add filters
            if log_filter.level:
                query += " AND level = ?"
                params.append(log_filter.level.value)

            if log_filter.operation:
                query += " AND operation = ?"
                params.append(log_filter.operation)

            if log_filter.start_time:
                query += " AND timestamp >= ?"
                params.append(log_filter.start_time)

            if log_filter.end_time:
                query += " AND timestamp <= ?"
                params.append(log_filter.end_time)

            if log_filter.user_id:
                query += " AND user_id = ?"
                params.append(log_filter.user_id)

            # Order by timestamp (newest first)
            query += " ORDER BY timestamp DESC"

            # Add limit if specified
            if log_filter.limit:
                query += " LIMIT ?"
                params.append(log_filter.limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to LogEntry objects
            logs = []
            for row in rows:
                log_entry = LogEntry(
                    id=row[0],
                    timestamp=row[1],
                    level=LogLevel(row[2]),
                    message=row[3],
                    dataset_id=row[4],
                    operation=row[5],
                    metadata=json.loads(row[6]) if row[6] else None,
                    user_id=row[7]
                )
                logs.append(log_entry)

            return logs

    def log_export_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export logs endpoint with various formats.

        Args:
            request_data: Dictionary containing export parameters

        Returns:
            Dictionary with export data or error
        """
        try:
            dataset_id = request_data.get('dataset_id')
            format_type = request_data.get('format', 'json')

            if not dataset_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: dataset_id',
                    'code': 400
                }

            # Get logs
            log_filter = LogFilter(dataset_id=dataset_id, limit=request_data.get('limit', 1000))
            logs = self._query_logs(log_filter)

            # Export in requested format
            if format_type.lower() == 'json':
                export_data = json.dumps([asdict(log) for log in logs], indent=2)
                content_type = 'application/json'
            elif format_type.lower() == 'csv':
                # Simple CSV format
                import io
                output = io.StringIO()
                output.write('id,timestamp,level,message,dataset_id,operation,user_id\n')
                for log in logs:
                    output.write(f'{log.id},{log.timestamp},{log.level.value},"{log.message}",{log.dataset_id},{log.operation},{log.user_id or ""}\n')
                export_data = output.getvalue()
                content_type = 'text/csv'
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported export format: {format_type}',
                    'code': 400
                }

            return {
                'status': 'success',
                'data': {
                    'export_data': export_data,
                    'format': format_type,
                    'content_type': content_type,
                    'record_count': len(logs)
                },
                'metadata': {
                    'export_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Export failed: {str(e)}',
                'code': 500
            }

    def log_aggregation_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log aggregation endpoint for statistics.

        Args:
            request_data: Dictionary containing aggregation parameters

        Returns:
            Dictionary with aggregated log statistics
        """
        try:
            dataset_id = request_data.get('dataset_id')
            if not dataset_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: dataset_id',
                    'code': 400
                }

            # Get all logs for dataset
            log_filter = LogFilter(dataset_id=dataset_id)
            logs = self._query_logs(log_filter)

            # Calculate aggregations
            aggregations = {
                'total_logs': len(logs),
                'by_level': {},
                'by_operation': {},
                'by_hour': {},
                'recent_errors': 0
            }

            # Group by level
            for log in logs:
                level = log.level.value
                aggregations['by_level'][level] = aggregations['by_level'].get(level, 0) + 1

                operation = log.operation
                aggregations['by_operation'][operation] = aggregations['by_operation'].get(operation, 0) + 1

                # Count recent errors (last 24 hours)
                if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    log_time = datetime.fromisoformat(log.timestamp)
                    if datetime.now() - log_time < timedelta(hours=24):
                        aggregations['recent_errors'] += 1

            return {
                'status': 'success',
                'data': aggregations,
                'metadata': {
                    'dataset_id': dataset_id,
                    'aggregation_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Aggregation failed: {str(e)}',
                'code': 500
            }

    def add_log_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add log entry endpoint.

        Args:
            request_data: Dictionary containing log entry data

        Returns:
            Dictionary with success/error status
        """
        try:
            # Validate required fields
            required_fields = ['level', 'message', 'dataset_id', 'operation']
            for field in required_fields:
                if field not in request_data:
                    return {
                        'status': 'error',
                        'message': f'Missing required field: {field}',
                        'code': 400
                    }

            # Create log entry
            log_entry = LogEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                level=LogLevel(request_data['level']),
                message=request_data['message'],
                dataset_id=request_data['dataset_id'],
                operation=request_data['operation'],
                metadata=request_data.get('metadata'),
                user_id=request_data.get('user_id')
            )

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (id, timestamp, level, message, dataset_id, operation, metadata, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log_entry.id,
                    log_entry.timestamp,
                    log_entry.level.value,
                    log_entry.message,
                    log_entry.dataset_id,
                    log_entry.operation,
                    json.dumps(log_entry.metadata) if log_entry.metadata else None,
                    log_entry.user_id
                ))
                conn.commit()

            return {
                'status': 'success',
                'data': {
                    'log_id': log_entry.id,
                    'timestamp': log_entry.timestamp
                },
                'metadata': {
                    'operation': 'log_added',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to add log: {str(e)}',
                'code': 500
            }