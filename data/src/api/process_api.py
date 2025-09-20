"""
Process API class for data processing operations.

This module provides API endpoints for data preprocessing,
validation, and transformation operations.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.feature_service import FeatureGenerator as FeatureService
from services.validation_service import ValidationService
from lib.cleaning import DataCleaner
from lib.validation import DataValidator


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    """Processing job data structure."""
    id: str
    status: ProcessingStatus
    input_data: Dict[str, Any]
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0


@dataclass
class ProcessingRequest:
    """Processing request data structure."""
    data: Dict[str, Any]
    operations: List[str]
    config: Optional[Dict[str, Any]] = None
    validation_rules: Optional[List[Dict[str, Any]]] = None
    output_format: str = "json"


class ProcessAPI:
    """API class for data processing operations."""

    def __init__(self, max_workers: int = 4):
        """Initialize ProcessAPI."""
        self.max_workers = max_workers
        self.jobs: Dict[str, ProcessingJob] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize services
        self.feature_service = FeatureService()
        self.validation_service = ValidationService()
        self.data_cleaner = DataCleaner()
        self.data_validator = DataValidator()

    def process_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main data processing endpoint.

        Args:
            request_data: Dictionary containing processing request

        Returns:
            Dictionary with processing response
        """
        try:
            # Validate required fields
            if not isinstance(request_data, dict):
                return {
                    'status': 'error',
                    'message': 'Request data must be a dictionary',
                    'code': 400
                }

            required_fields = ['data', 'operations']
            for field in required_fields:
                if field not in request_data:
                    return {
                        'status': 'error',
                        'message': f'Missing required field: {field}',
                        'code': 400
                    }

            data = request_data['data']
            operations = request_data['operations']
            config = request_data.get('config', {})

            # Validate data format
            if not isinstance(data, dict):
                return {
                    'status': 'error',
                    'message': 'Data must be a dictionary',
                    'code': 400
                }

            # Check data size limits
            estimated_size = self._estimate_data_size(data)
            if estimated_size > 10_000_000:  # 10MB limit
                return {
                    'status': 'error',
                    'message': f'Data size {estimated_size} bytes exceeds maximum allowed 10MB',
                    'code': 413
                }

            # Create processing job
            job_id = str(uuid.uuid4())
            job = ProcessingJob(
                id=job_id,
                status=ProcessingStatus.PENDING,
                input_data=data,
                config=config,
                created_at=datetime.now().isoformat()
            )

            with self.lock:
                self.jobs[job_id] = job

            # Start processing in background
            future = self.executor.submit(self._process_data, job)
            job.future = future  # Store future for cancellation

            return {
                'status': 'success',
                'data': {
                    'job_id': job_id,
                    'status': job.status.value,
                    'message': 'Processing started',
                    'estimated_duration': self._estimate_processing_duration(data, operations)
                },
                'metadata': {
                    'operation': 'process_data',
                    'timestamp': datetime.now().isoformat(),
                    'operations_queued': len(operations)
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Processing initialization failed: {str(e)}',
                'code': 500
            }

    def _process_data(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process data according to job specifications."""
        try:
            with self.lock:
                job.status = ProcessingStatus.PROCESSING
                job.started_at = datetime.now().isoformat()

            # Convert input data to DataFrame
            df = self._convert_to_dataframe(job.input_data)

            results = {}
            total_operations = len(job.config.get('operations', []))
            completed_operations = 0

            for operation in job.config.get('operations', []):
                try:
                    # Update progress
                    progress = (completed_operations / total_operations) * 100
                    with self.lock:
                        job.progress = progress

                    # Execute operation
                    if operation == 'clean':
                        result = self._clean_data(df)
                    elif operation == 'validate':
                        result = self._validate_data(df)
                    elif operation == 'generate_features':
                        features = job.config.get('features', ['returns', 'volatility'])
                        result = self._generate_features(df, features)
                    elif operation == 'normalize':
                        method = job.config.get('normalization_method', 'zscore')
                        result = self._normalize_data(df, method)
                    else:
                        raise ValueError(f'Unknown operation: {operation}')

                    results[operation] = result
                    completed_operations += 1

                except Exception as e:
                    with self.lock:
                        job.status = ProcessingStatus.FAILED
                        job.error_message = f'Operation {operation} failed: {str(e)}'
                        job.completed_at = datetime.now().isoformat()
                    return

            # Complete job
            with self.lock:
                job.status = ProcessingStatus.COMPLETED
                job.result = results
                job.progress = 100.0
                job.completed_at = datetime.now().isoformat()

        except Exception as e:
            with self.lock:
                job.status = ProcessingStatus.FAILED
                job.error_message = f'Processing failed: {str(e)}'
                job.completed_at = datetime.now().isoformat()

    def _convert_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame."""
        try:
            df = pd.DataFrame(data)

            # Handle dates if present
            if 'dates' in df.columns:
                df['dates'] = pd.to_datetime(df['dates'])
                df.set_index('dates', inplace=True)

            return df
        except Exception as e:
            raise ValueError(f'Data conversion failed: {str(e)}')

    def _clean_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Clean data using data cleaner."""
        try:
            # Clean missing values
            cleaned_df = self.data_cleaner.forward_fill(df)

            # Detect and handle outliers
            outliers = self.data_cleaner.detect_outliers_zscore(cleaned_df, threshold=3.0)
            cleaned_df = self.data_cleaner.clip_outliers(cleaned_df, outliers)

            # Calculate data quality score
            quality_score = self.data_cleaner.calculate_data_quality_score(cleaned_df)

            return {
                'cleaned_data': cleaned_df.to_dict('records'),
                'quality_score': quality_score,
                'outliers_detected': outliers,
                'rows_processed': len(cleaned_df)
            }
        except Exception as e:
            raise ValueError(f'Data cleaning failed: {str(e)}')

    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data using data validator."""
        try:
            # Validate price data
            price_validation = self.data_validator.validate_price_data(df)

            # Validate OHLC relationships if applicable
            ohlc_validation = {}
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_validation = self.data_validator.validate_ohlc_relationships(df)

            # Check for time gaps
            time_gaps = self.data_cleaner.detect_time_gaps(df)

            return {
                'price_validation': price_validation,
                'ohlc_validation': ohlc_validation,
                'time_gaps': time_gaps,
                'overall_validity': price_validation.get('is_valid', True) and not time_gaps
            }
        except Exception as e:
            raise ValueError(f'Data validation failed: {str(e)}')

    def _generate_features(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Generate features from data."""
        try:
            result = self.feature_service.generate_features_batch(df, features=features)

            # Convert to serializable format
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    serializable_result[key] = value.to_dict('records')
                elif isinstance(value, pd.Series):
                    serializable_result[key] = value.to_dict()
                else:
                    serializable_result[key] = value

            return {
                'features': serializable_result,
                'features_generated': features,
                'data_points': len(df)
            }
        except Exception as e:
            raise ValueError(f'Feature generation failed: {str(e)}')

    def _normalize_data(self, df: pd.DataFrame, method: str) -> Dict[str, Any]:
        """Normalize data using specified method."""
        try:
            # This would integrate with normalization library
            # For now, return basic normalization info
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            if method == 'zscore':
                normalized_df = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
            elif method == 'minmax':
                normalized_df = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
            else:
                raise ValueError(f'Unknown normalization method: {method}')

            return {
                'normalized_data': normalized_df.to_dict('records'),
                'method': method,
                'columns_normalized': list(numeric_columns)
            }
        except Exception as e:
            raise ValueError(f'Data normalization failed: {str(e)}')

    def _estimate_data_size(self, data: Dict[str, Any]) -> int:
        """Estimate data size in bytes."""
        return len(json.dumps(data))

    def _estimate_processing_duration(self, data: Dict[str, Any], operations: List[str]) -> float:
        """Estimate processing duration in seconds."""
        data_size = self._estimate_data_size(data)
        base_time = data_size / 1000000  # Base time per MB

        # Add time for each operation
        operation_times = {
            'clean': 1.0,
            'validate': 0.5,
            'generate_features': 2.0,
            'normalize': 1.0
        }

        total_time = base_time
        for operation in operations:
            total_time += operation_times.get(operation, 1.0)

        return total_time

    def get_job_status_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing job status."""
        try:
            job_id = request_data.get('job_id')
            if not job_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: job_id',
                    'code': 400
                }

            with self.lock:
                job = self.jobs.get(job_id)

            if not job:
                return {
                    'status': 'error',
                    'message': f'Job not found: {job_id}',
                    'code': 404
                }

            response_data = {
                'job_id': job.id,
                'status': job.status.value,
                'progress': job.progress,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at
            }

            if job.status == ProcessingStatus.COMPLETED:
                response_data['result'] = job.result
            elif job.status == ProcessingStatus.FAILED:
                response_data['error_message'] = job.error_message

            return {
                'status': 'success',
                'data': response_data,
                'metadata': {
                    'query_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Status query failed: {str(e)}',
                'code': 500
            }

    def cancel_job_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel processing job."""
        try:
            job_id = request_data.get('job_id')
            if not job_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: job_id',
                    'code': 400
                }

            with self.lock:
                job = self.jobs.get(job_id)

            if not job:
                return {
                    'status': 'error',
                    'message': f'Job not found: {job_id}',
                    'code': 404
                }

            if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
                return {
                    'status': 'error',
                    'message': f'Job cannot be cancelled (status: {job.status.value})',
                    'code': 400
                }

            # Cancel the job
            with self.lock:
                job.status = ProcessingStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()

            return {
                'status': 'success',
                'data': {
                    'job_id': job_id,
                    'status': job.status.value,
                    'message': 'Job cancelled successfully'
                },
                'metadata': {
                    'operation': 'cancel_job',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Job cancellation failed: {str(e)}',
                'code': 500
            }