"""
Quality API class for data quality metrics and monitoring.

This module provides API endpoints for data quality assessment,
historical quality tracking, and quality management.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
from statistics import mean, stdev

from services.validation_service import ValidationService
from lib.validation import DataValidator
from lib.cleaning import DataCleaner


class QualityMetric(Enum):
    """Quality metric types."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class QualityScore:
    """Quality score data structure."""
    metric: QualityMetric
    score: float
    weight: float = 1.0
    threshold: float = 0.8
    details: Optional[Dict[str, Any]] = None


@dataclass
class QualityReport:
    """Quality report data structure."""
    dataset_id: str
    overall_score: float
    metric_scores: List[QualityScore]
    timestamp: str
    data_points: int
    issues_found: int
    recommendations: List[str]


class QualityAPI:
    """API class for data quality metrics and monitoring."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize QualityAPI with database path."""
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', '..', 'storage', 'quality_metrics.db')
        self.validation_service = ValidationService()
        self.data_validator = DataValidator()
        self.data_cleaner = DataCleaner()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for quality metrics."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create quality_reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    metric_scores TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_points INTEGER NOT NULL,
                    issues_found INTEGER NOT NULL,
                    recommendations TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_id ON quality_reports(dataset_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON quality_reports(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_overall_score ON quality_reports(overall_score)')

            conn.commit()

    def quality_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main quality assessment endpoint.

        Args:
            request_data: Dictionary containing quality assessment request

        Returns:
            Dictionary with quality assessment results
        """
        try:
            dataset_id = request_data.get('dataset_id')
            if not dataset_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: dataset_id',
                    'code': 400
                }

            # Get data for assessment (would normally come from data store)
            data = request_data.get('data')
            if not data:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: data',
                    'code': 400
                }

            # Convert to DataFrame
            try:
                df = pd.DataFrame(data)
                if 'dates' in df.columns:
                    df['dates'] = pd.to_datetime(df['dates'])
                    df.set_index('dates', inplace=True)
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Invalid data format: {str(e)}',
                    'code': 400
                }

            # Calculate quality metrics
            quality_report = self._calculate_quality_metrics(dataset_id, df)

            # Store report in database
            self._store_quality_report(quality_report)

            return {
                'status': 'success',
                'data': {
                    'dataset_id': quality_report.dataset_id,
                    'overall_score': quality_report.overall_score,
                    'metric_scores': [asdict(score) for score in quality_report.metric_scores],
                    'data_points': quality_report.data_points,
                    'issues_found': quality_report.issues_found,
                    'recommendations': quality_report.recommendations
                },
                'metadata': {
                    'assessment_timestamp': quality_report.timestamp,
                    'metrics_calculated': len(quality_report.metric_scores)
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Quality assessment failed: {str(e)}',
                'code': 500
            }

    def _calculate_quality_metrics(self, dataset_id: str, df: pd.DataFrame) -> QualityReport:
        """Calculate comprehensive quality metrics for dataset."""
        metric_scores = []
        total_issues = 0
        recommendations = []

        # 1. Completeness Score
        completeness_score = self._calculate_completeness(df)
        metric_scores.append(completeness_score)
        total_issues += completeness_score.details.get('missing_values', 0)

        # 2. Accuracy Score (based on price validation)
        accuracy_score = self._calculate_accuracy(df)
        metric_scores.append(accuracy_score)
        total_issues += accuracy_score.details.get('validation_issues', 0)

        # 3. Consistency Score
        consistency_score = self._calculate_consistency(df)
        metric_scores.append(consistency_score)
        total_issues += consistency_score.details.get('inconsistencies', 0)

        # 4. Timeliness Score
        timeliness_score = self._calculate_timeliness(df)
        metric_scores.append(timeliness_score)
        total_issues += timeliness_score.details.get('timeliness_issues', 0)

        # 5. Validity Score
        validity_score = self._calculate_validity(df)
        metric_scores.append(validity_score)
        total_issues += validity_score.details.get('validity_issues', 0)

        # 6. Uniqueness Score
        uniqueness_score = self._calculate_uniqueness(df)
        metric_scores.append(uniqueness_score)
        total_issues += uniqueness_score.details.get('duplicate_rows', 0)

        # Calculate weighted overall score
        weighted_sum = sum(score.score * score.weight for score in metric_scores)
        total_weight = sum(score.weight for score in metric_scores)
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores)

        return QualityReport(
            dataset_id=dataset_id,
            overall_score=overall_score,
            metric_scores=metric_scores,
            timestamp=datetime.now().isoformat(),
            data_points=len(df),
            issues_found=total_issues,
            recommendations=recommendations
        )

    def _calculate_completeness(self, df: pd.DataFrame) -> QualityScore:
        """Calculate completeness score based on missing values."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()

        completeness_score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0

        details = {
            'missing_values': int(missing_cells),
            'total_cells': int(total_cells),
            'missing_by_column': df.isnull().sum().to_dict(),
            'completeness_ratio': float(completeness_score)
        }

        return QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=completeness_score,
            weight=0.25,
            threshold=0.9,
            details=details
        )

    def _calculate_accuracy(self, df: pd.DataFrame) -> QualityScore:
        """Calculate accuracy score based on data validation."""
        validation_issues = 0

        # Validate price data if available
        if 'close' in df.columns:
            price_validation = self.data_validator.validate_price_data(df)
            if not price_validation.get('is_valid', True):
                validation_issues += len(price_validation.get('validation_issues', []))

        # Validate OHLC relationships if available
        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_columns):
            ohlc_validation = self.data_validator.validate_ohlc_relationships(df)
            validation_issues += len(ohlc_validation.get('validation_issues', []))

        # Calculate accuracy score (fewer issues = higher score)
        max_possible_issues = len(df) * 2  # Rough estimate
        accuracy_score = 1.0 - (validation_issues / max_possible_issues) if max_possible_issues > 0 else 1.0

        details = {
            'validation_issues': validation_issues,
            'max_possible_issues': max_possible_issues,
            'accuracy_ratio': float(accuracy_score)
        }

        return QualityScore(
            metric=QualityMetric.ACCURACY,
            score=accuracy_score,
            weight=0.2,
            threshold=0.85,
            details=details
        )

    def _calculate_consistency(self, df: pd.DataFrame) -> QualityScore:
        """Calculate consistency score based on data consistency."""
        inconsistencies = 0

        # Check for inconsistent OHLC relationships
        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_columns):
            # High should be >= low
            inconsistent_hl = (df['high'] < df['low']).sum()
            inconsistencies += inconsistent_hl

            # High should be >= open and close
            inconsistent_high = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
            inconsistencies += inconsistent_high

            # Low should be <= open and close
            inconsistent_low = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
            inconsistencies += inconsistent_low

        # Check for negative prices/volumes
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        negative_values = (df[numeric_columns] < 0).sum().sum()
        inconsistencies += negative_values

        # Calculate consistency score
        max_possible_inconsistencies = len(df) * 3 + len(numeric_columns) * len(df)
        consistency_score = 1.0 - (inconsistencies / max_possible_inconsistencies) if max_possible_inconsistencies > 0 else 1.0

        details = {
            'inconsistencies': inconsistencies,
            'max_possible_inconsistencies': max_possible_inconsistencies,
            'consistency_ratio': float(consistency_score)
        }

        return QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=consistency_score,
            weight=0.15,
            threshold=0.9,
            details=details
        )

    def _calculate_timeliness(self, df: pd.DataFrame) -> QualityScore:
        """Calculate timeliness score based on data recency and gaps."""
        timeliness_issues = 0

        # Check for time gaps if DataFrame has datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            time_gaps = self.data_cleaner.detect_time_gaps(df)
            timeliness_issues += len(time_gaps)

            # Check data recency (if within last 7 days)
            latest_date = df.index.max()
            days_since_latest = (datetime.now() - latest_date).days
            if days_since_latest > 7:
                timeliness_issues += 1

        # Calculate timeliness score
        max_possible_issues = len(df) + 1  # Time gaps + recency
        timeliness_score = 1.0 - (timeliness_issues / max_possible_issues) if max_possible_issues > 0 else 1.0

        details = {
            'timeliness_issues': timeliness_issues,
            'max_possible_issues': max_possible_issues,
            'timeliness_ratio': float(timeliness_score)
        }

        return QualityScore(
            metric=QualityMetric.TIMELINESS,
            score=timeliness_score,
            weight=0.1,
            threshold=0.8,
            details=details
        )

    def _calculate_validity(self, df: pd.DataFrame) -> QualityScore:
        """Calculate validity score based on data type and range validation."""
        validity_issues = 0

        # Check for infinite values
        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        validity_issues += infinite_values

        # Check for extreme values (potential outliers)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if len(df[col]) > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                extreme_values = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                validity_issues += extreme_values

        # Calculate validity score
        max_possible_issues = len(numeric_columns) * len(df) * 2  # Infinite + extreme values
        validity_score = 1.0 - (validity_issues / max_possible_issues) if max_possible_issues > 0 else 1.0

        details = {
            'validity_issues': validity_issues,
            'max_possible_issues': max_possible_issues,
            'validity_ratio': float(validity_score)
        }

        return QualityScore(
            metric=QualityMetric.VALIDITY,
            score=validity_score,
            weight=0.15,
            threshold=0.85,
            details=details
        )

    def _calculate_uniqueness(self, df: pd.DataFrame) -> QualityScore:
        """Calculate uniqueness score based on duplicate detection."""
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()

        uniqueness_score = 1.0 - (duplicate_rows / total_rows) if total_rows > 0 else 1.0

        details = {
            'duplicate_rows': int(duplicate_rows),
            'total_rows': int(total_rows),
            'uniqueness_ratio': float(uniqueness_score)
        }

        return QualityScore(
            metric=QualityMetric.UNIQUENESS,
            score=uniqueness_score,
            weight=0.15,
            threshold=0.95,
            details=details
        )

    def _generate_recommendations(self, metric_scores: List[QualityScore]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []

        for score in metric_scores:
            if score.score < score.threshold:
                if score.metric == QualityMetric.COMPLETENESS:
                    recommendations.append("Consider data imputation strategies for missing values")
                elif score.metric == QualityMetric.ACCURACY:
                    recommendations.append("Implement data validation rules and outlier detection")
                elif score.metric == QualityMetric.CONSISTENCY:
                    recommendations.append("Review data consistency rules and validate OHLC relationships")
                elif score.metric == QualityMetric.TIMELINESS:
                    recommendations.append("Ensure regular data updates and monitor for time gaps")
                elif score.metric == QualityMetric.VALIDITY:
                    recommendations.append("Implement data range validation and outlier handling")
                elif score.metric == QualityMetric.UNIQUENESS:
                    recommendations.append("Remove duplicate records and implement uniqueness constraints")

        return recommendations

    def _store_quality_report(self, report: QualityReport):
        """Store quality report in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO quality_reports (
                    dataset_id, overall_score, metric_scores, timestamp,
                    data_points, issues_found, recommendations
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.dataset_id,
                report.overall_score,
                json.dumps([asdict(score) for score in report.metric_scores]),
                report.timestamp,
                report.data_points,
                report.issues_found,
                json.dumps(report.recommendations)
            ))
            conn.commit()

    def get_historical_quality_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical quality data for a dataset."""
        try:
            dataset_id = request_data.get('dataset_id')
            if not dataset_id:
                return {
                    'status': 'error',
                    'message': 'Missing required parameter: dataset_id',
                    'code': 400
                }

            # Get historical data from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM quality_reports
                    WHERE dataset_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', (dataset_id,))
                rows = cursor.fetchall()

            historical_data = []
            for row in rows:
                report = {
                    'id': row[0],
                    'dataset_id': row[1],
                    'overall_score': row[2],
                    'metric_scores': json.loads(row[3]),
                    'timestamp': row[4],
                    'data_points': row[5],
                    'issues_found': row[6],
                    'recommendations': json.loads(row[7]) if row[7] else []
                }
                historical_data.append(report)

            return {
                'status': 'success',
                'data': {
                    'dataset_id': dataset_id,
                    'historical_reports': historical_data,
                    'total_reports': len(historical_data)
                },
                'metadata': {
                    'query_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Historical quality query failed: {str(e)}',
                'code': 500
            }

    def get_quality_thresholds_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get quality threshold configuration."""
        try:
            # Default thresholds
            thresholds = {
                'completeness': 0.9,
                'accuracy': 0.85,
                'consistency': 0.9,
                'timeliness': 0.8,
                'validity': 0.85,
                'uniqueness': 0.95,
                'overall': 0.85
            }

            # Allow override from request
            if 'custom_thresholds' in request_data:
                thresholds.update(request_data['custom_thresholds'])

            return {
                'status': 'success',
                'data': {
                    'thresholds': thresholds,
                    'description': 'Quality score thresholds for different metrics'
                },
                'metadata': {
                    'threshold_version': '1.0',
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Threshold retrieval failed: {str(e)}',
                'code': 500
            }