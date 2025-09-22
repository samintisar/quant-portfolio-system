"""
Data validation service for financial data quality assessment.

This service provides comprehensive validation capabilities for financial data
including integrity checks, quality scoring, outlier detection, and data
cleansing recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

from models.financial_instrument import FinancialInstrument
from models.price_data import PriceData
from lib.cleaning import DataCleaner


class ValidationLevel(Enum):
    """Enumeration of validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class DataQuality(Enum):
    """Enumeration of data quality levels."""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    UNUSABLE = 1


class ValidationCategory(Enum):
    """Enumeration of validation categories."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    quality_score: float
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    category_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    category: ValidationCategory
    description: str
    severity: str  # 'error', 'warning', 'info'
    validator: callable
    threshold: Optional[float] = None
    enabled: bool = True


class DataValidator:
    """
    Comprehensive data validation service for financial data.

    This service provides validation capabilities for various types of financial
    data including price data, returns, and features with configurable
    validation rules and quality scoring.
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize the data validator.

        Args:
            validation_level: Level of strictness for validation
        """
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_validation_rules()
        self.validation_history: List[ValidationResult] = []

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules."""
        rules = [
            # Completeness rules
            ValidationRule(
                name="missing_values",
                category=ValidationCategory.COMPLETENESS,
                description="Check for missing values in data",
                severity="warning",
                validator=self._validate_missing_values,
                threshold=0.05  # 5% threshold
            ),
            ValidationRule(
                name="consecutive_missing",
                category=ValidationCategory.COMPLETENESS,
                description="Check for consecutive missing values",
                severity="error",
                validator=self._validate_consecutive_missing,
                threshold=3  # Max consecutive missing
            ),
            # Accuracy rules
            ValidationRule(
                name="negative_prices",
                category=ValidationCategory.ACCURACY,
                description="Check for negative prices",
                severity="error",
                validator=self._validate_negative_prices
            ),
            ValidationRule(
                name="negative_volumes",
                category=ValidationCategory.ACCURACY,
                description="Check for negative volumes",
                severity="error",
                validator=self._validate_negative_volumes
            ),
            ValidationRule(
                name="zero_prices",
                category=ValidationCategory.ACCURACY,
                description="Check for zero prices",
                severity="warning",
                validator=self._validate_zero_prices
            ),
            # Consistency rules
            ValidationRule(
                name="ohlc_consistency",
                category=ValidationCategory.CONSISTENCY,
                description="Check OHLC data consistency",
                severity="error",
                validator=self._validate_ohlc_consistency
            ),
            ValidationRule(
                name="monotonic_timestamps",
                category=ValidationCategory.CONSISTENCY,
                description="Check if timestamps are monotonic",
                severity="error",
                validator=self._validate_monotonic_timestamps
            ),
            # Timeliness rules
            ValidationRule(
                name="future_dates",
                category=ValidationCategory.TIMELINESS,
                description="Check for future dates in data",
                severity="error",
                validator=self._validate_future_dates
            ),
            ValidationRule(
                name="data_freshness",
                category=ValidationCategory.TIMELINESS,
                description="Check if data is recent enough",
                severity="warning",
                validator=self._validate_data_freshness,
                threshold=7  # Days threshold
            ),
            # Uniqueness rules
            ValidationRule(
                name="duplicate_timestamps",
                category=ValidationCategory.UNIQUENESS,
                description="Check for duplicate timestamps",
                severity="error",
                validator=self._validate_duplicate_timestamps
            ),
            # Validity rules
            ValidationRule(
                name="extreme_values",
                category=ValidationCategory.VALIDITY,
                description="Check for extreme values (outliers)",
                severity="warning",
                validator=self._validate_extreme_values,
                threshold=3.0  # Standard deviations
            ),
            ValidationRule(
                name="data_gaps",
                category=ValidationCategory.VALIDITY,
                description="Check for gaps in time series",
                severity="warning",
                validator=self._validate_data_gaps
            )
        ]
        return rules

    def validate_data(self, data: Union[pd.DataFrame, pd.Series, PriceData],
                    custom_rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """
        Validate financial data with comprehensive checks.

        Args:
            data: Data to validate (DataFrame, Series, or PriceData object)
            custom_rules: Additional validation rules to apply

        Returns:
            ValidationResult object with detailed validation results

        Examples:
        >>> validator = DataValidator()
        >>> prices = pd.Series([100, 101, 102, 103, 104])
        >>> result = validator.validate_data(prices)
        >>> print(f"Quality score: {result.quality_score:.2f}")
        """
        self.logger.info(f"Starting data validation for {type(data).__name__}")

        # Prepare data for validation
        if isinstance(data, PriceData):
            df = data.prices
            instrument = data.instrument
            metadata = {'source': data.source, 'frequency': data.frequency.value}
        elif isinstance(data, pd.DataFrame):
            df = data
            instrument = None
            metadata = {}
        elif isinstance(data, pd.Series):
            df = data.to_frame()
            instrument = None
            metadata = {}
        else:
            raise ValueError("Unsupported data type for validation")

        if df.empty:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                errors=["Data is empty"],
                warnings=[],
                recommendations=[],
                category_scores={},
                metadata=metadata
            )

        # Combine default and custom rules
        all_rules = self.rules.copy()
        if custom_rules:
            all_rules.extend(custom_rules)

        # Apply validation rules
        errors = []
        warnings = []
        recommendations = []
        category_scores = {}

        for rule in all_rules:
            if not rule.enabled:
                continue

            try:
                rule_result = rule.validator(df, rule)

                if rule_result['passed']:
                    # Rule passed, update category score
                    category = rule.category.value
                    if category not in category_scores:
                        category_scores[category] = 0.0
                    category_scores[category] += 1.0
                else:
                    # Rule failed, add to appropriate list
                    message = f"{rule.name}: {rule_result['message']}"

                    if rule.severity == 'error':
                        errors.append(message)
                        recommendations.append(f"Fix: {rule.description}")
                    elif rule.severity == 'warning':
                        warnings.append(message)
                        recommendations.append(f"Consider: {rule.description}")
                    else:
                        # Info level
                        recommendations.append(message)

            except Exception as e:
                error_msg = f"Error in rule {rule.name}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        # Calculate overall quality score
        total_rules = len([r for r in all_rules if r.enabled])
        if total_rules > 0:
            overall_score = sum(category_scores.values()) / total_rules
        else:
            overall_score = 0.0

        # Normalize category scores
        for category in category_scores:
            category_rules = len([r for r in all_rules if r.enabled and r.category.value == category])
            if category_rules > 0:
                category_scores[category] = category_scores[category] / category_rules

        # Determine overall validity based on validation level
        is_valid = self._determine_validity(errors, warnings, overall_score)

        # Add metadata
        metadata.update({
            'validation_timestamp': datetime.now().isoformat(),
            'data_points': len(df),
            'validation_level': self.validation_level.value,
            'total_rules': total_rules,
            'passed_rules': sum(category_scores.values())
        })

        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=overall_score,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            category_scores=category_scores,
            metadata=metadata
        )

        # Store in history
        self.validation_history.append(result)

        self.logger.info(f"Validation completed. Quality score: {overall_score:.3f}, "
                        f"Errors: {len(errors)}, Warnings: {len(warnings)}")

        return result

    def _determine_validity(self, errors: List[str], warnings: List[str], quality_score: float) -> bool:
        """Determine overall validity based on validation level."""
        if self.validation_level == ValidationLevel.STRICT:
            return len(errors) == 0 and len(warnings) == 0 and quality_score >= 0.9
        elif self.validation_level == ValidationLevel.MODERATE:
            return len(errors) == 0 and quality_score >= 0.7
        else:  # LENIENT
            return quality_score >= 0.5

    def _validate_missing_values(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate missing values in data."""
        missing_counts = data.isnull().sum()
        total_cells = data.shape[0] * data.shape[1]
        missing_percentage = missing_counts.sum() / total_cells if total_cells > 0 else 0

        threshold = rule.threshold or 0.05

        return {
            'passed': missing_percentage <= threshold,
            'message': f"{missing_percentage:.1%} missing values (threshold: {threshold:.1%})",
            'details': {
                'missing_percentage': missing_percentage,
                'missing_counts': missing_counts.to_dict(),
                'threshold': threshold
            }
        }

    def _validate_consecutive_missing(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate consecutive missing values."""
        max_consecutive = 0
        for column in data.columns:
            consecutive = 0
            for is_missing in data[column].isnull():
                if is_missing:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0

        threshold = rule.threshold or 3

        return {
            'passed': max_consecutive <= threshold,
            'message': f"Found {max_consecutive} consecutive missing values (threshold: {threshold})",
            'details': {
                'max_consecutive_missing': max_consecutive,
                'threshold': threshold
            }
        }

    def _validate_negative_prices(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate negative prices."""
        price_columns = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'price']]
        negative_prices = 0

        for col in price_columns:
            if col in data.columns:
                negative_prices += (data[col] < 0).sum()

        return {
            'passed': negative_prices == 0,
            'message': f"Found {negative_prices} negative price values",
            'details': {
                'negative_price_count': negative_prices,
                'price_columns': price_columns
            }
        }

    def _validate_negative_volumes(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate negative volumes."""
        negative_volumes = 0
        if 'volume' in data.columns:
            negative_volumes = (data['volume'] < 0).sum()

        return {
            'passed': negative_volumes == 0,
            'message': f"Found {negative_volumes} negative volume values",
            'details': {
                'negative_volume_count': negative_volumes
            }
        }

    def _validate_zero_prices(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate zero prices."""
        price_columns = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'price']]
        zero_prices = 0

        for col in price_columns:
            if col in data.columns:
                zero_prices += (data[col] == 0).sum()

        return {
            'passed': zero_prices == 0,
            'message': f"Found {zero_prices} zero price values",
            'details': {
                'zero_price_count': zero_prices,
                'price_columns': price_columns
            }
        }

    def _validate_ohlc_consistency(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate OHLC data consistency."""
        inconsistencies = 0

        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in ohlc_columns):
            # Check high >= low
            high_low_inconsistencies = (data['high'] < data['low']).sum()
            inconsistencies += high_low_inconsistencies

            # Check high >= open and close
            high_open_inconsistencies = (data['high'] < data['open']).sum()
            high_close_inconsistencies = (data['high'] < data['close']).sum()
            inconsistencies += high_open_inconsistencies + high_close_inconsistencies

            # Check low <= open and close
            low_open_inconsistencies = (data['low'] > data['open']).sum()
            low_close_inconsistencies = (data['low'] > data['close']).sum()
            inconsistencies += low_open_inconsistencies + low_close_inconsistencies

        return {
            'passed': inconsistencies == 0,
            'message': f"Found {inconsistencies} OHLC consistency violations",
            'details': {
                'inconsistency_count': inconsistencies,
                'ohlc_columns': ohlc_columns
            }
        }

    def _validate_monotonic_timestamps(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate if timestamps are monotonic increasing."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {
                'passed': False,
                'message': "Data index is not DatetimeIndex",
                'details': {'index_type': str(type(data.index))}
            }

        is_monotonic = data.index.is_monotonic_increasing

        return {
            'passed': is_monotonic,
            'message': "Timestamps are not monotonic increasing" if not is_monotonic else "Timestamps are monotonic",
            'details': {
                'is_monotonic': is_monotonic,
                'index_type': 'DatetimeIndex'
            }
        }

    def _validate_future_dates(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate for future dates in data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'passed': True, 'message': "Not a DatetimeIndex", 'details': {}}

        current_time = datetime.now()
        future_dates = (data.index > current_time).sum()

        return {
            'passed': future_dates == 0,
            'message': f"Found {future_dates} future dates in data",
            'details': {
                'future_date_count': future_dates,
                'current_time': current_time.isoformat()
            }
        }

    def _validate_data_freshness(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate if data is recent enough."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'passed': True, 'message': "Not a DatetimeIndex", 'details': {}}

        threshold = rule.threshold or 7  # days
        current_time = datetime.now()
        if len(data.index) > 0:
            latest_date = data.index.max()
            days_since_latest = (current_time - latest_date).days

            return {
                'passed': days_since_latest <= threshold,
                'message': f"Data is {days_since_latest} days old (threshold: {threshold} days)",
                'details': {
                    'days_since_latest': days_since_latest,
                    'latest_date': latest_date.isoformat(),
                    'threshold': threshold
                }
            }

        return {'passed': False, 'message': "No data available", 'details': {}}

    def _validate_duplicate_timestamps(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate duplicate timestamps."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'passed': True, 'message': "Not a DatetimeIndex", 'details': {}}

        duplicate_count = data.index.duplicated().sum()

        return {
            'passed': duplicate_count == 0,
            'message': f"Found {duplicate_count} duplicate timestamps",
            'details': {
                'duplicate_count': duplicate_count
            }
        }

    def _validate_extreme_values(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate extreme values using z-score method."""
        threshold = rule.threshold or 3.0
        extreme_count = 0
        extreme_details = {}

        for column in data.select_dtypes(include=[np.number]).columns:
            if data[column].std() > 0:  # Avoid division by zero
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                column_extremes = (z_scores > threshold).sum()
                extreme_count += column_extremes
                extreme_details[column] = column_extremes

        return {
            'passed': extreme_count == 0,
            'message': f"Found {extreme_count} extreme values (z-score > {threshold})",
            'details': {
                'extreme_count': extreme_count,
                'threshold': threshold,
                'by_column': extreme_details
            }
        }

    def _validate_data_gaps(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate gaps in time series data."""
        if not isinstance(data.index, pd.DatetimeIndex) or len(data.index) < 2:
            return {'passed': True, 'message': "Insufficient data for gap detection", 'details': {}}

        # Estimate expected frequency
        time_diffs = pd.Series(data.index[1:] - data.index[:-1])
        expected_freq = time_diffs.median()

        # Detect gaps
        gap_threshold = expected_freq * 1.5  # 50% tolerance
        gaps = time_diffs > gap_threshold
        gap_count = gaps.sum()

        return {
            'passed': gap_count == 0,
            'message': f"Found {gap_count} gaps in time series",
            'details': {
                'gap_count': gap_count,
                'expected_frequency': str(expected_freq),
                'gap_threshold': str(gap_threshold)
            }
        }

    def validate_financial_data(self, data: pd.DataFrame,
                              instrument: Optional[FinancialInstrument] = None) -> ValidationResult:
        """
        Validate financial data with financial-specific rules.

        Args:
            data: Financial data DataFrame
            instrument: Financial instrument for context

        Returns:
            ValidationResult with financial-specific validation
        """
        # Add financial-specific validation rules
        financial_rules = [
            ValidationRule(
                name="price_bounds",
                category=ValidationCategory.VALIDITY,
                description="Check if prices are within reasonable bounds",
                severity="warning",
                validator=self._validate_price_bounds
            ),
            ValidationRule(
                name="volume_bounds",
                category=ValidationCategory.VALIDITY,
                description="Check if volumes are within reasonable bounds",
                severity="warning",
                validator=self._validate_volume_bounds
            ),
            ValidationRule(
                name="return_bounds",
                category=ValidationCategory.VALIDITY,
                description="Check if returns are within reasonable bounds",
                severity="warning",
                validator=self._validate_return_bounds
            )
        ]

        return self.validate_data(data, custom_rules=financial_rules)

    def _validate_price_bounds(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate price bounds."""
        price_columns = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'price']]
        violations = 0

        for col in price_columns:
            if col in data.columns:
                # Check for extremely high prices (> $1,000,000)
                high_violations = (data[col] > 1000000).sum()
                violations += high_violations

                # Check for extremely low prices (< $0.01 for stocks)
                if 'stock' in str(data).lower() or 'equity' in str(data).lower():
                    low_violations = ((data[col] > 0) & (data[col] < 0.01)).sum()
                    violations += low_violations

        return {
            'passed': violations == 0,
            'message': f"Found {violations} price bound violations",
            'details': {
                'violation_count': violations,
                'price_columns': price_columns
            }
        }

    def _validate_volume_bounds(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate volume bounds."""
        violations = 0
        if 'volume' in data.columns:
            # Check for extremely high volumes (> 1B shares)
            high_violations = (data['volume'] > 1000000000).sum()
            violations += high_violations

            # Check for zero volumes (if not expected)
            zero_violations = (data['volume'] == 0).sum()
            violations += zero_violations

        return {
            'passed': violations == 0,
            'message': f"Found {violations} volume bound violations",
            'details': {
                'violation_count': violations
            }
        }

    def _validate_return_bounds(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate return bounds."""
        violations = 0

        # Calculate returns if we have price data
        price_columns = [col for col in data.columns if col in ['close', 'price']]
        for col in price_columns:
            if col in data.columns:
                returns = data[col].pct_change().dropna()
                if len(returns) > 0:
                    # Check for extreme returns (> 100% daily)
                    extreme_returns = (np.abs(returns) > 1.0).sum()
                    violations += extreme_returns

        return {
            'passed': violations == 0,
            'message': f"Found {violations} extreme return values",
            'details': {
                'violation_count': violations
            }
        }

    def generate_quality_report(self, data: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            data: Data to analyze

        Returns:
            Dictionary with detailed quality metrics and recommendations
        """
        validation_result = self.validate_data(data)

        report = {
            'summary': {
                'quality_score': validation_result.quality_score,
                'is_valid': validation_result.is_valid,
                'total_issues': len(validation_result.errors) + len(validation_result.warnings),
                'data_points': validation_result.metadata.get('data_points', 0)
            },
            'quality_breakdown': validation_result.category_scores,
            'issues': {
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            },
            'recommendations': validation_result.recommendations,
            'metadata': validation_result.metadata,
            'timestamp': datetime.now().isoformat()
        }

        return report

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about validation history.

        Returns:
            Dictionary with validation statistics
        """
        if not self.validation_history:
            return {'message': 'No validation history available'}

        quality_scores = [result.quality_score for result in self.validation_history]
        error_counts = [len(result.errors) for result in self.validation_history]
        warning_counts = [len(result.warnings) for result in self.validation_history]

        return {
            'total_validations': len(self.validation_history),
            'average_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'average_errors': np.mean(error_counts),
            'average_warnings': np.mean(warning_counts),
            'valid_with_errors': sum(1 for result in self.validation_history if result.errors),
            'valid_without_errors': sum(1 for result in self.validation_history if not result.errors)
        }

    def add_custom_rule(self, rule: ValidationRule):
        """
        Add a custom validation rule.

        Args:
            rule: ValidationRule to add
        """
        self.rules.append(rule)
        self.logger.info(f"Added custom validation rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """
        Remove a validation rule by name.

        Args:
            rule_name: Name of rule to remove
        """
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self.logger.info(f"Removed validation rule: {rule_name}")

    def set_validation_level(self, level: ValidationLevel):
        """
        Set the validation level.

        Args:
            level: New validation level
        """
        self.validation_level = level
        self.logger.info(f"Set validation level to: {level.value}")

    def clean_data(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Clean data using integrated preprocessing pipeline.

        This method combines validation and cleaning to produce high-quality data
        for feature generation and analysis.

        Args:
            data: Input DataFrame to clean
            config: Optional cleaning configuration

        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning with validation integration")

        if data.empty:
            return data

        # Initialize default cleaning configuration
        if config is None:
            config = {
                'missing_values': {
                    'method': 'forward_fill',
                    'threshold': 0.1,
                    'window_size': 5
                },
                'outliers': {
                    'method': 'iqr',
                    'threshold': 1.5,
                    'action': 'clip'
                },
                'remove_duplicates': True,
                'time_gaps': {
                    'expected_frequency': '1D',
                    'fill_method': 'interpolate'
                }
            }

        # Initialize data cleaner
        cleaner = DataCleaner()

        # Step 1: Initial validation to understand data issues
        initial_validation = self.validate_data(data)
        self.logger.info(f"Initial validation quality score: {initial_validation.quality_score:.3f}")

        # Step 2: Clean data based on validation results
        cleaning_config = {
            'missing_values': config.get('missing_values', {}),
            'outliers': config.get('outliers', {}),
            'remove_duplicates': config.get('remove_duplicates', True)
        }

        # Handle time gaps if timestamp is in index
        if isinstance(data.index, pd.DatetimeIndex):
            cleaning_config['time_gaps'] = config.get('time_gaps', {
                'expected_frequency': '1D',
                'fill_method': 'interpolate'
            })
            # Convert index to column for gap filling
            data_with_timestamp = data.reset_index()
            data_with_timestamp.columns.name = None  # Remove index name
            data_with_timestamp = data_with_timestamp.rename(columns={'index': 'timestamp'})
            cleaning_result = cleaner.clean_financial_data(data_with_timestamp, cleaning_config)
            cleaned_data = cleaning_result['cleaned_data']
            # Set timestamp back as index
            cleaned_data = cleaned_data.set_index('timestamp')
        else:
            cleaning_result = cleaner.clean_financial_data(data, cleaning_config)
            cleaned_data = cleaning_result['cleaned_data']

        # Step 3: Post-cleaning validation
        final_validation = self.validate_data(cleaned_data)
        self.logger.info(f"Post-cleaning validation quality score: {final_validation.quality_score:.3f}")

        # Log improvement
        quality_improvement = final_validation.quality_score - initial_validation.quality_score
        if quality_improvement > 0:
            self.logger.info(f"Data quality improved by {quality_improvement:.3f}")
        else:
            self.logger.warning(f"Data quality did not improve (change: {quality_improvement:.3f})")

        # Step 4: Return cleaned data with warnings if quality is still low
        if final_validation.quality_score < 0.7:
            self.logger.warning("Data quality is still below acceptable threshold after cleaning")

        return cleaned_data

    def analyze_time_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze time gaps in time series data.

        Args:
            data: DataFrame with DatetimeIndex

        Returns:
            Dictionary with time gap analysis results
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return {
                'gaps_detected': 0,
                'max_gap': pd.Timedelta(0),
                'avg_gap': pd.Timedelta(0),
                'total_gaps': 0,
                'gap_details': [],
                'error': 'Data index is not DatetimeIndex'
            }

        if len(data.index) < 2:
            return {
                'gaps_detected': 0,
                'max_gap': pd.Timedelta(0),
                'avg_gap': pd.Timedelta(0),
                'total_gaps': 0,
                'gap_details': [],
                'error': 'Insufficient data for gap analysis'
            }

        # Calculate time differences
        time_diffs = pd.Series(data.index[1:] - data.index[:-1])

        # Determine expected frequency (most common interval)
        expected_freq = time_diffs.mode()
        if len(expected_freq) > 0:
            expected_freq = expected_freq[0]
        else:
            expected_freq = time_diffs.median()

        # Find gaps (intervals significantly larger than expected)
        gap_threshold = expected_freq * 2  # Consider gaps larger than 2x expected frequency
        gaps = time_diffs[time_diffs > gap_threshold]

        gap_details = []
        for i, (idx, gap_size) in enumerate(gaps.items()):
            gap_details.append({
                'start_time': data.index[idx].isoformat(),
                'end_time': data.index[idx + 1].isoformat(),
                'gap_size': str(gap_size),
                'gap_size_seconds': gap_size.total_seconds()
            })

        return {
            'gaps_detected': len(gaps),
            'max_gap': gaps.max() if len(gaps) > 0 else pd.Timedelta(0),
            'avg_gap': gaps.mean() if len(gaps) > 0 else pd.Timedelta(0),
            'total_gaps': len(gaps),
            'expected_frequency': str(expected_freq),
            'gap_threshold': str(gap_threshold),
            'gap_details': gap_details
        }

    def check_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data integrity for financial data.

        Args:
            data: DataFrame to check

        Returns:
            Dictionary with integrity check results
        """
        integrity_results = {
            'price_integrity': True,
            'volume_integrity': True,
            'chronological_order': True,
            'data_types': True,
            'missing_values': True,
            'duplicate_rows': True,
            'index_integrity': True,
            'detailed_issues': []
        }

        # Check price integrity
        price_columns = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close', 'price']]
        for col in price_columns:
            if col in data.columns:
                # Check for negative prices
                negative_prices = (data[col] < 0).sum()
                if negative_prices > 0:
                    integrity_results['price_integrity'] = False
                    integrity_results['detailed_issues'].append(f'Found {negative_prices} negative prices in {col}')

                # Check for NaN prices
                nan_prices = data[col].isna().sum()
                if nan_prices > 0:
                    integrity_results['price_integrity'] = False
                    integrity_results['detailed_issues'].append(f'Found {nan_prices} missing prices in {col}')

        # Check OHLC consistency if all OHLC columns exist
        ohlc_columns = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in ohlc_columns):
            # High should be >= Low
            high_low_issues = (data['high'] < data['low']).sum()
            if high_low_issues > 0:
                integrity_results['price_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {high_low_issues} cases where high < low')

            # High should be >= Open and Close
            high_open_issues = (data['high'] < data['open']).sum()
            high_close_issues = (data['high'] < data['close']).sum()
            if high_open_issues > 0:
                integrity_results['price_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {high_open_issues} cases where high < open')
            if high_close_issues > 0:
                integrity_results['price_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {high_close_issues} cases where high < close')

            # Low should be <= Open and Close
            low_open_issues = (data['low'] > data['open']).sum()
            low_close_issues = (data['low'] > data['close']).sum()
            if low_open_issues > 0:
                integrity_results['price_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {low_open_issues} cases where low > open')
            if low_close_issues > 0:
                integrity_results['price_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {low_close_issues} cases where low > close')

        # Check volume integrity
        if 'volume' in data.columns:
            # Check for negative volumes
            negative_volumes = (data['volume'] < 0).sum()
            if negative_volumes > 0:
                integrity_results['volume_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {negative_volumes} negative volumes')

            # Check for NaN volumes
            nan_volumes = data['volume'].isna().sum()
            if nan_volumes > 0:
                integrity_results['volume_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {nan_volumes} missing volumes')

        # Check chronological order
        if isinstance(data.index, pd.DatetimeIndex):
            is_monotonic = data.index.is_monotonic_increasing
            if not is_monotonic:
                integrity_results['chronological_order'] = False
                integrity_results['detailed_issues'].append('Timestamps are not in chronological order')

        # Check data types
        for col in data.columns:
            if col.lower() in ['open', 'high', 'low', 'close', 'price', 'volume']:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    integrity_results['data_types'] = False
                    integrity_results['detailed_issues'].append(f'Column {col} should be numeric but is {data[col].dtype}')

        # Check missing values overall
        total_missing = data.isna().sum().sum()
        total_cells = data.shape[0] * data.shape[1]
        if total_missing > 0:
            missing_percentage = (total_missing / total_cells) * 100
            if missing_percentage > 10:  # More than 10% missing is an issue
                integrity_results['missing_values'] = False
                integrity_results['detailed_issues'].append(f'High missing value rate: {missing_percentage:.1f}%')

        # Check duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            integrity_results['duplicate_rows'] = False
            integrity_results['detailed_issues'].append(f'Found {duplicate_rows} duplicate rows')

        # Check index integrity
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_index = data.index.duplicated().sum()
            if duplicate_index > 0:
                integrity_results['index_integrity'] = False
                integrity_results['detailed_issues'].append(f'Found {duplicate_index} duplicate timestamps')

        return integrity_results


# Alias for backward compatibility with tests
ValidationService = DataValidator