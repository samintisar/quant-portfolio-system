"""
Error Handling and Graceful Degradation Service

Provides comprehensive error handling, recovery strategies, and
graceful degradation for preprocessing operations when data issues occur.
"""

import traceback
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
import numpy as np
from functools import wraps

from ..models.processing_log import ProcessingLog


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"          # Can continue with warnings
    MEDIUM = "medium"    # Needs recovery action
    HIGH = "high"        # Critical, may need to stop
    CRITICAL = "critical" # Must stop immediately


class ErrorCategory(Enum):
    """Error categories for better handling."""
    DATA_QUALITY = "data_quality"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL = "external"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for an error."""
    operation: str
    dataset_id: str
    input_shape: tuple
    timestamp: datetime
    environment: Dict[str, Any] = field(default_factory=dict)
    user_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation': self.operation,
            'dataset_id': self.dataset_id,
            'input_shape': self.input_shape,
            'timestamp': self.timestamp.isoformat(),
            'environment': self.environment,
            'user_metadata': self.user_metadata
        }


@dataclass
class ProcessingError:
    """Processing error with recovery information."""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    traceback: Optional[str] = None
    recovery_suggestion: str = ""
    can_continue: bool = True
    impacted_operations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'error_id': self.error_id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback,
            'recovery_suggestion': self.recovery_suggestion,
            'can_continue': self.can_continue,
            'impacted_operations': self.impacted_operations,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RecoveryStrategy:
    """Recovery strategy for handling errors."""
    strategy_id: str
    name: str
    description: str
    applicable_categories: List[ErrorCategory]
    applicable_severities: List[ErrorSeverity]
    recovery_function: Callable
    priority: int = 1
    success_rate: float = 0.0
    usage_count: int = 0

    def can_handle(self, error: ProcessingError) -> bool:
        """Check if this strategy can handle the given error."""
        return (
            error.category in self.applicable_categories and
            error.severity in self.applicable_severities
        )


class ErrorHandler:
    """Error handling service for preprocessing operations."""

    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ProcessingError] = []
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.error_patterns: Dict[str, List[ProcessingError]] = {}

        # Initialize recovery strategies
        self._initialize_recovery_strategies()

    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for common errors."""
        # Memory error recovery
        self.recovery_strategies.append(RecoveryStrategy(
            strategy_id="reduce_memory_usage",
            name="Reduce Memory Usage",
            description="Optimize memory usage by converting data types",
            applicable_categories=[ErrorCategory.MEMORY],
            applicable_severities=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._recover_memory_usage,
            priority=1
        ))

        # Missing data recovery
        self.recovery_strategies.append(RecoveryStrategy(
            strategy_id="handle_missing_data",
            name="Handle Missing Data",
            description="Apply missing data imputation strategies",
            applicable_categories=[ErrorCategory.DATA_QUALITY],
            applicable_severities=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            recovery_function=self._recover_missing_data,
            priority=2
        ))

        # Data type conversion recovery
        self.recovery_strategies.append(RecoveryStrategy(
            strategy_id="convert_data_types",
            name="Convert Data Types",
            description="Convert problematic data types to compatible formats",
            applicable_categories=[ErrorCategory.DATA_QUALITY, ErrorCategory.VALIDATION],
            applicable_severities=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            recovery_function=self._recover_data_types,
            priority=3
        ))

        # Performance degradation recovery
        self.recovery_strategies.append(RecoveryStrategy(
            strategy_id="reduce_processing_scope",
            name="Reduce Processing Scope",
            description="Reduce data size or processing complexity",
            applicable_categories=[ErrorCategory.PERFORMANCE],
            applicable_severities=[ErrorSeverity.MEDIUM, ErrorSeverity.HIGH],
            recovery_function=self._reduce_processing_scope,
            priority=4
        ))

        # Configuration error recovery
        self.recovery_strategies.append(RecoveryStrategy(
            strategy_id="use_default_config",
            name="Use Default Configuration",
            description="Fall back to default configuration",
            applicable_categories=[ErrorCategory.CONFIGURATION],
            applicable_severities=[ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            recovery_function=self._use_default_config,
            priority=5
        ))

    def handle_error(self, error: Exception, context: ErrorContext,
                    operation: str = "unknown") -> ProcessingError:
        """Handle an error with appropriate recovery strategies.

        Args:
            error: The exception that occurred
            context: Error context information
            operation: Operation that failed

        Returns:
            ProcessingError object
        """
        # Create processing error
        processing_error = self._create_processing_error(error, context, operation)

        # Add to history
        self.error_history.append(processing_error)

        # Update error patterns
        error_pattern = f"{processing_error.category.value}_{processing_error.error_type}"
        if error_pattern not in self.error_patterns:
            self.error_patterns[error_pattern] = []
        self.error_patterns[error_pattern].append(processing_error)

        # Log the error
        self._log_error(processing_error)

        return processing_error

    def _create_processing_error(self, error: Exception, context: ErrorContext,
                                operation: str) -> ProcessingError:
        """Create a processing error from an exception.

        Args:
            error: The exception
            context: Error context
            operation: Operation name

        Returns:
            ProcessingError object
        """
        import uuid

        # Determine error category and severity
        category, severity = self._classify_error(error, operation)

        # Generate recovery suggestion
        recovery_suggestion = self._generate_recovery_suggestion(error, category, severity)

        # Determine if processing can continue
        can_continue = severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]

        return ProcessingError(
            error_id=str(uuid.uuid4()),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context,
            traceback=traceback.format_exc(),
            recovery_suggestion=recovery_suggestion,
            can_continue=can_continue,
            impacted_operations=self._determine_impacted_operations(operation, category, severity)
        )

    def _classify_error(self, error: Exception, operation: str) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error by category and severity.

        Args:
            error: The exception
            operation: Operation that failed

        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_type = type(error).__name__

        # Memory errors
        if error_type in ["MemoryError", "OutOfMemoryError"]:
            return ErrorCategory.MEMORY, ErrorSeverity.HIGH

        # Data quality errors
        elif any(keyword in str(error).lower() for keyword in ["missing", "null", "na", "nan"]):
            return ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM

        # Data type errors
        elif error_type in ["TypeError", "ValueError"]:
            return ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM

        # Configuration errors
        elif "config" in str(error).lower() or "setting" in str(error).lower():
            return ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM

        # Performance errors
        elif any(keyword in str(error).lower() for keyword in ["timeout", "slow", "performance"]):
            return ErrorCategory.PERFORMANCE, ErrorSeverity.MEDIUM

        # Validation errors
        elif "validation" in str(error).lower() or "invalid" in str(error).lower():
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW

        # Processing errors
        elif error_type in ["ProcessingError", "ComputationError"]:
            return ErrorCategory.PROCESSING, ErrorSeverity.HIGH

        # System errors
        elif error_type in ["SystemError", "OSError", "IOError"]:
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL

        # Default classification
        else:
            return ErrorCategory.PROCESSING, ErrorSeverity.MEDIUM

    def _generate_recovery_suggestion(self, error: Exception, category: ErrorCategory,
                                     severity: ErrorSeverity) -> str:
        """Generate recovery suggestion for an error.

        Args:
            error: The exception
            category: Error category
            severity: Error severity

        Returns:
            Recovery suggestion string
        """
        suggestions = {
            ErrorCategory.MEMORY: [
                "Reduce data size by sampling or filtering",
                "Optimize data types to use less memory",
                "Process data in smaller chunks",
                "Free up system memory before processing"
            ],
            ErrorCategory.DATA_QUALITY: [
                "Clean data by removing or imputing missing values",
                "Convert data types to appropriate formats",
                "Validate data against expected schema",
                "Apply data quality rules and corrections"
            ],
            ErrorCategory.PERFORMANCE: [
                "Reduce processing scope or complexity",
                "Use more efficient algorithms",
                "Process data in parallel or batches",
                "Optimize system resources"
            ],
            ErrorCategory.CONFIGURATION: [
                "Use default configuration settings",
                "Validate configuration parameters",
                "Check configuration file format",
                "Reset to known good configuration"
            ],
            ErrorCategory.VALIDATION: [
                "Review validation rules and constraints",
                "Adjust validation thresholds",
                "Check data format and structure",
                "Apply data corrections before validation"
            ]
        }

        category_suggestions = suggestions.get(category, ["Check error details and try again"])
        return category_suggestions[0] if category_suggestions else "Review error and try again"

    def _determine_impacted_operations(self, operation: str, category: ErrorCategory,
                                    severity: ErrorSeverity) -> List[str]:
        """Determine which operations are impacted by the error.

        Args:
            operation: Current operation
            category: Error category
            severity: Error severity

        Returns:
            List of impacted operation names
        """
        impacted = [operation]

        # Add related operations based on category
        if category == ErrorCategory.MEMORY:
            impacted.extend(["preprocessing", "validation", "normalization"])
        elif category == ErrorCategory.DATA_QUALITY:
            impacted.extend(["cleaning", "validation", "quality_assessment"])
        elif category == ErrorCategory.PERFORMANCE:
            impacted.extend(["processing", "analysis", "reporting"])
        elif category == ErrorCategory.CONFIGURATION:
            impacted.extend(["configuration", "pipeline_setup"])

        return list(set(impacted))

    def _log_error(self, error: ProcessingError):
        """Log error with appropriate level.

        Args:
            error: Processing error to log
        """
        log_message = f"Error in {error.context.operation}: {error.error_message}"

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def attempt_recovery(self, error: ProcessingError, data: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> tuple[bool, Optional[pd.DataFrame]]:
        """Attempt to recover from an error.

        Args:
            error: Processing error to recover from
            data: Current data state
            config: Optional configuration for recovery

        Returns:
            Tuple of (success, recovered_data)
        """
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if strategy.can_handle(error)
        ]

        # Sort by priority
        applicable_strategies.sort(key=lambda s: s.priority)

        for strategy in applicable_strategies:
            try:
                self.logger.info(f"Attempting recovery strategy: {strategy.name}")
                recovered_data = strategy.recovery_function(data, error, config or {})

                # Update strategy statistics
                strategy.usage_count += 1
                strategy.success_rate = (
                    (strategy.success_rate * (strategy.usage_count - 1) + 1) / strategy.usage_count
                )

                self.logger.info(f"Recovery successful using {strategy.name}")
                return True, recovered_data

            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                # Update strategy statistics
                strategy.usage_count += 1
                strategy.success_rate = (
                    (strategy.success_rate * (strategy.usage_count - 1)) / strategy.usage_count
                )
                continue

        self.logger.error("All recovery strategies failed")
        return False, None

    # Recovery strategy implementations
    def _recover_memory_usage(self, data: pd.DataFrame, error: ProcessingError,
                             config: Dict[str, Any]) -> pd.DataFrame:
        """Recover from memory errors by optimizing data types."""
        self.logger.info("Optimizing data types to reduce memory usage")

        # Convert numeric columns to smaller types where possible
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='integer')

        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = pd.to_numeric(data[col], downcast='float')

        # Convert object columns to category if appropriate
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) < 0.5:  # Low cardinality
                data[col] = data[col].astype('category')

        return data

    def _recover_missing_data(self, data: pd.DataFrame, error: ProcessingError,
                            config: Dict[str, Any]) -> pd.DataFrame:
        """Recover from missing data by applying imputation."""
        self.logger.info("Applying missing data imputation")

        # Forward fill for time series data
        if 'timestamp' in data.columns or 'date' in data.columns:
            data = data.fillna(method='ffill')

        # Fill numeric columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())

        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'unknown')

        return data

    def _recover_data_types(self, data: pd.DataFrame, error: ProcessingError,
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Recover from data type conversion errors."""
        self.logger.info("Converting data types to compatible formats")

        # Try to convert object columns to appropriate types
        for col in data.select_dtypes(include=['object']).columns:
            try:
                # Try numeric conversion
                data[col] = pd.to_numeric(data[col], errors='ignore')
            except:
                # Keep as object if conversion fails
                pass

        return data

    def _reduce_processing_scope(self, data: pd.DataFrame, error: ProcessingError,
                                config: Dict[str, Any]) -> pd.DataFrame:
        """Reduce processing scope by sampling data."""
        self.logger.info("Reducing data scope through sampling")

        # Sample 50% of data
        sample_size = max(1000, len(data) // 2)
        return data.sample(n=min(sample_size, len(data)), random_state=42)

    def _use_default_config(self, data: pd.DataFrame, error: ProcessingError,
                          config: Dict[str, Any]) -> pd.DataFrame:
        """Use default configuration to recover."""
        self.logger.info("Using default configuration")

        # Return data as-is - config changes should be handled by caller
        return data

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors.

        Returns:
            Error statistics dictionary
        """
        if not self.error_history:
            return {'message': 'No errors recorded'}

        # Count by category
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Recovery success rates
        strategy_stats = {}
        for strategy in self.recovery_strategies:
            if strategy.usage_count > 0:
                strategy_stats[strategy.name] = {
                    'usage_count': strategy.usage_count,
                    'success_rate': strategy.success_rate
                }

        return {
            'total_errors': len(self.error_history),
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'recovery_strategies': strategy_stats,
            'error_patterns': {pattern: len(errors) for pattern, errors in self.error_patterns.items()}
        }

    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.error_patterns.clear()
        self.logger.info("Error history cleared")


# Error handling decorator
def handle_errors(operation_name: Optional[str] = None,
                  continue_on_error: bool = True):
    """Decorator for automatic error handling.

    Args:
        operation_name: Operation name, or None to use function name
        continue_on_error: Whether to continue processing after errors
    """
    def decorator(func):
        name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    operation=name,
                    dataset_id=kwargs.get('dataset_id', 'unknown'),
                    input_shape=kwargs.get('input_shape', (0, 0)),
                    timestamp=datetime.now()
                )

                # Handle error
                error_handler = ErrorHandler()
                error = error_handler.handle_error(e, context, name)

                # Attempt recovery if possible
                if continue_on_error and error.can_continue:
                    data = kwargs.get('data', None)
                    config = kwargs.get('config', {})

                    if data is not None:
                        success, recovered_data = error_handler.attempt_recovery(error, data, config)
                        if success:
                            # Update kwargs with recovered data and continue
                            kwargs['data'] = recovered_data
                            kwargs['recovery_applied'] = True
                            kwargs['recovery_strategy'] = error.recovery_suggestion
                            return func(*args, **kwargs)

                # Re-raise if can't continue or recovery failed
                raise

        return wrapper
    return decorator


# Global error handler instance
error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return error_handler