"""
Feature API class for feature generation and management.

This module provides API endpoints for financial feature generation,
validation, and metadata management.
"""

import json
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from services.feature_service import FeatureGenerator as FeatureService
from services.validation_service import ValidationService
from models.financial_instrument import InstrumentType, FinancialInstrument
from models.price_data import Frequency, PriceType, PriceData


@dataclass
class APIResponse:
    """Standard API response structure."""
    status: str
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FeatureAPI:
    """API class for feature generation and management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FeatureAPI with optional configuration."""
        self.config = config or {}
        self.feature_service = FeatureService()
        self.validation_service = ValidationService()
        self._initialize_default_config()
        self._initialize_rate_limiting()

    def _initialize_default_config(self):
        """Initialize default configuration values."""
        default_config = {
            'max_data_points': 100000,
            'max_window_size': 252,
            'strict_validation': True,
            'enable_caching': True,
            'default_features': ['returns', 'volatility', 'momentum']
        }

        # Merge with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

    def _initialize_rate_limiting(self):
        """Initialize rate limiting configuration."""
        # Rate limiting settings
        self.max_requests_per_minute = self.config.get('api_limits', {}).get('rate_limit_requests_per_minute', 100)
        self.request_timestamps = deque()
        self.rate_limit_enabled = self.max_requests_per_minute > 0

    def _check_rate_limit(self) -> bool:
        """Check if the current request is within rate limits."""
        if not self.rate_limit_enabled:
            return True

        current_time = time.time()
        minute_ago = current_time - 60

        # Remove timestamps older than 1 minute
        while self.request_timestamps and self.request_timestamps[0] < minute_ago:
            self.request_timestamps.popleft()

        # Check if we're within the limit
        if len(self.request_timestamps) < self.max_requests_per_minute:
            self.request_timestamps.append(current_time)
            return True

        return False

    def generate_features_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate features endpoint.

        Args:
            request_data: Dictionary containing:
                - data: Dictionary with price, volume, dates
                - features: List of features to generate
                - config: Optional configuration overrides

        Returns:
            Dictionary with generated features or error message
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                return {
                    'status': 'error',
                    'message': 'Rate limited: Too many requests. Please wait before making another request.',
                    'data': None,
                    'metadata': None
                }
            # Validate required fields
            if not isinstance(request_data, dict):
                return {
                    'status': 'error',
                    'message': 'Request data must be a dictionary',
                    'data': None,
                    'metadata': None
                }

            if 'data' not in request_data:
                return {
                    'status': 'error',
                    'message': 'Missing required field: data',
                    'data': None,
                    'metadata': None
                }

            if 'features' not in request_data:
                return {
                    'status': 'error',
                    'message': 'Missing required field: features',
                    'data': None,
                    'metadata': None
                }

            data = request_data['data']
            features = request_data['features']
            config = request_data.get('config', {})

            # Validate data structure
            if not isinstance(data, dict):
                return {
                    'status': 'error',
                    'message': 'Data must be a dictionary',
                    'data': None,
                    'metadata': None
                }

            # Convert data to DataFrame
            try:
                df = pd.DataFrame(data)

                # Convert dates to datetime if present
                if 'dates' in df.columns:
                    df['dates'] = pd.to_datetime(df['dates'])
                    df.set_index('dates', inplace=True)

            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Invalid data format: {str(e)}',
                    'data': None,
                    'metadata': None
                }

            # Check data size limits
            if len(df) > self.config['max_data_points']:
                return {
                    'status': 'error',
                    'message': f'Data size {len(df)} exceeds maximum allowed {self.config["max_data_points"]}',
                    'data': None,
                    'metadata': None
                }

            # Merge config with defaults
            merged_config = {**self.config, **config}

            # Convert DataFrame to PriceData for feature service
            try:
                from models.financial_instrument import InstrumentType, FinancialInstrument
                from models.price_data import Frequency, PriceType

                # Create OHLC DataFrame from input data
                ohlc_data = pd.DataFrame({
                    'open': df['price'] if 'price' in df.columns else df['close'],
                    'high': df['price'] if 'price' in df.columns else df['high'],
                    'low': df['price'] if 'price' in df.columns else df['low'],
                    'close': df['price'] if 'price' in df.columns else df['close'],
                    'volume': df['volume'] if 'volume' in df.columns else [1000] * len(df)
                }, index=df.index)

                # Create PriceData object
                price_data = PriceData(
                    prices=ohlc_data,
                    instrument=FinancialInstrument(
                        symbol="API_TEST",
                        name="API Test Instrument",
                        instrument_type=InstrumentType.EQUITY
                    ),
                    frequency=Frequency.DAILY,
                    price_type=PriceType.OHLC
                )

                # Generate features using proper method
                from services.feature_service import FeatureGenerationConfig
                feature_config = FeatureGenerationConfig(
                    return_periods=[1, 5, 21],
                    volatility_windows=[5, 21],
                    momentum_periods=[5, 14]
                )

                feature_set = self.feature_service.generate_features(
                    price_data=price_data,
                    custom_config=feature_config
                )

                return {
                    'status': 'success',
                    'data': {
                        'feature_set': feature_set.to_dict(),
                        'features': feature_set.feature_names,
                        'quality_score': feature_set.quality_score.value
                    },
                    'metadata': {
                        'features_generated': features,
                        'data_points': len(df),
                        'config_used': merged_config
                    }
                }

            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Feature generation failed: {str(e)}',
                    'data': None,
                    'metadata': None
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Internal server error: {str(e)}',
                'data': None,
                'metadata': None
            }

    def validate_features_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate features endpoint.

        Args:
            request_data: Dictionary containing features to validate

        Returns:
            Dictionary with validation results
        """
        try:
            if 'features' not in request_data:
                return {
                    'status': 'error',
                    'message': 'Missing required field: features',
                    'data': None,
                    'metadata': None
                }

            features = request_data['features']

            # Validate features using validation service
            validation_result = self.validation_service.validate_features(features)

            return {
                'status': 'success',
                'data': validation_result,
                'metadata': {
                    'features_validated': len(features) if isinstance(features, list) else 1,
                    'validation_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Validation failed: {str(e)}',
                'data': None,
                'metadata': None
            }

    def get_feature_metadata_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get feature metadata endpoint.

        Args:
            request_data: Dictionary containing optional feature name

        Returns:
            Dictionary with feature metadata
        """
        try:
            feature_name = request_data.get('feature_name')

            # Get metadata from feature service
            metadata = self.feature_service.get_feature_metadata(feature_name)

            return {
                'status': 'success',
                'data': metadata,
                'metadata': {
                    'feature_name': feature_name or 'all',
                    'metadata_timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Metadata retrieval failed: {str(e)}',
                'data': None,
                'metadata': None
            }

    def get_available_features_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of available features endpoint.

        Returns:
            Dictionary with list of available features
        """
        try:
            available_features = self.feature_service.get_available_features()

            return {
                'status': 'success',
                'data': {'available_features': available_features},
                'metadata': {
                    'total_features': len(available_features),
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to retrieve available features: {str(e)}',
                'data': None,
                'metadata': None
            }

    def health_check_endpoint(self) -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns:
            Dictionary with service health status
        """
        try:
            # Check if services are available
            feature_service_ok = self.feature_service is not None
            validation_service_ok = self.validation_service is not None

            health_status = {
                'status': 'healthy' if feature_service_ok and validation_service_ok else 'unhealthy',
                'services': {
                    'feature_service': 'ok' if feature_service_ok else 'error',
                    'validation_service': 'ok' if validation_service_ok else 'error'
                },
                'timestamp': datetime.now().isoformat()
            }

            return {
                'status': 'success',
                'data': health_status
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'data': None,
                'metadata': None
            }