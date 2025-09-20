"""
Feature Generation Service

This service orchestrates the generation of financial features from market data,
integrating calculation libraries, validation, and quality assurance.

Key Features:
- Automated feature generation pipeline
- Configurable feature sets and parameters
- Quality validation and scoring
- Performance monitoring
- Integration with all calculation libraries

Author: Claude Code
Date: 2025-09-19
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from models.financial_instrument import FinancialInstrument
from models.price_data import PriceData
from models.feature_set import FeatureSet
from models.return_series import ReturnSeries
from models.volatility_measure import VolatilityMeasure
from models.momentum_indicator import MomentumIndicator
from services.validation_service import DataValidator, ValidationResult
from lib.returns import (
    calculate_simple_returns, calculate_log_returns,
    calculate_cumulative_returns, calculate_annualized_returns
)
from lib.volatility import (
    calculate_rolling_volatility, calculate_parkinson_volatility,
    calculate_garman_klass_volatility, calculate_garch11_volatility,
    calculate_ewma_volatility, calculate_yang_zhang_volatility
)
from lib.momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_roc,
    calculate_money_flow, calculate_ultimate_oscillator
)
from lib.structured_logging import (
    StructuredLogger, CalculationStage, get_logger, LogLevel
)


@dataclass
class FeatureGenerationConfig:
    """Configuration for feature generation"""
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 21])
    volatility_windows: List[int] = field(default_factory=lambda: [21, 63, 252])
    volatility_methods: List[str] = field(default_factory=lambda: ['rolling', 'ewma'])
    momentum_periods: List[int] = field(default_factory=lambda: [14, 21, 63])
    momentum_indicators: List[str] = field(default_factory=lambda: ['rsi', 'macd', 'stochastic'])
    quality_threshold: float = 0.7
    max_missing_ratio: float = 0.1
    enable_garch: bool = True
    enable_multifeature: bool = True


class FeatureGenerator:
    """Service for generating financial features from market data"""

    def __init__(self, config: Optional[FeatureGenerationConfig] = None,
                 structured_logger: Optional[StructuredLogger] = None):
        self.config = config or FeatureGenerationConfig()
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        self.structured_logger = structured_logger or get_logger()

    def generate_features(self, price_data: PriceData, custom_config: Optional[FeatureGenerationConfig] = None) -> FeatureSet:
        """
        Generate comprehensive features from price data

        Args:
            price_data: Input price data with OHLCV information
            custom_config: Optional custom configuration

        Returns:
            FeatureSet with all calculated features
        """
        # Input validation
        if price_data is None:
            raise ValueError("price_data cannot be None")

        if not isinstance(price_data, PriceData):
            raise TypeError("price_data must be a PriceData object")

        if price_data.prices is None or price_data.prices.empty:
            raise ValueError("price_data.prices cannot be None or empty")

        config = custom_config or self.config

        # Start structured logging
        self.structured_logger.start_stage(CalculationStage.FEATURE_GENERATION)
        self.structured_logger.log_calculation_decision(
            stage=CalculationStage.FEATURE_GENERATION,
            decision="Initialize feature generation pipeline",
            parameters={
                "instrument": price_data.instrument.symbol,
                "data_points": len(price_data.prices),
                "quality_threshold": config.quality_threshold,
                "enable_garch": config.enable_garch,
                "enable_multifeature": config.enable_multifeature
            }
        )

        self.logger.info(f"Starting feature generation for {price_data.instrument.symbol}")

        # Data validation stage
        self.structured_logger.start_stage(CalculationStage.DATA_VALIDATION)
        validation_result = self.validator.validate_financial_data(
            price_data.prices, price_data.instrument
        )

        # Log validation results
        validation_dict = {
            "is_valid": validation_result.is_valid,
            "quality_score": validation_result.quality_score,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "category_scores": validation_result.category_scores
        }
        self.structured_logger.log_data_validation(validation_dict)

        if not validation_result.is_valid:
            self.logger.warning(f"Data validation failed: {validation_result.summary}")
            # Apply cleaning if quality is too low
            if validation_result.quality_score < config.quality_threshold:
                self.structured_logger.start_stage(CalculationStage.DATA_CLEANING)
                price_data.prices = self._clean_data(price_data.prices)
                self.structured_logger.end_stage(CalculationStage.DATA_CLEANING)

        self.structured_logger.end_stage(CalculationStage.DATA_VALIDATION)

        # Initialize features DataFrame
        features_df = price_data.prices.copy()

        # Generate feature groups
        feature_types = []

        # 1. Returns features
        self.structured_logger.log_calculation_decision(
            stage=CalculationStage.FEATURE_GENERATION,
            decision="Generate returns features",
            parameters={"return_periods": config.return_periods}
        )
        returns_features = self._generate_returns_features(price_data, config)
        features_df = pd.concat([features_df, returns_features], axis=1)
        feature_types.extend(['returns'] * len(returns_features.columns))

        # 2. Volatility features
        self.structured_logger.log_calculation_decision(
            stage=CalculationStage.FEATURE_GENERATION,
            decision="Generate volatility features",
            parameters={
                "volatility_windows": config.volatility_windows,
                "volatility_methods": config.volatility_methods,
                "enable_garch": config.enable_garch
            }
        )
        volatility_features = self._generate_volatility_features(price_data, config)
        features_df = pd.concat([features_df, volatility_features], axis=1)
        feature_types.extend(['volatility'] * len(volatility_features.columns))

        # 3. Momentum features
        self.structured_logger.log_calculation_decision(
            stage=CalculationStage.FEATURE_GENERATION,
            decision="Generate momentum features",
            parameters={
                "momentum_periods": config.momentum_periods,
                "momentum_indicators": config.momentum_indicators
            }
        )
        momentum_features = self._generate_momentum_features(price_data, config)
        features_df = pd.concat([features_df, momentum_features], axis=1)
        feature_types.extend(['momentum'] * len(momentum_features.columns))

        # 4. Multi-feature combinations
        if config.enable_multifeature:
            self.structured_logger.log_calculation_decision(
                stage=CalculationStage.FEATURE_GENERATION,
                decision="Generate multi-feature combinations",
                parameters={"enable_multifeature": config.enable_multifeature}
            )
            multifeature_df = self._generate_multifeature_features(
                features_df, returns_features, volatility_features, momentum_features, config
            )
            features_df = pd.concat([features_df, multifeature_df], axis=1)
            feature_types.extend(['multifeature'] * len(multifeature_df.columns))

        # Quality assessment
        self.structured_logger.start_stage(CalculationStage.QUALITY_ASSESSMENT)
        quality_metrics = self._assess_feature_quality(features_df, feature_types)
        self.structured_logger.log_quality_metrics(quality_metrics)
        self.structured_logger.end_stage(CalculationStage.QUALITY_ASSESSMENT)

        # Create FeatureSet
        feature_set = FeatureSet(
            features=features_df,
            instrument=price_data.instrument,
            feature_types=feature_types,
            quality_metrics=quality_metrics,
            calculation_params=config.__dict__,
            frequency=price_data.frequency.value,
            metadata={
                'start_date': str(price_data.prices.index[0]),
                'end_date': str(price_data.prices.index[-1]),
                'price_type': price_data.price_type.value,
                'api_version': '1.0.0'
            }
        )

        # Log feature generation results
        self.structured_logger.log_feature_generation(
            features=features_df,
            instrument=price_data.instrument.symbol,
            quality_metrics=quality_metrics
        )

        # Performance monitoring
        self.structured_logger.log_performance_metrics(
            data_points=len(features_df),
            features_generated=len(feature_types)
        )

        self.structured_logger.end_stage(CalculationStage.FEATURE_GENERATION)

        self.logger.info(f"Feature generation completed. Generated {len(feature_types)} features.")
        return feature_set

    def _generate_returns_features(self, price_data: PriceData, config: FeatureGenerationConfig) -> pd.DataFrame:
        """Generate returns-based features"""
        features = pd.DataFrame(index=price_data.prices.index)

        # Calculate returns for different periods
        for period in config.return_periods:
            # Simple returns
            simple_returns = calculate_simple_returns(price_data.prices['close'], period=period)
            features[f'returns_{period}d'] = simple_returns

            # Log returns
            log_returns = calculate_log_returns(price_data.prices['close'], period=period)
            features[f'log_returns_{period}d'] = log_returns

            # Cumulative returns
            cumulative_returns = calculate_cumulative_returns(simple_returns)
            features[f'cum_returns_{period}d'] = cumulative_returns

            # Annualized returns
            if period >= 21:
                annualized_returns = calculate_annualized_returns(simple_returns, periods_per_year=252//period)
                features[f'annualized_returns_{period}d'] = annualized_returns

        return features

    def _generate_volatility_features(self, price_data: PriceData, config: FeatureGenerationConfig) -> pd.DataFrame:
        """Generate volatility-based features"""
        features = pd.DataFrame(index=price_data.prices.index)

        # Calculate base returns for volatility calculations
        returns = calculate_simple_returns(price_data.prices['close'])

        for window in config.volatility_windows:
            # Rolling volatility
            if 'rolling' in config.volatility_methods:
                rolling_vol = calculate_rolling_volatility(returns, window=window)
                features[f'volatility_{window}d'] = rolling_vol

            # EWMA volatility
            if 'ewma' in config.volatility_methods:
                ewma_vol = calculate_ewma_volatility(returns, span=window)
                features[f'ewma_volatility_{window}d'] = ewma_vol

            # Parkinson volatility
            parkinson_vol = calculate_parkinson_volatility(
                price_data.prices['high'], price_data.prices['low'], window=window
            )
            features[f'parkinson_volatility_{window}d'] = parkinson_vol

            # Garman-Klass volatility
            gk_vol = calculate_garman_klass_volatility(
                price_data.prices['open'], price_data.prices['high'],
                price_data.prices['low'], price_data.prices['close'], window=window
            )
            features[f'garman_klass_volatility_{window}d'] = gk_vol

            # Yang-Zhang volatility
            yz_vol = calculate_yang_zhang_volatility(
                price_data.prices['open'], price_data.prices['high'],
                price_data.prices['low'], price_data.prices['close'], window=window
            )
            features[f'yang_zhang_volatility_{window}d'] = yz_vol

        # GARCH volatility for longer windows
        if config.enable_garch:
            for window in [63, 252]:
                if window in config.volatility_windows:
                    garch_vol = calculate_garch11_volatility(returns)
                    features[f'garch_volatility_{window}d'] = garch_vol

        return features

    def _generate_momentum_features(self, price_data: PriceData, config: FeatureGenerationConfig) -> pd.DataFrame:
        """Generate momentum-based features"""
        features = pd.DataFrame(index=price_data.prices.index)

        for period in config.momentum_periods:
            # RSI
            if 'rsi' in config.momentum_indicators:
                rsi = calculate_rsi(price_data.prices['close'], period=period)
                features[f'rsi_{period}'] = rsi

            # MACD
            if 'macd' in config.momentum_indicators and period >= 12:
                macd_line, signal_line, histogram = calculate_macd(
                    price_data.prices['close'],
                    fast_period=min(12, period),
                    slow_period=min(26, period),
                    signal_period=min(9, period//2)
                )
                features[f'macd_{period}'] = macd_line
                features[f'macd_signal_{period}'] = signal_line
                features[f'macd_histogram_{period}'] = histogram

            # Stochastic
            if 'stochastic' in config.momentum_indicators:
                stoch_k, stoch_d = calculate_stochastic(
                    price_data.prices['high'], price_data.prices['low'],
                    price_data.prices['close'], k_period=period
                )
                features[f'stoch_k_{period}'] = stoch_k
                features[f'stoch_d_{period}'] = stoch_d

            # Williams %R
            if 'williams_r' in config.momentum_indicators:
                williams_r = calculate_williams_r(
                    price_data.prices['high'], price_data.prices['low'],
                    price_data.prices['close'], period=period
                )
                features[f'williams_r_{period}'] = williams_r

            # Commodity Channel Index
            if 'cci' in config.momentum_indicators:
                cci = calculate_cci(
                    price_data.prices['high'], price_data.prices['low'],
                    price_data.prices['close'], period=period
                )
                features[f'cci_{period}'] = cci

            # Rate of Change
            if 'roc' in config.momentum_indicators:
                roc = calculate_roc(price_data.prices['close'], period=period)
                features[f'roc_{period}'] = roc

            # Money Flow Index
            if 'mfi' in config.momentum_indicators:
                mfi = calculate_money_flow_index(
                    price_data.prices['high'], price_data.prices['low'],
                    price_data.prices['close'], price_data.prices['volume'], period=period
                )
                features[f'mfi_{period}'] = mfi

        # TRIX and DPO not yet implemented
        # if 'trix' in config.momentum_indicators:
        #     trix = calculate_trix(price_data.prices['close'])
        #     features['trix'] = trix
        #
        # if 'dpo' in config.momentum_indicators:
        #     dpo = calculate_dpo(price_data.prices['close'])
        #     features['dpo'] = dpo

        return features

    def _generate_multifeature_features(self, features_df: pd.DataFrame,
                                      returns_features: pd.DataFrame,
                                      volatility_features: pd.DataFrame,
                                      momentum_features: pd.DataFrame,
                                      config: FeatureGenerationConfig) -> pd.DataFrame:
        """Generate multi-feature combinations"""
        multifeatures = pd.DataFrame(index=features_df.index)

        # Volatility-adjusted returns
        for window in config.volatility_windows:
            if f'volatility_{window}d' in volatility_features.columns:
                # Sharpe-like ratio
                if f'returns_{window}d' in returns_features.columns:
                    multifeatures[f'sharpe_{window}d'] = (
                        returns_features[f'returns_{window}d'] /
                        volatility_features[f'volatility_{window}d']
                    )

                # Momentum-volatility interaction
                if f'rsi_{window}' in momentum_features.columns:
                    multifeatures[f'rsi_vol_{window}d'] = (
                        momentum_features[f'rsi_{window}'] *
                        volatility_features[f'volatility_{window}d']
                    )

        # Relative strength indicators
        if len(config.return_periods) >= 2:
            short_period = min(config.return_periods)
            long_period = max(config.return_periods)

            if f'returns_{short_period}d' in returns_features.columns and \
               f'returns_{long_period}d' in returns_features.columns:
                multifeatures['relative_strength'] = (
                    returns_features[f'returns_{short_period}d'] -
                    returns_features[f'returns_{long_period}d']
                )

        # Volatility regime
        if len(config.volatility_windows) >= 2:
            short_window = min(config.volatility_windows)
            long_window = max(config.volatility_windows)

            if f'volatility_{short_window}d' in volatility_features.columns and \
               f'volatility_{long_window}d' in volatility_features.columns:
                multifeatures['vol_regime'] = (
                    volatility_features[f'volatility_{short_window}d'] -
                    volatility_features[f'volatility_{long_window}d']
                )

        return multifeatures

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers"""
        cleaned_data = data.copy()

        # Forward fill missing values
        cleaned_data = cleaned_data.fillna(method='ffill')

        # Backward fill remaining values
        cleaned_data = cleaned_data.fillna(method='bfill')

        return cleaned_data

    def _assess_feature_quality(self, features_df: pd.DataFrame, feature_types: List[str]) -> Dict[str, Any]:
        """Assess the quality of generated features"""
        quality_metrics = {
            'total_features': len(features_df.columns),
            'completeness_score': 1.0 - (features_df.isnull().sum().sum() / features_df.size),
            'feature_type_distribution': {}
        }

        # Count features by type
        for feature_type in set(feature_types):
            quality_metrics['feature_type_distribution'][feature_type] = feature_types.count(feature_type)

        # Check for extreme values
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                q1 = features_df[col].quantile(0.25)
                q3 = features_df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((features_df[col] < (q1 - 1.5 * iqr)) |
                           (features_df[col] > (q3 + 1.5 * iqr))).sum()
                quality_metrics[f'{col}_outliers'] = outliers

        return quality_metrics

    def generate_custom_features(self, price_data: PriceData,
                               feature_definitions: List[Dict[str, Any]]) -> FeatureSet:
        """
        Generate custom features based on user-defined specifications

        Args:
            price_data: Input price data
            feature_definitions: List of feature definitions

        Returns:
            FeatureSet with custom features
        """
        custom_features = pd.DataFrame(index=price_data.prices.index)
        feature_types = []

        for feature_def in feature_definitions:
            feature_name = feature_def['name']
            feature_type = feature_def['type']
            params = feature_def.get('params', {})

            if feature_type == 'returns':
                period = params.get('period', 1)
                return_type = params.get('return_type', 'simple')

                if return_type == 'simple':
                    custom_features[feature_name] = calculate_simple_returns(
                        price_data.prices['close'], period=period
                    )
                elif return_type == 'log':
                    custom_features[feature_name] = calculate_log_returns(
                        price_data.prices['close'], period=period
                    )

            elif feature_type == 'volatility':
                method = params.get('method', 'rolling')
                window = params.get('window', 21)
                returns = calculate_simple_returns(price_data.prices['close'])

                if method == 'rolling':
                    custom_features[feature_name] = calculate_rolling_volatility(
                        returns, window=window
                    )
                elif method == 'ewma':
                    custom_features[feature_name] = calculate_ewma_volatility(
                        returns, window=window
                    )

            elif feature_type == 'momentum':
                indicator = params.get('indicator', 'rsi')
                period = params.get('period', 14)

                if indicator == 'rsi':
                    custom_features[feature_name] = calculate_rsi(
                        price_data.prices['close'], period=period
                    )
                elif indicator == 'macd':
                    macd_line, signal_line, histogram = calculate_macd(
                        price_data.prices['close'], **params
                    )
                    custom_features[feature_name] = macd_line

            feature_types.append(feature_type)

        # Quality assessment
        quality_metrics = self._assess_feature_quality(custom_features, feature_types)

        return FeatureSet(
            features=custom_features,
            instrument=price_data.instrument,
            feature_types=feature_types,
            quality_metrics=quality_metrics,
            generation_config=self.config,
            created_at=datetime.now(),
            source_data_info={
                'start_date': str(price_data.prices.index[0]),
                'end_date': str(price_data.prices.index[-1]),
                'frequency': price_data.frequency.value,
                'price_type': price_data.price_type.value
            }
        )

    def generate_features_batch(self, data, features=None, config=None):
        """
        Generate features from DataFrame data (for API compatibility)

        Args:
            data: pandas DataFrame with price data
            features: list of features to generate
            config: optional configuration

        Returns:
            Dictionary with generated features
        """
        try:
            # Convert DataFrame to PriceData for processing
            # This is a simplified version for API compatibility
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")

            # Create a simple feature dictionary for now
            result = {}

            if 'returns' in (features or ['returns']):
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    result['returns'] = returns.to_dict()

            if 'volatility' in (features or ['volatility']):
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    volatility = returns.rolling(window=21).std().dropna()
                    result['volatility'] = volatility.to_dict()

            return result

        except Exception as e:
            raise ValueError(f"Feature generation failed: {str(e)}")

    def validate_features(self, features):
        """Validate features (for API compatibility)"""
        # Simple validation for now
        if not features:
            return {'is_valid': False, 'errors': ['No features provided']}

        return {
            'is_valid': True,
            'features_validated': len(features) if isinstance(features, list) else 1,
            'validation_timestamp': datetime.now().isoformat()
        }

    def get_feature_metadata(self, feature_name=None):
        """Get feature metadata (for API compatibility)"""
        metadata = {
            'returns': {
                'description': 'Price returns calculation',
                'methods': ['simple', 'log', 'percentage'],
                'default_period': 1
            },
            'volatility': {
                'description': 'Price volatility calculation',
                'methods': ['rolling', 'ewma', 'garch'],
                'default_window': 21
            },
            'momentum': {
                'description': 'Momentum indicators',
                'indicators': ['rsi', 'macd', 'stochastic'],
                'default_period': 14
            }
        }

        if feature_name:
            return metadata.get(feature_name, {})
        return metadata

    def get_available_features(self):
        """Get list of available features"""
        return ['returns', 'volatility', 'momentum', 'risk_metrics']

    def batch_generate_features(self, price_data_list: List[PriceData],
                              config: Optional[FeatureGenerationConfig] = None) -> List[FeatureSet]:
        """
        Generate features for multiple instruments

        Args:
            price_data_list: List of PriceData objects
            config: Optional configuration

        Returns:
            List of FeatureSet objects
        """
        feature_sets = []

        for price_data in price_data_list:
            try:
                feature_set = self.generate_features(price_data, config)
                feature_sets.append(feature_set)
                self.logger.info(f"Successfully generated features for {price_data.instrument.symbol}")
            except Exception as e:
                self.logger.error(f"Failed to generate features for {price_data.instrument.symbol}: {str(e)}")
                continue

        return feature_sets


# Alias for backward compatibility with tests
FeatureService = FeatureGenerator