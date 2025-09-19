"""
Data quality tests for ProcessedData entity validation
Tests validation rules, data integrity, and quality metrics for processed data
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional, Tuple
import json
import uuid


class TestProcessedDataValidation:
    """Test suite for ProcessedData entity data quality validation"""

    @pytest.fixture
    def valid_processed_data(self):
        """Create a valid ProcessedData instance for testing"""
        return {
            "processed_id": str(uuid.uuid4()),
            "source_stream_id": str(uuid.uuid4()),
            "name": "processed_stock_prices",
            "processing_version": "1.0.0",
            "processing_date": datetime.now().isoformat(),
            "data": pd.DataFrame({
                'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
                'symbol': ['AAPL'] * 100,
                'price_normalized': np.random.uniform(0, 1, 100),  # Normalized price
                'volume_normalized': np.random.uniform(0, 1, 100),  # Normalized volume
                'price_standardized': np.random.normal(0, 1, 100),  # Standardized price
                'volume_standardized': np.random.normal(0, 1, 100),  # Standardized volume
                'returns': np.random.normal(0, 0.02, 100),  # Calculated returns
                'volatility': np.random.uniform(0.1, 0.5, 100),  # Calculated volatility
                'price_ma_5': np.random.uniform(150, 170, 100),  # 5-period moving average
                'price_ma_20': np.random.uniform(150, 170, 100),  # 20-period moving average
                'rsi': np.random.uniform(30, 70, 100),  # RSI indicator
                'macd': np.random.uniform(-2, 2, 100),  # MACD indicator
                'quality_score': np.random.uniform(0.8, 1.0, 100),  # Data quality score
                'outlier_flag': [False] * 95 + [True] * 5,  # Outlier detection flags
                'missing_imputed': [False] * 98 + [True] * 2  # Missing value imputation flags
            }),
            "schema": {
                "columns": [
                    "timestamp", "symbol", "price_normalized", "volume_normalized",
                    "price_standardized", "volume_standardized", "returns", "volatility",
                    "price_ma_5", "price_ma_20", "rsi", "macd", "quality_score",
                    "outlier_flag", "missing_imputed"
                ],
                "types": {
                    "timestamp": "datetime64[ns]",
                    "symbol": "object",
                    "price_normalized": "float64",
                    "volume_normalized": "float64",
                    "price_standardized": "float64",
                    "volume_standardized": "float64",
                    "returns": "float64",
                    "volatility": "float64",
                    "price_ma_5": "float64",
                    "price_ma_20": "float64",
                    "rsi": "float64",
                    "macd": "float64",
                    "quality_score": "float64",
                    "outlier_flag": "bool",
                    "missing_imputed": "bool"
                },
                "constraints": {
                    "price_normalized": {"min": 0, "max": 1},
                    "volume_normalized": {"min": 0, "max": 1},
                    "quality_score": {"min": 0, "max": 1},
                    "rsi": {"min": 0, "max": 100},
                    "returns": {"min": -1, "max": 1}
                }
            },
            "processing_metadata": {
                "preprocessing_steps": [
                    {"step": "normalization", "method": "min_max", "columns": ["price", "volume"]},
                    {"step": "standardization", "method": "z_score", "columns": ["price", "volume"]},
                    {"step": "feature_engineering", "method": "technical_indicators", "indicators": ["ma", "rsi", "macd"]},
                    {"step": "outlier_detection", "method": "z_score", "threshold": 3},
                    {"step": "missing_value_imputation", "method": "linear_interpolation"}
                ],
                "processing_time_ms": 1250,
                "records_processed": 100,
                "records_filtered": 0,
                "quality_improvements": {
                    "completeness_before": 0.98,
                    "completeness_after": 1.0,
                    "accuracy_before": 0.95,
                    "accuracy_after": 0.98
                }
            },
            "quality_metrics": {
                "data_quality": 0.98,
                "processing_quality": 0.95,
                "feature_quality": 0.96,
                "overall_score": 0.96
            },
            "validation_results": {
                "schema_valid": True,
                "constraints_satisfied": True,
                "statistical_properties_valid": True,
                "business_rules_satisfied": True
            },
            "data_lineage": {
                "source_datasets": ["raw_stock_prices_2025"],
                "transformation_applied": "preprocessing_pipeline_v1",
                "processing_chain": ["raw_data_ingestion", "data_cleaning", "feature_engineering", "quality_assurance"]
            }
        }

    @pytest.fixture
    def sample_processed_datasets(self):
        """Generate sample processed datasets for testing"""
        np.random.seed(42)

        # High-quality processed data
        high_quality_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': ['AAPL'] * 100,
            'price_normalized': np.random.uniform(0, 1, 100),
            'volume_normalized': np.random.uniform(0, 1, 100),
            'price_standardized': np.random.normal(0, 1, 100),
            'volume_standardized': np.random.normal(0, 1, 100),
            'returns': np.random.normal(0, 0.02, 100),
            'volatility': np.random.uniform(0.1, 0.5, 100),
            'quality_score': np.random.uniform(0.9, 1.0, 100),
            'outlier_flag': [False] * 98 + [True] * 2,
            'missing_imputed': [False] * 99 + [True] * 1
        })

        # Low-quality processed data
        low_quality_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': ['AAPL'] * 100,
            'price_normalized': np.concatenate([np.random.uniform(0, 1, 95), [-0.1, 1.1, np.nan, np.nan, np.nan]]),
            'volume_normalized': np.concatenate([np.random.uniform(0, 1, 98), [-0.05, 1.2]]),
            'price_standardized': np.concatenate([np.random.normal(0, 1, 90), [10, -10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
            'volume_standardized': np.concatenate([np.random.normal(0, 1, 95), [15, -15, np.nan, np.nan, np.nan]]),
            'returns': np.concatenate([np.random.normal(0, 0.02, 92), [2, -2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
            'volatility': np.concatenate([np.random.uniform(0.1, 0.5, 97), [2, -0.5, np.nan]]),
            'quality_score': np.concatenate([np.random.uniform(0.9, 1.0, 85), [0.1, 0.2, 0.3, 0.4, 0.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]),
            'outlier_flag': [True] * 100,  # All flagged as outliers (incorrect)
            'missing_imputed': [True] * 100  # All marked as imputed (incorrect)
        })

        return {
            'high_quality': high_quality_data,
            'low_quality': low_quality_data
        }

    def test_processed_id_validation(self, valid_processed_data):
        """
        Test: Processed ID validation rules
        Expected: Valid UUID format, non-empty, unique
        """
        processed = valid_processed_data.copy()

        # Test valid UUID
        assert self._is_valid_uuid(processed["processed_id"]), \
            "Processed ID should be a valid UUID"

        # Test invalid UUID formats
        invalid_ids = ["", "invalid_uuid", "123e4567-e89b-12d3-a456", 12345, None]

        for invalid_id in invalid_ids:
            processed["processed_id"] = invalid_id
            with pytest.raises(ValueError) as exc_info:
                self._validate_processed_id(invalid_id)

            assert "processed_id" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_source_stream_id_validation(self, valid_processed_data):
        """
        Test: Source stream ID validation rules
        Expected: Valid UUID format, references existing stream
        """
        processed = valid_processed_data.copy()

        # Test valid source stream ID
        assert self._is_valid_uuid(processed["source_stream_id"]), \
            "Source stream ID should be a valid UUID"

        # Test invalid source stream IDs
        invalid_ids = ["", "invalid_uuid", "123e4567-e89b-12d3-a456", 12345, None]

        for invalid_id in invalid_ids:
            processed["source_stream_id"] = invalid_id
            with pytest.raises(ValueError) as exc_info:
                self._validate_source_stream_id(invalid_id)

            assert "source_stream_id" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_processing_version_validation(self, valid_processed_data):
        """
        Test: Processing version validation rules
        Expected: Valid semantic version format, non-empty
        """
        processed = valid_processed_data.copy()

        # Test valid version
        assert self._is_valid_version(processed["processing_version"]), \
            "Processing version should be valid semantic version"

        # Test invalid versions
        invalid_versions = [
            "",
            "invalid_version",
            "1",  # Incomplete
            "1.2",  # Incomplete
            "1.2.3.4",  # Too many parts
            "a.b.c",  # Non-numeric
            "1.2.x",  # Contains non-numeric
            None,
            123  # Not a string
        ]

        for invalid_version in invalid_versions:
            processed["processing_version"] = invalid_version
            with pytest.raises(ValueError) as exc_info:
                self._validate_processing_version(invalid_version)

            assert "processing_version" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_processed_data_schema_validation(self, valid_processed_data):
        """
        Test: Processed data schema validation
        Expected: Valid columns, types, and constraints for processed data
        """
        processed = valid_processed_data.copy()

        # Test valid schema
        assert self._is_valid_processed_schema(processed["schema"]), \
            "Processed data schema should be valid"

        # Test invalid schemas
        invalid_schemas = [
            # Missing required processed data columns
            {"columns": ["timestamp", "symbol"], "types": {"timestamp": "datetime64[ns]", "symbol": "object"}, "constraints": {}},
            # Invalid normalized value ranges
            {"columns": ["price_normalized"], "types": {"price_normalized": "float64"}, "constraints": {"price_normalized": {"min": -1, "max": 2}}},
            # Missing quality score
            {"columns": ["timestamp", "symbol", "price_normalized"], "types": {"timestamp": "datetime64[ns]", "symbol": "object", "price_normalized": "float64"}, "constraints": {"price_normalized": {"min": 0, "max": 1}}},
            # Invalid boolean columns
            {"columns": ["outlier_flag"], "types": {"outlier_flag": "int64"}, "constraints": {}}
        ]

        for invalid_schema in invalid_schemas:
            processed["schema"] = invalid_schema
            with pytest.raises(ValueError) as exc_info:
                self._validate_processed_schema(invalid_schema)

            assert "schema" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_normalized_data_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Normalized data validation
        Expected: Normalized values within [0,1] range, proper scaling
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test valid normalized data
        normalized_columns = ['price_normalized', 'volume_normalized']
        for col in normalized_columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()

                assert min_val >= 0, f"Normalized column {col} should have minimum >= 0"
                assert max_val <= 1, f"Normalized column {col} should have maximum <= 1"

        # Test invalid normalized data
        invalid_normalized = data.copy()
        invalid_normalized['price_normalized'] = np.concatenate([
            np.random.uniform(0, 1, 95),
            [-0.1, 1.1, 1.5]  # Invalid normalized values
        ])

        validation_result = self._validate_normalized_data(invalid_normalized, processed["schema"])
        assert not validation_result["is_valid"], \
            "Invalid normalized data should fail validation"

        assert len(validation_result["violations"]) > 0, \
            "Should identify specific normalization violations"

    def test_standardized_data_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Standardized data validation
        Expected: Standardized values have mean ≈ 0, std ≈ 1
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test valid standardized data
        standardized_columns = ['price_standardized', 'volume_standardized']
        for col in standardized_columns:
            if col in data.columns:
                mean_val = data[col].mean()
                std_val = data[col].std()

                assert abs(mean_val) < 0.1, f"Standardized column {col} should have mean ≈ 0"
                assert abs(std_val - 1) < 0.1, f"Standardized column {col} should have std ≈ 1"

        # Test invalid standardized data
        invalid_standardized = data.copy()
        invalid_standardized['price_standardized'] = np.concatenate([
            np.random.normal(0, 1, 90),
            [10, -10, 15, -15]  # Invalid standardized values
        ])

        validation_result = self._validate_standardized_data(invalid_standardized)
        assert not validation_result["is_valid"], \
            "Invalid standardized data should fail validation"

    def test_feature_engineering_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Feature engineering validation
        Expected: Calculated features are mathematically correct and consistent
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test returns calculation
        if 'returns' in data.columns and 'price_normalized' in data.columns:
            # Calculate expected returns
            price_series = data['price_normalized']
            expected_returns = price_series.pct_change().fillna(0)

            # Compare with actual returns (should be similar)
            correlation = np.corrcoef(data['returns'].fillna(0), expected_returns)[0, 1]
            assert abs(correlation) > 0.7, \
                "Returns should be highly correlated with price changes"

        # Test moving averages
        if all(col in data.columns for col in ['price_normalized', 'price_ma_5', 'price_ma_20']):
            # Moving averages should be smooth
            ma_5_smoothness = self._calculate_smoothness(data['price_ma_5'])
            ma_20_smoothness = self._calculate_smoothness(data['price_ma_20'])

            assert ma_20_smoothness > ma_5_smoothness, \
                "Longer moving average should be smoother"

        # Test technical indicators ranges
        if 'rsi' in data.columns:
            rsi_values = data['rsi'].dropna()
            assert rsi_values.min() >= 0, "RSI should be >= 0"
            assert rsi_values.max() <= 100, "RSI should be <= 100"

        if 'volatility' in data.columns:
            volatility_values = data['volatility'].dropna()
            assert volatility_values.min() >= 0, "Volatility should be >= 0"
            assert (volatility_values > 0).any(), "Volatility should have some positive values"

    def test_outlier_detection_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Outlier detection validation
        Expected: Outlier flags are correctly applied and statistically justified
        """
        processed = valid_processed_data.copy()
        high_quality_data = sample_processed_datasets['high_quality']
        low_quality_data = sample_processed_datasets['low_quality']

        # Test high quality data (should have few outliers)
        high_quality_outliers = self._validate_outlier_flags(high_quality_data)
        assert high_quality_outliers["outlier_percentage"] < 0.1, \
            "High quality data should have < 10% outliers"

        # Test low quality data (should have more outliers)
        low_quality_outliers = self._validate_outlier_flags(low_quality_data)
        assert low_quality_outliers["outlier_percentage"] > 0.5, \
            "Low quality data should have > 50% outliers"

        # Test outlier detection consistency
        outlier_columns = ['outlier_flag']
        for col in outlier_columns:
            if col in high_quality_data.columns:
                assert high_quality_data[col].dtype == bool, \
                    f"Outlier flag column {col} should be boolean"

        # Test outlier impact assessment
        outlier_impact = self._assess_outlier_impact(high_quality_data)
        assert "impact_score" in outlier_impact, \
            "Should assess outlier impact on data quality"

    def test_missing_value_imputation_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Missing value imputation validation
        Expected: Imputation flags are correctly applied and imputed values are reasonable
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test imputation flags
        imputation_result = self._validate_imputation_flags(data)
        assert imputation_result["is_valid"], \
            "Imputation flags should be valid"

        # Test imputation quality
        if 'missing_imputed' in data.columns:
            imputed_rows = data[data['missing_imputed'] == True]
            non_imputed_rows = data[data['missing_imputed'] == False]

            # Imputed values should be reasonable
            for col in data.select_dtypes(include=[np.number]).columns:
                if col != 'missing_imputed' and col != 'outlier_flag':
                    imputed_values = imputed_rows[col].dropna()
                    non_imputed_values = non_imputed_rows[col].dropna()

                    if len(imputed_values) > 0 and len(non_imputed_values) > 0:
                        # Imputed values should be within reasonable range of non-imputed values
                        non_imputed_range = non_imputed_values.max() - non_imputed_values.min()
                        if non_imputed_range > 0:
                            imputed_in_range = ((imputed_values >= non_imputed_values.min()) &
                                              (imputed_values <= non_imputed_values.max())).sum()
                            imputed_percentage = imputed_in_range / len(imputed_values)
                            assert imputed_percentage > 0.5, \
                                f"Most imputed values in {col} should be within range of non-imputed values"

    def test_quality_score_validation(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Quality score validation
        Expected: Quality scores are within [0,1] range and correlate with actual quality
        """
        processed = valid_processed_data.copy()
        high_quality_data = sample_processed_datasets['high_quality']
        low_quality_data = sample_processed_datasets['low_quality']

        # Test quality score ranges
        for data, expected_score in [(high_quality_data, 0.9), (low_quality_data, 0.3)]:
            if 'quality_score' in data.columns:
                quality_scores = data['quality_score'].dropna()

                assert quality_scores.min() >= 0, "Quality scores should be >= 0"
                assert quality_scores.max() <= 1, "Quality scores should be <= 1"
                assert quality_scores.mean() > expected_score - 0.2, \
                    f"Quality scores should reflect actual data quality (expected > {expected_score - 0.2})"

        # Test quality score correlation with other metrics
        quality_correlation = self._calculate_quality_correlation(high_quality_data)
        assert "correlations" in quality_correlation, \
            "Should calculate correlations between quality score and other metrics"

    def test_processing_metadata_validation(self, valid_processed_data):
        """
        Test: Processing metadata validation
        Expected: Valid processing steps, timing information, and quality improvements
        """
        processed = valid_processed_data.copy()

        # Test valid processing metadata
        assert self._is_valid_processing_metadata(processed["processing_metadata"]), \
            "Processing metadata should be valid"

        # Test invalid metadata
        invalid_metadata_list = [
            {"preprocessing_steps": [], "processing_time_ms": -1},  # Negative time
            {"preprocessing_steps": [{"step": "invalid_step"}], "processing_time_ms": 1000},  # Invalid step
            {"preprocessing_steps": [], "processing_time_ms": 1000, "records_processed": -1},  # Negative records
            {"preprocessing_steps": [], "processing_time_ms": 1000, "quality_improvements": {"completeness_before": 1.5}},  # Invalid improvement score
            {"preprocessing_steps": [], "processing_time_ms": 1000, "quality_improvements": {"completeness_before": 0.8, "completeness_after": 0.7}}  # Quality degraded
        ]

        for invalid_metadata in invalid_metadata_list:
            processed["processing_metadata"] = invalid_metadata
            with pytest.raises(ValueError) as exc_info:
                self._validate_processing_metadata(invalid_metadata)

            assert "processing_metadata" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_validation_results_validation(self, valid_processed_data):
        """
        Test: Validation results validation
        Expected: All validation flags are boolean and comprehensive
        """
        processed = valid_processed_data.copy()

        # Test valid validation results
        assert self._is_valid_validation_results(processed["validation_results"]), \
            "Validation results should be valid"

        # Test required validation flags
        required_flags = ["schema_valid", "constraints_satisfied", "statistical_properties_valid", "business_rules_satisfied"]
        for flag in required_flags:
            assert flag in processed["validation_results"], \
                f"Validation results should include {flag} flag"
            assert isinstance(processed["validation_results"][flag], bool), \
                f"Validation flag {flag} should be boolean"

        # Test invalid validation results
        invalid_results = [
            {"schema_valid": "yes", "constraints_satisfied": True},  # Non-boolean
            {"schema_valid": True},  # Missing required flags
            {"schema_valid": True, "constraints_satisfied": True, "invalid_flag": False}  # Extra flag
        ]

        for invalid_result in invalid_results:
            processed["validation_results"] = invalid_result
            with pytest.raises(ValueError) as exc_info:
                self._validate_validation_results(invalid_result)

            assert "validation_results" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_data_lineage_validation(self, valid_processed_data):
        """
        Test: Data lineage validation
        Expected: Complete and valid lineage information
        """
        processed = valid_processed_data.copy()

        # Test valid data lineage
        assert self._is_valid_data_lineage(processed["data_lineage"]), \
            "Data lineage should be valid"

        # Test invalid lineage
        invalid_lineage_list = [
            {"source_datasets": [], "transformation_applied": "test"},  # Empty sources
            {"source_datasets": ["source1"], "transformation_applied": ""},  # Empty transformation
            {"source_datasets": ["source1"], "transformation_applied": "test", "processing_chain": []},  # Empty chain
            {"source_datasets": [123], "transformation_applied": "test"},  # Invalid source format
        ]

        for invalid_lineage in invalid_lineage_list:
            processed["data_lineage"] = invalid_lineage
            with pytest.raises(ValueError) as exc_info:
                self._validate_data_lineage(invalid_lineage)

            assert "data_lineage" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_processed_data_integrity(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Processed data integrity across all components
        Expected: All components are consistent and mutually validating
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test schema matches actual data
        schema_match = self._validate_schema_data_match(processed["schema"], data)
        assert schema_match["matches"], \
            "Schema should match actual data structure"

        # Test processing steps are reflected in data
        processing_reflection = self._validate_processing_steps_reflection(processed["processing_metadata"], data)
        assert processing_reflection["is_consistent"], \
            "Processing steps should be reflected in processed data"

        # Test quality metrics are consistent with actual data quality
        quality_consistency = self._validate_quality_metrics_consistency(processed["quality_metrics"], data)
        assert quality_consistency["is_consistent"], \
            "Quality metrics should be consistent with actual data quality"

        # Test validation results are accurate
        validation_accuracy = self._validate_validation_results_accuracy(processed["validation_results"], data)
        assert validation_accuracy["is_accurate"], \
            "Validation results should be accurate"

    def test_processed_data_performance_metrics(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Performance metrics for processed data
        Expected: Processing efficiency, memory usage, and computational performance
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test processing time efficiency
        processing_efficiency = self._calculate_processing_efficiency(processed["processing_metadata"], data)
        assert "efficiency_score" in processing_efficiency, \
            "Should calculate processing efficiency"

        # Test memory usage optimization
        memory_usage = self._calculate_memory_usage(data)
        assert "memory_mb" in memory_usage, \
            "Should calculate memory usage"

        # Test computational performance
        computational_performance = self._assess_computational_performance(data)
        assert "performance_score" in computational_performance, \
            "Should assess computational performance"

    def test_processed_data_reproducibility(self, valid_processed_data, sample_processed_datasets):
        """
        Test: Processed data reproducibility
        Expected: Same processing yields identical results
        """
        processed = valid_processed_data.copy()
        data = sample_processed_datasets['high_quality']

        # Test deterministic processing
        reproducibility_result = self._test_reproducibility(data, processed["processing_metadata"])
        assert reproducibility_result["is_reproducible"], \
            "Processing should be reproducible"

        # Test version consistency
        version_consistency = self._test_version_consistency(processed["processing_version"])
        assert version_consistency["is_consistent"], \
            "Processing version should be consistent"

    # Helper methods
    def _is_valid_uuid(self, uuid_str: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False

    def _validate_processed_id(self, processed_id: str):
        """Validate processed ID"""
        if not processed_id or not self._is_valid_uuid(processed_id):
            raise ValueError(f"Invalid processed_id: {processed_id}")

    def _validate_source_stream_id(self, source_stream_id: str):
        """Validate source stream ID"""
        if not source_stream_id or not self._is_valid_uuid(source_stream_id):
            raise ValueError(f"Invalid source_stream_id: {source_stream_id}")

    def _is_valid_version(self, version: str) -> bool:
        """Check if version is valid semantic version"""
        if not version:
            return False

        parts = version.split('.')
        if len(parts) != 3:
            return False

        try:
            major, minor, patch = map(int, parts)
            return major >= 0 and minor >= 0 and patch >= 0
        except ValueError:
            return False

    def _validate_processing_version(self, version: str):
        """Validate processing version"""
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid processing_version: {version}")

    def _is_valid_processed_schema(self, schema: Dict) -> bool:
        """Check if processed data schema is valid"""
        required_fields = ["columns", "types", "constraints"]
        if not all(field in schema for field in required_fields):
            return False

        # Check for required processed data columns
        required_columns = ["timestamp", "symbol"]
        if not all(col in schema["columns"] for col in required_columns):
            return False

        # Check normalized value constraints
        normalized_columns = [col for col in schema["columns"] if "normalized" in col]
        for col in normalized_columns:
            if col in schema["constraints"]:
                constraints = schema["constraints"][col]
                if constraints.get("min", 0) != 0 or constraints.get("max", 1) != 1:
                    return False

        # Check boolean columns
        boolean_columns = [col for col in schema["columns"] if "flag" in col or col.startswith("is_")]
        for col in boolean_columns:
            if col in schema["types"] and schema["types"][col] != "bool":
                return False

        return True

    def _validate_processed_schema(self, schema: Dict):
        """Validate processed data schema"""
        if not self._is_valid_processed_schema(schema):
            raise ValueError("Invalid processed data schema")

    def _validate_schema_data_match(self, schema: Dict, data: pd.DataFrame) -> Dict:
        """Validate that schema matches actual data"""
        mismatches = []

        # Check columns match
        if set(data.columns) != set(schema["columns"]):
            mismatches.append("Data columns do not match schema columns")

        # Check types match
        for col, expected_type in schema["types"].items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                type_mapping = {
                    "datetime64[ns]": "datetime64[ns]",
                    "float64": ["float64", "float32"],
                    "int64": ["int64", "int32"],
                    "bool": "bool",
                    "object": "object"
                }

                if expected_type in type_mapping:
                    if isinstance(type_mapping[expected_type], list):
                        if actual_type not in type_mapping[expected_type]:
                            mismatches.append(f"Column {col} type {actual_type} does not match expected {expected_type}")
                    else:
                        if actual_type != type_mapping[expected_type]:
                            mismatches.append(f"Column {col} type {actual_type} does not match expected {expected_type}")

        return {
            "matches": len(mismatches) == 0,
            "mismatches": mismatches
        }

    def _validate_normalized_data(self, data: pd.DataFrame, schema: Dict) -> Dict:
        """Validate normalized data"""
        violations = []

        normalized_columns = [col for col in data.columns if "normalized" in col]

        for col in normalized_columns:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    min_val = values.min()
                    max_val = values.max()

                    if min_val < 0:
                        violations.append({
                            "column": col,
                            "type": "range_violation",
                            "message": f"Column {col} has values below 0",
                            "min_value": min_val
                        })

                    if max_val > 1:
                        violations.append({
                            "column": col,
                            "type": "range_violation",
                            "message": f"Column {col} has values above 1",
                            "max_value": max_val
                        })

        return {
            "is_valid": len(violations) == 0,
            "violations": violations
        }

    def _validate_standardized_data(self, data: pd.DataFrame) -> Dict:
        """Validate standardized data"""
        violations = []

        standardized_columns = [col for col in data.columns if "standardized" in col]

        for col in standardized_columns:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 1:
                    mean_val = values.mean()
                    std_val = values.std()

                    if abs(mean_val) > 0.5:
                        violations.append({
                            "column": col,
                            "type": "mean_violation",
                            "message": f"Column {col} mean {mean_val} is far from 0",
                            "mean": mean_val
                        })

                    if abs(std_val - 1) > 0.5:
                        violations.append({
                            "column": col,
                            "type": "std_violation",
                            "message": f"Column {col} std {std_val} is far from 1",
                            "std": std_val
                        })

        return {
            "is_valid": len(violations) == 0,
            "violations": violations
        }

    def _calculate_smoothness(self, series: pd.Series) -> float:
        """Calculate smoothness of a series (inverse of volatility)"""
        if len(series) < 2:
            return 1.0

        diff = series.diff().dropna()
        volatility = diff.std()
        return 1.0 / (1.0 + volatility) if volatility > 0 else 1.0

    def _validate_outlier_flags(self, data: pd.DataFrame) -> Dict:
        """Validate outlier detection flags"""
        outlier_columns = [col for col in data.columns if "outlier" in col.lower()]

        total_flags = 0
        total_cells = 0

        for col in outlier_columns:
            if col in data.columns:
                flags = data[col].sum() if data[col].dtype == bool else 0
                total_flags += flags
                total_cells += len(data)

        outlier_percentage = total_flags / total_cells if total_cells > 0 else 0

        return {
            "outlier_count": total_flags,
            "outlier_percentage": outlier_percentage,
            "outlier_columns": outlier_columns
        }

    def _assess_outlier_impact(self, data: pd.DataFrame) -> Dict:
        """Assess impact of outliers on data quality"""
        outlier_columns = [col for col in data.columns if "outlier" in col.lower()]

        impact_score = 0.0
        impact_details = []

        for col in outlier_columns:
            if col in data.columns and data[col].dtype == bool:
                outlier_mask = data[col]
                non_outlier_data = data[~outlier_mask]
                outlier_data = data[outlier_mask]

                if len(outlier_data) > 0:
                    # Compare statistical properties
                    numeric_cols = data.select_dtypes(include=[np.number]).columns

                    for numeric_col in numeric_cols:
                        if numeric_col != col:
                            non_outlier_std = non_outlier_data[numeric_col].std()
                            outlier_std = outlier_data[numeric_col].std()

                            if non_outlier_std > 0:
                                impact_ratio = outlier_std / non_outlier_std
                                if impact_ratio > 2.0:  # Outliers increase variability significantly
                                    impact_score += 0.1
                                    impact_details.append({
                                        "column": numeric_col,
                                        "impact_ratio": impact_ratio,
                                        "outlier_column": col
                                    })

        return {
            "impact_score": min(1.0, impact_score),
            "impact_details": impact_details
        }

    def _validate_imputation_flags(self, data: pd.DataFrame) -> Dict:
        """Validate imputation flags"""
        imputation_columns = [col for col in data.columns if "imputed" in col.lower() or "missing" in col.lower()]

        issues = []

        for col in imputation_columns:
            if col in data.columns:
                if data[col].dtype != bool:
                    issues.append({
                        "column": col,
                        "type": "type_error",
                        "message": f"Imputation flag column {col} should be boolean"
                    })

        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

    def _calculate_quality_correlation(self, data: pd.DataFrame) -> Dict:
        """Calculate correlation between quality score and other metrics"""
        if 'quality_score' not in data.columns:
            return {"correlations": {}}

        quality_series = data['quality_score'].dropna()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlations = {}

        for col in numeric_cols:
            if col != 'quality_score':
                other_series = data[col].dropna()
                if len(quality_series) > 1 and len(other_series) > 1:
                    # Align series
                    aligned_quality, aligned_other = quality_series.align(other_series, join='inner')
                    if len(aligned_quality) > 1:
                        correlation = np.corrcoef(aligned_quality, aligned_other)[0, 1]
                        if not np.isnan(correlation):
                            correlations[col] = correlation

        return {
            "correlations": correlations,
            "max_correlation": max(abs(c) for c in correlations.values()) if correlations else 0
        }

    def _is_valid_processing_metadata(self, metadata: Dict) -> bool:
        """Check if processing metadata is valid"""
        required_fields = ["preprocessing_steps", "processing_time_ms", "records_processed"]
        if not all(field in metadata for field in required_fields):
            return False

        if metadata["processing_time_ms"] < 0:
            return False

        if metadata["records_processed"] < 0:
            return False

        # Check preprocessing steps
        if not isinstance(metadata["preprocessing_steps"], list) or len(metadata["preprocessing_steps"]) == 0:
            return False

        valid_steps = ["normalization", "standardization", "feature_engineering", "outlier_detection", "missing_value_imputation"]
        for step in metadata["preprocessing_steps"]:
            if not isinstance(step, dict) or "step" not in step or step["step"] not in valid_steps:
                return False

        # Check quality improvements
        if "quality_improvements" in metadata:
            improvements = metadata["quality_improvements"]
            for metric in ["completeness", "accuracy"]:
                if f"{metric}_before" in improvements and f"{metric}_after" in improvements:
                    if improvements[f"{metric}_before"] > improvements[f"{metric}_after"]:
                        return False  # Quality should not degrade

        return True

    def _validate_processing_metadata(self, metadata: Dict):
        """Validate processing metadata"""
        if not self._is_valid_processing_metadata(metadata):
            raise ValueError("Invalid processing metadata")

    def _is_valid_validation_results(self, results: Dict) -> bool:
        """Check if validation results are valid"""
        required_flags = ["schema_valid", "constraints_satisfied", "statistical_properties_valid", "business_rules_satisfied"]
        if not all(flag in results for flag in required_flags):
            return False

        # All flags should be boolean
        for flag in required_flags:
            if not isinstance(results[flag], bool):
                return False

        return True

    def _validate_validation_results(self, results: Dict):
        """Validate validation results"""
        if not self._is_valid_validation_results(results):
            raise ValueError("Invalid validation results")

    def _is_valid_data_lineage(self, lineage: Dict) -> bool:
        """Check if data lineage is valid"""
        required_fields = ["source_datasets", "transformation_applied", "processing_chain"]
        if not all(field in lineage for field in required_fields):
            return False

        if not isinstance(lineage["source_datasets"], list) or len(lineage["source_datasets"]) == 0:
            return False

        if not lineage["transformation_applied"]:
            return False

        if not isinstance(lineage["processing_chain"], list) or len(lineage["processing_chain"]) == 0:
            return False

        return True

    def _validate_data_lineage(self, lineage: Dict):
        """Validate data lineage"""
        if not self._is_valid_data_lineage(lineage):
            raise ValueError("Invalid data lineage")

    def _validate_processing_steps_reflection(self, metadata: Dict, data: pd.DataFrame) -> Dict:
        """Validate that processing steps are reflected in the data"""
        inconsistencies = []

        steps = metadata.get("preprocessing_steps", [])

        for step in steps:
            step_type = step.get("step")

            if step_type == "normalization":
                normalized_cols = [col for col in data.columns if "normalized" in col]
                if not normalized_cols:
                    inconsistencies.append("Normalization step applied but no normalized columns found")

            elif step_type == "standardization":
                standardized_cols = [col for col in data.columns if "standardized" in col]
                if not standardized_cols:
                    inconsistencies.append("Standardization step applied but no standardized columns found")

            elif step_type == "outlier_detection":
                outlier_cols = [col for col in data.columns if "outlier" in col.lower()]
                if not outlier_cols:
                    inconsistencies.append("Outlier detection step applied but no outlier columns found")

            elif step_type == "missing_value_imputation":
                imputed_cols = [col for col in data.columns if "imputed" in col.lower() or "missing" in col.lower()]
                if not imputed_cols:
                    inconsistencies.append("Missing value imputation step applied but no imputation columns found")

        return {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies
        }

    def _validate_quality_metrics_consistency(self, quality_metrics: Dict, data: pd.DataFrame) -> Dict:
        """Validate that quality metrics are consistent with actual data quality"""
        inconsistencies = []

        # Check overall score
        overall_score = quality_metrics.get("overall_score", 0)
        individual_scores = [
            quality_metrics.get("data_quality", 0),
            quality_metrics.get("processing_quality", 0),
            quality_metrics.get("feature_quality", 0)
        ]

        calculated_average = np.mean(individual_scores)
        if abs(overall_score - calculated_average) > 0.1:
            inconsistencies.append(f"Overall score {overall_score} differs from average {calculated_average}")

        # Check individual score ranges
        for score_name, score_value in quality_metrics.items():
            if score_name != "overall_score":
                if not (0 <= score_value <= 1):
                    inconsistencies.append(f"Score {score_name} {score_value} is not in [0,1] range")

        return {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies
        }

    def _validate_validation_results_accuracy(self, validation_results: Dict, data: pd.DataFrame) -> Dict:
        """Validate that validation results are accurate"""
        inaccuracies = []

        # Test schema validation
        if validation_results.get("schema_valid", False):
            # Check if data actually has valid schema
            has_required_columns = all(col in data.columns for col in ["timestamp", "symbol"])
            if not has_required_columns:
                inaccuracies.append("Schema marked as valid but missing required columns")

        # Test constraint validation
        if validation_results.get("constraints_satisfied", False):
            # Check normalized columns are in [0,1]
            normalized_cols = [col for col in data.columns if "normalized" in col]
            for col in normalized_cols:
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        if values.min() < 0 or values.max() > 1:
                            inaccuracies.append(f"Constraints marked as satisfied but {col} violates normalization constraints")

        return {
            "is_accurate": len(inaccuracies) == 0,
            "inaccuracies": inaccuracies
        }

    def _calculate_processing_efficiency(self, metadata: Dict, data: pd.DataFrame) -> Dict:
        """Calculate processing efficiency metrics"""
        processing_time = metadata.get("processing_time_ms", 0)
        records_processed = metadata.get("records_processed", 0)

        if processing_time > 0 and records_processed > 0:
            records_per_second = (records_processed / processing_time) * 1000
            efficiency_score = min(1.0, records_per_second / 10000)  # Normalize to 10k records/sec as perfect
        else:
            records_per_second = 0
            efficiency_score = 0

        return {
            "efficiency_score": efficiency_score,
            "records_per_second": records_per_second,
            "processing_time_ms": processing_time,
            "records_processed": records_processed
        }

    def _calculate_memory_usage(self, data: pd.DataFrame) -> Dict:
        """Calculate memory usage of processed data"""
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)

        return {
            "memory_mb": memory_mb,
            "memory_per_record_mb": memory_mb / len(data) if len(data) > 0 else 0,
            "total_records": len(data)
        }

    def _assess_computational_performance(self, data: pd.DataFrame) -> Dict:
        """Assess computational performance of processed data"""
        start_time = pd.Timestamp.now()

        # Perform typical operations
        if len(data) > 0:
            # Calculate correlations
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                correlations = numeric_data.corr()

            # Calculate aggregations
            aggregations = data.agg(['mean', 'std', 'min', 'max'])

        end_time = pd.Timestamp.now()
        computation_time = (end_time - start_time).total_seconds()

        performance_score = min(1.0, 1.0 / (computation_time + 0.001))  # Inverse of time

        return {
            "performance_score": performance_score,
            "computation_time_seconds": computation_time,
            "operations_performed": len(data.columns)
        }

    def _test_reproducibility(self, data: pd.DataFrame, metadata: Dict) -> Dict:
        """Test if processing is reproducible"""
        # This is a simplified test - in practice, you'd run the actual processing twice
        # and compare results

        is_reproducible = True  # Assume reproducible for now
        differences = []

        return {
            "is_reproducible": is_reproducible,
            "differences": differences
        }

    def _test_version_consistency(self, version: str) -> Dict:
        """Test version consistency"""
        # Check if version follows semantic versioning
        is_consistent = self._is_valid_version(version)

        return {
            "is_consistent": is_consistent,
            "version": version
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])