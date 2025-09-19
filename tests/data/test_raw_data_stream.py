"""
Data quality tests for RawDataStream entity validation
Tests validation rules, constraints, and quality metrics for raw data streams
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional
import json
import uuid


class TestRawDataStreamValidation:
    """Test suite for RawDataStream entity data quality validation"""

    @pytest.fixture
    def valid_raw_data_stream(self):
        """Create a valid RawDataStream instance for testing"""
        return {
            "stream_id": str(uuid.uuid4()),
            "name": "stock_prices_stream",
            "source": "yahoo_finance",
            "data_type": "time_series",
            "schema": {
                "columns": ["timestamp", "symbol", "price", "volume", "open", "high", "low"],
                "types": {
                    "timestamp": "datetime64[ns]",
                    "symbol": "object",
                    "price": "float64",
                    "volume": "int64",
                    "open": "float64",
                    "high": "float64",
                    "low": "float64"
                },
                "constraints": {
                    "price": {"min": 0, "max": 100000},
                    "volume": {"min": 0, "max": 1000000000},
                    "high": {"min": 0, "max": 100000},
                    "low": {"min": 0, "max": 100000}
                }
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "data_frequency": "1min",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "currency": "USD",
                "timezone": "UTC"
            },
            "quality_metrics": {
                "completeness": 0.98,
                "accuracy": 0.95,
                "consistency": 0.97,
                "timeliness": 0.99,
                "uniqueness": 0.96
            },
            "processing_config": {
                "batch_size": 1000,
                "validation_rules": ["price_range_check", "volume_positive_check"],
                "error_handling": "log_and_continue"
            }
        }

    @pytest.fixture
    def sample_data_frames(self):
        """Generate sample data frames for testing"""
        np.random.seed(42)

        # Valid data frame
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 100),
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 100000, 100),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100)
        })

        # Ensure high >= low and high/low are close to price
        valid_data['high'] = np.maximum(valid_data['high'], valid_data['price'])
        valid_data['low'] = np.minimum(valid_data['low'], valid_data['price'])

        # Invalid data frame (with various quality issues)
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'INVALID'], 100),
            'price': np.concatenate([np.random.uniform(100, 200, 95), [-50, 500000, np.nan, np.inf, -np.inf]]),
            'volume': np.concatenate([np.random.randint(1000, 100000, 97), [-100, 0, np.nan]]),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100)
        })

        return {
            'valid': valid_data,
            'invalid': invalid_data
        }

    def test_stream_id_validation(self, valid_raw_data_stream):
        """
        Test: Stream ID validation rules
        Expected: Valid UUID format, non-empty, unique
        """
        # Test valid UUID
        stream = valid_raw_data_stream.copy()
        assert self._is_valid_uuid(stream["stream_id"]), \
            "Stream ID should be a valid UUID"

        # Test invalid UUID formats
        invalid_ids = [
            "",
            "invalid_uuid",
            "123e4567-e89b-12d3-a456-42661417400",  # Too short
            "123e4567-e89b-12d3-a456-42661417400x",  # Invalid character
            12345,  # Not a string
            None
        ]

        for invalid_id in invalid_ids:
            stream["stream_id"] = invalid_id
            with pytest.raises(ValueError) as exc_info:
                self._validate_stream_id(stream["stream_id"])

            assert "stream_id" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_stream_name_validation(self, valid_raw_data_stream):
        """
        Test: Stream name validation rules
        Expected: Non-empty, reasonable length, valid characters
        """
        stream = valid_raw_data_stream.copy()

        # Test valid name
        assert self._is_valid_stream_name(stream["name"]), \
            "Stream name should be valid"

        # Test invalid names
        invalid_names = [
            "",
            "a" * 256,  # Too long
            "name with spaces and special chars!",
            "name\nwith\nnewlines",
            "name\twith\ttabs",
            None,
            12345  # Not a string
        ]

        for invalid_name in invalid_names:
            stream["name"] = invalid_name
            with pytest.raises(ValueError) as exc_info:
                self._validate_stream_name(stream["name"])

            assert "name" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_schema_validation(self, valid_raw_data_stream):
        """
        Test: Schema validation rules
        Expected: Valid column names, data types, and constraints
        """
        stream = valid_raw_data_stream.copy()

        # Test valid schema
        assert self._is_valid_schema(stream["schema"]), \
            "Schema should be valid"

        # Test invalid schemas
        invalid_schemas = [
            # Missing required fields
            {"columns": ["col1"], "types": {"col1": "int64"}},  # Missing constraints
            {"columns": [], "types": {}, "constraints": {}},  # Empty
            # Invalid column names
            {"columns": ["", "valid_col"], "types": {"": "int64", "valid_col": "int64"}, "constraints": {}},
            # Invalid data types
            {"columns": ["col1"], "types": {"col1": "invalid_type"}, "constraints": {}},
            # Missing column definitions
            {"columns": ["col1", "col2"], "types": {"col1": "int64"}, "constraints": {}},
            # Invalid constraints
            {"columns": ["col1"], "types": {"col1": "int64"}, "constraints": {"col1": {"min": "not_a_number"}}}
        ]

        for invalid_schema in invalid_schemas:
            stream["schema"] = invalid_schema
            with pytest.raises(ValueError) as exc_info:
                self._validate_schema(invalid_schema)

            assert "schema" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_data_type_constraints_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data type and constraint validation
        Expected: Data conforms to schema types and constraints
        """
        stream = valid_raw_data_stream.copy()
        data = sample_data_frames['valid']

        # Test valid data
        validation_result = self._validate_data_against_schema(data, stream["schema"])
        assert validation_result["is_valid"], \
            "Valid data should pass schema validation"

        # Test invalid data
        invalid_data = sample_data_frames['invalid']
        invalid_result = self._validate_data_against_schema(invalid_data, stream["schema"])
        assert not invalid_result["is_valid"], \
            "Invalid data should fail schema validation"

        # Check specific validation errors
        assert len(invalid_result["errors"]) > 0, \
            "Should identify specific validation errors"

        # Test constraint violations
        constraint_errors = [e for e in invalid_result["errors"] if "constraint" in e["type"].lower()]
        assert len(constraint_errors) > 0, \
            "Should identify constraint violations"

    def test_completeness_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data completeness validation
        Expected: Measures missing values, handles different missing patterns
        """
        stream = valid_raw_data_stream.copy()

        # Test with complete data
        complete_data = sample_data_frames['valid']
        completeness = self._calculate_completeness(complete_data)
        assert completeness == 1.0, \
            "Complete data should have completeness score of 1.0"

        # Test with missing data
        data_with_missing = sample_data_frames['valid'].copy()
        data_with_missing.loc[10:15, 'price'] = np.nan
        data_with_missing.loc[20:25, 'volume'] = np.nan

        completeness_missing = self._calculate_completeness(data_with_missing)
        assert 0 < completeness_missing < 1.0, \
            "Data with missing values should have completeness score between 0 and 1"

        # Test column-wise completeness
        column_completeness = self._calculate_column_completeness(data_with_missing)
        assert 'price' in column_completeness, \
            "Should calculate completeness for each column"
        assert 'volume' in column_completeness, \
            "Should calculate completeness for each column"
        assert column_completeness['price'] < 1.0, \
            "Column with missing values should have completeness < 1.0"

    def test_accuracy_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data accuracy validation
        Expected: Validates data ranges, business rules, and data quality
        """
        stream = valid_raw_data_stream.copy()

        # Test with accurate data
        accurate_data = sample_data_frames['valid']
        accuracy_result = self._validate_accuracy(accurate_data, stream["schema"])

        assert accuracy_result["score"] > 0.9, \
            "Accurate data should have high accuracy score"

        # Test with inaccurate data
        inaccurate_data = sample_data_frames['invalid'].copy()
        accuracy_result_invalid = self._validate_accuracy(inaccurate_data, stream["schema"])

        assert accuracy_result_invalid["score"] < accuracy_result["score"], \
            "Inaccurate data should have lower accuracy score"

        # Check specific accuracy issues
        assert len(accuracy_result_invalid["issues"]) > 0, \
            "Should identify specific accuracy issues"

        # Test business rule validation
        business_rules = [
            {"rule": "high_low_validation", "description": "high >= low"},
            {"rule": "price_reasonable", "description": "price between 0 and 100000"}
        ]

        rule_results = self._validate_business_rules(accurate_data, business_rules)
        assert all(result["passed"] for result in rule_results), \
            "All business rules should pass for valid data"

    def test_consistency_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data consistency validation
        Expected: Validates internal consistency, cross-field relationships
        """
        stream = valid_raw_data_stream.copy()

        # Test consistent data
        consistent_data = sample_data_frames['valid'].copy()
        # Ensure high >= low (consistency rule)
        consistent_data['high'] = np.maximum(consistent_data['high'], consistent_data['low'])

        consistency_result = self._validate_consistency(consistent_data)
        assert consistency_result["score"] > 0.95, \
            "Consistent data should have high consistency score"

        # Test inconsistent data
        inconsistent_data = sample_data_frames['valid'].copy()
        # Create inconsistency: high < low for some rows
        inconsistent_data.loc[10:20, 'high'] = inconsistent_data.loc[10:20, 'low'] - 10

        consistency_result_invalid = self._validate_consistency(inconsistent_data)
        assert consistency_result_invalid["score"] < consistency_result["score"], \
            "Inconsistent data should have lower consistency score"

        # Test cross-field consistency
        cross_field_rules = [
            {"fields": ["high", "low"], "rule": "high >= low"},
            {"fields": ["price", "high", "low"], "rule": "low <= price <= high"}
        ]

        cross_field_results = self._validate_cross_field_consistency(consistent_data, cross_field_rules)
        assert all(result["passed"] for result in cross_field_results), \
            "All cross-field rules should pass for consistent data"

    def test_timeliness_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data timeliness validation
        Expected: Validates data freshness, timestamp consistency, latency
        """
        stream = valid_raw_data_stream.copy()

        # Test timely data
        timely_data = sample_data_frames['valid'].copy()
        timeliness_result = self._validate_timeliness(timely_data, stream["metadata"])

        assert timeliness_result["score"] > 0.9, \
            "Timely data should have high timeliness score"

        # Test outdated data
        outdated_data = sample_data_frames['valid'].copy()
        outdated_data['timestamp'] = pd.to_datetime(outdated_data['timestamp']) - pd.Timedelta(days=7)

        timeliness_result_outdated = self._validate_timeliness(outdated_data, stream["metadata"])
        assert timeliness_result_outdated["score"] < timeliness_result["score"], \
            "Outdated data should have lower timeliness score"

        # Test timestamp consistency
        timestamp_result = self._validate_timestamp_consistency(timely_data)
        assert timestamp_result["is_consistent"], \
            "Timestamps should be consistent in valid data"

        # Test data frequency validation
        expected_frequency = stream["metadata"]["data_frequency"]
        frequency_result = self._validate_data_frequency(timely_data, expected_frequency)
        assert frequency_result["matches_expected"], \
            "Data frequency should match expected frequency"

    def test_uniqueness_validation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Data uniqueness validation
        Expected: Validates duplicate records, key uniqueness
        """
        stream = valid_raw_data_stream.copy()

        # Test unique data
        unique_data = sample_data_frames['valid'].copy()
        uniqueness_result = self._validate_uniqueness(unique_data, ["timestamp", "symbol"])

        assert uniqueness_result["score"] > 0.95, \
            "Unique data should have high uniqueness score"

        # Test data with duplicates
        duplicate_data = sample_data_frames['valid'].copy()
        # Add duplicate rows
        duplicate_rows = duplicate_data.iloc[0:5].copy()
        duplicate_data = pd.concat([duplicate_data, duplicate_rows], ignore_index=True)

        uniqueness_result_duplicates = self._validate_uniqueness(duplicate_data, ["timestamp", "symbol"])
        assert uniqueness_result_duplicates["score"] < uniqueness_result["score"], \
            "Data with duplicates should have lower uniqueness score"

        # Test key uniqueness
        key_columns = ["timestamp", "symbol"]
        key_uniqueness = self._validate_key_uniqueness(unique_data, key_columns)
        assert key_uniqueness["is_unique"], \
            "Key columns should be unique in valid data"

    def test_quality_metrics_aggregation(self, valid_raw_data_stream, sample_data_frames):
        """
        Test: Quality metrics aggregation and scoring
        Expected: Combines individual metrics into overall quality score
        """
        stream = valid_raw_data_stream.copy()
        data = sample_data_frames['valid']

        # Calculate individual quality metrics
        completeness = self._calculate_completeness(data)
        accuracy = self._validate_accuracy(data, stream["schema"])["score"]
        consistency = self._validate_consistency(data)["score"]
        timeliness = self._validate_timeliness(data, stream["metadata"])["score"]
        uniqueness = self._validate_uniqueness(data, ["timestamp", "symbol"])["score"]

        # Aggregate metrics
        aggregated_metrics = self._aggregate_quality_metrics({
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "timeliness": timeliness,
            "uniqueness": uniqueness
        })

        # Validate aggregation
        assert "overall_score" in aggregated_metrics, \
            "Should calculate overall quality score"
        assert 0 <= aggregated_metrics["overall_score"] <= 1, \
            "Overall score should be between 0 and 1"
        assert "grade" in aggregated_metrics, \
            "Should assign quality grade"
        assert aggregated_metrics["grade"] in ["A", "B", "C", "D", "F"], \
            "Quality grade should be valid"

        # Test with poor quality data
        poor_data = sample_data_frames['invalid']
        poor_metrics = {
            "completeness": self._calculate_completeness(poor_data),
            "accuracy": self._validate_accuracy(poor_data, stream["schema"])["score"],
            "consistency": self._validate_consistency(poor_data)["score"],
            "timeliness": self._validate_timeliness(poor_data, stream["metadata"])["score"],
            "uniqueness": self._validate_uniqueness(poor_data, ["timestamp", "symbol"])["score"]
        }

        poor_aggregated = self._aggregate_quality_metrics(poor_metrics)
        assert poor_aggregated["overall_score"] < aggregated_metrics["overall_score"], \
            "Poor quality data should have lower overall score"

    def test_processing_configuration_validation(self, valid_raw_data_stream):
        """
        Test: Processing configuration validation
        Expected: Validates batch size, rules, and error handling settings
        """
        stream = valid_raw_data_stream.copy()

        # Test valid processing configuration
        assert self._is_valid_processing_config(stream["processing_config"]), \
            "Processing configuration should be valid"

        # Test invalid configurations
        invalid_configs = [
            {"batch_size": -1, "validation_rules": [], "error_handling": "log_and_continue"},  # Negative batch size
            {"batch_size": 0, "validation_rules": [], "error_handling": "log_and_continue"},  # Zero batch size
            {"batch_size": 1000000, "validation_rules": [], "error_handling": "log_and_continue"},  # Too large
            {"batch_size": 1000, "validation_rules": ["invalid_rule"], "error_handling": "invalid_handler"},  # Invalid rule/handler
            {"batch_size": 1000, "validation_rules": [], "error_handling": ""},  # Empty error handling
        ]

        for invalid_config in invalid_configs:
            stream["processing_config"] = invalid_config
            with pytest.raises(ValueError) as exc_info:
                self._validate_processing_config(invalid_config)

            assert "processing_config" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_metadata_validation(self, valid_raw_data_stream):
        """
        Test: Metadata validation
        Expected: Validates creation dates, frequencies, and other metadata
        """
        stream = valid_raw_data_stream.copy()

        # Test valid metadata
        assert self._is_valid_metadata(stream["metadata"]), \
            "Metadata should be valid"

        # Test invalid metadata
        invalid_metadata_list = [
            {"created_at": "invalid_date", "updated_at": "2025-09-18T10:00:00Z"},  # Invalid date format
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-17T10:00:00Z"},  # Updated before created
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-18T10:00:00Z", "data_frequency": "invalid_frequency"},  # Invalid frequency
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-18T10:00:00Z", "symbols": []},  # Empty symbols
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-18T10:00:00Z", "currency": ""},  # Empty currency
        ]

        for invalid_metadata in invalid_metadata_list:
            stream["metadata"] = invalid_metadata
            with pytest.raises(ValueError) as exc_info:
                self._validate_metadata(invalid_metadata)

            assert "metadata" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_data_stream_serialization(self, valid_raw_data_stream):
        """
        Test: Data stream serialization and deserialization
        Expected: Can be serialized to/from JSON without loss of information
        """
        stream = valid_raw_data_stream.copy()

        # Serialize to JSON
        json_str = json.dumps(stream, default=str)
        assert isinstance(json_str, str), \
            "Should serialize to JSON string"

        # Deserialize from JSON
        deserialized_stream = json.loads(json_str)
        assert deserialized_stream == stream, \
            "Deserialized stream should match original"

        # Test with datetime objects
        stream_with_datetime = stream.copy()
        stream_with_datetime["metadata"]["created_at"] = datetime.now()
        stream_with_datetime["metadata"]["updated_at"] = datetime.now()

        # Should handle datetime serialization
        json_str_with_datetime = json.dumps(stream_with_datetime, default=str)
        deserialized_with_datetime = json.loads(json_str_with_datetime)

        assert "created_at" in deserialized_with_datetime["metadata"], \
            "Should preserve datetime fields during serialization"

    # Helper methods
    def _is_valid_uuid(self, uuid_str) -> bool:
        """Check if string is a valid UUID"""
        if not isinstance(uuid_str, str):
            return False
        try:
            uuid.UUID(uuid_str)
            return True
        except (ValueError, AttributeError):
            return False

    def _validate_stream_id(self, stream_id: str):
        """Validate stream ID"""
        if not stream_id or not self._is_valid_uuid(stream_id):
            raise ValueError(f"Invalid stream_id: {stream_id}")

    def _is_valid_stream_name(self, name) -> bool:
        """Check if stream name is valid"""
        if not isinstance(name, str) or not name or len(name) > 255:
            return False
        # Allow alphanumeric, spaces, underscores, hyphens
        return all(c.isalnum() or c in ' _-' for c in name)

    def _validate_stream_name(self, name: str):
        """Validate stream name"""
        if not self._is_valid_stream_name(name):
            raise ValueError(f"Invalid stream name: {name}")

    def _is_valid_schema(self, schema: Dict) -> bool:
        """Check if schema is valid"""
        required_fields = ["columns", "types", "constraints"]
        if not all(field in schema for field in required_fields):
            return False

        if not schema["columns"] or len(schema["columns"]) == 0:
            return False

        # Check that all columns have types
        if set(schema["columns"]) != set(schema["types"].keys()):
            return False

        return True

    def _validate_schema(self, schema: Dict):
        """Validate schema"""
        if not self._is_valid_schema(schema):
            raise ValueError("Invalid schema")

    def _validate_data_against_schema(self, data: pd.DataFrame, schema: Dict) -> Dict:
        """Validate data against schema"""
        errors = []

        # Check columns match
        if set(data.columns) != set(schema["columns"]):
            errors.append({
                "type": "schema_mismatch",
                "message": "Data columns do not match schema columns"
            })

        # Check data types
        for col, expected_type in schema["types"].items():
            if col in data.columns:
                try:
                    if expected_type == "datetime64[ns]":
                        pd.to_datetime(data[col])
                    elif expected_type == "float64":
                        pd.to_numeric(data[col], errors='coerce')
                    elif expected_type == "int64":
                        pd.to_numeric(data[col], errors='coerce').astype('int64')
                except Exception as e:
                    errors.append({
                        "type": "type_mismatch",
                        "column": col,
                        "message": f"Column {col} does not match expected type {expected_type}"
                    })

        # Check constraints
        for col, constraints in schema["constraints"].items():
            if col in data.columns:
                if "min" in constraints:
                    if data[col].min() < constraints["min"]:
                        errors.append({
                            "type": "constraint_violation",
                            "column": col,
                            "message": f"Column {col} has values below minimum constraint"
                        })

                if "max" in constraints:
                    if data[col].max() > constraints["max"]:
                        errors.append({
                            "type": "constraint_violation",
                            "column": col,
                            "message": f"Column {col} has values above maximum constraint"
                        })

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        total_cells = data.size
        missing_cells = data.isna().sum().sum()

        if total_cells == 0:
            return 1.0

        return 1.0 - (missing_cells / total_cells)

    def _calculate_column_completeness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate completeness for each column"""
        completeness = {}
        for col in data.columns:
            total_cells = len(data)
            missing_cells = data[col].isna().sum()
            completeness[col] = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0
        return completeness

    def _validate_accuracy(self, data: pd.DataFrame, schema: Dict) -> Dict:
        """Validate data accuracy"""
        issues = []
        score = 1.0

        for col, constraints in schema.get("constraints", {}).items():
            if col in data.columns:
                # Check range constraints
                if "min" in constraints:
                    invalid_count = (data[col] < constraints["min"]).sum()
                    if invalid_count > 0:
                        issues.append({
                            "type": "range_violation",
                            "column": col,
                            "invalid_count": invalid_count,
                            "constraint": f"min={constraints['min']}"
                        })
                        score -= (invalid_count / len(data)) * 0.1

                if "max" in constraints:
                    invalid_count = (data[col] > constraints["max"]).sum()
                    if invalid_count > 0:
                        issues.append({
                            "type": "range_violation",
                            "column": col,
                            "invalid_count": invalid_count,
                            "constraint": f"max={constraints['max']}"
                        })
                        score -= (invalid_count / len(data)) * 0.1

        return {
            "score": max(0.0, score),
            "issues": issues
        }

    def _validate_consistency(self, data: pd.DataFrame) -> Dict:
        """Validate data consistency"""
        issues = []
        score = 1.0

        # Check high >= low consistency
        if 'high' in data.columns and 'low' in data.columns:
            inconsistent_rows = (data['high'] < data['low']).sum()
            if inconsistent_rows > 0:
                issues.append({
                    "type": "consistency_violation",
                    "rule": "high >= low",
                    "violation_count": inconsistent_rows
                })
                score -= (inconsistent_rows / len(data)) * 0.2

        # Check price is between high and low
        if all(col in data.columns for col in ['price', 'high', 'low']):
            price_out_of_bounds = ((data['price'] < data['low']) | (data['price'] > data['high'])).sum()
            if price_out_of_bounds > 0:
                issues.append({
                    "type": "consistency_violation",
                    "rule": "low <= price <= high",
                    "violation_count": price_out_of_bounds
                })
                score -= (price_out_of_bounds / len(data)) * 0.1

        return {
            "score": max(0.0, score),
            "issues": issues
        }

    def _validate_business_rules(self, data: pd.DataFrame, rules: List[Dict]) -> List[Dict]:
        """Validate business rules"""
        results = []

        for rule in rules:
            if rule["rule"] == "high_low_validation":
                passed = (data['high'] >= data['low']).all()
            elif rule["rule"] == "price_reasonable":
                passed = ((data['price'] >= 0) & (data['price'] <= 100000)).all()
            else:
                passed = False

            results.append({
                "rule": rule["rule"],
                "description": rule["description"],
                "passed": passed
            })

        return results

    def _validate_cross_field_consistency(self, data: pd.DataFrame, rules: List[Dict]) -> List[Dict]:
        """Validate cross-field consistency rules"""
        results = []

        for rule in rules:
            if rule["rule"] == "high >= low":
                passed = (data['high'] >= data['low']).all()
            elif rule["rule"] == "low <= price <= high":
                passed = ((data['low'] <= data['price']) & (data['price'] <= data['high'])).all()
            else:
                passed = False

            results.append({
                "fields": rule["fields"],
                "rule": rule["rule"],
                "passed": passed
            })

        return results

    def _validate_timeliness(self, data: pd.DataFrame, metadata: Dict) -> Dict:
        """Validate data timeliness"""
        issues = []
        score = 1.0

        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            now = pd.Timestamp.now()

            # Check data freshness (within 1 day)
            max_timestamp = timestamps.max()
            if (now - max_timestamp) > pd.Timedelta(days=1):
                issues.append({
                    "type": "freshness_violation",
                    "message": "Data is older than 1 day"
                })
                score -= 0.3

            # Check timestamp consistency
            if not timestamps.is_monotonic_increasing:
                issues.append({
                    "type": "timestamp_order_violation",
                    "message": "Timestamps are not in increasing order"
                })
                score -= 0.2

        return {
            "score": max(0.0, score),
            "issues": issues
        }

    def _validate_timestamp_consistency(self, data: pd.DataFrame) -> Dict:
        """Validate timestamp consistency"""
        if 'timestamp' not in data.columns:
            return {"is_consistent": False, "message": "No timestamp column"}

        timestamps = pd.to_datetime(data['timestamp'])
        is_consistent = timestamps.is_monotonic_increasing

        return {
            "is_consistent": is_consistent,
            "message": "Timestamps are consistent" if is_consistent else "Timestamps are not in order"
        }

    def _validate_data_frequency(self, data: pd.DataFrame, expected_frequency: str) -> Dict:
        """Validate data frequency"""
        if 'timestamp' not in data.columns:
            return {"matches_expected": False, "message": "No timestamp column"}

        timestamps = pd.to_datetime(data['timestamp'])
        if len(timestamps) < 2:
            return {"matches_expected": True, "message": "Insufficient data for frequency validation"}

        # Calculate actual frequency
        actual_frequency = pd.infer_freq(timestamps)

        frequency_mapping = {
            "1min": "T",
            "5min": "5T",
            "15min": "15T",
            "1hour": "H",
            "1day": "D"
        }

        expected_freq_code = frequency_mapping.get(expected_frequency, expected_frequency)
        matches_expected = actual_frequency == expected_freq_code

        return {
            "matches_expected": matches_expected,
            "actual_frequency": actual_frequency,
            "expected_frequency": expected_frequency
        }

    def _validate_uniqueness(self, data: pd.DataFrame, key_columns: List[str]) -> Dict:
        """Validate data uniqueness"""
        issues = []
        score = 1.0

        # Check for duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            issues.append({
                "type": "duplicate_rows",
                "count": duplicate_rows
            })
            score -= (duplicate_rows / len(data)) * 0.2

        # Check key uniqueness
        if all(col in data.columns for col in key_columns):
            key_duplicates = data.duplicated(subset=key_columns).sum()
            if key_duplicates > 0:
                issues.append({
                    "type": "key_duplicates",
                    "key_columns": key_columns,
                    "count": key_duplicates
                })
                score -= (key_duplicates / len(data)) * 0.3

        return {
            "score": max(0.0, score),
            "issues": issues
        }

    def _validate_key_uniqueness(self, data: pd.DataFrame, key_columns: List[str]) -> Dict:
        """Validate key uniqueness"""
        if not all(col in data.columns for col in key_columns):
            return {"is_unique": False, "message": "Key columns not found in data"}

        is_unique = not data.duplicated(subset=key_columns).any()

        return {
            "is_unique": is_unique,
            "message": "Key columns are unique" if is_unique else "Key columns have duplicates"
        }

    def _aggregate_quality_metrics(self, metrics: Dict) -> Dict:
        """Aggregate quality metrics into overall score"""
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "consistency": 0.20,
            "timeliness": 0.15,
            "uniqueness": 0.15
        }

        overall_score = sum(metrics[metric] * weights[metric] for metric in weights)

        # Determine grade
        if overall_score >= 0.95:
            grade = "A"
        elif overall_score >= 0.85:
            grade = "B"
        elif overall_score >= 0.75:
            grade = "C"
        elif overall_score >= 0.65:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "individual_metrics": metrics,
            "weights": weights
        }

    def _is_valid_processing_config(self, config: Dict) -> bool:
        """Check if processing configuration is valid"""
        required_fields = ["batch_size", "validation_rules", "error_handling"]
        if not all(field in config for field in required_fields):
            return False

        if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
            return False

        if config["batch_size"] > 100000:  # Reasonable upper limit
            return False

        valid_error_handlers = ["log_and_continue", "stop_on_error", "skip_and_continue"]
        if config["error_handling"] not in valid_error_handlers:
            return False

        return True

    def _validate_processing_config(self, config: Dict):
        """Validate processing configuration"""
        if not self._is_valid_processing_config(config):
            raise ValueError("Invalid processing configuration")

    def _is_valid_metadata(self, metadata: Dict) -> bool:
        """Check if metadata is valid"""
        required_fields = ["created_at", "updated_at", "data_frequency"]
        if not all(field in metadata for field in required_fields):
            return False

        try:
            created_at = pd.to_datetime(metadata["created_at"])
            updated_at = pd.to_datetime(metadata["updated_at"])

            if updated_at < created_at:
                return False

        except Exception:
            return False

        valid_frequencies = ["1min", "5min", "15min", "1hour", "1day"]
        if metadata["data_frequency"] not in valid_frequencies:
            return False

        return True

    def _validate_metadata(self, metadata: Dict):
        """Validate metadata"""
        if not self._is_valid_metadata(metadata):
            raise ValueError("Invalid metadata")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])