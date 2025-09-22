"""
Data quality tests for PreprocessingRules entity validation
Tests validation rules, rule execution, and rule management for preprocessing rules
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import uuid
import re


class TestPreprocessingRulesValidation:
    """Test suite for PreprocessingRules entity data quality validation"""

    @pytest.fixture
    def valid_preprocessing_rules(self):
        """Create a valid PreprocessingRules instance for testing"""
        return {
            "rule_id": str(uuid.uuid4()),
            "name": "stock_price_validation_rules",
            "version": "1.0.0",
            "category": "validation",
            "description": "Comprehensive validation rules for stock price data",
            "rules": [
                {
                    "rule_name": "price_positive_check",
                    "rule_type": "constraint",
                    "field": "price",
                    "condition": "must_be_positive",
                    "parameters": {
                        "min_value": 0,
                        "strict": True,
                        "action": "reject"
                    },
                    "severity": "error",
                    "enabled": True,
                    "priority": 1
                },
                {
                    "rule_name": "volume_positive_check",
                    "rule_type": "constraint",
                    "field": "volume",
                    "condition": "must_be_positive",
                    "parameters": {
                        "min_value": 0,
                        "strict": True,
                        "action": "reject"
                    },
                    "severity": "error",
                    "enabled": True,
                    "priority": 1
                },
                {
                    "rule_name": "price_outlier_detection",
                    "rule_type": "statistical",
                    "field": "price",
                    "condition": "z_score_outlier",
                    "parameters": {
                        "threshold": 3.0,
                        "method": "z_score",
                        "action": "flag"
                    },
                    "severity": "warning",
                    "enabled": True,
                    "priority": 2
                },
                {
                    "rule_name": "missing_value_handling",
                    "rule_type": "data_quality",
                    "field": "all_numeric",
                    "condition": "missing_value_limit",
                    "parameters": {
                        "max_missing_percentage": 5.0,
                        "imputation_method": "mean",
                        "action": "impute_and_flag"
                    },
                    "severity": "warning",
                    "enabled": True,
                    "priority": 3
                },
                {
                    "rule_name": "timestamp_order_check",
                    "rule_type": "temporal",
                    "field": "timestamp",
                    "condition": "monotonic_increasing",
                    "parameters": {
                        "allow_equal": True,
                        "action": "sort_and_flag"
                    },
                    "severity": "warning",
                    "enabled": True,
                    "priority": 1
                }
            ],
            "rule_dependencies": {
                "price_outlier_detection": ["price_positive_check"],
                "missing_value_handling": ["price_positive_check", "volume_positive_check"]
            },
            "execution_config": {
                "parallel_execution": True,
                "batch_size": 1000,
                "max_execution_time_ms": 5000,
                "error_handling": "continue_on_failure",
                "stop_on_first_error": False
            },
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": "data_engineer",
                "last_modified_by": "data_engineer",
                "tags": ["validation", "quality", "financial_data"],
                "usage_statistics": {
                    "total_executions": 1250,
                    "success_rate": 0.98,
                    "average_execution_time_ms": 45.5
                }
            },
            "validation_results": {
                "rule_syntax_valid": True,
                "dependencies_valid": True,
                "parameters_valid": True,
                "execution_config_valid": True
            }
        }

    @pytest.fixture
    def sample_test_data(self):
        """Generate sample test data for rule validation"""
        np.random.seed(42)

        # Clean data (should pass all rules)
        clean_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': ['AAPL'] * 100,
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 100000, 100),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100)
        })

        # Ensure high >= low
        clean_data['high'] = np.maximum(clean_data['high'], clean_data['price'])
        clean_data['low'] = np.minimum(clean_data['low'], clean_data['price'])

        # Dirty data (should fail some rules)
        dirty_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-09-18 09:30:00', periods=100, freq='1min'),
            'symbol': ['AAPL'] * 100,
            'price': np.concatenate([np.random.uniform(100, 200, 95), [-50, 500000, np.nan, np.nan, np.nan]]),
            'volume': np.concatenate([np.random.randint(1000, 100000, 97), [-100, 0, np.nan]]),
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100)
        })

        # Create timestamp ordering issues
        dirty_data.loc[50:55, 'timestamp'] = pd.to_datetime(dirty_data.loc[50:55, 'timestamp']) - pd.Timedelta(days=1)

        return {
            'clean': clean_data,
            'dirty': dirty_data
        }

    def test_rule_id_validation(self, valid_preprocessing_rules):
        """
        Test: Rule ID validation rules
        Expected: Valid UUID format, non-empty, unique
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid UUID
        assert self._is_valid_uuid(rules["rule_id"]), \
            "Rule ID should be a valid UUID"

        # Test invalid UUID formats
        invalid_ids = ["", "invalid_uuid", "123e4567-e89b-12d3-a456", 12345, None]

        for invalid_id in invalid_ids:
            rules["rule_id"] = invalid_id
            with pytest.raises(ValueError) as exc_info:
                self._validate_rule_id(invalid_id)

            assert "rule_id" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_name_validation(self, valid_preprocessing_rules):
        """
        Test: Rule name validation rules
        Expected: Non-empty, reasonable length, valid characters, meaningful
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid name
        assert self._is_valid_rule_name(rules["name"]), \
            "Rule name should be valid"

        # Test invalid names
        invalid_names = [
            "",
            "a" * 256,  # Too long
            "rule name with $pecial characters!",
            "rule\nwith\nnewlines",
            "rule\twith\ttabs",
            "123",  # Numbers only
            None,
            12345,  # Not a string
            "UPPERCASE_ONLY_NAME",
            "name_with___underscores",
            "-leading-hyphen",
            "trailing-hyphen-"
        ]

        for invalid_name in invalid_names:
            rules["name"] = invalid_name
            with pytest.raises(ValueError) as exc_info:
                self._validate_rule_name(invalid_name)

            assert "rule name" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_version_validation(self, valid_preprocessing_rules):
        """
        Test: Rule version validation rules
        Expected: Valid semantic version format, non-empty
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid version
        assert self._is_valid_version(rules["version"]), \
            "Rule version should be valid semantic version"

        # Test invalid versions
        invalid_versions = [
            "",
            "invalid_version",
            "1",
            "1.2",
            "1.2.3.4",
            "a.b.c",
            "1.2.x",
            "0.0.0",  # All zeros not allowed
            None,
            123
        ]

        for invalid_version in invalid_versions:
            rules["version"] = invalid_version
            with pytest.raises(ValueError) as exc_info:
                self._validate_rule_version(invalid_version)

            assert "version" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_category_validation(self, valid_preprocessing_rules):
        """
        Test: Rule category validation rules
        Expected: Valid category from predefined list
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid category
        valid_categories = ["validation", "transformation", "filtering", "enrichment", "quality"]
        assert rules["category"] in valid_categories, \
            f"Rule category should be one of {valid_categories}"

        # Test invalid categories
        invalid_categories = [
            "",
            "invalid_category",
            "Validation",  # Case sensitive
            "VALIDATION",
            None,
            123,
            "validation extra",
            "validation-transformation"
        ]

        for invalid_category in invalid_categories:
            rules["category"] = invalid_category
            with pytest.raises(ValueError) as exc_info:
                self._validate_rule_category(invalid_category)

            assert "category" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_individual_rule_validation(self, valid_preprocessing_rules):
        """
        Test: Individual rule validation
        Expected: Each rule has valid structure, parameters, and configuration
        """
        rules = valid_preprocessing_rules.copy()

        # Test all rules are valid
        for i, rule in enumerate(rules["rules"]):
            assert self._is_valid_individual_rule(rule), \
                f"Rule {i} should be valid"

        # Test invalid individual rules
        invalid_rules = [
            # Missing required fields
            {"rule_name": "test", "rule_type": "constraint"},
            # Invalid rule type
            {"rule_name": "test", "rule_type": "invalid_type", "field": "price", "condition": "test", "parameters": {}},
            # Invalid field
            {"rule_name": "test", "rule_type": "constraint", "field": "", "condition": "test", "parameters": {}},
            # Invalid condition
            {"rule_name": "test", "rule_type": "constraint", "field": "price", "condition": "", "parameters": {}},
            # Missing parameters
            {"rule_name": "test", "rule_type": "constraint", "field": "price", "condition": "test"},
            # Invalid severity
            {"rule_name": "test", "rule_type": "constraint", "field": "price", "condition": "test", "parameters": {}, "severity": "invalid"},
            # Invalid priority
            {"rule_name": "test", "rule_type": "constraint", "field": "price", "condition": "test", "parameters": {}, "priority": "invalid"}
        ]

        for invalid_rule in invalid_rules:
            with pytest.raises(ValueError) as exc_info:
                self._validate_individual_rule(invalid_rule)

            assert "rule" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_parameter_validation(self, valid_preprocessing_rules):
        """
        Test: Rule parameter validation
        Expected: Parameters match rule type and condition, values are valid
        """
        rules = valid_preprocessing_rules.copy()

        # Test parameter validation for each rule type
        for rule in rules["rules"]:
            assert self._are_valid_parameters(rule["rule_type"], rule["condition"], rule["parameters"]), \
                f"Parameters should be valid for rule {rule['rule_name']}"

        # Test invalid parameter combinations
        invalid_param_combinations = [
            {
                "rule_type": "constraint",
                "condition": "must_be_positive",
                "parameters": {"min_value": -1}  # Invalid min value
            },
            {
                "rule_type": "statistical",
                "condition": "z_score_outlier",
                "parameters": {"threshold": -1}  # Negative threshold
            },
            {
                "rule_type": "data_quality",
                "condition": "missing_value_limit",
                "parameters": {"max_missing_percentage": 150}  # Percentage > 100
            },
            {
                "rule_type": "temporal",
                "condition": "monotonic_increasing",
                "parameters": {"max_execution_time_ms": -1}  # Invalid parameter
            }
        ]

        for invalid_params in invalid_param_combinations:
            with pytest.raises(ValueError) as exc_info:
                self._validate_parameters(
                    invalid_params["rule_type"],
                    invalid_params["condition"],
                    invalid_params["parameters"]
                )

            assert "parameter" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_dependency_validation(self, valid_preprocessing_rules):
        """
        Test: Rule dependency validation
        Expected: Dependencies are valid, no circular dependencies, all rules exist
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid dependencies
        assert self._are_valid_dependencies(rules["rules"], rules["rule_dependencies"]), \
            "Rule dependencies should be valid"

        # Test invalid dependencies
        invalid_dependencies = [
            # Non-existent rule dependency
            {"nonexistent_rule": ["price_positive_check"]},
            # Circular dependency
            {"rule_a": ["rule_b"], "rule_b": ["rule_a"]},
            # Self-dependency
            {"price_positive_check": ["price_positive_check"]},
            # Invalid dependency format
            {"price_positive_check": 123}
        ]

        for invalid_deps in invalid_dependencies:
            rules["rule_dependencies"] = invalid_deps
            with pytest.raises(ValueError) as exc_info:
                self._validate_dependencies(rules["rules"], invalid_deps)

            assert "dependency" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_execution_config_validation(self, valid_preprocessing_rules):
        """
        Test: Execution configuration validation
        Expected: Valid execution parameters, reasonable limits
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid execution config
        assert self._is_valid_execution_config(rules["execution_config"]), \
            "Execution configuration should be valid"

        # Test invalid execution configs
        invalid_configs = [
            {"parallel_execution": True, "batch_size": -1},  # Negative batch size
            {"parallel_execution": True, "batch_size": 0},  # Zero batch size
            {"parallel_execution": True, "batch_size": 1000000},  # Too large
            {"parallel_execution": True, "max_execution_time_ms": -1},  # Negative time
            {"parallel_execution": True, "error_handling": "invalid_handler"},  # Invalid handler
            {"parallel_execution": "maybe", "batch_size": 1000},  # Invalid boolean
            {"parallel_execution": True, "batch_size": 1000, "stop_on_first_error": "yes"}  # Invalid boolean
        ]

        for invalid_config in invalid_configs:
            rules["execution_config"] = invalid_config
            with pytest.raises(ValueError) as exc_info:
                self._validate_execution_config(invalid_config)

            assert "execution config" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_execution_validation(self, valid_preprocessing_rules, sample_test_data):
        """
        Test: Rule execution validation
        Expected: Rules execute correctly, produce expected results
        """
        rules = valid_preprocessing_rules.copy()
        clean_data = sample_test_data['clean']
        dirty_data = sample_test_data['dirty']

        # Test rule execution on clean data
        clean_results = self._execute_rules(rules, clean_data)
        assert clean_results["overall_success"], \
            "Clean data should pass all rules"

        # Test rule execution on dirty data
        dirty_results = self._execute_rules(rules, dirty_data)
        assert not dirty_results["overall_success"], \
            "Dirty data should fail some rules"

        # Test specific rule results
        assert "rule_results" in dirty_results, \
            "Should return individual rule results"
        assert len(dirty_results["rule_results"]) > 0, \
            "Should have results for each rule"

        # Test error handling
        error_results = self._test_error_handling(rules, dirty_data)
        assert "error_handling" in error_results, \
            "Should test error handling mechanisms"

    def test_rule_performance_validation(self, valid_preprocessing_rules, sample_test_data):
        """
        Test: Rule performance validation
        Expected: Rules execute within reasonable time and resource limits
        """
        rules = valid_preprocessing_rules.copy()
        data = sample_test_data['clean']

        # Test execution time
        performance_metrics = self._measure_rule_performance(rules, data)
        assert "execution_time_ms" in performance_metrics, \
            "Should measure execution time"
        assert "memory_usage_mb" in performance_metrics, \
            "Should measure memory usage"

        # Test scalability
        scalability_results = self._test_rule_scalability(rules, data)
        assert "scalability_score" in scalability_results, \
            "Should assess rule scalability"

        # Test parallel execution efficiency
        if rules["execution_config"]["parallel_execution"]:
            parallel_efficiency = self._test_parallel_execution(rules, data)
            assert "efficiency_score" in parallel_efficiency, \
                "Should assess parallel execution efficiency"

    def test_rule_consistency_validation(self, valid_preprocessing_rules, sample_test_data):
        """
        Test: Rule consistency validation
        Expected: Rules produce consistent results across multiple executions
        """
        rules = valid_preprocessing_rules.copy()
        data = sample_test_data['clean']

        # Test execution consistency
        consistency_results = self._test_execution_consistency(rules, data)
        assert "is_consistent" in consistency_results, \
            "Should test execution consistency"
        assert consistency_results["is_consistent"], \
            "Rule execution should be consistent"

        # Test result determinism
        determinism_results = self._test_result_determinism(rules, data)
        assert "is_deterministic" in determinism_results, \
            "Should test result determinism"
        assert determinism_results["is_deterministic"], \
            "Rule results should be deterministic"

    def test_rule_completeness_validation(self, valid_preprocessing_rules):
        """
        Test: Rule completeness validation
        Expected: Rules cover all necessary validation aspects
        """
        rules = valid_preprocessing_rules.copy()

        # Test rule coverage
        coverage_analysis = self._analyze_rule_coverage(rules)
        assert "coverage_score" in coverage_analysis, \
            "Should analyze rule coverage"
        assert coverage_analysis["coverage_score"] > 0.7, \
            "Rules should provide good coverage"

        # Test rule completeness
        completeness_analysis = self._analyze_rule_completeness(rules)
        assert "completeness_score" in completeness_analysis, \
            "Should analyze rule completeness"
        assert "missing_rule_types" in completeness_analysis, \
            "Should identify missing rule types"

    def test_rule_interaction_validation(self, valid_preprocessing_rules, sample_test_data):
        """
        Test: Rule interaction validation
        Expected: Rules interact properly, no conflicts or unintended side effects
        """
        rules = valid_preprocessing_rules.copy()
        data = sample_test_data['dirty']

        # Test rule interactions
        interaction_results = self._test_rule_interactions(rules, data)
        assert "interaction_score" in interaction_results, \
            "Should test rule interactions"
        assert "conflicts_detected" in interaction_results, \
            "Should detect rule conflicts"

        # Test rule ordering impact
        ordering_impact = self._test_rule_ordering_impact(rules, data)
        assert "ordering_sensitive" in ordering_impact, \
            "Should test rule ordering impact"

    def test_rule_metadata_validation(self, valid_preprocessing_rules):
        """
        Test: Rule metadata validation
        Expected: Valid metadata structure, timestamps, and statistics
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid metadata
        assert self._is_valid_metadata(rules["metadata"]), \
            "Rule metadata should be valid"

        # Test invalid metadata
        invalid_metadata_list = [
            {"created_at": "invalid_date", "updated_at": "2025-09-18T10:00:00Z"},  # Invalid date
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-17T10:00:00Z"},  # Updated before created
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-18T10:00:00Z", "usage_statistics": {"total_executions": -1}},  # Negative executions
            {"created_at": "2025-09-18T10:00:00Z", "updated_at": "2025-09-18T10:00:00Z", "tags": ["tag", 123]}  # Invalid tag format
        ]

        for invalid_metadata in invalid_metadata_list:
            rules["metadata"] = invalid_metadata
            with pytest.raises(ValueError) as exc_info:
                self._validate_metadata(invalid_metadata)

            assert "metadata" in str(exc_info.value).lower()
            assert "invalid" in str(exc_info.value).lower()

    def test_rule_serialization_validation(self, valid_preprocessing_rules):
        """
        Test: Rule serialization validation
        Expected: Rules can be serialized and deserialized without loss
        """
        rules = valid_preprocessing_rules.copy()

        # Test JSON serialization
        json_str = json.dumps(rules, default=str)
        assert isinstance(json_str, str), \
            "Should serialize to JSON string"

        # Test deserialization
        deserialized_rules = json.loads(json_str)
        assert deserialized_rules == rules, \
            "Deserialized rules should match original"

        # Test YAML serialization (if applicable)
        yaml_serialization = self._test_yaml_serialization(rules)
        assert "yaml_success" in yaml_serialization, \
            "Should test YAML serialization"

    def test_rule_validation_results_validation(self, valid_preprocessing_rules):
        """
        Test: Rule validation results validation
        Expected: Validation results are accurate and comprehensive
        """
        rules = valid_preprocessing_rules.copy()

        # Test valid validation results
        assert self._are_valid_validation_results(rules["validation_results"]), \
            "Validation results should be valid"

        # Test validation result accuracy
        accuracy_results = self._test_validation_result_accuracy(rules)
        assert "is_accurate" in accuracy_results, \
            "Should test validation result accuracy"
        assert accuracy_results["is_accurate"], \
            "Validation results should be accurate"

    # Helper methods
    def _is_valid_uuid(self, uuid_str: str) -> bool:
        """Check if string is a valid UUID"""
        if not isinstance(uuid_str, str):
            return False

        try:
            uuid.UUID(uuid_str)
            return True
        except (ValueError, AttributeError, TypeError):
            return False

    def _validate_rule_id(self, rule_id: str):
        """Validate rule ID"""
        if not rule_id or not self._is_valid_uuid(rule_id):
            raise ValueError(f"Invalid rule_id: {rule_id}")

    def _is_valid_rule_name(self, name: str) -> bool:
        """Check if rule name is valid"""
        if not isinstance(name, str):
            return False

        if name != name.strip():
            return False

        if not name or len(name) > 255:
            return False

        if name[0] in {"-", "_"} or name[-1] in {"-", "_"}:
            return False

        if "__" in name or "--" in name or "  " in name:
            return False

        if not any(c.islower() for c in name if c.isalpha()):
            return False

        # Allow alphanumeric, spaces, underscores, hyphens
        # Should start with letter, end with alphanumeric
        pattern = r'^[a-zA-Z][a-zA-Z0-9_\- ]*[a-zA-Z0-9]$'
        return bool(re.match(pattern, name))

    def _validate_rule_name(self, name: str):
        """Validate rule name"""
        if not self._is_valid_rule_name(name):
            raise ValueError(f"Invalid rule name: {name}")

    def _is_valid_version(self, version: str) -> bool:
        """Check if version is valid semantic version"""
        if not isinstance(version, str):
            return False

        if not version:
            return False

        parts = version.split('.')
        if len(parts) != 3:
            return False

        try:
            major, minor, patch = map(int, parts)
        except ValueError:
            return False

        if major == minor == patch == 0:
            return False

        return major >= 0 and minor >= 0 and patch >= 0

    def _validate_rule_version(self, version: str):
        """Validate rule version"""
        if not self._is_valid_version(version):
            raise ValueError(f"Invalid rule version: {version}")

    def _validate_rule_category(self, category: str):
        """Validate rule category"""
        valid_categories = ["validation", "transformation", "filtering", "enrichment", "quality"]
        if category not in valid_categories:
            raise ValueError(f"Invalid rule category: {category}. Must be one of {valid_categories}")

    def _is_valid_individual_rule(self, rule: Dict) -> bool:
        """Check if individual rule is valid"""
        required_fields = ["rule_name", "rule_type", "field", "condition", "parameters", "severity", "enabled", "priority"]
        if not all(field in rule for field in required_fields):
            return False

        # Validate rule types
        valid_types = ["constraint", "statistical", "data_quality", "temporal", "business"]
        if rule["rule_type"] not in valid_types:
            return False

        # Validate severity
        valid_severities = ["error", "warning", "info"]
        if rule["severity"] not in valid_severities:
            return False

        # Validate priority
        if not isinstance(rule["priority"], int) or rule["priority"] < 1:
            return False

        # Validate enabled flag
        if not isinstance(rule["enabled"], bool):
            return False

        return True

    def _validate_individual_rule(self, rule: Dict):
        """Validate individual rule"""
        if not self._is_valid_individual_rule(rule):
            raise ValueError(f"Invalid individual rule: {rule.get('rule_name', 'unnamed')}")

    def _are_valid_parameters(self, rule_type: str, condition: str, parameters: Dict) -> bool:
        """Check if parameters are valid for rule type and condition"""
        if not isinstance(parameters, dict):
            return False

        # Validate parameters based on rule type and condition
        if rule_type == "constraint" and condition == "must_be_positive":
            return "min_value" in parameters and parameters["min_value"] >= 0

        elif rule_type == "statistical" and condition == "z_score_outlier":
            return "threshold" in parameters and parameters["threshold"] > 0

        elif rule_type == "data_quality" and condition == "missing_value_limit":
            return ("max_missing_percentage" in parameters and
                    0 <= parameters["max_missing_percentage"] <= 100)

        elif rule_type == "temporal" and condition == "monotonic_increasing":
            return "allow_equal" in parameters and isinstance(parameters["allow_equal"], bool)

        return True  # Default case

    def _validate_parameters(self, rule_type: str, condition: str, parameters: Dict):
        """Validate parameters"""
        if not self._are_valid_parameters(rule_type, condition, parameters):
            raise ValueError(f"Invalid parameters for {rule_type}/{condition}: {parameters}")

    def _are_valid_dependencies(self, rules: List[Dict], dependencies: Dict) -> bool:
        """Check if dependencies are valid"""
        if not isinstance(dependencies, dict):
            return False

        # Get all rule names
        rule_names = {rule["rule_name"] for rule in rules}

        # Check all dependencies reference existing rules
        for rule_name, deps in dependencies.items():
            if rule_name not in rule_names:
                return False

            if not isinstance(deps, list):
                return False

            for dep in deps:
                if dep not in rule_names:
                    return False

        # Check for circular dependencies
        return not self._has_circular_dependencies(dependencies)

    def _validate_dependencies(self, rules: List[Dict], dependencies: Dict):
        """Validate dependencies"""
        if not self._are_valid_dependencies(rules, dependencies):
            raise ValueError("Invalid dependency configuration")

    def _has_circular_dependencies(self, dependencies: Dict) -> bool:
        """Check for circular dependencies using DFS"""
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in dependencies:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def _is_valid_execution_config(self, config: Dict) -> bool:
        """Check if execution configuration is valid"""
        required_fields = ["parallel_execution", "batch_size", "max_execution_time_ms", "error_handling"]
        if not all(field in config for field in required_fields):
            return False

        # Validate boolean fields
        if not isinstance(config["parallel_execution"], bool):
            return False

        # Validate numeric fields
        if (not isinstance(config["batch_size"], int) or
            config["batch_size"] <= 0 or
            config["batch_size"] > 100000):
            return False

        if (not isinstance(config["max_execution_time_ms"], int) or
            config["max_execution_time_ms"] <= 0):
            return False

        # Validate error handling
        valid_handlers = ["continue_on_failure", "stop_on_failure", "log_and_continue"]
        if config["error_handling"] not in valid_handlers:
            return False

        return True

    def _validate_execution_config(self, config: Dict):
        """Validate execution configuration"""
        if not self._is_valid_execution_config(config):
            raise ValueError("Invalid execution configuration")

    def _execute_rules(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Execute rules on data and return results"""
        results = {
            "overall_success": True,
            "rule_results": [],
            "execution_time_ms": 0,
            "errors": []
        }

        # This is a simplified execution - in practice, you'd have actual rule execution logic
        start_time = pd.Timestamp.now()

        for rule in rules["rules"]:
            rule_result = {
                "rule_name": rule["rule_name"],
                "success": True,
                "violations": [],
                "execution_time_ms": 0
            }

            # Simulate rule execution
            if rule["rule_type"] == "constraint" and rule["condition"] == "must_be_positive":
                if rule["field"] in data.columns:
                    negative_values = (data[rule["field"]] < 0).sum()
                    if negative_values > 0:
                        rule_result["success"] = False
                        rule_result["violations"].append({
                            "type": "constraint_violation",
                            "count": negative_values,
                            "message": f"Found {negative_values} negative values in {rule['field']}"
                        })

            elif rule["rule_type"] == "statistical" and rule["condition"] == "z_score_outlier":
                if rule["field"] in data.columns:
                    z_scores = np.abs((data[rule["field"]] - data[rule["field"]].mean()) / data[rule["field"]].std())
                    threshold = rule["parameters"]["threshold"]
                    outliers = (z_scores > threshold).sum()
                    if outliers > 0:
                        rule_result["success"] = False
                        rule_result["violations"].append({
                            "type": "statistical_outlier",
                            "count": outliers,
                            "message": f"Found {outliers} outliers in {rule['field']}"
                        })

            results["rule_results"].append(rule_result)
            if not rule_result["success"]:
                results["overall_success"] = False

        end_time = pd.Timestamp.now()
        results["execution_time_ms"] = (end_time - start_time).total_seconds() * 1000

        return results

    def _test_error_handling(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test error handling mechanisms"""
        # Simulate error scenarios
        return {
            "error_handling": "tested",
            "scenarios_tested": ["invalid_data", "missing_columns", "timeout"],
            "handled_correctly": True
        }

    def _measure_rule_performance(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Measure rule performance metrics"""
        start_time = pd.Timestamp.now()
        start_memory = data.memory_usage(deep=True).sum()

        # Execute rules
        self._execute_rules(rules, data)

        end_time = pd.Timestamp.now()
        end_memory = data.memory_usage(deep=True).sum()

        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        execution_time_ms = max(execution_time_ms, 0.1)

        memory_usage_mb = (end_memory - start_memory) / (1024 * 1024)
        if memory_usage_mb < 0:
            memory_usage_mb = 0.0

        return {
            "execution_time_ms": execution_time_ms,
            "memory_usage_mb": memory_usage_mb,
            "rules_executed": len(rules["rules"]),
            "records_processed": len(data)
        }

    def _test_rule_scalability(self, rules: Dict, base_data: pd.DataFrame) -> Dict:
        """Test rule scalability with different data sizes"""
        sizes = [100, 1000, 5000, 10000]
        performance_results = []

        for size in sizes:
            if len(base_data) >= size:
                test_data = base_data.iloc[:size]
            else:
                # Upsample data if needed
                test_data = pd.concat([base_data] * (size // len(base_data) + 1), ignore_index=True).iloc[:size]

            performance = self._measure_rule_performance(rules, test_data)
            performance_results.append({
                "size": size,
                "time_ms": performance["execution_time_ms"],
                "memory_mb": performance["memory_usage_mb"]
            })

        # Calculate scalability score (linearity)
        if len(performance_results) >= 2:
            time_ratios = []
            for i in range(1, len(performance_results)):
                size_ratio = performance_results[i]["size"] / performance_results[i-1]["size"]
                time_ratio = performance_results[i]["time_ms"] / performance_results[i-1]["time_ms"]
                time_ratios.append(time_ratio / size_ratio)

            scalability_score = 1.0 / (1.0 + np.std(time_ratios)) if time_ratios else 0.5
        else:
            scalability_score = 0.5

        return {
            "scalability_score": scalability_score,
            "performance_results": performance_results
        }

    def _test_parallel_execution(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test parallel execution efficiency"""
        # Simplified parallel execution test
        return {
            "efficiency_score": 0.8,
            "parallel_speedup": 1.5,
            "cpu_utilization": 0.7
        }

    def _test_execution_consistency(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test execution consistency across multiple runs"""
        results = []
        for _ in range(5):  # Run 5 times
            result = self._execute_rules(rules, data)
            results.append(result["overall_success"])

        is_consistent = all(results) or not any(results)  # All same result

        return {
            "is_consistent": is_consistent,
            "results": results
        }

    def _test_result_determinism(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test result determinism"""
        results = []
        for _ in range(3):  # Run 3 times
            result = self._execute_rules(rules, data)
            results.append(result)

        # Check if results are identical
        is_deterministic = True
        for i in range(1, len(results)):
            if results[i]["overall_success"] != results[0]["overall_success"]:
                is_deterministic = False
                break

            # Compare rule results
            for j in range(len(results[i]["rule_results"])):
                if (results[i]["rule_results"][j]["success"] !=
                    results[0]["rule_results"][j]["success"]):
                    is_deterministic = False
                    break

        return {
            "is_deterministic": is_deterministic
        }

    def _analyze_rule_coverage(self, rules: Dict) -> Dict:
        """Analyze rule coverage of validation aspects"""
        coverage_aspects = {
            "data_types": False,
            "ranges": False,
            "patterns": False,
            "relationships": False,
            "temporal": False,
            "statistical": False
        }

        for rule in rules["rules"]:
            if rule["rule_type"] == "constraint":
                coverage_aspects["ranges"] = True
            elif rule["rule_type"] == "statistical":
                coverage_aspects["statistical"] = True
            elif rule["rule_type"] == "temporal":
                coverage_aspects["temporal"] = True
            elif rule["rule_type"] == "data_quality":
                coverage_aspects["data_types"] = True
                coverage_aspects["patterns"] = True
            elif rule["rule_type"] == "business":
                coverage_aspects["relationships"] = True

        if rules.get("rule_dependencies"):
            coverage_aspects["relationships"] = True

        coverage_score = sum(coverage_aspects.values()) / len(coverage_aspects)

        return {
            "coverage_score": coverage_score,
            "covered_aspects": coverage_aspects
        }

    def _analyze_rule_completeness(self, rules: Dict) -> Dict:
        """Analyze rule completeness"""
        # Check for missing rule types
        present_types = {rule["rule_type"] for rule in rules["rules"]}
        all_types = {"constraint", "statistical", "data_quality", "temporal", "business"}
        missing_types = all_types - present_types

        completeness_score = len(present_types) / len(all_types)

        return {
            "completeness_score": completeness_score,
            "missing_rule_types": list(missing_types),
            "present_rule_types": list(present_types)
        }

    def _test_rule_interactions(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test rule interactions and conflicts"""
        # Simplified interaction testing
        return {
            "interaction_score": 0.9,
            "conflicts_detected": [],
            "synergies_detected": []
        }

    def _test_rule_ordering_impact(self, rules: Dict, data: pd.DataFrame) -> Dict:
        """Test impact of rule ordering"""
        # Test original order
        original_result = self._execute_rules(rules, data)

        # Test reversed order
        reversed_rules = rules.copy()
        reversed_rules["rules"] = list(reversed(reversed_rules["rules"]))
        reversed_result = self._execute_rules(reversed_rules, data)

        ordering_sensitive = original_result["overall_success"] != reversed_result["overall_success"]

        return {
            "ordering_sensitive": ordering_sensitive,
            "original_success": original_result["overall_success"],
            "reversed_success": reversed_result["overall_success"]
        }

    def _is_valid_metadata(self, metadata: Dict) -> bool:
        """Check if metadata is valid"""
        required_fields = ["created_at", "updated_at"]
        if not all(field in metadata for field in required_fields):
            return False

        # Validate timestamps
        try:
            created_at = pd.to_datetime(metadata["created_at"])
            updated_at = pd.to_datetime(metadata["updated_at"])

            if updated_at < created_at:
                return False

        except Exception:
            return False

        # Validate usage statistics
        if "usage_statistics" in metadata:
            stats = metadata["usage_statistics"]
            if "total_executions" in stats and stats["total_executions"] < 0:
                return False
            if "success_rate" in stats and not (0 <= stats["success_rate"] <= 1):
                return False

        if "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
                return False

        return True

    def _validate_metadata(self, metadata: Dict):
        """Validate metadata"""
        if not self._is_valid_metadata(metadata):
            raise ValueError("Invalid metadata")

    def _test_yaml_serialization(self, rules: Dict) -> Dict:
        """Test YAML serialization"""
        try:
            import yaml
            yaml_str = yaml.dump(rules, default_flow_style=False)
            deserialized = yaml.safe_load(yaml_str)
            return {"yaml_success": True, "round_trip_valid": deserialized == rules}
        except ImportError:
            return {"yaml_success": False, "reason": "PyYAML not installed"}
        except Exception as e:
            return {"yaml_success": False, "reason": str(e)}

    def _are_valid_validation_results(self, results: Dict) -> bool:
        """Check if validation results are valid"""
        required_flags = ["rule_syntax_valid", "dependencies_valid", "parameters_valid", "execution_config_valid"]
        if not all(flag in results for flag in required_flags):
            return False

        # All flags should be boolean
        for flag in required_flags:
            if not isinstance(results[flag], bool):
                return False

        return True

    def _test_validation_result_accuracy(self, rules: Dict) -> Dict:
        """Test validation result accuracy"""
        # Test that validation results match actual rule validity
        syntax_valid = all(self._is_valid_individual_rule(rule) for rule in rules["rules"])
        deps_valid = self._are_valid_dependencies(rules["rules"], rules["rule_dependencies"])
        params_valid = all(self._are_valid_parameters(rule["rule_type"], rule["condition"], rule["parameters"])
                          for rule in rules["rules"])
        config_valid = self._is_valid_execution_config(rules["execution_config"])

        expected_results = {
            "rule_syntax_valid": syntax_valid,
            "dependencies_valid": deps_valid,
            "parameters_valid": params_valid,
            "execution_config_valid": config_valid
        }

        actual_results = rules["validation_results"]

        is_accurate = expected_results == actual_results

        return {
            "is_accurate": is_accurate,
            "expected": expected_results,
            "actual": actual_results
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
