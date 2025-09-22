"""
API contract tests for /preprocessing/rules endpoints
Tests preprocessing rules API contract compliance
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from data.src.api.preprocessing_api import (
    validate_rule_definition,
    create_rule,
    list_rules,
    get_rule as fetch_rule,
    update_rule as apply_rule_update,
    delete_rule as remove_rule,
    test_rule as execute_rule_test,
    perform_bulk_rule_operation,
    export_rules as export_rules_data,
    import_rules as import_rules_data,
    validate_type_specific_rule,
)


class TestRulesAPIContract:
    """Test suite for /preprocessing/rules endpoints contract compliance"""

    def test_rules_endpoint_create_rule_validation(self):
        """
        Test: Create rule endpoint validates rule structure
        Expected: 400 Bad Request for invalid rule definitions
        """
        invalid_rules = [
            {"name": "", "type": "validation", "conditions": []},  # Empty name
            {"type": "unknown_type", "conditions": [{"field": "price", "operator": ">", "value": 0}]},  # Unknown type
            {"name": "test", "type": "validation"},  # Missing conditions
            {"name": "test", "type": "validation", "conditions": [{"field": "", "operator": ">", "value": 0}]},  # Empty field
            {"name": "test", "type": "validation", "conditions": [{"field": "price", "operator": "invalid", "value": 0}]},  # Invalid operator
        ]

        for invalid_rule in invalid_rules:
            with pytest.raises(ValueError) as exc_info:
                validate_rule_definition(invalid_rule)

            error_message = str(exc_info.value).lower()
            assert "validation" in error_message
            assert "invalid" in error_message

    def test_rules_endpoint_create_rule_success(self):
        """
        Test: Create rule endpoint successfully creates new rules
        Expected: 201 Created with rule ID and confirmation
        """
        new_rule = {
            "name": "price_validation",
            "type": "validation",
            "description": "Validate price ranges for financial instruments",
            "conditions": [
                {"field": "price", "operator": ">", "value": 0},
                {"field": "price", "operator": "<", "value": 10000}
            ],
            "severity": "error",
            "action": "reject",
            "enabled": True
        }

        expected_response = {
            "rule_id": "rule_12345",
            "name": "price_validation",
            "type": "validation",
            "status": "created",
            "version": 1,
            "created_at": "2025-09-18T10:00:00Z",
            "created_by": "system",
            "conditions_count": 2,
            "validation_passed": True
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.create_rule.return_value = expected_response

            response = create_rule(new_rule)

            assert response == expected_response
            assert response["status"] == "created"
            assert response["version"] == 1
            assert response["conditions_count"] == 2

    def test_rules_endpoint_get_rules_list(self):
        """
        Test: Get rules endpoint returns paginated list of rules
        Expected: 200 OK with rules list and pagination metadata
        """
        expected_response = {
            "rules": [
                {
                    "rule_id": "rule_001",
                    "name": "price_validation",
                    "type": "validation",
                    "description": "Validate price ranges",
                    "enabled": True,
                    "version": 1,
                    "last_modified": "2025-09-18T09:00:00Z",
                    "usage_count": 150
                },
                {
                    "rule_id": "rule_002",
                    "name": "volume_cleaning",
                    "type": "cleaning",
                    "description": "Clean volume data",
                    "enabled": True,
                    "version": 2,
                    "last_modified": "2025-09-17T14:00:00Z",
                    "usage_count": 89
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_rules": 25,
                "total_pages": 2,
                "has_next": True,
                "has_prev": False
            },
            "filters": {
                "type": "all",
                "enabled": True,
                "search_query": None
            }
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.get_rules.return_value = expected_response

            response = list_rules()

            assert response == expected_response
            rules = response["rules"]
            assert len(rules) > 0
            for rule in rules:
                assert "rule_id" in rule
                assert "name" in rule
                assert "type" in rule
                assert "enabled" in rule

    def test_rules_endpoint_get_rule_by_id(self):
        """
        Test: Get rule by ID endpoint returns specific rule details
        Expected: 200 OK with complete rule definition
        """
        rule_id = "rule_12345"
        expected_response = {
            "rule_id": rule_id,
            "name": "price_validation",
            "type": "validation",
            "description": "Validate price ranges for financial instruments",
            "version": 2,
            "enabled": True,
            "conditions": [
                {
                    "field": "price",
                    "operator": ">",
                    "value": 0,
                    "data_type": "numeric"
                },
                {
                    "field": "price",
                    "operator": "<",
                    "value": 10000,
                    "data_type": "numeric"
                }
            ],
            "actions": [
                {
                    "type": "reject",
                    "message": "Price out of valid range",
                    "severity": "error"
                }
            ],
            "metadata": {
                "created_at": "2025-09-18T10:00:00Z",
                "created_by": "system",
                "last_modified": "2025-09-18T11:00:00Z",
                "modified_by": "admin",
                "usage_count": 156,
                "success_rate": 0.98
            },
            "performance_metrics": {
                "average_execution_time_ms": 2.5,
                "memory_usage_bytes": 1024,
                "last_executed": "2025-09-18T11:30:00Z"
            }
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.get_rule.return_value = expected_response

            response = fetch_rule(rule_id)

            assert response == expected_response
            conditions = response["conditions"]
            for condition in conditions:
                assert "field" in condition
                assert "operator" in condition
                assert "value" in condition

    def test_rules_endpoint_update_rule(self):
        """
        Test: Update rule endpoint modifies existing rules
        Expected: 200 OK with updated rule information
        """
        rule_id = "rule_12345"
        update_data = {
            "name": "enhanced_price_validation",
            "description": "Enhanced price validation with additional constraints",
            "enabled": True,
            "conditions": [
                {"field": "price", "operator": ">", "value": 0},
                {"field": "price", "operator": "<", "value": 10000},
                {"field": "volume", "operator": ">=", "value": 0}
            ]
        }

        expected_response = {
            "rule_id": rule_id,
            "name": "enhanced_price_validation",
            "type": "validation",
            "version": 3,  # Version incremented
            "status": "updated",
            "updated_at": "2025-09-18T12:00:00Z",
            "changes": [
                {"field": "name", "old_value": "price_validation", "new_value": "enhanced_price_validation"},
                {"field": "conditions", "old_count": 2, "new_count": 3}
            ],
            "validation_result": {
                "syntax_valid": True,
                "semantic_valid": True,
                "performance_impact": "low"
            }
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.update_rule.return_value = expected_response

            response = apply_rule_update(rule_id, update_data)

            assert response == expected_response
            assert response["version"] == 3
            assert response["status"] == "updated"

    def test_rules_endpoint_delete_rule(self):
        """
        Test: Delete rule endpoint removes rules permanently
        Expected: 200 OK with deletion confirmation
        """
        rule_id = "rule_12345"
        expected_response = {
            "rule_id": rule_id,
            "status": "deleted",
            "deleted_at": "2025-09-18T12:00:00Z",
            "deleted_by": "admin",
            "backup_created": True,
            "backup_location": "/backups/rules/rule_12345_2025-09-18.json",
            "affected_datasets": ["dataset_1", "dataset_2"],
            "cascade_effects": "none"
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.delete_rule.return_value = expected_response

            response = remove_rule(rule_id)

            assert response == expected_response
            assert response["status"] == "deleted"
            assert response["backup_created"] is True

    def test_rules_endpoint_test_rule(self):
        """
        Test: Test rule endpoint validates rule against sample data
        Expected: 200 OK with test results and validation report
        """
        rule_id = "rule_12345"
        test_data = {
            "dataset_id": "test_dataset",
            "sample_size": 100,
            "sample_data": [
                {"price": 150.25, "volume": 1000},
                {"price": -50.0, "volume": 500},  # Should fail validation
                {"price": 50000.0, "volume": 2000}  # Should fail validation
            ]
        }

        expected_response = {
            "rule_id": rule_id,
            "test_status": "completed",
            "test_results": {
                "total_records": 3,
                "passed_records": 1,
                "failed_records": 2,
                "pass_rate": 0.333,
                "failures": [
                    {
                        "record_index": 1,
                        "field": "price",
                        "value": -50.0,
                        "violation": "Price must be greater than 0",
                        "severity": "error"
                    },
                    {
                        "record_index": 2,
                        "field": "price",
                        "value": 50000.0,
                        "violation": "Price must be less than 10000",
                        "severity": "error"
                    }
                ]
            },
            "performance_metrics": {
                "execution_time_ms": 5.2,
                "memory_usage_bytes": 512,
                "records_per_second": 576.9
            },
            "recommendations": [
                "Consider adjusting price thresholds for market conditions",
                "Add data cleaning for negative prices"
            ]
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.test_rule.return_value = expected_response

            response = execute_rule_test(rule_id, test_data["dataset_id"], test_data["sample_data"])

            assert response == expected_response
            test_results = response["test_results"]
            assert "total_records" in test_results
            assert "passed_records" in test_results
            assert "failed_records" in test_results
            assert "pass_rate" in test_results

    def test_rules_endpoint_bulk_operations(self):
        """
        Test: Bulk operations endpoint handles multiple rules
        Expected: 200 OK with bulk operation results
        """
        bulk_request = {
            "operation": "enable",
            "rule_ids": ["rule_001", "rule_002", "rule_003"],
            "dry_run": False
        }

        expected_response = {
            "operation_id": "bulk_12345",
            "operation": "enable",
            "total_rules": 3,
            "processed_rules": 3,
            "successful_rules": 3,
            "failed_rules": 0,
            "results": [
                {"rule_id": "rule_001", "status": "success", "new_state": "enabled"},
                {"rule_id": "rule_002", "status": "success", "new_state": "enabled"},
                {"rule_id": "rule_003", "status": "success", "new_state": "enabled"}
            ],
            "execution_time_ms": 25.5,
            "errors": []
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.bulk_operation.return_value = expected_response

            response = perform_bulk_rule_operation(bulk_request)

            assert response == expected_response
            assert response["total_rules"] == 3
            assert response["processed_rules"] == 3
            assert response["successful_rules"] == 3
            assert response["failed_rules"] == 0

    def test_rules_endpoint_export_import(self):
        """
        Test: Export/Import endpoints handle rule management
        Expected: 200 OK with export/import confirmation
        """
        # Test export
        export_response = {
            "export_id": "export_12345",
            "format": "json",
            "rules_exported": 25,
            "export_url": "/exports/rules_2025-09-18.json",
            "expires_at": "2025-09-19T12:00:00Z",
            "checksum": "abc123def456"
        }

        # Test import
        import_request = {
            "import_url": "/imports/rules_backup.json",
            "conflict_resolution": "overwrite",
            "validate_only": False
        }

        import_response = {
            "import_id": "import_12345",
            "rules_imported": 25,
            "rules_updated": 5,
            "rules_created": 20,
            "validation_errors": [],
            "conflicts_resolved": 3,
            "import_status": "completed"
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            mock_service.export_rules.return_value = export_response
            mock_service.import_rules.return_value = import_response

            export_result = export_rules_data()
            assert export_result == export_response

            import_result = import_rules_data(import_request)
            assert import_result == import_response

    @pytest.mark.parametrize("rule_type", ["validation", "cleaning", "transformation", "enrichment"])
    def test_rules_endpoint_type_specific_behavior(self, rule_type):
        """
        Test: Rules endpoint handles different rule types appropriately
        Expected: Type-specific validation and behavior
        """
        rule_config = {
            "name": f"{rule_type}_rule",
            "type": rule_type,
            "conditions": [{"field": "price", "operator": ">", "value": 0}]
        }

        with patch('data.src.api.preprocessing_api.rules_service') as mock_service:
            validation_response = {
                "rule_type": rule_type,
                "validation_passed": True,
                "type_specific_requirements": {
                    "validation": ["conditions", "actions"],
                    "cleaning": ["conditions", "transformation_rules"],
                    "transformation": ["input_fields", "output_fields"],
                    "enrichment": ["source_fields", "enrichment_logic"]
                },
                "recommended_config": {}
            }

            mock_service.validate_type_specific.return_value = validation_response

            validation = validate_type_specific_rule(rule_type, rule_config)
            assert validation == validation_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
