"""
API contract tests for /preprocessing/process endpoint
Tests preprocessing pipeline API contract compliance
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from typing import Dict, Any
import pandas as pd
import numpy as np

from data.src.api.preprocessing_api import (
    validate_process_request_payload,
    parse_process_data_payload,
    validate_preprocessing_rules,
    execute_processing_request,
    start_async_processing_request,
    enforce_processing_rate_limit,
    ensure_request_authenticated,
    reset_processing_rate_limits,
    estimate_processing_duration,
)


class TestProcessAPIContract:
    """Test suite for /preprocessing/process endpoint contract compliance"""

    def test_process_endpoint_missing_required_fields(self):
        """
        Test: Process endpoint rejects requests missing required fields
        Expected: 400 Bad Request with validation error
        """
        # Missing dataset_id
        request_data = {
            "preprocessing_config": {
                "cleaning": {"missing_value_strategy": "mean"},
                "validation": {"min_value": 0, "max_value": 100}
            }
        }

        with pytest.raises(ValueError) as exc_info:
            validate_process_request_payload(request_data)

        error_message = str(exc_info.value).lower()
        assert "dataset_id" in error_message
        assert "required" in error_message

        # Dataset id present but data missing
        incomplete_payload = {
            "dataset_id": "test_dataset",
            "preprocessing_config": {
                "cleaning": {"missing_value_strategy": "mean"}
            }
        }

        with pytest.raises(ValueError) as exc_info:
            validate_process_request_payload(incomplete_payload)

        error_message = str(exc_info.value).lower()
        assert "dataset_id" in error_message
        assert "required" in error_message

    def test_process_endpoint_invalid_data_format(self):
        """
        Test: Process endpoint rejects invalid data formats
        Expected: 400 Bad Request with format validation error
        """
        invalid_formats = [
            {"data": "not_json", "dataset_id": "test", "preprocessing_config": {}},
            {"data": "[]", "dataset_id": "test", "preprocessing_config": {}},  # Empty array
            {"data": '{"invalid": "structure"}', "dataset_id": "test", "preprocessing_config": {}}
        ]

        for invalid_data in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                parse_process_data_payload(invalid_data["data"])

            message = str(exc_info.value).lower()
            assert "format" in message
            assert "invalid" in message

    def test_process_endpoint_success_response_structure(self):
        """
        Test: Process endpoint returns correct success response structure
        Expected: 200 OK with processing_id, status, and metrics
        """
        # Mock successful processing response
        expected_response = {
            "processing_id": "proc_12345",
            "status": "completed",
            "dataset_id": "test_dataset",
            "metrics": {
                "rows_processed": 1000,
                "columns_processed": 10,
                "missing_values_handled": 50,
                "outliers_detected": 15,
                "processing_time_ms": 250,
                "quality_score": 0.92
            },
            "warnings": ["Some values were imputed using mean"],
            "timestamp": "2025-09-18T10:00:00Z"
        }

        with patch('data.src.api.preprocessing_api.preprocessing_orchestrator') as mock_orchestrator:
            mock_orchestrator.process_data.return_value = expected_response

            response = execute_processing_request(
                dataset_id="test_dataset",
                raw_data=json.dumps({"records": [{"price": 100.0, "volume": 1000}]}),
                config={"cleaning": {"missing_value_strategy": "mean"}},
            )

            assert response == expected_response

            metrics = response["metrics"]
            assert "rows_processed" in metrics
            assert "columns_processed" in metrics
            assert "processing_time_ms" in metrics
            assert "quality_score" in metrics
            assert isinstance(metrics["quality_score"], float)
            assert 0 <= metrics["quality_score"] <= 1

    def test_process_endpoint_error_handling(self):
        """
        Test: Process endpoint handles processing errors gracefully
        Expected: 500 Internal Server Error with error details
        """
        with patch('data.src.api.preprocessing_api.preprocessing_orchestrator') as mock_orchestrator:
            mock_orchestrator.process_data.side_effect = Exception("Data processing failed")

            with pytest.raises(RuntimeError) as exc_info:
                execute_processing_request(
                    dataset_id="test_dataset",
                    raw_data=json.dumps({"records": [{"price": 100.0}]}),
                    config={"cleaning": {"missing_value_strategy": "mean"}},
                )

            error_message = str(exc_info.value).lower()
            assert "failed" in error_message

    def test_process_endpoint_concurrent_processing(self):
        """
        Test: Process endpoint handles concurrent processing requests
        Expected: 202 Accepted with processing_id for async handling
        """
        # Mock async processing response
        async_response = {
            "processing_id": "proc_async_123",
            "status": "processing",
            "dataset_id": "test_dataset",
            "message": "Processing started in background",
            "estimated_completion_ms": 15000,
            "timestamp": "2025-09-18T10:00:00Z"
        }

        with patch('data.src.api.preprocessing_api.preprocessing_orchestrator') as mock_orchestrator:
            mock_orchestrator.start_async_processing.return_value = async_response

            response = start_async_processing_request(
                dataset_id="test_dataset",
                raw_data=json.dumps({"values": [1, 2, 3]}),
                config={"cleaning": {"missing_value_strategy": "mean"}},
            )

            assert response["processing_id"] == async_response["processing_id"]
            assert response["status"] == "processing"
            assert isinstance(response["estimated_completion_ms"], int)

    def test_process_endpoint_validation_rules(self):
        """
        Test: Process endpoint validates preprocessing rules
        Expected: 400 Bad Request for invalid preprocessing configurations
        """
        invalid_configs = [
            {"cleaning": {"missing_value_strategy": "invalid_strategy"}},
            {"validation": {"min_value": 100, "max_value": 50}},  # min > max
            {"normalization": {"method": "unknown_method"}}
        ]

        for invalid_config in invalid_configs:
            with pytest.raises(ValueError) as exc_info:
                validate_preprocessing_rules(invalid_config)

            message = str(exc_info.value).lower()
            assert "validation" in message
            assert "invalid" in message

    def test_process_endpoint_rate_limiting(self):
        """
        Test: Process endpoint enforces rate limiting
        Expected: 429 Too Many Requests for excessive calls
        """
        reset_processing_rate_limits("rate_test")

        for _ in range(5):
            enforce_processing_rate_limit("rate_test")

        with pytest.raises(RuntimeError) as exc_info:
            enforce_processing_rate_limit("rate_test")

        error_message = str(exc_info.value).lower()
        assert "rate" in error_message or "limit" in error_message

    def test_process_endpoint_authentication(self):
        """
        Test: Process endpoint requires authentication
        Expected: 401 Unauthorized for unauthenticated requests
        """
        with pytest.raises(PermissionError) as exc_info:
            ensure_request_authenticated(None)

        error_message = str(exc_info.value).lower()
        assert "auth" in error_message or "unauthorized" in error_message

    @pytest.mark.parametrize("data_size", [100, 1000, 10000])
    def test_process_endpoint_data_size_handling(self, data_size):
        """
        Test: Process endpoint handles different data sizes appropriately
        Expected: Correct processing strategy based on data size
        """
        # Mock data of different sizes
        test_data = {
            "dataset_id": f"test_{data_size}",
            "data": json.dumps({"values": list(range(data_size))}),
            "preprocessing_config": {
                "cleaning": {"missing_value_strategy": "mean"}
            }
        }

        with patch('data.src.api.preprocessing_api.preprocessing_orchestrator') as mock_orchestrator:
            if data_size > 1000:
                expected_processing_time = 5000
            else:
                expected_processing_time = 500

            mock_orchestrator.estimate_processing_time.return_value = expected_processing_time

            estimated_time = estimate_processing_duration(test_data)
            assert isinstance(estimated_time, int)
            assert estimated_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
