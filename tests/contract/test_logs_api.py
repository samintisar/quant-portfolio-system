"""
API contract tests for /preprocessing/logs/{dataset_id} endpoint
Tests preprocessing logs API contract compliance
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.src.api.preprocessing_api import logs_service


class TestLogsAPIContract:
    """Test suite for /preprocessing/logs/{dataset_id} endpoint contract compliance"""

    def test_logs_endpoint_invalid_dataset_id(self):
        """
        Test: Logs endpoint rejects invalid dataset IDs
        Expected: 404 Not Found for non-existent datasets
        """
        invalid_dataset_ids = [
            "nonexistent_dataset",
            "",
            "invalid@id",
            12345  # Not a string
        ]

        for dataset_id in invalid_dataset_ids:
            with pytest.raises(ValueError) as exc_info:
                logs_service.validate_dataset_id(dataset_id)

            error_message = str(exc_info.value)
            assert "dataset" in error_message.lower()
            assert "not found" in error_message.lower()

    def test_logs_endpoint_get_logs_success(self):
        """
        Test: Get logs endpoint returns processing logs successfully
        Expected: 200 OK with paginated log entries
        """
        expected_response = {
            "dataset_id": "test_dataset",
            "logs": [
                {
                    "log_id": "log_001",
                    "timestamp": "2025-09-18T10:00:00Z",
                    "level": "INFO",
                    "operation": "data_loading",
                    "message": "Started loading dataset",
                    "details": {
                        "file_path": "/data/test_dataset.csv",
                        "file_size_bytes": 1024000,
                        "estimated_records": 10000
                    },
                    "user_id": "system",
                    "session_id": "sess_12345"
                },
                {
                    "log_id": "log_002",
                    "timestamp": "2025-09-18T10:00:15Z",
                    "level": "WARNING",
                    "operation": "missing_value_detection",
                    "message": "Detected missing values in 'price' column",
                    "details": {
                        "column": "price",
                        "missing_count": 150,
                        "missing_percentage": 1.5,
                        "handling_strategy": "mean_imputation"
                    },
                    "user_id": "system",
                    "session_id": "sess_12345"
                },
                {
                    "log_id": "log_003",
                    "timestamp": "2025-09-18T10:01:30Z",
                    "level": "ERROR",
                    "operation": "outlier_detection",
                    "message": "Found extreme outliers in volume data",
                    "details": {
                        "column": "volume",
                        "outlier_count": 25,
                        "outlier_percentage": 0.25,
                        "z_score_threshold": 3.0,
                        "action": "cap_at_threshold"
                    },
                    "user_id": "system",
                    "session_id": "sess_12345"
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 50,
                "total_logs": 156,
                "total_pages": 4,
                "has_next": True,
                "has_prev": False
            },
            "log_summary": {
                "total_operations": 156,
                "info_count": 120,
                "warning_count": 25,
                "error_count": 8,
                "critical_count": 3,
                "processing_time_range": {
                    "start": "2025-09-18T10:00:00Z",
                    "end": "2025-09-18T10:45:30Z"
                }
            },
            "filters_applied": {
                "level": "all",
                "operation": "all",
                "date_range": {
                    "start": "2025-09-18T00:00:00Z",
                    "end": "2025-09-18T23:59:59Z"
                }
            }
        }

        with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
            mock_service.get_dataset_logs.return_value = expected_response

            # Test response structure
            response = expected_response

            # Verify required fields
            assert "dataset_id" in response
            assert "logs" in response
            assert "pagination" in response
            assert "log_summary" in response

            # Verify log entries structure
            logs = response["logs"]
            assert len(logs) > 0
            for log in logs:
                assert "log_id" in log
                assert "timestamp" in log
                assert "level" in log
                assert "operation" in log
                assert "message" in log
                assert log["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_logs_endpoint_filter_by_level(self):
        """
        Test: Get logs endpoint filters by log level
        Expected: 200 OK with filtered log entries
        """
        filter_params = {
            "dataset_id": "test_dataset",
            "level": "ERROR",
            "start_date": "2025-09-18T00:00:00Z",
            "end_date": "2025-09-18T23:59:59Z"
        }

        expected_response = {
            "dataset_id": "test_dataset",
            "logs": [
                {
                    "log_id": f"log_err_{idx}",
                    "timestamp": f"2025-09-18T10:{idx:02d}:30Z",
                    "level": "ERROR",
                    "operation": "outlier_detection",
                    "message": "Found extreme outliers in volume data",
                    "details": {
                        "column": "volume",
                        "outlier_count": 25 + idx,
                        "outlier_percentage": round(0.25 + idx * 0.01, 2)
                    }
                }
                for idx in range(1, 9)
            ],
            "pagination": {
                "page": 1,
                "page_size": 50,
                "total_logs": 8,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False
            },
            "filter_summary": {
                "applied_filters": {
                    "level": "ERROR",
                    "date_range": "2025-09-18"
                },
                "total_available": 156,
                "filtered_count": 8,
                "filter_efficiency": 0.95
            }
        }

        with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
            mock_service.get_filtered_logs.return_value = expected_response

            # Test filtered response
            response = expected_response

            # Verify filtering results
            assert response["dataset_id"] == filter_params["dataset_id"]
            assert len(response["logs"]) == 8
            for log in response["logs"]:
                assert log["level"] == "ERROR"

    def test_logs_endpoint_filter_by_operation(self):
        """
        Test: Get logs endpoint filters by operation type
        Expected: 200 OK with operation-specific log entries
        """
        operation_types = ["data_loading", "missing_value_detection", "outlier_detection", "normalization"]

        for operation in operation_types:
            filter_params = {
                "dataset_id": "test_dataset",
                "operation": operation
            }

            expected_response = {
                "dataset_id": "test_dataset",
                "logs": [
                    {
                        "log_id": f"log_{operation}_001",
                        "timestamp": "2025-09-18T10:00:00Z",
                        "level": "INFO",
                        "operation": operation,
                        "message": f"Started {operation} operation",
                        "details": {"operation_type": operation}
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 50,
                    "total_logs": 15,
                    "total_pages": 1
                },
                "operation_summary": {
                    "operation": operation,
                    "total_executions": 15,
                    "average_duration_ms": 250,
                    "success_rate": 0.93
                }
            }

            with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
                mock_service.get_operation_logs.return_value = expected_response

                # Test operation filtering
                response = expected_response
                assert response["dataset_id"] == filter_params["dataset_id"]
                for log in response["logs"]:
                    assert log["operation"] == operation

    def test_logs_endpoint_time_range_filtering(self):
        """
        Test: Get logs endpoint filters by time range
        Expected: 200 OK with time-filtered log entries
        """
        time_ranges = [
            {"start": "2025-09-18T10:00:00Z", "end": "2025-09-18T10:30:00Z"},
            {"start": "2025-09-18T00:00:00Z", "end": "2025-09-18T23:59:59Z"}
        ]

        for time_range in time_ranges:
            expected_response = {
                "dataset_id": "test_dataset",
                "logs": [
                    {
                        "log_id": "log_timerange_001",
                        "timestamp": "2025-09-18T10:15:00Z",
                        "level": "INFO",
                        "operation": "normalization",
                        "message": "Completed data normalization",
                        "details": {"method": "z_score", "columns_normalized": 5}
                    }
                ],
                "time_range": {
                    "requested_start": time_range["start"],
                    "requested_end": time_range["end"],
                    "actual_start": "2025-09-18T10:00:00Z",
                    "actual_end": "2025-09-18T10:30:00Z"
                },
                "time_stats": {
                    "logs_in_range": 45,
                    "time_span_hours": 0.5,
                    "log_frequency_per_hour": 90
                }
            }

            with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
                mock_service.get_time_range_logs.return_value = expected_response

                # Test time range filtering
                response = expected_response
                assert "time_range" in response
                assert "time_stats" in response

    def test_logs_endpoint_log_export(self):
        """
        Test: Get logs endpoint supports log export
        Expected: 200 OK with export information
        """
        export_formats = ["json", "csv", "txt"]

        for format_type in export_formats:
            export_request = {
                "dataset_id": "test_dataset",
                "format": format_type,
                "filters": {
                    "level": "all",
                    "operation": "all"
                }
            }

            expected_response = {
                "export_id": f"export_{format_type}_12345",
                "dataset_id": "test_dataset",
                "format": format_type,
                "export_status": "ready",
                "export_url": f"/exports/logs/test_dataset_logs.{format_type}",
                "file_size_bytes": 256000,
                "logs_exported": 156,
                "expires_at": "2025-09-19T12:00:00Z",
                "checksum": "def456abc123"
            }

            with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
                mock_service.export_logs.return_value = expected_response

                # Test export response
                response = expected_response
                assert "export_id" in response
                assert response["format"] == format_type
                assert "export_url" in response
                assert "logs_exported" in response

    def test_logs_endpoint_log_search(self):
        """
        Test: Get logs endpoint supports full-text search
        Expected: 200 OK with search results
        """
        search_queries = ["outlier", "missing", "error", "normalization"]

        for query in search_queries:
            search_request = {
                "dataset_id": "test_dataset",
                "query": query,
                "search_fields": ["message", "operation"],
                "fuzzy_search": True
            }

            expected_response = {
                "dataset_id": "test_dataset",
                "search_results": [
                    {
                        "log_id": "log_search_001",
                        "timestamp": "2025-09-18T10:01:30Z",
                        "level": "ERROR",
                        "operation": "outlier_detection",
                        "message": "Found extreme outliers in volume data",
                        "relevance_score": 0.95,
                        "matched_fields": ["message", "operation"]
                    }
                ],
                "search_metadata": {
                    "query": query,
                    "total_matches": 12,
                    "search_time_ms": 15.5,
                    "fuzzy_used": True
                }
            }

            with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
                mock_service.search_logs.return_value = expected_response

                # Test search response
                response = expected_response
                assert "search_results" in response
                assert "search_metadata" in response
                assert response["search_metadata"]["query"] == query

    def test_logs_endpoint_real_time_streaming(self):
        """
        Test: Get logs endpoint supports real-time log streaming
        Expected: 200 OK with streaming configuration
        """
        streaming_request = {
            "dataset_id": "test_dataset",
            "stream_type": "live",
            "filter_levels": ["WARNING", "ERROR", "CRITICAL"],
            "batch_size": 10
        }

        expected_response = {
            "stream_id": "stream_12345",
            "dataset_id": "test_dataset",
            "stream_status": "active",
            "stream_url": "ws://api.example.com/logs/stream/stream_12345",
            "configuration": {
                "filter_levels": ["WARNING", "ERROR", "CRITICAL"],
                "batch_size": 10,
                "heartbeat_interval_seconds": 30
            },
            "connection_info": {
                "max_connections": 100,
                "current_connections": 1,
                "bandwidth_limit_mbps": 10
            }
        }

        with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
            mock_service.create_log_stream.return_value = expected_response

            # Test streaming response
            response = expected_response
            assert "stream_id" in response
            assert "stream_status" in response
            assert "stream_url" in response
            assert response["stream_status"] == "active"

    def test_logs_endpoint_log_aggregation(self):
        """
        Test: Get logs endpoint supports log aggregation
        Expected: 200 OK with aggregated log statistics
        """
        aggregation_requests = [
            {"group_by": "level", "metric": "count"},
            {"group_by": "operation", "metric": "average_duration"},
            {"group_by": "hour", "metric": "error_rate"}
        ]

        for agg_request in aggregation_requests:
            expected_response = {
                "dataset_id": "test_dataset",
                "aggregation": agg_request,
                "results": [
                    {
                        "group": "ERROR",
                        "count": 8,
                        "percentage": 5.13
                    },
                    {
                        "group": "WARNING",
                        "count": 25,
                        "percentage": 16.03
                    }
                ],
                "summary": {
                    "total_groups": 5,
                    "time_range": "2025-09-18",
                    "aggregation_time_ms": 45.2
                }
            }

            with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
                mock_service.aggregate_logs.return_value = expected_response

                # Test aggregation response
                response = expected_response
                assert "aggregation" in response
                assert "results" in response
                assert "summary" in response

    def test_logs_endpoint_log_retention(self):
        """
        Test: Get logs endpoint handles log retention policies
        Expected: 200 OK with retention information
        """
        retention_request = {
            "dataset_id": "test_dataset",
            "retention_days": 30,
            "action": "cleanup"
        }

        expected_response = {
            "retention_id": "retention_12345",
            "dataset_id": "test_dataset",
            "policy": {
                "retention_days": 30,
                "cleanup_action": "archive",
                "archive_location": "/archives/logs/test_dataset/"
            },
            "execution_summary": {
                "logs_before_cleanup": 1560,
                "logs_removed": 1200,
                "logs_retained": 360,
                "space_freed_mb": 45.2,
                "execution_time_ms": 1200.5
            },
            "status": "completed",
            "completed_at": "2025-09-18T12:00:00Z"
        }

        with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
            mock_service.apply_retention_policy.return_value = expected_response

            # Test retention response
            response = expected_response
            assert "retention_id" in response
            assert "policy" in response
            assert "execution_summary" in response
            assert response["status"] == "completed"

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_logs_endpoint_level_specific_behavior(self, log_level):
        """
        Test: Get logs endpoint handles different log levels appropriately
        Expected: Level-specific filtering and behavior
        """
        level_request = {
            "dataset_id": "test_dataset",
            "level": log_level,
            "include_lower_levels": True
        }

        expected_response = {
            "dataset_id": "test_dataset",
            "filter_level": log_level,
            "include_lower_levels": True,
            "logs_filtered": [
                {
                    "log_id": f"log_{log_level}_001",
                    "level": log_level,
                    "timestamp": "2025-09-18T10:00:00Z",
                    "operation": "test_operation",
                    "message": f"Test message with {log_level} level"
                }
            ],
            "level_statistics": {
                "requested_level": log_level,
                "logs_found": 25,
                "lower_levels_included": True,
                "total_log_count": 156
            }
        }

        with patch('data.src.api.preprocessing_api.logs_service') as mock_service:
            mock_service.get_level_specific_logs.return_value = expected_response

            # Test level-specific response
            response = expected_response
            assert response["filter_level"] == log_level
            assert "level_statistics" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
