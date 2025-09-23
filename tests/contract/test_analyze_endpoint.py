"""
Contract tests for POST /analyze endpoint.

These tests validate the API contract for portfolio analysis.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestAnalyzeEndpoint:
    """Test suite for the portfolio analysis endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        from portfolio.api.main import app
        self.client = TestClient(app)

    def test_analyze_endpoint_exists(self):
        """Test that the analyze endpoint exists."""
        response = self.client.post("/analyze", json={})
        # Should be 422 for empty request due to validation
        assert response.status_code == 422

    def test_analyze_with_valid_portfolio(self):
        """Test portfolio analysis with valid data."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "weights": {"AAPL": 0.6, "GOOGL": 0.4},
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02,
            "lookback_period": 252
        }

        response = self.client.post("/analyze", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "success" in result
        assert "weights" in result
        assert "execution_time" in result
        assert "timestamp" in result

    def test_analyze_with_invalid_weights(self):
        """Test analysis with invalid weights (don't sum to 1.0)."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "weights": {"AAPL": 0.8, "GOOGL": 0.3},  # Sums to 1.1
            "benchmark_symbol": "SPY"
        }

        response = self.client.post("/analyze", json=request_data)
        assert response.status_code == 422

    def test_analyze_with_missing_assets(self):
        """Test analysis with missing assets in weights."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "weights": {"AAPL": 0.6, "MSFT": 0.4},  # MSFT not in assets
            "benchmark_symbol": "SPY"
        }

        response = self.client.post("/analyze", json=request_data)
        assert response.status_code == 422

    def test_analyze_response_structure(self):
        """Test that analysis response has correct structure."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "weights": {"AAPL": 0.6, "GOOGL": 0.4},
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02
        }

        response = self.client.post("/analyze", json=request_data)
        assert response.status_code == 200

        result = response.json()

        # Check required fields
        required_fields = ["success", "weights", "execution_time", "timestamp"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check types
        assert isinstance(result["success"], bool)
        assert isinstance(result["weights"], dict)
        assert isinstance(result["execution_time"], (int, float))

    def test_analyze_without_required_fields(self):
        """Test analysis without required fields."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"}
            ]
            # Missing weights
        }

        response = self.client.post("/analyze", json=request_data)
        assert response.status_code == 422