"""
Contract tests for POST /optimize endpoint.

These tests validate the API contract for portfolio optimization.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestOptimizeEndpoint:
    """Test suite for the portfolio optimization endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        from portfolio.api.main import app
        self.client = TestClient(app)

    def test_optimize_endpoint_exists(self):
        """Test that the optimize endpoint exists."""
        response = self.client.post("/optimize", json={})
        # Should be 422 for empty request due to validation
        assert response.status_code == 422

    def test_optimize_with_mean_variance_method(self):
        """Test optimization with mean-variance method."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "method": "mean_variance",
            "objective": "sharpe",
            "constraints": {
                "max_position_size": 0.3,
                "max_sector_concentration": 0.6,
                "max_volatility": 0.25,
                "risk_free_rate": 0.02
            },
            "lookback_period": 252
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert "success" in result
        assert "method" in result
        assert "objective" in result
        assert "optimal_weights" in result
        assert "execution_time" in result
        assert "timestamp" in result

    def test_optimize_with_insufficient_assets(self):
        """Test optimization with insufficient assets (should fail validation)."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"}
            ],
            "method": "mean_variance",
            "objective": "sharpe"
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 422

    def test_optimize_with_cvar_method(self):
        """Test optimization with CVaR method."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "JPM", "name": "JP Morgan", "sector": "Financial"}
            ],
            "method": "cvar",
            "objective": "min_cvar",
            "constraints": {
                "max_position_size": 0.4,
                "max_volatility": 0.25,
                "risk_free_rate": 0.02
            }
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"]
        assert result["method"] == "cvar"

    def test_optimize_with_black_litterman_method(self):
        """Test optimization with Black-Litterman method."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "method": "black_litterman",
            "objective": "sharpe",
            "market_views": [
                {
                    "symbol": "AAPL",
                    "view_type": "absolute",
                    "confidence": 0.7,
                    "expected_return": 0.12
                }
            ],
            "constraints": {
                "max_position_size": 0.5,
                "max_volatility": 0.25,
                "risk_free_rate": 0.02
            }
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"]
        assert result["method"] == "black_litterman"

    def test_optimize_response_structure(self):
        """Test that optimization response has correct structure."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "method": "mean_variance",
            "objective": "sharpe",
            "constraints": {
                "max_position_size": 0.5,
                "max_volatility": 0.25,
                "risk_free_rate": 0.02
            }
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 200

        result = response.json()

        # Check required fields
        required_fields = ["success", "method", "objective", "optimal_weights", "execution_time", "timestamp"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        # Check types
        assert isinstance(result["success"], bool)
        assert isinstance(result["method"], str)
        assert isinstance(result["objective"], str)
        assert isinstance(result["optimal_weights"], dict)
        assert isinstance(result["execution_time"], (int, float))

        # Check weights sum to approximately 1.0
        weights_sum = sum(result["optimal_weights"].values())
        assert abs(weights_sum - 1.0) < 0.01, f"Weights sum to {weights_sum}, not 1.0"

    def test_optimize_with_invalid_method(self):
        """Test optimization with invalid method."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"},
                {"symbol": "GOOGL", "name": "Google", "sector": "Technology"}
            ],
            "method": "invalid_method",
            "objective": "sharpe"
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 422

    def test_optimize_without_required_fields(self):
        """Test optimization without required fields."""
        request_data = {
            "assets": [
                {"symbol": "AAPL", "name": "Apple", "sector": "Technology"}
            ]
            # Missing method and objective
        }

        response = self.client.post("/optimize", json=request_data)
        assert response.status_code == 422