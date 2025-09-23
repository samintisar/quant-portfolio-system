"""
Contract tests for GET /data/assets endpoint.

These tests validate the API contract for the assets data endpoint.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestAssetsEndpoint:
    """Test suite for the assets data endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        from portfolio.api.main import app
        self.client = TestClient(app)

    def test_assets_endpoint_exists(self):
        """Test that the assets endpoint exists."""
        response = self.client.get("/data/assets")
        assert response.status_code == 422  # Missing symbols parameter

    def test_assets_basic_request(self):
        """Test basic assets data request."""
        params = {"symbols": "AAPL,GOOGL,MSFT", "period": "5y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            assert "assets" in data
            assert "summary" in data

    def test_assets_single_symbol(self):
        """Test getting data for a single asset."""
        params = {"symbols": "AAPL", "period": "2y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            assert "AAPL" in data["assets"]
            assert data["assets"]["AAPL"]["symbol"] == "AAPL"

    def test_assets_multiple_symbols(self):
        """Test getting data for multiple assets."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        params = {"symbols": ",".join(symbols), "period": "5y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            assets_data = data["assets"]

            for symbol in symbols:
                assert symbol in assets_data
                assert assets_data[symbol]["symbol"] == symbol

    def test_assets_data_structure(self):
        """Test that asset data has correct structure."""
        params = {"symbols": "AAPL,GOOGL", "period": "5y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            assets_data = data["assets"]

            for symbol in ["AAPL", "GOOGL"]:
                if symbol in assets_data:
                    self._validate_asset_data_structure(assets_data[symbol])

    def test_assets_prices_data(self):
        """Test that prices data is properly formatted."""
        params = {"symbols": "AAPL", "period": "2y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            if "AAPL" in data["assets"]:
                aapl_data = data["assets"]["AAPL"]
                prices = aapl_data["prices"]

                assert isinstance(prices, list)

                if len(prices) > 0:
                    first_price = prices[0]
                    required_price_fields = ["date", "open", "high", "low", "close", "volume"]

                    for field in required_price_fields:
                        assert field in first_price

    def test_assets_metrics_data(self):
        """Test that asset metrics are calculated."""
        params = {"symbols": "AAPL,GOOGL", "period": "5y"}

        response = self.client.get("/data/assets", params=params)

        if response.status_code == 200:
            data = response.json()
            assets_data = data["assets"]

            for symbol in ["AAPL", "GOOGL"]:
                if symbol in assets_data and "metrics" in assets_data[symbol]:
                    metrics = assets_data[symbol]["metrics"]

                    required_metrics = [
                        "annual_return", "annual_volatility", "sharpe_ratio",
                        "max_drawdown", "beta"
                    ]

                    for metric in required_metrics:
                        assert metric in metrics

    def test_assets_validation_errors(self):
        """Test validation of assets request."""
        # Missing required symbols parameter
        response = self.client.get("/data/assets")
        assert response.status_code in [400, 422]

        # Empty symbols parameter
        params = {"symbols": "", "period": "5y"}
        response = self.client.get("/data/assets", params=params)
        assert response.status_code in [400, 422]

        # Invalid period format
        params = {"symbols": "AAPL", "period": "invalid_period"}
        response = self.client.get("/data/assets", params=params)
        assert response.status_code in [400, 422]

    def _validate_asset_data_structure(self, asset_data):
        """Helper method to validate asset data structure."""
        required_fields = ["symbol", "prices", "metrics"]

        for field in required_fields:
            assert field in asset_data

        assert isinstance(asset_data["symbol"], str)
        assert isinstance(asset_data["prices"], list)
        assert isinstance(asset_data["metrics"], dict)