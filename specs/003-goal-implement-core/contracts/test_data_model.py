"""
Contract tests for data model validation.
These tests validate the data model contracts before implementation.
"""

import pytest
from datetime import datetime, date
from typing import Dict, Any, List
import json


class TestDataModelContract:
    """Test suite for validating data model contracts."""

    def test_asset_entity_validation(self):
        """Test Asset entity validation rules."""
        # Valid asset
        valid_asset = {
            "asset_id": "AAPL.US",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "Equity",
            "sector": "Technology",
            "country": "US",
            "currency": "USD",
            "min_data_points": 252,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        assert self._validate_asset(valid_asset)

        # Invalid asset_id format
        invalid_asset = valid_asset.copy()
        invalid_asset["asset_id"] = "INVALID_FORMAT"
        assert not self._validate_asset(invalid_asset)

        # Invalid asset_class
        invalid_asset = valid_asset.copy()
        invalid_asset["asset_class"] = "INVALID_CLASS"
        assert not self._validate_asset(invalid_asset)

        # Insufficient min_data_points
        invalid_asset = valid_asset.copy()
        invalid_asset["min_data_points"] = 100
        assert not self._validate_asset(invalid_asset)

    def test_forecast_entity_validation(self):
        """Test Forecast entity validation rules."""
        # Valid forecast
        valid_forecast = {
            "forecast_id": "forecast_123",
            "asset_id": "AAPL.US",
            "model_type": "ARIMA",
            "forecast_horizon": 30,
            "created_at": datetime.now().isoformat(),
            "parameters": {"p": 2, "d": 1, "q": 1},
            "forecast_values": [0.01, 0.02, 0.015],
            "confidence_intervals": {
                "0.95": [[-0.02, 0.04], [-0.01, 0.05], [-0.015, 0.045]]
            },
            "metrics": {
                "rmse": 0.02,
                "mae": 0.015,
                "direction_accuracy": 0.65
            }
        }

        assert self._validate_forecast(valid_forecast)

        # Invalid forecast_horizon
        invalid_forecast = valid_forecast.copy()
        invalid_forecast["forecast_horizon"] = 300  # Too large
        assert not self._validate_forecast(invalid_forecast)

        # Missing 95% confidence interval
        invalid_forecast = valid_forecast.copy()
        invalid_forecast["confidence_intervals"] = {"0.90": [[-0.01, 0.03]]}
        assert not self._validate_forecast(invalid_forecast)

        # Missing required metrics
        invalid_forecast = valid_forecast.copy()
        invalid_forecast["metrics"] = {"rmse": 0.02}  # Missing MAE and direction accuracy
        assert not self._validate_forecast(invalid_forecast)

    def test_volatility_forecast_validation(self):
        """Test VolatilityForecast entity validation rules."""
        # Valid volatility forecast
        valid_vol_forecast = {
            "volatility_id": "vol_123",
            "asset_id": "AAPL.US",
            "model_type": "GARCH",
            "forecast_horizon": 21,
            "created_at": datetime.now().isoformat(),
            "parameters": {"omega": 0.01, "alpha": 0.1, "beta": 0.85},
            "volatility_forecasts": [0.15, 0.16, 0.14],
            "long_run_variance": 0.025,
            "persistence": 0.95
        }

        assert self._validate_volatility_forecast(valid_vol_forecast)

        # Negative volatility forecast
        invalid_vol_forecast = valid_vol_forecast.copy()
        invalid_vol_forecast["volatility_forecasts"] = [-0.1, 0.16, 0.14]
        assert not self._validate_volatility_forecast(invalid_vol_forecast)

        # Negative long_run_variance
        invalid_vol_forecast = valid_vol_forecast.copy()
        invalid_vol_forecast["long_run_variance"] = -0.01
        assert not self._validate_volatility_forecast(invalid_vol_forecast)

        # Invalid persistence (outside [0,1])
        invalid_vol_forecast = valid_vol_forecast.copy()
        invalid_vol_forecast["persistence"] = 1.5
        assert not self._validate_volatility_forecast(invalid_forecast)

    def test_market_regime_validation(self):
        """Test MarketRegime entity validation rules."""
        # Valid market regime
        valid_regime = {
            "regime_id": "regime_1",
            "regime_name": "Bull",
            "regime_description": "Rising market with moderate volatility",
            "expected_duration": 120,
            "volatility_level": "Medium",
            "return_characteristics": {"mean": 0.08, "volatility": 0.12},
            "transition_probabilities": {
                "regime_1": 0.95,
                "regime_2": 0.04,
                "regime_3": 0.01
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        assert self._validate_market_regime(valid_regime)

        # Invalid transition probabilities (don't sum to 1)
        invalid_regime = valid_regime.copy()
        invalid_regime["transition_probabilities"] = {
            "regime_1": 0.8,
            "regime_2": 0.3,
            "regime_3": 0.1
        }
        assert not self._validate_market_regime(invalid_regime)

        # Invalid volatility_level
        invalid_regime = valid_regime.copy()
        invalid_regime["volatility_level"] = "INVALID"
        assert not self._validate_market_regime(invalid_regime)

    def test_regime_label_validation(self):
        """Test RegimeLabel entity validation rules."""
        # Valid regime label
        valid_label = {
            "label_id": "label_123",
            "asset_id": "AAPL.US",
            "regime_id": "regime_1",
            "date": date.today().isoformat(),
            "probability": 0.85,
            "confidence": 0.92,
            "features_used": {"returns": 0.02, "volatility": 0.15},
            "created_at": datetime.now().isoformat()
        }

        assert self._validate_regime_label(valid_label)

        # Invalid probability (outside [0,1])
        invalid_label = valid_label.copy()
        invalid_label["probability"] = 1.5
        assert not self._validate_regime_label(invalid_label)

        # Invalid confidence (outside [0,1])
        invalid_label = valid_label.copy()
        invalid_label["confidence"] = -0.1
        assert not self._validate_regime_label(invalid_label)

    def test_economic_scenario_validation(self):
        """Test EconomicScenario entity validation rules."""
        # Valid economic scenario
        valid_scenario = {
            "scenario_id": "scenario_1",
            "scenario_name": "Recession",
            "scenario_description": "Economic downturn with negative growth",
            "probability": 0.2,
            "economic_indicators": {
                "gdp": -0.02,
                "inflation": 0.01,
                "unemployment": 0.08
            },
            "duration_estimate": 6,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        assert self._validate_economic_scenario(valid_scenario)

        # Invalid probability (outside [0,1])
        invalid_scenario = valid_scenario.copy()
        invalid_scenario["probability"] = 1.5
        assert not self._validate_economic_scenario(invalid_scenario)

        # Missing required economic indicators
        invalid_scenario = valid_scenario.copy()
        invalid_scenario["economic_indicators"] = {"gdp": -0.02}
        assert not self._validate_economic_scenario(invalid_scenario)

    def test_scenario_impact_validation(self):
        """Test ScenarioImpact entity validation rules."""
        # Valid scenario impact
        valid_impact = {
            "impact_id": "impact_123",
            "scenario_id": "scenario_1",
            "asset_id": "AAPL.US",
            "expected_return": -0.15,
            "return_volatility": 0.25,
            "correlation_change": 0.1,
            "probability_adjustment": -0.05,
            "confidence_interval": {"lower": -0.25, "upper": -0.05},
            "created_at": datetime.now().isoformat()
        }

        assert self._validate_scenario_impact(valid_impact)

        # Unreasonable expected return
        invalid_impact = valid_impact.copy()
        invalid_impact["expected_return"] = -2.0  # -200% return
        assert not self._validate_scenario_impact(invalid_impact)

        # Negative volatility
        invalid_impact = valid_impact.copy()
        invalid_impact["return_volatility"] = -0.1
        assert not self._validate_scenario_impact(invalid_impact)

        # Invalid correlation change
        invalid_impact = valid_impact.copy()
        invalid_impact["correlation_change"] = 1.5
        assert not self._validate_scenario_impact(invalid_impact)

    def test_signal_validation_validation(self):
        """Test SignalValidation entity validation rules."""
        # Valid signal validation
        valid_validation = {
            "validation_id": "validation_123",
            "forecast_id": "forecast_123",
            "validation_type": "Statistical",
            "validation_metrics": {
                "p_value": 0.03,
                "test_statistic": 2.5,
                "critical_value": 1.96
            },
            "passed_checks": True,
            "warnings": ["High volatility detected"],
            "errors": [],
            "created_at": datetime.now().isoformat()
        }

        assert self._validate_signal_validation(valid_validation)

        # Invalid validation type
        invalid_validation = valid_validation.copy()
        invalid_validation["validation_type"] = "INVALID_TYPE"
        assert not self._validate_signal_validation(invalid_validation)

        # Missing significance test in metrics
        invalid_validation = valid_validation.copy()
        invalid_validation["validation_metrics"] = {"test_statistic": 2.5}
        assert not self._validate_signal_validation(invalid_validation)

    def test_entity_relationships(self):
        """Test entity relationship constraints."""
        # Test that foreign keys reference valid entities
        assets = [
            {"asset_id": "AAPL.US", "asset_class": "Equity"},
            {"asset_id": "MSFT.US", "asset_class": "Equity"}
        ]

        regimes = [
            {"regime_id": "regime_1", "regime_name": "Bull"},
            {"regime_id": "regime_2", "regime_name": "Bear"}
        ]

        # Valid regime label with proper foreign keys
        valid_label = {
            "asset_id": "AAPL.US",
            "regime_id": "regime_1"
        }
        assert self._validate_relationships(valid_label, assets, regimes)

        # Invalid asset reference
        invalid_label = {
            "asset_id": "INVALID.US",
            "regime_id": "regime_1"
        }
        assert not self._validate_relationships(invalid_label, assets, regimes)

        # Invalid regime reference
        invalid_label = {
            "asset_id": "AAPL.US",
            "regime_id": "INVALID_REGIME"
        }
        assert not self._validate_relationships(invalid_label, assets, regimes)

    def test_data_quality_constraints(self):
        """Test data quality and business rule validation."""
        # Test minimum data requirements
        data_points = 300  # Sufficient data
        assert self._validate_data_sufficiency(data_points)

        data_points = 200  # Insufficient data
        assert not self._validate_data_sufficiency(data_points)

        # Test missing value constraints
        missing_value_pct = 0.03  # Acceptable
        assert self._validate_missing_values(missing_value_pct)

        missing_value_pct = 0.08  # Too many missing values
        assert not self._validate_missing_values(missing_value_pct)

        # Test price and volume constraints
        price_data = [100.0, 101.5, 102.0]  # Valid prices
        assert self._validate_price_data(price_data)

        price_data = [-100.0, 101.5, 102.0]  # Invalid negative price
        assert not self._validate_price_data(price_data)

        volume_data = [1000000, 1200000, 900000]  # Valid volumes
        assert self._validate_volume_data(volume_data)

        volume_data = [1000000, -500000, 900000]  # Invalid negative volume
        assert not self._validate_volume_data(volume_data)

    # Helper validation methods (will be implemented with actual business logic)
    def _validate_asset(self, asset: Dict[str, Any]) -> bool:
        """Validate Asset entity."""
        # Placeholder implementation
        required_fields = ["asset_id", "symbol", "asset_class", "min_data_points"]

        if not all(field in asset for field in required_fields):
            return False

        # Validate asset_id format (SYMBOL.COUNTRY)
        if not "." in asset["asset_id"]:
            return False

        # Validate asset_class enum
        valid_classes = ["Equity", "Bond", "Commodity", "Currency", "ETF"]
        if asset["asset_class"] not in valid_classes:
            return False

        # Validate min_data_points
        if asset["min_data_points"] < 252:
            return False

        return True

    def _validate_forecast(self, forecast: Dict[str, Any]) -> bool:
        """Validate Forecast entity."""
        required_fields = ["forecast_id", "asset_id", "model_type",
                          "forecast_horizon", "forecast_values", "metrics"]

        if not all(field in forecast for field in required_fields):
            return False

        # Validate forecast_horizon
        if not (1 <= forecast["forecast_horizon"] <= 252):
            return False

        # Validate confidence_intervals include 95% level
        if "confidence_intervals" in forecast:
            if "0.95" not in forecast["confidence_intervals"]:
                return False

        # Validate required metrics
        required_metrics = ["rmse", "mae", "direction_accuracy"]
        if not all(metric in forecast["metrics"] for metric in required_metrics):
            return False

        return True

    def _validate_volatility_forecast(self, vol_forecast: Dict[str, Any]) -> bool:
        """Validate VolatilityForecast entity."""
        required_fields = ["volatility_id", "asset_id", "model_type",
                          "volatility_forecasts", "long_run_variance", "persistence"]

        if not all(field in vol_forecast for field in required_fields):
            return False

        # Validate all volatility forecasts are positive
        if any(v < 0 for v in vol_forecast["volatility_forecasts"]):
            return False

        # Validate long_run_variance is positive
        if vol_forecast["long_run_variance"] <= 0:
            return False

        # Validate persistence is between 0 and 1
        if not (0 <= vol_forecast["persistence"] <= 1):
            return False

        return True

    def _validate_market_regime(self, regime: Dict[str, Any]) -> bool:
        """Validate MarketRegime entity."""
        required_fields = ["regime_id", "regime_name", "expected_duration",
                          "transition_probabilities"]

        if not all(field in regime for field in required_fields):
            return False

        # Validate expected_duration is positive
        if regime["expected_duration"] <= 0:
            return False

        # Validate transition probabilities sum to 1
        trans_probs = regime["transition_probabilities"]
        if not abs(sum(trans_probs.values()) - 1.0) < 0.001:
            return False

        # Validate volatility_level enum
        valid_levels = ["Low", "Medium", "High"]
        if "volatility_level" in regime and regime["volatility_level"] not in valid_levels:
            return False

        return True

    def _validate_regime_label(self, label: Dict[str, Any]) -> bool:
        """Validate RegimeLabel entity."""
        required_fields = ["asset_id", "regime_id", "date", "probability", "confidence"]

        if not all(field in label for field in required_fields):
            return False

        # Validate probability and confidence are in [0,1]
        if not (0 <= label["probability"] <= 1):
            return False
        if not (0 <= label["confidence"] <= 1):
            return False

        return True

    def _validate_economic_scenario(self, scenario: Dict[str, Any]) -> bool:
        """Validate EconomicScenario entity."""
        required_fields = ["scenario_id", "scenario_name", "probability",
                          "economic_indicators"]

        if not all(field in scenario for field in required_fields):
            return False

        # Validate probability is in [0,1]
        if not (0 <= scenario["probability"] <= 1):
            return False

        # Validate required economic indicators
        required_indicators = ["gdp", "inflation", "unemployment"]
        indicators = scenario["economic_indicators"]
        if not all(indicator in indicators for indicator in required_indicators):
            return False

        return True

    def _validate_scenario_impact(self, impact: Dict[str, Any]) -> bool:
        """Validate ScenarioImpact entity."""
        required_fields = ["scenario_id", "asset_id", "expected_return", "return_volatility"]

        if not all(field in impact for field in required_fields):
            return False

        # Validate expected_return is reasonable
        if not (-0.5 <= impact["expected_return"] <= 1.0):  # -50% to +100%
            return False

        # Validate return_volatility is positive
        if impact["return_volatility"] <= 0:
            return False

        # Validate correlation_change is in [-1,1]
        if "correlation_change" in impact:
            if not (-1 <= impact["correlation_change"] <= 1):
                return False

        return True

    def _validate_signal_validation(self, validation: Dict[str, Any]) -> bool:
        """Validate SignalValidation entity."""
        required_fields = ["forecast_id", "validation_type", "validation_metrics", "passed_checks"]

        if not all(field in validation for field in required_fields):
            return False

        # Validate validation_type
        valid_types = ["Statistical", "Economic", "Backtest"]
        if validation["validation_type"] not in valid_types:
            return False

        # For statistical validation, ensure significance tests are present
        if validation["validation_type"] == "Statistical":
            metrics = validation["validation_metrics"]
            if "p_value" not in metrics:
                return False

        return True

    def _validate_relationships(self, entity: Dict[str, Any], assets: List[Dict],
                               regimes: List[Dict]) -> bool:
        """Validate entity relationships."""
        asset_ids = [asset["asset_id"] for asset in assets]
        regime_ids = [regime["regime_id"] for regime in regimes]

        if "asset_id" in entity and entity["asset_id"] not in asset_ids:
            return False

        if "regime_id" in entity and entity["regime_id"] not in regime_ids:
            return False

        return True

    def _validate_data_sufficiency(self, data_points: int) -> bool:
        """Validate sufficient historical data."""
        return data_points >= 252

    def _validate_missing_values(self, missing_pct: float) -> bool:
        """Validate acceptable missing value percentage."""
        return missing_pct <= 0.05

    def _validate_price_data(self, prices: List[float]) -> bool:
        """Validate price data constraints."""
        return all(price > 0 for price in prices)

    def _validate_volume_data(self, volumes: List[int]) -> bool:
        """Validate volume data constraints."""
        return all(volume >= 0 for volume in volumes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])