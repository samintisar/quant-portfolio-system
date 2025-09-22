from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..models.configuration import RiskConfiguration


class RiskConfigurationLoader:
    """Load base configuration profiles and apply overrides."""

    def __init__(self, defaults_path: Path | str | None = None) -> None:
        self.defaults_path = Path(defaults_path or "config/risk/defaults.json")
        self._data = json.loads(self.defaults_path.read_text(encoding="utf-8-sig"))

    def build(self, overrides: Optional[Dict[str, Any]] = None, profile: Optional[str] = None) -> RiskConfiguration:
        overrides = overrides or {}
        profile_name = profile or overrides.pop("profile", None) or self._data["defaults"].get("profile")
        if profile_name not in self._data["profiles"]:
            raise ValueError(f"Unknown configuration profile '{profile_name}'")

        profile_data = dict(self._data["profiles"][profile_name])
        payload: Dict[str, Any] = {}
        payload.update(profile_data)
        payload.update({k: v for k, v in overrides.items() if v is not None})

        for key, value in self._data.get("defaults", {}).items():
            payload.setdefault(key, value)

        payload.setdefault("stress_scenarios", [])
        payload.setdefault("covariance_methods", ["sample"])
        payload.setdefault("var_methods", ["historical"])
        payload.setdefault("cvar_methods", ["historical"])

        allowed = set(RiskConfiguration.__dataclass_fields__.keys())
        filtered = {key: value for key, value in payload.items() if key in allowed}
        return RiskConfiguration(**filtered)

    def available_profiles(self) -> Dict[str, str]:
        return {
            name: details.get("description", "")
            for name, details in self._data.get("profiles", {}).items()
        }

