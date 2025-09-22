from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from ..models.stress_scenario import StressScenario


@dataclass
class ScenarioCatalog:
    scenarios: Dict[str, StressScenario]
    catalogs: Dict[str, List[str]]

    @classmethod
    def from_path(cls, path: Path) -> "ScenarioCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        scenarios = {
            item["scenario_id"]: StressScenario(
                scenario_id=item["scenario_id"],
                name=item.get("name", item["scenario_id"]),
                description=item.get("description", ""),
                shock_type=item.get("shock_type", "macro"),
                parameters=item.get("parameters", {}),
                horizon=item.get("horizon", 1),
                tags=tuple(item.get("tags", [])),
            )
            for item in data.get("scenarios", [])
        }
        catalogs = {name: list(ids) for name, ids in data.get("catalogs", {}).items()}
        return cls(scenarios=scenarios, catalogs=catalogs)

    def get(self, scenario_id: str) -> StressScenario:
        try:
            return self.scenarios[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Scenario '{scenario_id}' not found") from exc

    def ensure_ids_exist(self, scenario_ids: Iterable[str]) -> None:
        missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in self.scenarios]
        if missing:
            raise ValueError(f"Unknown stress scenarios: {missing}")

    def get_catalog(self, name: str) -> List[StressScenario]:
        if name not in self.catalogs:
            raise KeyError(f"Catalog '{name}' is not defined")
        return [self.get(scenario_id) for scenario_id in self.catalogs[name]]

    def iter_selected(self, scenario_ids: Iterable[str]) -> Iterator[StressScenario]:
        for scenario_id in scenario_ids:
            yield self.get(scenario_id)

