from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..models.snapshot import PortfolioSnapshot
from ..models.stress_impact import StressImpact
from ..models.stress_scenario import StressScenario


def evaluate_stress_scenarios(
    snapshot: PortfolioSnapshot,
    scenarios: Iterable[StressScenario],
) -> List[StressImpact]:
    """Apply stress scenarios and return impact summaries.

    The implementation keeps calculations simple yet deterministic by scaling
    scenario parameters against portfolio volatility. Expected loss proxies the
    average contribution magnitude, while worst-case adds a conservative buffer.
    """

    impacts: List[StressImpact] = []
    portfolio_returns = snapshot.returns.to_numpy() @ snapshot.weights.to_numpy()
    volatility = float(np.std(portfolio_returns, ddof=1)) if portfolio_returns.size else 0.0
    volatility = max(volatility, 1e-3)

    for scenario in scenarios:
        contributions = _parameter_contributions(scenario, volatility)
        expected_loss = float(sum(contributions.values()))
        worst_case_loss = expected_loss * 1.5
        impacts.append(
            StressImpact(
                scenario_id=scenario.scenario_id,
                expected_loss=expected_loss,
                worst_case_loss=worst_case_loss,
                factor_contributions=contributions,
                notes=scenario.description,
            )
        )
    return impacts


def _parameter_contributions(scenario: StressScenario, scale: float) -> dict[str, float]:
    contributions: dict[str, float] = {}
    for key, value in scenario.parameters.items():
        magnitude = abs(float(value))
        contributions[key] = magnitude * scale
    if not contributions:
        contributions[scenario.scenario_id] = scale
    return contributions

