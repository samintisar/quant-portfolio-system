from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..models.risk_metric import RiskMetricEntry
from ..models.snapshot import PortfolioSnapshot
from ..models.visualization import VisualizationArtifact


def generate_visualizations(
    snapshot: PortfolioSnapshot,
    metrics: Sequence[RiskMetricEntry],
    output_dir: Path,
) -> List[VisualizationArtifact]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[VisualizationArtifact] = []
    generated_at = datetime.utcnow()

    artifacts.append(_render_factor_exposure(snapshot, output_dir, generated_at))
    var_metrics = [metric for metric in metrics if metric.metric == "var"]
    if var_metrics:
        artifacts.append(_render_var_series(var_metrics, output_dir, generated_at))
    cvar_metrics = [metric for metric in metrics if metric.metric == "cvar"]
    if cvar_metrics:
        artifacts.append(_render_cvar_distribution(cvar_metrics, output_dir, generated_at))

    return artifacts


def _render_factor_exposure(
    snapshot: PortfolioSnapshot,
    output_dir: Path,
    generated_at: datetime,
) -> VisualizationArtifact:
    if snapshot.factor_exposures is not None:
        exposures = snapshot.factor_exposures.sum(axis=1)
    else:
        exposures = snapshot.weights
    exposures = exposures.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    exposures.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Factor Exposure")
    ax.set_ylabel("Exposure")
    fig.tight_layout()

    path = output_dir / f"factor_exposure_{generated_at.strftime('%Y%m%d%H%M%S')}.png"
    fig.savefig(path, format="png")
    plt.close(fig)

    artifact = VisualizationArtifact(
        artifact_id=str(uuid4()),
        type="factor_exposure",
        path=path,
        format="png",
        generated_at=generated_at,
    )
    artifact.ensure_within(output_dir)
    return artifact


def _render_var_series(
    metrics: Sequence[RiskMetricEntry],
    output_dir: Path,
    generated_at: datetime,
) -> VisualizationArtifact:
    methods = [metric.method for metric in metrics]
    values = [metric.value for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(methods, values, marker="o", color="darkorange")
    ax.set_title("VaR by Method")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    path = output_dir / f"var_series_{generated_at.strftime('%Y%m%d%H%M%S')}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)

    artifact = VisualizationArtifact(
        artifact_id=str(uuid4()),
        type="var_timeseries",
        path=path,
        format="svg",
        generated_at=generated_at,
    )
    artifact.ensure_within(output_dir)
    return artifact


def _render_cvar_distribution(
    metrics: Sequence[RiskMetricEntry],
    output_dir: Path,
    generated_at: datetime,
) -> VisualizationArtifact:
    path = output_dir / f"cvar_distribution_{generated_at.strftime('%Y%m%d%H%M%S')}.html"
    rows = "".join(
        f"<tr><td>{metric.method}</td><td>{metric.confidence:.2%}</td><td>{metric.value:.6f}</td></tr>"
        for metric in metrics
    )
    html = (
        "<html><head><title>CVaR Distribution</title></head><body>"
        "<h2>CVaR by Method</h2>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<thead><tr><th>Method</th><th>Confidence</th><th>Loss</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></body></html>"
    )
    path.write_text(html)

    artifact = VisualizationArtifact(
        artifact_id=str(uuid4()),
        type="cvar_distribution",
        path=path,
        format="html",
        generated_at=generated_at,
    )
    artifact.ensure_within(output_dir)
    return artifact

