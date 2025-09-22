from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from ..models.report import RiskReport


@dataclass(slots=True)
class StoredReportRecord:
    """Structured metadata describing a persisted risk report."""

    report_id: str
    json_path: Path
    metrics_path: Optional[Path]
    generated_at: str
    confidence_levels: List[float]
    horizons: List[int]
    stress_scenarios: List[str]
    reports: List[str]

    def to_json(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "json_path": self.json_path.as_posix(),
            "metrics_path": self.metrics_path.as_posix() if self.metrics_path else None,
            "generated_at": self.generated_at,
            "confidence_levels": list(self.confidence_levels),
            "horizons": list(self.horizons),
            "stress_scenarios": list(self.stress_scenarios),
            "reports": list(self.reports),
        }

    def to_parquet_row(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "json_path": self.json_path.as_posix(),
            "metrics_path": self.metrics_path.as_posix() if self.metrics_path else None,
            "generated_at": self.generated_at,
            "confidence_levels": ",".join(str(level) for level in self.confidence_levels),
            "horizons": ",".join(str(horizon) for horizon in self.horizons),
            "stress_scenarios": ",".join(self.stress_scenarios),
            "reports": ",".join(self.reports),
        }


class ReportStore:
    """Persistence helper that manages risk report artifacts and index files."""

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._json_index = self.base_path / "index.json"
        self._parquet_index = self.base_path / "index.parquet"

    def save(self, report: RiskReport) -> StoredReportRecord:
        json_path = self._write_json_report(report)
        metrics_path = self._write_metrics_parquet(report)
        record = StoredReportRecord(
            report_id=report.report_id,
            json_path=json_path,
            metrics_path=metrics_path,
            generated_at=report.generated_at.isoformat(),
            confidence_levels=report.configuration.confidence_levels,
            horizons=report.configuration.horizons,
            stress_scenarios=report.configuration.stress_scenarios,
            reports=report.configuration.reports,
        )
        self._upsert_index(record)
        return record

    def load(self, report_id: str) -> Dict[str, Any]:
        path = self.base_path / f"{report_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Report '{report_id}' not found under {self.base_path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_metrics(self, report_id: str) -> pd.DataFrame:
        path = self.base_path / f"{report_id}_metrics.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Metrics parquet for '{report_id}' not found under {self.base_path}")
        return pd.read_parquet(path)

    def load_index(self) -> List[Dict[str, Any]]:
        if not self._json_index.exists():
            return []
        try:
            return json.loads(self._json_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def list_records(self) -> List[StoredReportRecord]:
        """Return persisted report metadata as structured records."""

        return [StoredReportRecord(**data) for data in self._iter_index_records()]

    def get_record(self, report_id: str) -> StoredReportRecord:
        for record in self.list_records():
            if record.report_id == report_id:
                return record
        raise FileNotFoundError(f"Report metadata '{report_id}' not found in index")

    def _write_json_report(self, report: RiskReport) -> Path:
        path = self.base_path / f"{report.report_id}.json"
        payload = report.to_dict()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_metrics_parquet(self, report: RiskReport) -> Optional[Path]:
        if not report.risk_metrics:
            return None
        rows = [
            {
                "report_id": report.report_id,
                "metric": metric.metric,
                "method": metric.method,
                "confidence": metric.confidence,
                "horizon": metric.horizon,
                "value": metric.value,
                "sample_size": metric.sample_size,
                "metadata": json.dumps(metric.metadata, sort_keys=True),
            }
            for metric in report.risk_metrics
        ]
        frame = pd.DataFrame(rows)
        metrics_path = self.base_path / f"{report.report_id}_metrics.parquet"
        frame.to_parquet(metrics_path, index=False)
        return metrics_path

    def _upsert_index(self, record: StoredReportRecord) -> None:
        records = [
            StoredReportRecord(**self._coerce_record(raw))
            for raw in self._load_json_records()
            if raw.get("report_id") != record.report_id
        ]
        records.append(record)
        records.sort(key=lambda item: item.generated_at, reverse=True)
        self._write_json_records([item.to_json() for item in records])
        self._write_parquet_records([item.to_parquet_row() for item in records])

    def _coerce_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "report_id": raw["report_id"],
            "json_path": Path(raw["json_path"]),
            "metrics_path": Path(raw["metrics_path"]) if raw.get("metrics_path") else None,
            "generated_at": raw["generated_at"],
            "confidence_levels": [float(value) for value in raw.get("confidence_levels", [])],
            "horizons": [int(value) for value in raw.get("horizons", [])],
            "stress_scenarios": list(raw.get("stress_scenarios", [])),
            "reports": list(raw.get("reports", [])),
        }

    def _load_json_records(self) -> List[Dict[str, Any]]:
        if not self._json_index.exists():
            return []
        try:
            return json.loads(self._json_index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _write_json_records(self, records: List[Dict[str, Any]]) -> None:
        self._json_index.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _write_parquet_records(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            if self._parquet_index.exists():
                self._parquet_index.unlink()
            return
        frame = pd.DataFrame(rows)
        frame.to_parquet(self._parquet_index, index=False)

    def _iter_index_records(self) -> Iterable[Dict[str, Any]]:
        for raw in self._load_json_records():
            yield self._coerce_record(raw)


__all__ = ["ReportStore", "StoredReportRecord"]
