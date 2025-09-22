"""Service layer for preprocessing log retrieval.

Provides lightweight validation and accessors used by the CLI/API contract
suite. The implementation keeps storage in-memory for now and can be swapped
for a persistent backend when the real service lands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

_DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")


@dataclass
class LogEntry:
    """Simple representation of a preprocessing log entry."""

    log_id: str
    timestamp: datetime
    level: str
    operation: str
    message: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class LogsService:
    """Manage preprocessing logs and dataset validation."""

    def __init__(self) -> None:
        self._datasets: Dict[str, List[LogEntry]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_dataset(self, dataset_id: str, logs: Optional[Iterable[LogEntry]] = None) -> None:
        """Register a dataset so lookups and validations succeed."""

        if not self._is_valid_identifier(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        self._datasets.setdefault(dataset_id, [])
        if logs:
            self._datasets[dataset_id].extend(list(logs))

    def dataset_exists(self, dataset_id: str) -> bool:
        """Return True when the dataset is known to the service."""

        if not isinstance(dataset_id, str):
            return False
        return dataset_id in self._datasets

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_dataset_id(self, dataset_id: Any) -> None:
        """Validate dataset identifier and raise ValueError when invalid."""

        dataset_str = str(dataset_id)

        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(f"Dataset '{dataset_str}' not found.")

        if not self._is_valid_identifier(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        if not self.dataset_exists(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found.")

    def _is_valid_identifier(self, dataset_id: str) -> bool:
        return bool(_DATASET_ID_PATTERN.match(dataset_id))

    # ------------------------------------------------------------------
    # Retrieval helpers (currently simple placeholders)
    # ------------------------------------------------------------------
    def get_dataset_logs(self, dataset_id: str) -> Dict[str, Any]:
        """Return logs for a dataset.

        For now this returns a minimal structure so contract tests can patch
        over it with richer responses.
        """

        self.validate_dataset_id(dataset_id)
        logs = [
            entry.__dict__ | {"timestamp": entry.timestamp.isoformat()}
            for entry in self._datasets.get(dataset_id, [])
        ]
        return {
            "dataset_id": dataset_id,
            "logs": logs,
            "pagination": {"page": 1, "page_size": len(logs), "total_logs": len(logs)},
        }

    def get_filtered_logs(self, dataset_id: str, level: Optional[str] = None, **_: Any) -> Dict[str, Any]:
        """Return filtered logs for a dataset."""

        response = self.get_dataset_logs(dataset_id)
        if level:
            response["logs"] = [log for log in response["logs"] if log.get("level") == level]
        return response

    def stream_logs(self, dataset_id: str) -> Iterable[Dict[str, Any]]:
        """Yield logs for streaming endpoints."""

        for entry in self._datasets.get(dataset_id, []):
            yield entry.__dict__ | {"timestamp": entry.timestamp.isoformat()}


__all__ = ["LogsService", "LogEntry"]
