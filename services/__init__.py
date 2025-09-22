"""Compatibility shim exposing data.src.services as a top-level package."""

from __future__ import annotations

from pathlib import Path

_service_path = (Path(__file__).resolve().parent.parent / "data" / "src" / "services").resolve()
__path__ = [str(_service_path)] if _service_path.exists() else []
