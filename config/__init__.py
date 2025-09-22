"""Compatibility shim exposing data.src.config as a top-level package."""

from __future__ import annotations

from pathlib import Path

_config_path = (Path(__file__).resolve().parent.parent / "data" / "src" / "config").resolve()
__path__ = [str(_config_path)] if _config_path.exists() else []
