"""Ensure project root is always available on sys.path for tests and tooling."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
_root_str = str(_project_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
