#!/usr/bin/env python3
"""Scalability log parsing entrypoint (runs ``experiments/parse_logs.py`` implementation)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_fp_parse_logs", Path(__file__).resolve().parent / "parse_logs.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    _mod.main()
