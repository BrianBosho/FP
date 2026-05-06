"""Provenance bundle: captures the environment fingerprint at experiment time."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_info() -> dict[str, str | None]:
    """Return git commit SHA and dirty flag; silently returns None values if git unavailable."""
    info: dict[str, str | None] = {"commit": None, "branch": None, "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def _torch_info() -> dict[str, str | None]:
    info: dict[str, str | None] = {"torch": None, "cuda": None, "cuda_device": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            try:
                info["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except ImportError:
        pass
    return info


def build_provenance(config_hash: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a provenance dict capturing the current environment."""
    git = _git_info()
    torch_info = _torch_info()
    bundle: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
        "torch_version": torch_info["torch"],
        "cuda_version": torch_info["cuda"],
        "cuda_device": torch_info["cuda_device"],
        "config_hash": config_hash,
        "cwd": os.getcwd(),
    }
    if extra:
        bundle.update(extra)
    return bundle


def write_provenance(
    result_dir: str | Path,
    config_hash: str | None = None,
    extra: dict[str, Any] | None = None,
    filename: str = "provenance.json",
) -> Path:
    """Write a provenance.json into *result_dir* and return its path."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_provenance(config_hash=config_hash, extra=extra)
    out_path = result_dir / filename
    out_path.write_text(json.dumps(bundle, indent=2, default=str))
    return out_path
