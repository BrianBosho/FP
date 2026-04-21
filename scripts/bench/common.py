"""Shared helpers for the FL benchmark harness.

Kept intentionally small and stdlib-heavy so it runs anywhere the project runs.
No imports from `src/` so it can be invoked without triggering heavy deps
(torch, Ray, PyG) when we only want to aggregate or compare results.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Environment / provenance                                                    #
# --------------------------------------------------------------------------- #

def git_commit(cwd: Path) -> Dict[str, Any]:
    """Best-effort git provenance.  Never raises."""
    out: Dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        )
        out["dirty"] = bool(diff.strip())
    except Exception:
        pass
    return out


def env_probe() -> Dict[str, Any]:
    """Capture only things that plausibly affect numerical results."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    # Avoid importing heavy libraries here (torch/ray can block on CUDA init in
    # some environments). Prefer package metadata.
    try:
        from importlib.metadata import version as pkg_version  # py>=3.8
    except Exception:
        pkg_version = None  # type: ignore

    for pkg in ("torch", "torch-geometric", "torch-sparse", "numpy", "scipy",
                "pandas", "ray", "omegaconf"):
        try:
            info[pkg] = pkg_version(pkg) if pkg_version is not None else None
        except Exception:
            info[pkg] = None

    # GPU info (best-effort). Only query nvidia-smi; never import torch here.
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["nvidia_smi"] = [line.strip() for line in smi.splitlines() if line.strip()]
        info["cuda_available"] = bool(info["nvidia_smi"])
    except Exception:
        info["nvidia_smi"] = None
        info["cuda_available"] = False
    return info


# --------------------------------------------------------------------------- #
# Run record shape                                                            #
# --------------------------------------------------------------------------- #

# Keys that uniquely identify a single experiment (one combination) across seeds.
EXPERIMENT_KEY_FIELDS: Tuple[str, ...] = (
    "dataset", "data_loading_option", "model_type",
    "num_clients", "beta", "hop", "use_pe", "fulltraining_flag",
)


@dataclass
class RunRow:
    """One row per (variant, seed, experiment_combination)."""
    variant: str
    seed: int
    dataset: str
    data_loading_option: str
    model_type: str
    num_clients: int
    beta: float
    hop: int
    use_pe: bool
    fulltraining_flag: bool
    avg_global_acc: float
    avg_client_acc: float
    std_global: float
    std_client: float
    num_rounds_recorded: int
    duration_s: float
    results_json_path: str


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #

_RESULTS_JSON_RE = re.compile(r"^results_.*\.json$")


def iter_results_json(root: Path) -> Iterable[Path]:
    """Yield all `results_*.json` files written by run_experiments."""
    for p in root.rglob("*.json"):
        if _RESULTS_JSON_RE.match(p.name):
            yield p


def parse_results_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single run_experiments result file; return a flat dict or None."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    cfg = data.get("experiment_config", {})
    summary = data.get("summary", {})
    duration = data.get("duration", {})
    rounds = data.get("rounds", [])

    flat: Dict[str, Any] = {
        "dataset": cfg.get("dataset"),
        "data_loading_option": cfg.get("data_loading_option"),
        "model_type": cfg.get("model_type"),
        "num_clients": cfg.get("num_clients"),
        "beta": cfg.get("beta"),
        "hop": cfg.get("hop"),
        "use_pe": bool(cfg.get("use_pe", False)),
        "fulltraining_flag": bool(cfg.get("fulltraining_flag", False)),
        "avg_global_acc": summary.get("average_global_result"),
        "avg_client_acc": summary.get("average_client_result"),
        "std_global": summary.get("std_global"),
        "std_client": summary.get("std_client"),
        "num_rounds_recorded": len(rounds),
        "duration_s": duration.get("seconds"),
        "results_json_path": str(path),
        "rounds": rounds,  # full round-by-round for curve plotting
    }
    return flat


def collect_variant_rows(variant_dir: Path, variant: str) -> List[Dict[str, Any]]:
    """Walk a variant run directory (bench/runs/<variant>_<ts>/) and return
    one row per (seed, experiment_combination).  Skips seeds whose results
    didn't parse so a partial run is still usable."""
    rows: List[Dict[str, Any]] = []
    for seed_dir in sorted(p for p in variant_dir.glob("seed_*") if p.is_dir()):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        for rjson in iter_results_json(seed_dir):
            flat = parse_results_json(rjson)
            if flat is None:
                continue
            flat["variant"] = variant
            flat["seed"] = seed
            rows.append(flat)
    return rows


def experiment_key(row: Dict[str, Any]) -> Tuple:
    return tuple(row.get(k) for k in EXPERIMENT_KEY_FIELDS)


def experiment_key_str(row: Dict[str, Any]) -> str:
    parts = [f"{k}={row.get(k)}" for k in EXPERIMENT_KEY_FIELDS]
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Small utilities                                                             #
# --------------------------------------------------------------------------- #

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    import csv
    if not rows:
        path.write_text("")
        return
    cols = columns or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Parse `key=value` pairs.  Values are parsed as YAML so lists/bools work.

    Examples::

        lr=0.1                    -> {'lr': 0.1}
        num_clients=[3]           -> {'num_clients': [3]}
        data_loading=[zero_hop]   -> {'data_loading': ['zero_hop']}
        aggregation=fedavg_weighted -> {'aggregation': 'fedavg_weighted'}
    """
    import yaml  # type: ignore
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got: {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k.strip()] = yaml.safe_load(v)
        except Exception:
            out[k.strip()] = v
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    import math
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Returns (t_stat, p_value) for Welch's two-sample t-test.  Falls back
    to (nan, nan) if scipy isn't available or samples are too small."""
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return float("nan"), float("nan")
    a_clean = [x for x in a if x is not None]
    b_clean = [x for x in b if x is not None]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return float("nan"), float("nan")
    res = stats.ttest_ind(a_clean, b_clean, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def cohens_d(a: List[float], b: List[float]) -> float:
    import math
    a_clean = [x for x in a if x is not None]
    b_clean = [x for x in b if x is not None]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return float("nan")
    ma, sa = mean_std(a_clean)
    mb, sb = mean_std(b_clean)
    na, nb = len(a_clean), len(b_clean)
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    if pooled == 0.0:
        return float("nan")
    return (mb - ma) / pooled
