"""Staged experiment policy for statistical efficiency.

Three-stage structure:

  smoke  — 1 client, 1 round, 1 rep.  Validates wiring.
  pilot  — 1 rep per config.  Reject clearly bad configs early.
  full   — all seeds for promising / ambiguous configs.

Helpers also compute 95 % confidence intervals for summary tables
so result CSVs can show ``mean ± CI`` instead of bare averages.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

STAGE_SMOKE = "smoke"
STAGE_PILOT = "pilot"
STAGE_FULL  = "full"


# --------------------------------------------------------------------------
# Stage overrides
# --------------------------------------------------------------------------

def smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return overrides that shrink *cfg* to a fast smoke test."""
    clients = cfg.get("num_clients", 1)
    if isinstance(clients, (list, tuple)) and clients:
        clients = [1]
    else:
        clients = 1
    return {
        "num_rounds":             1,
        "epochs":                 1,
        "repetitions":            1,
        "num_clients":            clients,
        "use_wandb":              False,
        "wandb_mode":             "disabled",
        "save_results":           False,
        "early_stopping_patience": 9999,
        "debug":                  False,
    }


def pilot_overrides() -> dict[str, Any]:
    """Return overrides for a single-seed pilot run."""
    return {
        "repetitions": 1,
        "use_wandb":   False,
        "wandb_mode":  "disabled",
    }


def should_promote_to_full(
    pilot_acc: float,
    baseline_acc: float | None = None,
    *,
    min_acc_threshold: float = 0.0,
    headroom: float = 0.02,
) -> bool:
    """Return True when a pilot result is worth running with all seeds.

    Promotion conditions (either is sufficient):
    - ``pilot_acc`` exceeds ``min_acc_threshold`` (absolute floor).
    - ``pilot_acc`` is within ``headroom`` of ``baseline_acc``
      (i.e. not clearly worse than the best known result).
    """
    if pilot_acc < min_acc_threshold:
        return False
    if baseline_acc is None:
        return True
    return pilot_acc >= baseline_acc - headroom


# --------------------------------------------------------------------------
# Confidence intervals
# --------------------------------------------------------------------------

def ci_95(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return ``(mean, std, ci_low, ci_high)`` using the normal approximation.

    For *n* < 2 the CI collapses to [mean, mean].
    NaN values are silently dropped.
    """
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        nan = float("nan")
        return nan, nan, nan, nan
    n    = len(clean)
    mean = statistics.mean(clean)
    if n < 2:
        return mean, 0.0, mean, mean
    std = statistics.stdev(clean)
    se  = std / math.sqrt(n)
    hw  = 1.96 * se          # half-width at 95 %
    return mean, std, mean - hw, mean + hw


def format_ci(values: Sequence[float], digits: int = 4) -> str:
    """Return ``'mean ± half_CI'`` as a string, e.g. ``'0.8312 ± 0.0041'``."""
    mean, _, lo, hi = ci_95(values)
    if math.isnan(mean):
        return "n/a"
    hw = (hi - lo) / 2
    return f"{mean:.{digits}f} ± {hw:.{digits}f}"


def enrich_summary_with_ci(summary: dict[str, Any]) -> dict[str, Any]:
    """Add CI fields alongside the existing mean/std in a *summary* dict.

    Expects ``summary["global_results"]`` and ``summary["client_results"]``
    to be lists of floats (one per repetition).  Returns a copy with extra
    ``*_mean``, ``*_std``, ``*_ci95_low``, ``*_ci95_high``, ``*_ci95_str``,
    ``*_n`` keys added.
    """
    out = dict(summary)
    for key in ("global_results", "client_results"):
        vals = summary.get(key) or []
        if not vals:
            continue
        mean, std, lo, hi = ci_95(vals)
        prefix = key.replace("_results", "")
        out[f"{prefix}_mean"]       = round(mean, 6)
        out[f"{prefix}_std"]        = round(std,  6)
        out[f"{prefix}_ci95_low"]   = round(lo,   6)
        out[f"{prefix}_ci95_high"]  = round(hi,   6)
        out[f"{prefix}_ci95_str"]   = format_ci(vals)
        out[f"{prefix}_n"]          = len(vals)
    return out
